import torch
import cv2
import os
import time
from typing import NamedTuple
from utils import out_of_room
import data_utils
from edge_utils import (
    generate_sphere_pts,
    extract_img_line,
    filterEdgeByTopK,
    extract_principal_2d,
)
import warnings
from itertools import permutations
from fgpl.pose_estimation import (
    xdf_cost,
    refine_from_sphere_icp,
)
from log_utils import save_logger, PoseLogger
from fgpl.line_intersection import intersections_2d, intersections_3d, intersections_idx
import numpy as np
from fgpl.xdf_canonical_precompute import XDFCanonicalPrecompute
from map_utils import generate_line_map


warnings.filterwarnings("ignore", category=UserWarning)


def localize(cfg: NamedTuple, log_dir: str):
    """
    Main function for performing localization.

    Args:
        cfg: Config file
        log_dir: Directory in which logs will be saved
    
    Returns:
        None
    """
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    dataset = cfg.dataset
    supported_datasets = ['omniscenes', 'stanford']  # Currently supported datasets

    if dataset not in supported_datasets:
        raise ValueError("Invalid dataset")

    out_of_room_quantile = getattr(cfg, 'out_of_room_quantile', 0.05)

    # Dataset configs
    if dataset == 'omniscenes':
        room_name = getattr(cfg, 'room_name', None)
        scene_number = getattr(cfg, 'scene_number', None)
        split_name = getattr(cfg, 'split_name', 'extreme')

    elif dataset == 'stanford':
        room_name = getattr(cfg, 'room_name', None)
        scene_number = None
        split_name = getattr(cfg, 'split_name', None)

    # Algorithm configs
    top_k_candidate = getattr(cfg, 'top_k_candidate', 1)  # Number of poses per each room used directly for refinement
    sample_per_room = getattr(cfg, 'sample_per_room', top_k_candidate)  # Number of poses per each room to average during room selection
    selection_num_room = getattr(cfg, 'selection_num_room', 1)  # Number of rooms to select for refinement
    inlier_thres_2d = getattr(cfg, 'inlier_thres_2d', 0.5)
    intersect_thres_2d = getattr(cfg, 'intersect_thres_2d', 0.1)
    point_gamma = getattr(cfg, 'point_gamma', 0.2)
    loc_exp_mode = getattr(cfg, 'loc_exp_mode', 'single_room')

    # Multi room setup
    if dataset in supported_datasets:
        default_room_list = data_utils.get_room_list(dataset, split_name=split_name)

        # Optionally limit room names
        if getattr(cfg, 'room_name_list', None) is not None:
            full_room_list = [rm for rm in default_room_list if rm in cfg.room_name_list]
        elif room_name is not None:
            full_room_list = [rm for rm in default_room_list if room_name in rm]
        else:
            full_room_list = default_room_list

        # Additionally filter by split_name for stanford dataset
        if dataset == 'stanford':
            if split_name is not None:
                full_room_list = [rm for rm in full_room_list if split_name in rm]

        if getattr(cfg, 'exclude_room', None) is not None:
            if isinstance(cfg.exclude_room, str):
                exclude_room_list = [cfg.exclude_room]
            else:
                exclude_room_list = cfg.exclude_room
            full_room_list = [rm for rm in full_room_list if all([exc not in rm for exc in exclude_room_list])]

    else:
        raise NotImplementedError("Other datasets not supported")

    # Generate list of query and mapping rooms
    query_room_list = []
    map_room_list = []
    if loc_exp_mode == 'single_room':  # Single room localization
        room_list = full_room_list
        for q_id in range(len(room_list)):
            query_room_list.append(room_list[q_id])
            map_room_list.append([room_list[q_id]])
    else:  # Multi room localization
        num_multi_rooms = getattr(cfg, 'num_multi_rooms', 4)
        if num_multi_rooms == -1:  # Use all rooms as maps
            num_multi_rooms = len(full_room_list)

        room_perm = np.random.permutation(len(full_room_list))[:num_multi_rooms]
        room_list = [full_room_list[i] for i in room_perm]

        multi_room_repeats = getattr(cfg, 'multi_room_repeats', 5)
        if multi_room_repeats == -1:  # Evaluate for all rooms
            multi_room_repeats = len(room_list)

        for q_id in range(multi_room_repeats):
            query_room_list.append(room_list[q_id])
            map_room_list.append(room_list)

    # Optimization configs
    opt_dict = {
        'optimizer': getattr(cfg, 'optimizer', 'Adam'),
        'total_iter': getattr(cfg, 'total_iter', 100),
        'step_size': getattr(cfg, 'step_size', 0.1),
        'decay_patience': getattr(cfg, 'decay_patience', 5),
        'decay_factor': getattr(cfg, 'decay_factor', 0.9),
        'nn_dist_thres': getattr(cfg, 'nn_dist_thres', 0.5),
        'match_thres': getattr(cfg, 'match_thres', 0),
    }

    save_scores = getattr(cfg, 'save_scores', False)

    logger = PoseLogger(log_dir)

    filenames = data_utils.get_filename(dataset, split_name=split_name)

    if dataset == 'omniscenes':
        filenames = [file_name for file_name in filenames if any([rm in file_name for rm in full_room_list])]
    else:
        filenames = [file_name for file_name in filenames if any([rm.split('/')[0] in file_name and rm.split('/')[1] in file_name for rm in full_room_list])]

    if scene_number is not None:
        filenames = [file_name for file_name in filenames if "scene_{}".format(scene_number) in file_name]

    # Track localization
    well_posed = 0
    valid_trial = 0

    # Track room estimation
    well_selected = 0

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Set configs for principal 2d, 3d direction detection
    num_principal = 3
    perms = list(permutations(range(num_principal)))
    perms = torch.tensor(perms, device=device, dtype=torch.long)
    bin_mask = torch.ones([len(perms) * 4, 3, 1], device=device)  # Ambiguity in permutation and sign

    for perm_idx, perm in enumerate(perms):
        for idx in range(4):
            bin_mask[perm_idx * 4 + idx, 0, 0] = (-1) ** (idx // 2)
            bin_mask[perm_idx * 4 + idx, 1, 0] = (-1) ** (idx % 2)
            bin_mask[perm_idx * 4 + idx, 2, 0] = (-1) ** (idx // 2 + idx % 2)
            if perm_idx in [1, 2, 5]:
                bin_mask[perm_idx * 4 + idx, 2, 0] *= -1

    perms = torch.repeat_interleave(perms, repeats=torch.tensor([4] * len(perms), dtype=torch.long, device=device), dim=0)

    # Sphere for voting principal directions
    vote_sphere_pts = generate_sphere_pts(5, device=device)
    vote_sphere_pts = vote_sphere_pts[:vote_sphere_pts.shape[0] // 2]

    # Sphere for distance function evaluation
    query_points_level = getattr(cfg, 'query_points_level', 1)
    query_points = generate_sphere_pts(level=query_points_level, type='torch', device=device)

    # Optionally generate all maps at first run and keep them
    if loc_exp_mode == 'single_room':
        gen_map_once = False
    else:
        gen_map_once = True
    print(f"Generating map once flag set to {gen_map_once}...")

    # Main localization loop
    for room_idx, query_room in enumerate(query_room_list):
        curr_map_room_list = map_room_list[room_idx]
        print(f"Query Room: {query_room}, Map Room(s): {curr_map_room_list}")
        if dataset == 'omniscenes':
            query_filenames = [file_name for file_name in filenames if query_room in file_name]
        else:
            query_filenames = [file_name for file_name in filenames if query_room.split('/')[0] in file_name and query_room.split('/')[1] in file_name]

        if len(query_filenames) == 0:  # If no query exists, skip room
            continue

        # Read point cloud and dataset preparation
        print("STEP 1: Point Cloud Loading & 3D Map Setup")
        if room_idx == 0 or not gen_map_once:
            map_dict, topk_ratio, sparse_topk_ratio = generate_line_map(cfg, curr_map_room_list)
            precomputed_dist_3d = {room: None for room in curr_map_room_list}  # 3D distance functions computed for each room
            precomputed_point_dist_3d = {room: None for room in curr_map_room_list}  # 3D distance functions computed for each room
            precomputed_mask_3d = {room: None for room in curr_map_room_list}

            xdf_precompute = XDFCanonicalPrecompute(cfg, log_dir, map_dict)  # Canonical pre-computer for point/line distance functions            

            precompute_start = time.time()
            precomputed_dist_3d, precomputed_mask_3d = xdf_precompute.generate_ldf_3d()  # Pre-compute 3D line distance functions
            precomputed_point_dist_3d = xdf_precompute.generate_pdf_3d()  # Pre-compute 3D point distance functions

            # Choose which 2D XDF to use during room selection
            search_dist_3d = {room: torch.cat([precomputed_dist_3d[room], precomputed_point_dist_3d[room] ** point_gamma], axis=-1) for room in curr_map_room_list}

            elapsed = time.time() - precompute_start
            print(f"Precomputing 3D finished in {elapsed:.5f}s")

        for filename in query_filenames:
            print(f"Filename: {filename}")

            if dataset == 'omniscenes':
                video_name = filename.split('/')[-2]
                img_seq = filename.split('/')[-1]
                img_name = '{}/{}'.format(video_name, img_seq)
                room_type = video_name.split('_')[1]
                split_type = video_name.split('_')[0]
                room_no = video_name.split('_')[2]

                pcd_name = data_utils.get_pcd_name(dataset, room_type=room_type, room_no=room_no)
            elif dataset == 'stanford':
                split_type = filename.split('/')[-2]
                area_num = int(filename.split('/')[-2].split('_')[-1])
                img_name = filename.split('/')[-1]
                room_type = img_name.split('_')[2]
                room_no = img_name.split('_')[3]

                pcd_name = data_utils.get_pcd_name(dataset, area_name=split_type, room_type=room_type, room_no=room_no)
            
            # Update logger
            logger.add_filename(filename, f"{room_type}_{room_no}", split_type)

            # Load and check validity of gt pose
            if dataset == 'omniscenes':
                gt_trans, gt_rot = data_utils.read_gt(dataset, filename=filename)
            elif dataset == 'stanford':
                gt_trans, gt_rot = data_utils.read_gt(dataset, area_num=area_num, img_name=img_name)
            gt_trans = torch.from_numpy(gt_trans).float().to(device)
            gt_rot = torch.from_numpy(gt_rot).float().to(device)

            if out_of_room(map_dict[query_room]['xyz'], gt_trans, out_of_room_quantile):
                print(f'{filename} gt_trans is out of the room\n')
                logger.add_skipped_room(filename)
                continue
            else:
                valid_trial += 1

            # Read image
            orig_img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
            edge_tgt_img = cv2.resize(orig_img, (1024, 512))

            # 2D line extraction (edge_lines are used for refinement + PDFs, sparse_edge_lines are used for LDF)
            if dataset == 'stanford':
                # Eliminate the top, bottom dark regions
                edge_tgt_img[:512 // 6] = 0
                edge_tgt_img[-512 // 6:] = 0
                edge_lines, edge_img, full_edge_lines = extract_img_line(edge_tgt_img, view_size=120, return_full_lines=True, return_edge_img=True, length_ratio=topk_ratio)
                edge_lines = torch.from_numpy(edge_lines).float().to(device)
                start_z = edge_lines[:, 5]
                end_z = edge_lines[:, 8]

                on_up = ((start_z - np.cos(np.pi / 6)).abs() < 0.01) & ((end_z - np.cos(np.pi / 6)).abs() < 0.01)
                on_down = ((start_z - np.cos(5 * np.pi / 6)).abs() < 0.01) & ((end_z - np.cos(5 * np.pi / 6)).abs() < 0.01)

                edge_lines = edge_lines[torch.bitwise_not(on_up | on_down)]

                sparse_edge_lines = filterEdgeByTopK(full_edge_lines, int(full_edge_lines.shape[0] * sparse_topk_ratio), False)
                sparse_edge_lines = torch.from_numpy(sparse_edge_lines).float().to(device)
                start_z = sparse_edge_lines[:, 5]
                end_z = sparse_edge_lines[:, 8]

                on_up = ((start_z - np.cos(np.pi / 6)).abs() < 0.01) & ((end_z - np.cos(np.pi / 6)).abs() < 0.01)
                on_down = ((start_z - np.cos(5 * np.pi / 6)).abs() < 0.01) & ((end_z - np.cos(5 * np.pi / 6)).abs() < 0.01)

                sparse_edge_lines = sparse_edge_lines[torch.bitwise_not(on_up | on_down)]

            else:
                edge_lines, edge_img, full_edge_lines = extract_img_line(edge_tgt_img, view_size=120, return_full_lines=True, return_edge_img=True, length_ratio=topk_ratio)
                edge_lines = torch.from_numpy(edge_lines).float().to(device)

                sparse_edge_lines = filterEdgeByTopK(full_edge_lines, int(full_edge_lines.shape[0] * sparse_topk_ratio), False)
                sparse_edge_lines = torch.from_numpy(sparse_edge_lines).float().to(device)

            # Principal direction extraction
            print("STEP 2: Extracting principal directions and line intersections in 2D")
            start = time.time()
            principal_2d = extract_principal_2d(sparse_edge_lines, vote_sphere_pts)  # Use sparser lines for principal direction estimation

            # Extract 2D intersections considering rotation
            full_inter_2d = []
            full_inter_2d_mask = []
            full_inter_2d_idx = []

            # 2D intersections and mask before considering principal direction permutation
            raw_inter_2d, raw_inter_2d_idx = intersections_2d(edge_lines, principal_2d, inlier_thres=inlier_thres_2d, intersect_thres=intersect_thres_2d, return_idx=True)

            # Extract 2D intersections considering principal directions
            for perm_idx in range(perms.shape[0]):
                principal_perm = perms[perm_idx].tolist()  # Principal direction permutation
                intersection_perm = [intersections_idx(principal_perm[i % 3], principal_perm[(i + 1) % 3]) for i in range(3)]

                inter_2d_rot = torch.cat([raw_inter_2d[p_idx] for p_idx in intersection_perm], dim=0)
                full_inter_2d.append(inter_2d_rot)
                inter_2d_rot_idx = torch.cat([raw_inter_2d_idx[p_idx] for p_idx in intersection_perm], dim=0)
                full_inter_2d_idx.append(inter_2d_rot_idx)

                mask_2d_rot = []
                for k, p_idx in enumerate(intersection_perm):  # Make masks according to principal direction permutation
                    # inter_2d_rot[k]: intersection of kth principal direction line and (k+1)th principal direction line
                    mask_temp = torch.zeros_like(raw_inter_2d[p_idx])
                    mask_temp[:, k] = 1
                    mask_temp[:, (k + 1) % 3] = 1
                    mask_2d_rot.append(mask_temp.bool())

                mask_2d_rot = torch.cat(mask_2d_rot, dim=0)
                full_inter_2d_mask.append(mask_2d_rot)

            elapsed = time.time() - start
            print(f"Finished in {elapsed:.5f}s \n")

            # Search multiple rooms to find the top-k pose
            total_search_start = time.time()

            print("STEP 2.5: Optional room selction")
            start = time.time()
            precomputed_rot = {room: None for room in curr_map_room_list}
            precomputed_mask_2d = {room: None for room in curr_map_room_list}
            precomputed_dist_2d = {room: None for room in curr_map_room_list}  # 2D line distance functions computed for each room
            precomputed_point_dist_2d = {room: None for room in curr_map_room_list}  # 2D point distance functions computed for each room

            precompute_start = time.time()

            # Pre-compute 2D line distance functions
            precomputed_rot, precomputed_mask_2d, precomputed_dist_2d = \
                xdf_precompute.generate_ldf_2d(principal_2d, sparse_edge_lines, perms, bin_mask, single_pose_compute=True)

            # Pre-compute 2D point distance functions
            total_full_inter_2d = torch.stack(full_inter_2d, dim=0)  # (N_r, N_2D, L)
            total_full_inter_2d_mask = torch.stack(full_inter_2d_mask, dim=0)  # (N_r, N_2D, L)

            precomputed_point_dist_2d = xdf_precompute.generate_pdf_2d(total_full_inter_2d, total_full_inter_2d_mask, precomputed_rot, single_pose_compute=True)

            elapsed = time.time() - precompute_start
            print(f"Precomputing 2D finished in {elapsed:.5f}s")

            # Choose which 2D XDF to use during room selection
            search_dist_2d = {room: torch.cat([precomputed_dist_2d[room], precomputed_point_dist_2d[room] ** point_gamma], axis=-1) for room in curr_map_room_list}

            # Apply room selection
            search_room_list = []
            search_room_idx_list = xdf_precompute.infer_room(search_dist_2d, search_dist_3d, selection_num_room, sample_per_room=sample_per_room)
            search_room_list = [curr_map_room_list[r_id] for r_id in search_room_idx_list]

            elapsed = time.time() - start
            print(f"Finished in {elapsed:.5f}s \n")

            # Cache all top-k predictions made for each room
            cand_info_dict = {
                room: {
                    'sample_cost': None,
                    'trimmed_trans': None,
                    'trimmed_rot': None,
                    'inter_2d': None,
                    'inter_2d_mask': None,
                    'inter_2d_idx': None,
                } for room in search_room_list
            }

            for room in search_room_list:
                print(f"===============Searching {room}...===============")

                # Rotation Estimation and line masking
                print("STEP 3: Distance function preparation")
                start = time.time()

                # Load pre-computed quantities to GPU
                estim_rot = precomputed_rot[room].to(device)
                batch_mask_2d, batch_mask_3d = precomputed_mask_2d[room].to(device), precomputed_mask_3d[room].to(device)
                N_r = batch_mask_2d.shape[0]
                batch_mask_3d = batch_mask_3d[None, ...].repeat(N_r, 1, 1)  # (N_r, N_3D, 3)

                # Load pre-computed XDFs
                precomputed_dist_2d[room] = precomputed_dist_2d[room].to(device)
                precomputed_dist_3d[room] = precomputed_dist_3d[room].to(device)
                precomputed_point_dist_2d[room] = precomputed_point_dist_2d[room].to(device)
                precomputed_point_dist_3d[room] = precomputed_point_dist_3d[room].to(device)

                elapsed = time.time() - start
                print(f"Finished in {elapsed:.5f}s \n")

                # Rank distance functions at candidate poses
                print("STEP 4: Pose Search")
                start = time.time()
                cost_mtx = xdf_cost(
                    map_dict[room]['trans_tensors'], 
                    estim_rot,
                    precomputed_dist_3d=precomputed_dist_3d[room],
                    precomputed_dist_2d=precomputed_dist_2d[room],
                    precomputed_point_dist_3d=precomputed_point_dist_3d[room],
                    precomputed_point_dist_2d=precomputed_point_dist_2d[room],
                    point_gamma=point_gamma)

                elapsed = time.time() - start
                print(f"Finished in {elapsed:.5f}s \n")

                # Sample cost values for room selection
                sample_inds = cost_mtx.flatten().argsort()[:sample_per_room]
                sample_cost = cost_mtx.flatten()[sample_inds].float().mean()

                # Filter out top-k poses
                min_inds = cost_mtx.flatten().argsort()[:top_k_candidate]

                trimmed_trans = map_dict[room]['trans_tensors'][min_inds // len(estim_rot)]
                trimmed_rot = estim_rot[min_inds % len(estim_rot)]

                # Align canonicalized rotations back to original frame
                trimmed_rot = trimmed_rot @ xdf_precompute.canonical_rot_3d[room].unsqueeze(0)

                # Select 2D intersections
                inter_2d = [full_inter_2d[i] for i in min_inds % len(estim_rot)]
                inter_2d_mask = [full_inter_2d_mask[i] for i in min_inds % len(estim_rot)]
                inter_2d_idx = [full_inter_2d_idx[i] for i in min_inds % len(estim_rot)]

                cand_info_dict[room]['sample_cost'] = sample_cost
                cand_info_dict[room]['trimmed_trans'] = trimmed_trans
                cand_info_dict[room]['trimmed_rot'] = trimmed_rot
                cand_info_dict[room]['inter_2d'] = inter_2d
                cand_info_dict[room]['inter_2d_mask'] = inter_2d_mask
                cand_info_dict[room]['inter_2d_idx'] = inter_2d_idx
                cand_info_dict[room]['min_inds'] = min_inds
                cand_info_dict[room]['estim_rot'] = estim_rot

            # Select final poses to refine
            total_cost = torch.stack([cand_info_dict[room]['sample_cost'] for room in search_room_list])  # type: ignore # (N_k * N_room, )
            selected_room_idx_list = total_cost.argsort()[:selection_num_room].tolist()
            selected_room_inds = []
            selected_pose_inds = []
            for selected_room_idx in selected_room_idx_list:
                selected_room_inds.extend([selected_room_idx] * top_k_candidate)
                selected_pose_inds.extend(list(range(top_k_candidate)))

            cand_trimmed_trans = torch.stack([
                cand_info_dict[search_room_list[r_id]]['trimmed_trans'][p_id]
                    for r_id, p_id in zip(selected_room_inds, selected_pose_inds)
            ])
            cand_trimmed_rot = torch.stack([
                cand_info_dict[search_room_list[r_id]]['trimmed_rot'][p_id]
                    for r_id, p_id in zip(selected_room_inds, selected_pose_inds)
            ])
            cand_trimmed_room_list = [search_room_list[r_id] for r_id in selected_room_inds]

            total_search_elapsed = time.time() - total_search_start
            print(f"Total search finished in {total_search_elapsed:.5f}s \n")

            # Refine top-k poses
            print(f"STEP 5: Refining pose estimates with spherical alignment")
            start = time.time()

            # Prepare 2D intersections for refinement
            cand_inter_2d = [
                cand_info_dict[search_room_list[r_id]]['inter_2d'][p_id] \
                    for r_id, p_id in zip(selected_room_inds, selected_pose_inds)
            ]
            cand_inter_2d_mask = [
                cand_info_dict[search_room_list[r_id]]['inter_2d_mask'][p_id] \
                    for r_id, p_id in zip(selected_room_inds, selected_pose_inds)
            ]
            cand_inter_2d_idx = [
                cand_info_dict[search_room_list[r_id]]['inter_2d_idx'][p_id] \
                    for r_id, p_id in zip(selected_room_inds, selected_pose_inds)
            ]

            # Prepare 3D intersections for refinement
            cand_inter_3d = [map_dict[search_room_list[r_id]]['inter_3d'] for r_id in selected_room_inds]
            cand_inter_3d_mask = [map_dict[search_room_list[r_id]]['inter_3d_mask'] for r_id in selected_room_inds]
            cand_inter_3d_idx = [map_dict[search_room_list[r_id]]['inter_3d_idx'] for r_id in selected_room_inds]

            # Lines used for refining rotation
            line_dict = {
                'dense_dirs': [map_dict[search_room_list[r_id]]['dense_dirs'] for r_id in selected_room_inds],
                'edge_lines': edge_lines,
                'inter_2d_idx': cand_inter_2d_idx,
                'inter_3d_idx': cand_inter_3d_idx
            }

            refined_trans, refined_rot, _ = refine_from_sphere_icp(cand_trimmed_trans, cand_trimmed_rot,
                cand_inter_2d, cand_inter_3d, opt_dict, cand_inter_2d_mask, cand_inter_3d_mask, line_dict=line_dict)
            elapsed = time.time() - start
            print(f"Finished in {elapsed:.5f}s \n")

            # Log error metrics
            trans_error = (refined_trans.cpu() - gt_trans.cpu().squeeze()).norm().item()
            rot_error = torch.matmul(torch.transpose(refined_rot, dim0=-2, dim1=-1), gt_rot.unsqueeze(0))
            rot_error = torch.diagonal(rot_error, dim1=-2, dim2=-1).sum(-1)
            if rot_error < -1:
                rot_error = -2 - rot_error
            elif rot_error > 3:
                rot_error = 6 - rot_error
            rot_error = torch.rad2deg(torch.abs(torch.arccos((rot_error - 1) / 2))).item()

            logger.add_error(trans_error, rot_error, filename, f"{room_type}_{room_no}", split_type)
            logger.add_estimate(filename, refined_trans.cpu().numpy(), refined_rot.cpu().numpy(), f"{room_type}_{room_no}", split_type)

            print("=============== CURRENT RESULTS ===============")
            print("t-error: ", trans_error)
            print("r-error: ", rot_error)

            if (trans_error < 0.1) and (rot_error < 5):
                well_posed += 1
            print("Accuracy: ", well_posed / valid_trial)
            print("===============================================")

    # Calculate statistics and save logger
    logger.calc_statistics('room')
    logger.calc_statistics('split')
    logger.calc_statistics('room_split')
    logger.calc_statistics('total')
    save_logger(os.path.join(log_dir, getattr(cfg, 'log_name', 'result.pkl')), logger)
