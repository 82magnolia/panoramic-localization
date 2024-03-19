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
    make_pano_line_3d
)
import warnings
from itertools import permutations
from fgpl.pose_estimation import (
    xdf_cost,
    refine_from_sphere_icp,
)
from fgpl.line_intersection import intersections_2d, intersections_3d, intersections_idx
from fgpl.xdf_canonical_precompute import XDFCanonicalPrecompute
from map_utils import generate_line_map_single
import numpy as np


warnings.filterwarnings("ignore", category=UserWarning)


def localize(cfg: NamedTuple, log_dir: str, query_img_path: str, line_pcd_path: str, crop_up_down: bool):
    """
    Main function for performing localization.

    Args:
        cfg: Config file
        log_dir: Directory in which logs will be saved
        query_img_path: Path to query image
        line_pcd_path: Path to line cloud, saved in .txt format
        crop_up_down: If True, crops upper and lower parts of panorama during localization
    
    Returns:
        None
    """
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # Algorithm configs
    top_k_candidate = getattr(cfg, 'top_k_candidate', 1)  # Number of poses per each room used directly for refinement
    sample_per_room = getattr(cfg, 'sample_per_room', top_k_candidate)  # Number of poses per each room to average during room selection
    selection_num_room = getattr(cfg, 'selection_num_room', 1)  # Number of rooms to select for refinement
    inlier_thres_2d = getattr(cfg, 'inlier_thres_2d', 0.5)
    intersect_thres_2d = getattr(cfg, 'intersect_thres_2d', 0.1)
    point_gamma = getattr(cfg, 'point_gamma', 0.2)

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
    curr_map_room_list = ['test_room']  # Room fixed for single testing

    # Read point cloud and dataset preparation
    print("STEP 1: Point Cloud Loading & 3D Map Setup")
    map_dict, topk_ratio, sparse_topk_ratio = generate_line_map_single(cfg, line_pcd_path, curr_map_room_list)
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

    # Read image
    orig_img = cv2.cvtColor(cv2.imread(query_img_path), cv2.COLOR_BGR2RGB)
    edge_tgt_img = cv2.resize(orig_img, (1024, 512))

    # 2D line extraction (edge_lines are used for refinement + PDFs, sparse_edge_lines are used for LDF)
    if crop_up_down:
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

    # Rotation Estimation and line masking
    print("STEP 3: Distance function preparation")
    start = time.time()

    # Load pre-computed quantities to GPU
    room = curr_map_room_list[0]
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

    # Visualize localization
    vis_img = cv2.resize(edge_tgt_img, (400, 200))
    result_img = make_pano_line_3d(map_dict[curr_map_room_list[0]]['starts'], map_dict[curr_map_room_list[0]]['ends'], trans_mtx=refined_trans, rot_mtx=refined_rot, resolution=(200, 400))
    margin_img = np.zeros_like(vis_img[:, :100, :])
    final_img = np.concatenate([vis_img, margin_img, result_img], axis=1)
    final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(log_dir, 'result.png'), final_img)
