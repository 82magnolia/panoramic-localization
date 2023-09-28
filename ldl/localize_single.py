import torch
import cv2
import os
import time
from typing import NamedTuple
import numpy as np
from ldl.utils import (
    generate_trans_points,
    out_of_room,
    rgb_to_grayscale,
    sample_from_img,
    make_pano,
)
import ldl.data_utils as data_utils
from ldl.edge_utils import (
    generate_sphere_pts,
    extract_img_line,
    split_by_axes,
)
import warnings
from itertools import permutations
from ldl.pose_estimation import (
    split_distance_cost,
    get_matcher,
    refine_from_point_ransac,
)
from ldl.dict_utils import get_init_dict
from log_utils import save_logger, PoseLogger
from tqdm import tqdm


warnings.filterwarnings("ignore", category=UserWarning) 


def localize(cfg: NamedTuple, log_dir: str, query_img_path: str, color_pcd_path: str, line_pcd_path: str):
    """
    Main function for performing localization against a single 3D map and image.

    Args:
        cfg: Config file
        log_dir: Directory in which logs will be saved
        query_img_path: Path to query image
        color_pcd_path: Path to colored point cloud
        line_pcd_path: Path to line cloud
    
    Returns:
        None
    """

    """
    TODO: We need to implement the followings

    1. Read single image / point cloud
    2. Perform localization for a single iteration
    3. Visualize localized results using make_pano
    """
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    out_of_room_quantile = getattr(cfg, 'out_of_room_quantile', 0.05)

    # Algorithm configs
    sample_rate = getattr(cfg, 'sample_rate', 1)
    max_edge_count = getattr(cfg, 'max_edge_count', 1000)  # Max edge count to prevent GPU overload
    top_k_candidate = getattr(cfg, 'top_k_candidate', 5)
    refine_mode = getattr(cfg, 'refine_mode', 'match')
    print_retrieval = getattr(cfg, 'print_retrieval', False)
    coord_h = getattr(cfg, 'coord_h', 320)
    coord_w = getattr(cfg, 'coord_w', 640)
    render_h = getattr(cfg, 'render_h', 320)
    render_w = getattr(cfg, 'render_w', 640)

    well_posed = 0
    valid_trial = 0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    past_pcd_name = ""

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

    # Set matching module
    matcher = get_matcher(cfg, device=device)

    # Sphere for voting principal directions
    vote_sphere_pts = generate_sphere_pts(5, device=device)
    vote_sphere_pts = vote_sphere_pts[:vote_sphere_pts.shape[0] // 2]

    # Main localization
    if trial == 0:  # Assume a single 3D map to be localized against
        print("STEP 1: Point Cloud Loading & Feature Extraction & 3D Principal Direction Extraction")
        start = time.time()

        xyz_np, pcd_rgb_np = data_utils.read_pcd(dataset, pcd_name=pcd_name, sample_rate=sample_rate)
        mean_size = (xyz_np.max(0) - xyz_np.min(0)).mean()
        length_thres = mean_size * 0.10  # Adaptive length thres
        dirs_np, starts_np, ends_np, num_unq = data_utils.read_line(pcd_name, length_thres, max_edge_count)
        topk_ratio = starts_np.shape[0] / num_unq  # Top-k ratio to select from 2D lines
        dirs = torch.from_numpy(dirs_np).float().to(device)
        starts = torch.from_numpy(starts_np).float().to(device)
        ends = torch.from_numpy(ends_np).float().to(device)
        xyz = torch.from_numpy(xyz_np).float().to(device)
        rgb = torch.from_numpy(pcd_rgb_np).float().to(device)

        # Set 3D principal directions
        principal_3d = []
        counts_3d = []
        vote_dirs = dirs.clone().detach()
        for _ in range(num_principal):
            vote_3d = torch.abs(vote_dirs[:, :3] @ vote_sphere_pts.t()).argmax(-1).bincount(minlength=vote_sphere_pts.shape[0])
            max_idx = vote_3d.argmax()
            counts_3d.append(vote_3d.max().item())
            principal_3d.append(vote_sphere_pts[max_idx])
            outlier_idx = (torch.abs(vote_dirs[:, :3] @ vote_sphere_pts[max_idx: max_idx+1].t()) < 0.95).squeeze()
            vote_dirs = vote_dirs[outlier_idx]
        principal_3d = torch.stack(principal_3d, dim=0)

        if torch.det(principal_3d) < 0:
            principal_3d[-1, :] *= -1

        # Set translation start points
        init_dict = get_init_dict(cfg)
        trans_tensors = generate_trans_points(xyz, init_dict, device=device)

        kpts_xyz_list = []
        desc_list = []
        score_list = []

        # Feature extraction
        num_pts = 0
        for trans in tqdm(trans_tensors):
            transform_xyz = xyz - trans.unsqueeze(0)
            render_img = make_pano(transform_xyz, rgb, resolution=(render_h, render_w), return_torch=True).float() / 255.
            coord_img = make_pano(transform_xyz, xyz, return_torch=True, resolution=(coord_h, coord_w)).float() / 255.

            with torch.no_grad():
                render_img = rgb_to_grayscale(render_img.permute(2, 0, 1).unsqueeze(0))
                pred = matcher.superpoint({'image': render_img})
                kpts_ij = pred['keypoints'][0]  # (N_kpts, 2)
                kpts_ij[:, 0] = (kpts_ij[:, 0] - coord_w // 2) / (coord_w // 2)
                kpts_ij[:, 1] = (kpts_ij[:, 1] - coord_h // 2) / (coord_h // 2)
                kpts_xyz = sample_from_img(coord_img, kpts_ij, mode='nearest')
                valid_mask = kpts_xyz.norm(dim=-1) > 0.1
                kpts_xyz = kpts_xyz[valid_mask]                    
                scores = pred['scores'][0][valid_mask]
                desc = pred['descriptors'][0][:, valid_mask]
            num_pts += kpts_xyz.shape[0]
            kpts_xyz_list.append(kpts_xyz)
            desc_list.append(desc)
            score_list.append(scores)
        elapsed = time.time() - start
        print(f"Finished in {elapsed:.3f}s \n")
    
    # Update past_pcd_name
    past_pcd_name = pcd_name

    # Load and check validity of gt pose
    if dataset == 'omniscenes':
        gt_trans, gt_rot = data_utils.read_gt(dataset, filename=filename)
    elif dataset == 'stanford':
        gt_trans, gt_rot = data_utils.read_gt(dataset, area_num=area_num, img_name=img_name)
    gt_trans = torch.from_numpy(gt_trans).float().to(device)
    gt_rot = torch.from_numpy(gt_rot).float().to(device)

    if out_of_room(xyz, gt_trans, out_of_room_quantile):
        print(f'{filename} gt_trans is out of the room\n')
        logger.add_skipped_room(filename)
        continue
    else:
        valid_trial += 1

    # Read image
    orig_img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)

    edge_tgt_img = cv2.resize(orig_img, (1024, 512))

    if dataset == 'stanford':
        # Eliminate the top, bottom dark regions
        edge_tgt_img[:512 // 6] = 0
        edge_tgt_img[-512 // 6:] = 0
        edge_lines, edge_img = extract_img_line(edge_tgt_img, view_size=120, return_edge_img=True, length_ratio=topk_ratio)
        edge_lines = torch.from_numpy(edge_lines).float().to(device)
        start_z = edge_lines[:, 5]
        end_z = edge_lines[:, 8]

        on_up = ((start_z - np.cos(np.pi / 6)).abs() < 0.01) & ((end_z - np.cos(np.pi / 6)).abs() < 0.01)
        on_down = ((start_z - np.cos(5 * np.pi / 6)).abs() < 0.01) & ((end_z - np.cos(5 * np.pi / 6)).abs() < 0.01)

        edge_lines = edge_lines[torch.bitwise_not(on_up | on_down)]
    else:
        edge_lines, edge_img = extract_img_line(edge_tgt_img, view_size=120, return_edge_img=True, length_ratio=topk_ratio)
        edge_lines = torch.from_numpy(edge_lines).float().to(device)

    # Principal direction extraction
    print("STEP 2: Extracting principal directions in 2D")

    tot_directions = []
    counts_2d = []
    start = time.time()
    vote_edge_lines = edge_lines.clone().detach()
    max_search_iter = 20
    for idx in range(max_search_iter):
        if len(vote_edge_lines) == 0:
            break
        vote_2d = torch.where(torch.abs(vote_edge_lines[:, :3] @ vote_sphere_pts.t()) < 0.05)[1].bincount(minlength=vote_sphere_pts.shape[0])
        max_idx = vote_2d.argmax()
        counts_2d.append(vote_2d.max().item())
        cand_direction = vote_sphere_pts[max_idx]
        tot_directions.append(cand_direction)
        outlier_idx = (torch.abs(vote_edge_lines[:, :3] @ vote_sphere_pts[max_idx: max_idx+1].t()) > 0.05).squeeze()
        vote_edge_lines = vote_edge_lines[outlier_idx]

    tot_directions = torch.stack(tot_directions, dim=0)

    # Search through all combinations to find a match with principal_3d
    combs = torch.combinations(torch.arange(tot_directions.shape[0]), r=3)

    comb_directions = tot_directions[combs]
    comb_dots = torch.stack([(comb_directions[:, i % 3] * comb_directions[:, (i + 1) % 3]).sum(-1).abs()
        for i in range(3)], dim=-1)
    valid_comb_idx = (comb_dots < 0.1).sum(-1) == 3  # Assume perpendicular 3D, but can be extended

    if valid_comb_idx.sum() == 0:  # Failed to find from total directions, get 2 directions and obtain rest from cross
        if (comb_dots < 0.15).sum() != 0:  # At least one pair is perpendicular
            new_valid_comb_idx = torch.where(comb_dots < 0.15)  # Assume perpendicular 3D, but can be extended
            vec_0 = comb_directions[new_valid_comb_idx[0][0], new_valid_comb_idx[1][0]]
            vec_1 = comb_directions[new_valid_comb_idx[0][0], (new_valid_comb_idx[1][0] + 1) % 3]
            two_vec = torch.stack([vec_0, vec_1], dim=0)
            third_vec = torch.cross(two_vec[0], two_vec[1]).unsqueeze(0)
            principal_2d = torch.cat([two_vec, third_vec])
        else:  # Worst case: pick two most frequent directions
            two_vec = tot_directions[:2]
            third_vec = torch.cross(two_vec[0], two_vec[1]).unsqueeze(0)
            principal_2d = torch.cat([two_vec, third_vec])
    else:
        extracted_comb = torch.where(valid_comb_idx)[0][0]
        principal_2d = comb_directions[extracted_comb]

    if torch.det(principal_2d) < 0:
        principal_2d[-1, :] *= -1

    # Optionally apply splitting for finding keypoints near lines
    elapsed = time.time() - start
    print(f"Finished in {elapsed:.3f}s \n")

    # Rotation Estimation
    print("STEP 3: Rotation estimation up to ambiguity from principal directions")
    start = time.time()

    # Kabsch algorithm
    pts_2d = principal_2d[perms]  # (N_perms, 3, 3)
    H = (principal_3d.t().unsqueeze(0) @ (bin_mask * pts_2d))
    U, S, V = torch.svd(H)
    U_t = torch.transpose(U, 1, 2)
    d = torch.sign(torch.det(V @ U_t))
    diag_mtx = torch.eye(3, device=device).unsqueeze(0).repeat(perms.shape[0], 1, 1)  # (N_perms, 3, 3)
    diag_mtx[:, 2, 2] = d
    estim_rot = V @ diag_mtx @ U_t

    elapsed = time.time() - start
    print(f"Finished in {elapsed:.3f}s \n")
    
    # Rank distance functions at candidate views 
    sparse_sphere_points = generate_sphere_pts(level=1, type='torch', device=device)

    print("STEP 4: Distance function computation for candidate poses")
    start = time.time()
    batch_mask_2d, batch_mask_3d = split_by_axes(edge_lines, dirs, principal_3d, estim_rot)
    cost_mtx = split_distance_cost(sparse_sphere_points, trans_tensors, estim_rot, 
        batch_mask_2d, batch_mask_3d, edge_lines, starts, ends, method='inlier')

    elapsed = time.time() - start
    print(f"Finished in {elapsed:.3f}s \n")

    # Filter out top-k poses
    min_inds = cost_mtx.flatten().argsort()[:top_k_candidate]

    trimmed_trans = trans_tensors[min_inds // len(estim_rot)]
    trimmed_rot = estim_rot[min_inds % len(estim_rot)]

    # Evaluate estimated rotations with ground truth
    r_error = torch.matmul(torch.transpose(estim_rot, dim0=-2, dim1=-1), gt_rot.unsqueeze(0))
    r_error = torch.diagonal(r_error, dim1=-2, dim2=-1).sum(-1)
    r_error[r_error < -1] = -2 - r_error[r_error < -1]
    r_error[r_error > 3] = 6 - r_error[r_error > 3]
    r_error = torch.rad2deg(torch.abs(torch.arccos((r_error - 1) / 2)))

    rot_k_error = r_error[min_inds % len(estim_rot)]

    # Refine top-k poses
    print(f"STEP 5: Refining pose estimates with using {refine_mode} mode")
    start = time.time()
    if refine_mode == 'pnp_ransac':
        refined_trans, refined_rot = refine_from_point_ransac(trimmed_trans, trimmed_rot, cv2.resize(orig_img, (render_w, render_h)),
            matcher, kpts_xyz_list, desc_list, score_list, min_inds // len(estim_rot))
    else:
        raise NotImplementedError("Other refine modes not implemented")
    elapsed = time.time() - start
    print(f"Finished in {elapsed:.3f}s \n")

    # Visualize localization
