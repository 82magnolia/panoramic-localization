import torch
import numpy as np
from utils import ypr_from_rot, rot_from_ypr
from tqdm import tqdm


def refine_from_sphere_icp(trimmed_trans, trimmed_rot, inter_2d, inter_3d, opt_dict,
    inter_2d_mask=None, inter_3d_mask=None, line_dict=None):
    """
    Refine top-k poses by matching lines and intersections: here both rotation and translation is optimized

    Args:
        trimmed_trans: (N_k, 3) tensor containing top-k translations
        trimmed_rot: (N_k, 3, 3) tensor containing top-k rotations
        inter_2d: List of (N_2D, 3) points on sphere containing intersections of 2D lines
        inter_3d: (N_3D, 3) points or list of (N_3D, 3) points on sphere containing intersections of 2D lines
        opt_dict: Dictionary containing optimization options
        inter_2d_mask: List of (N_2D, 3) masks specifying the class of 2D line intersections
        inter_3d_mask: (N_3D, 3) mask or list of (N_3D, 3) mask specifying the class of 3D line intersections
        line_dict: Dictionary containing line information optionally used for rotation refinement

    Returns:
        tgt_trans: (3, ) tensor containing optimized translation
        tgt_rot: (3, 3) tensor containing optimized rotation
    """
    # trimmed_trans is (K, 3) and trimmed_rot is (K, 3, 3)
    num_k = trimmed_trans.shape[0]
    device = trimmed_trans.device

    loss_list = []
    trans_list = []
    rot_list = []
    idx_2d_list = []
    idx_3d_list = []

    # Copy intersections 2D and 3D, along with mask
    inter_2d = [i.clone().detach() for i in inter_2d]
    if inter_2d_mask is not None:
        inter_2d_mask = [i.clone().detach() for i in inter_2d_mask]
    
    # Clone intersections depending on its type
    if isinstance(inter_3d, list):
        inter_3d = [i3d.clone().detach() for i3d in inter_3d]
        if inter_3d_mask is not None:
            inter_3d_mask = [i3d_mask.clone().detach() for i3d_mask in inter_3d_mask]
    else:
        inter_3d = inter_3d.clone().detach()
        if inter_3d_mask is not None:
            inter_3d_mask = inter_3d_mask.clone().detach()

    for idx in range(num_k):
        input_trans = trimmed_trans[idx: idx+1].clone().detach().requires_grad_()
        input_ypr = ypr_from_rot(trimmed_rot[idx])

        # Tracking variables of optimization for each pose
        track_cost = np.inf
        track_trans = input_trans
        track_rot = rot_from_ypr(input_ypr.clone().detach())
        track_2d_idx = None
        track_3d_idx = None

        optimizer = torch.optim.Adam([input_trans], lr=opt_dict['step_size'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', \
            patience=opt_dict['decay_patience'], factor=opt_dict['decay_factor'])

        pts_2d_sphere = inter_2d[idx]  # (N_2D, 3)
        
        # Select 3D intersections to use for optimization
        if isinstance(inter_3d, list):
            curr_inter_3d = inter_3d[idx]
            curr_inter_3d_mask = inter_3d_mask[idx]
        else:
            curr_inter_3d = inter_3d
            curr_inter_3d_mask = inter_3d_mask
        
        range_2d = torch.arange(pts_2d_sphere.shape[0], device=device)
        range_3d = torch.arange(curr_inter_3d.shape[0], device=device)

        # Masks inverted for ease in indexing intersection points
        inv_inter_2d_mask = torch.bitwise_not(inter_2d_mask[idx])
        inv_inter_3d_mask = torch.bitwise_not(curr_inter_3d_mask)

        for it in tqdm(range(opt_dict['total_iter']), desc=f"Translation {idx}"):
            optimizer.zero_grad()
            input_rot = rot_from_ypr(input_ypr)

            transform_inter_3d = (curr_inter_3d - input_trans) @ input_rot.t()
            pts_3d_sphere = transform_inter_3d / transform_inter_3d.norm(dim=-1, keepdim=True)  # (N_3D, 3)
            
            with torch.no_grad():
                match_2d_idx_list = []
                match_3d_idx_list = []
                for mask_idx in [2, 0, 1]:  # Note that mask_idx == i indicates the intersections from i - 1 & i + 1 directions
                    mask_2d_idx = range_2d[inv_inter_2d_mask[:, mask_idx]]
                    mask_3d_idx = range_3d[inv_inter_3d_mask[:, mask_idx]]

                    if mask_2d_idx.shape[0] == 0 or mask_3d_idx.shape[0] == 0:
                        continue  # Skip if insufficient matches
                    mask_2d_range = torch.arange(mask_2d_idx.shape[0], device=device)
                    dist_mtx = (pts_2d_sphere[inv_inter_2d_mask[:, mask_idx]].unsqueeze(1) - \
                        pts_3d_sphere[inv_inter_3d_mask[:, mask_idx]].unsqueeze(0)).norm(dim=-1)  # (N_2D_mask, N_3D_mask)
                
                    # Mutual NN assignment (https://gist.github.com/mihaidusmanu/20fd0904b2102acc1330bad9b4badab8)
                    match_2d_to_3d = dist_mtx.argmin(-1)  # (N_2D_mask)
                    match_3d_to_2d = dist_mtx.argmin(0)  # (N_3D_mask)

                    if opt_dict['nn_dist_thres'] is not None:
                        valid_matches = (match_3d_to_2d[match_2d_to_3d] == mask_2d_range) & (dist_mtx.min(-1).values < opt_dict['nn_dist_thres'])
                    else:
                        valid_matches = match_3d_to_2d[match_2d_to_3d] == mask_2d_range

                    match_2d_idx = mask_2d_idx[valid_matches]
                    match_3d_idx = mask_3d_idx[match_2d_to_3d[valid_matches]]

                    match_2d_idx_list.append(match_2d_idx)
                    match_3d_idx_list.append(match_3d_idx)
                
                cluster_match_2d_idx = torch.cat(match_2d_idx_list)
                cluster_match_3d_idx = torch.cat(match_3d_idx_list)

                # Additionally extract distance NNs to consider corner cases
                full_dist_mtx = (pts_2d_sphere.unsqueeze(1) - pts_3d_sphere.unsqueeze(0)).norm(dim=-1)  # (N_2D, N_3D)

                # Mutual NN assignment (https://gist.github.com/mihaidusmanu/20fd0904b2102acc1330bad9b4badab8)
                match_2d_to_3d = full_dist_mtx.argmin(-1)  # (N_2D)
                match_3d_to_2d = full_dist_mtx.argmin(0)  # (N_3D)

                if opt_dict['nn_dist_thres'] is not None:
                    valid_matches = (match_3d_to_2d[match_2d_to_3d] == range_2d) & (full_dist_mtx.min(-1).values < opt_dict['nn_dist_thres'] / 5)
                else:
                    valid_matches = match_3d_to_2d[match_2d_to_3d] == range_2d

                dist_match_2d_idx = range_2d[valid_matches]
                dist_match_3d_idx = match_2d_to_3d[valid_matches]

                match_2d_idx = torch.cat([cluster_match_2d_idx, dist_match_2d_idx], axis=0)
                match_3d_idx = torch.cat([cluster_match_3d_idx, dist_match_3d_idx], axis=0)

                if match_2d_idx.shape[0] == 0:  # No valid matches found
                    break

            match_2d_sphere = pts_2d_sphere[match_2d_idx]
            match_3d_sphere = pts_3d_sphere[match_3d_idx]

            cost = torch.abs(match_2d_sphere - match_3d_sphere)
            cost = cost.sum(-1).mean()

            cost.backward()
            optimizer.step()
            scheduler.step(cost.item())

            # Keep track of current cost / translations / rotations
            track_cost = cost.item()
            track_trans = input_trans
            track_rot = input_rot
            track_2d_idx = cluster_match_2d_idx  # Use cluster matches for rotation refinement
            track_3d_idx = cluster_match_3d_idx  # Use cluster matches for rotation refinement

        loss_list.append(track_cost)
        trans_list.append(track_trans)
        rot_list.append(track_rot)
        idx_2d_list.append(track_2d_idx)
        idx_3d_list.append(track_3d_idx)

    best_idx = np.argmin(np.array(loss_list))
    best_trans = trans_list[best_idx]
    best_rot = rot_list[best_idx]
    best_idx_2d = idx_2d_list[best_idx]
    best_idx_3d = idx_3d_list[best_idx]
    optimize_rot = best_idx_2d is not None and best_idx_3d is not None

    # Optionally refine rotation if lines are provided
    if line_dict is not None and optimize_rot:
        input_ypr = ypr_from_rot(best_rot).requires_grad_()

        optimizer = torch.optim.Adam([input_ypr], lr=opt_dict['step_size'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', \
            patience=opt_dict['decay_patience'], factor=opt_dict['decay_factor'])

        full_dirs_2d = line_dict['edge_lines'][:, :3]  # (N_2D, 3)
        full_dirs_3d = line_dict['dense_dirs'][best_idx]  # (N_3D, 3)

        line_idx_2d_0 = line_dict['inter_2d_idx'][best_idx][best_idx_2d, 0]  # First line indices producing each intersection
        line_idx_2d_1 = line_dict['inter_2d_idx'][best_idx][best_idx_2d, 1]  # Second line indices producing each intersection
        line_idx_3d_0 = line_dict['inter_3d_idx'][best_idx][best_idx_3d, 0]  # First line indices producing each intersection
        line_idx_3d_1 = line_dict['inter_3d_idx'][best_idx][best_idx_3d, 1]  # Second line indices producing each intersection

        match_dirs_2d_0 = full_dirs_2d[line_idx_2d_0]
        match_dirs_2d_1 = full_dirs_2d[line_idx_2d_1]
        match_dirs_3d_0 = full_dirs_3d[line_idx_3d_0]
        match_dirs_3d_1 = full_dirs_3d[line_idx_3d_1]

        match_dirs_2d = torch.cat([match_dirs_2d_0, match_dirs_2d_1], dim=0)  # (2 * N_match, 3)

        # Order ambiguity exists for each intersection match
        match_dirs_3d = torch.cat([match_dirs_3d_0, match_dirs_3d_1], dim=0)  # (2 * N_match, 3)
        match_dirs_3d_inv = torch.cat([match_dirs_3d_1, match_dirs_3d_0], dim=0)  # (2 * N_match, 3)
        match_dirs_3d = torch.stack([match_dirs_3d, match_dirs_3d_inv], dim=-1)  # (2 * N_match, 3, 2)

        inlier_thres = 0.2
        for it in tqdm(range(opt_dict['total_iter']), desc="Rotation"):
            optimizer.zero_grad()
            input_rot = rot_from_ypr(input_ypr)
            rot_prod_2d = match_dirs_2d @ input_rot  # (N_match, 3)

            # Cost takes the minimum angle between the two types of line associations
            cost = torch.abs(rot_prod_2d.unsqueeze(-1) * match_dirs_3d).sum(1).min(-1).values  # (N_match, )
            cost = cost[cost < inlier_thres].mean()

            cost.backward()
            optimizer.step()
            scheduler.step(cost.item())
        best_rot = rot_from_ypr(input_ypr).clone().detach()

    max_t_error = 1.0
    max_r_error = 60
    trans_error = (best_trans.cpu() - trimmed_trans[best_idx: best_idx + 1].cpu().squeeze()).norm().item()
    rot_error = torch.matmul(torch.transpose(best_rot, dim0=-2, dim1=-1), trimmed_rot[best_idx].unsqueeze(0))
    rot_error = torch.diagonal(rot_error, dim1=-2, dim2=-1).sum(-1)
    if rot_error < -1:
        rot_error = -2 - rot_error
    elif rot_error > 3:
        rot_error = 6 - rot_error
    rot_error = torch.rad2deg(torch.abs(torch.arccos((rot_error - 1) / 2))).item()

    if trans_error > max_t_error or rot_error > max_r_error:
        best_trans = trimmed_trans[best_idx].clone().detach().unsqueeze(0)
        best_rot = trimmed_rot[best_idx].clone().detach()
    else:
        best_trans = best_trans.clone().detach()
        best_rot = best_rot.clone().detach()

    return best_trans, best_rot, best_idx


def xdf_cost(trans_tensor, rot_mtx, precomputed_dist_2d, precomputed_dist_3d, precomputed_point_dist_2d, precomputed_point_dist_3d, point_gamma=0.2):
    """
    Measure cost functions from point and line distance functions

    Args:
        trans_tensor: (N_t, 3) tensor containing translation candidate poses
        rot_mtx: (N_r, 3, 3) tensor containing rotation candidate poses
        precomputed_dist_2d: (N_r, N_q, K) pre-computed distance function tensor used for decomposed batch-wise search
        precomputed_dist_3d: (N_t, N_q, K) pre-computed distance function tensor used for decomposed batch-wise search
        precomputed_point_dist_2d: (N_r, N_q, K) pre-computed point distance function tensor used for decomposed batch-wise search
        precomputed_point_dist_3d: (N_t, N_q, K) pre-computed point distance function tensor used for decomposed batch-wise search
        point_gamma: Power factor applied to point distance function values

    Returns:
        cost: (N_p,) or (N_t, N_r) tensor containing cost functions for each candidate translation and rotation
    """
    # point distance function
    dist_2d = precomputed_point_dist_2d[None, ...].repeat(trans_tensor.shape[0], 1, 1, 1) ** point_gamma  # (N_t, N_r, N_q, L)
    dist_3d = precomputed_point_dist_3d[:, None, ...].repeat(1, rot_mtx.shape[0], 1, 1) ** point_gamma  # (N_t, N_r, N_q, L)

    # line distance function
    line_dist_2d = precomputed_dist_2d[None, ...].repeat(trans_tensor.shape[0], 1, 1, 1)  # (N_t, N_r, N_q, L)
    line_dist_3d = precomputed_dist_3d[:, None, ...].repeat(1, rot_mtx.shape[0], 1, 1)  # (N_t, N_r, N_q, L)
    dist_2d = torch.cat([line_dist_2d, dist_2d], dim=-1)  # (N_t, N_r, N_q, 2*L)
    dist_3d = torch.cat([line_dist_3d, dist_3d], dim=-1)  # (N_t, N_r, N_q, 2*L)

    cost = torch.abs(dist_2d - dist_3d)  # (N_t, N_r, N_q, L) or (N_t, N_r, N_q, 2*L)

    # Inlier-based cost
    cost = (cost < 0.1).sum(-2).sum(-1)  # (N_t, N_r)
    cost = -cost
    return cost
