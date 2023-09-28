import torch
from ldl.utils import (
    cloud2idx,
    ij2coord,
    make_pano,
    sample_from_img,
)
import cv2
try:
    import kornia as K
    import kornia.feature as KF
except ImportError:
    pass
from ldl.superglue_models.matching import Matching
import numpy as np
from ldl.utils import rgb_to_grayscale
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from scipy.ndimage import binary_dilation
import torch.nn as nn
from ldl.edge_utils import split_func_2d_batch, split_func_3d_batch


def split_distance_cost(query_points, trans_tensor, rot_mtx, mask_2d, mask_3d, edge_2d, starts_3d, ends_3d, method='mean'):
    """
    Measure cost functions from decomposed distance functions

    Args:
        query_points: (N_q, 3) tensor containing points to query
        trans_tensor: (N_t, 3) tensor containing translation candidate poses
        rot_mtx: (N_r, 3, 3) tensor containing rotation candidate poses
        mask_2d: (N_2D, K) or (N_r, N_2D, K) tensor containing masks for each 2D line, which may belong to K edge types
        mask_3d: (N_3D, K) or (N_r, N_3D, K) tensor containing masks for each 3D line, which may belong to K edge types
        edge_2d: (N_2D, 9) tensor containing 2D edges in [normals starts ends] format
        starts_3d: (N_3D, 3) tensor containing 3D edge start points
        ends_3d: (N_3D, 3) tensor containing 3D edge end points
        method: String specifying how to compute discepancies between 2D, 3D distance functions
    Returns:
        cost: (N_t, N_r) tensor containing cost functions for each candidate translation and rotation
    """
    dist_2d = split_func_2d_batch(query_points, edge_2d, mask_2d)  # (N_r, N_q, K)
    step = 100
    dist_3d = []
    num_loop = trans_tensor.shape[0] // step + 1 if trans_tensor.shape[0] % step != 0 else trans_tensor.shape[0] // step
    for i in range(num_loop):
        dist_3d.append(split_func_3d_batch(query_points, starts_3d, ends_3d, trans_tensor[step * i: step * (i + 1)], rot_mtx, mask_3d))
    dist_3d = torch.cat(dist_3d, dim=0)  # (N_t, N_r, N_q, K)
    cost = torch.abs(dist_2d.unsqueeze(0) - dist_3d)  # (N_t, N_r, N_q, K)
    if method == 'mean':
        cost = cost.mean(-2)  # (N_t, N_r, K)
        cost = cost.mean(-1)  # (N_t, N_r)
    elif method == 'median':
        cost = cost.median(-2).values  # (N_t, N_r, K)
        cost = cost.mean(-1)  # (N_t, N_r)
    elif method == 'inlier':
        cost = (cost < 0.1).sum(-2).sum(-1)  # (N_t, N_r)
        cost = -cost
    elif method == 'weighted_inlier':
        weight = (cost < 0.1).reshape(-1, 3).sum(0) / (cost < 0.1).sum()
        cost = ((cost < 0.1).sum(-2) * weight.reshape(1, 1, -1)).sum(-1)
        cost = -cost
    elif method == 'smooth_l1':
        cost = torch.nn.functional.smooth_l1_loss(cost, torch.zeros_like(cost), reduction='none', beta=0.1)
        cost = cost.mean(-2).mean(-1)
    elif method == 'huber':
        delta = 0.1
        cost[cost < delta] = 0.5 * cost[cost < delta] ** 2
        cost[cost >= delta] = delta * (cost[cost >= delta] - 0.5 * delta)
        cost = cost.mean(-2).mean(-1)
    elif method == 'bell_l1':
        delta = .1
        cost[cost > delta] = delta
        cost = cost.mean(-2).mean(-1)
    elif method == 'bell_l2':
        delta = 1.
        cost[cost > delta] = 0.5 * delta ** 2
        cost[cost <= delta] = 0.5 * cost[cost <= delta] ** 2
        cost = cost.mean(-2).mean(-1)
    return cost


def refine_from_point_ransac(trimmed_trans, trimmed_rot, query_img, match_model, map_kpts, map_desc, map_scores, min_trans_idx):
    """
    Refine top-k poses by using feature matching, here both rotation and translation is optimized

    Args:
        trimmed_trans: (N_k, 3) tensor containing top-k translations
        trimmed_rot: (N_k, 3, 3) tensor containing top-k rotations
        query_img: (H, W, 3) numpy array containing query image
        match_model: Torch pretrained network containing facilities for matching image features
        match_method: String specifying which matcher to used (currently SuperGlue or LoFTR)
        map_kpts: List containing keypoint xyz coordinates for each translation
        map_desc: List containing keypoint descriptors for each translation
        map_scores: List containing keypoint scores for each translation
        min_trans_idx: List containing indices of top-k trimmed translations

    Returns:
        tgt_trans: (3, ) tensor containing optimized translation
        tgt_rot: (3, 3) tensor containing optimized rotation
    """
    # trimmed_trans is (K, 3) and trimmed_rot is (K, 3, 3)
    num_k = trimmed_trans.shape[0]
    H, W = query_img.shape[:2]
    device = trimmed_trans.device
    if match_model is None:
        config = {
            'superpoint': {
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': -1
            },
            'superglue': {
                'weights': 'indoor',
                'sinkhorn_iterations': 20,
                'match_threshold': 0.2,
            }
        }
        matching = Matching(config).eval().to(device)
    else:
        matching = match_model
    
    tgt_img = torch.from_numpy(query_img).float() / 255.
    tgt_img = tgt_img.to(device).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

    tgt_img = rgb_to_grayscale(tgt_img)
    keys = ['keypoints', 'scores', 'descriptors']
    tgt_pred = matching.superpoint({'image': tgt_img[0:1]})
    tgt_pred = {k+'0': tgt_pred[k] for k in keys}
    tgt_pred['image0'] = tgt_img[0:1]

    last_data = {k+'0': [] for k in keys}
    last_data['image0'] = tgt_img[0:1]
    last_data['scores0'].append(tgt_pred['scores0'][0])
    last_data['keypoints0'].append(tgt_pred['keypoints0'][0])
    last_data['descriptors0'].append(tgt_pred['descriptors0'][0])

    # Create input dict & put it as input
    matches = {'batch_indexes': [], 'keypoints0': [], 'keypoints1': [], 'confidence': []}
    ref_match_list = []
    for idx in range(num_k):
        ref_data = {}
        ref_data['image1'] = torch.zeros_like(tgt_img[0: 1])
        ref_kpts = map_kpts[min_trans_idx[idx]]
        transform_kpts = (ref_kpts - trimmed_trans[idx: idx+1]) @ trimmed_rot[idx].T

        # Project 3D points to 2D image frame
        map_coords = (cloud2idx(transform_kpts) + 1) / 2
        map_coords[:, 0] *= (ref_data['image1'].shape[-1] - 1)
        map_coords[:, 1] *= (ref_data['image1'].shape[-2] - 1)
        map_coords = map_coords.round()
        ref_data['keypoints1'] = [map_coords]
        ref_data['scores1'] = [map_scores[min_trans_idx[idx]]]
        ref_data['descriptors1'] = [map_desc[min_trans_idx[idx]]]
        with torch.no_grad():
            pred = matching({**last_data, **ref_data})
        kpts0 = last_data['keypoints0'][0]
        kpts1 = ref_data['keypoints1'][0]
        pred_matches = pred['matches0'][0]
        confidence = pred['matching_scores0'][0]

        valid = pred_matches > -1
        num_keypts = valid.sum()
        mkpts0 = kpts0[valid]
        if pred_matches[valid].shape[0] != 0:
            mkpts1 = kpts1[pred_matches[valid]]
            ref_match_list.append(pred_matches[valid].tolist())
        else:  # Match failure
            mkpts1 = mkpts0
            ref_match_list.append([0 for i in range(mkpts1.shape[0])])
        confidence = confidence[valid]

        matches['batch_indexes'].append(idx * torch.ones(num_keypts, dtype=torch.int))
        matches['keypoints0'].append(mkpts0)
        matches['keypoints1'].append(mkpts1)
        matches['confidence'].append(confidence)
    matches['batch_indexes'] = torch.cat(matches['batch_indexes'], dim=0)
    matches['keypoints0'] = torch.cat(matches['keypoints0'], dim=0)
    matches['keypoints1'] = torch.cat(matches['keypoints1'], dim=0)
    matches['confidence'] = torch.cat(matches['confidence'], dim=0)

    batch_idx = matches['batch_indexes']

    # Choose view with largest number of matches
    batch_confidence = torch.tensor([matches['confidence'][batch_idx == idx].sum() for idx in range(num_k)])
    best_idx = batch_confidence.argmax()

    best_trans = trimmed_trans[best_idx]
    best_rot = trimmed_rot[best_idx]

    # Note that matches['keypoints'] are in (x, y) order, similar to torch.nn.grid_sample
    tgt_sphere = ij2coord(torch.flip(matches['keypoints0'][batch_idx == best_idx], [-1]), (H, W))  # (N, 3)
    best_coord = map_kpts[min_trans_idx[best_idx]][ref_match_list[best_idx]]
    best_conf = matches['confidence'][batch_idx == best_idx]

    tgt_trans = best_trans.clone().detach().unsqueeze(0)  # (1, 3)
    tgt_rot = best_rot.clone().detach()  # (3, 3)

    # Skip optimization for empty best_coord_arr
    if best_coord.numel() == 0 or best_conf.shape[0] < 5:  # These are cases where PnP cannot be applied
        optimize = False
    else:
        optimize = True

    # Optimize

    if optimize:
        try:
            conf_idx = torch.argsort(best_conf, descending=True)  # Use SuperGlue confidences for guiding search

            # RANSAC Loop
            x = best_coord[conf_idx[:100]].cpu().numpy()  # (N_3D, 3)
            y = tgt_sphere[conf_idx[:100]].cpu().numpy()  # (N_2D, 3)

            # Renormalize y for putting into efficient epnp
            y /= y[..., -1:]
            y = y[..., :2]
            
            sol = cv2.solvePnPRansac(x, y, np.eye(3), distCoeffs=None, iterationsCount=1000, reprojectionError=.1, flags=cv2.SOLVEPNP_ITERATIVE, rvec=np.zeros(3), tvec=np.zeros(3))

            tgt_rot = torch.from_numpy(cv2.Rodrigues(sol[1])[0]).float().to(device)  # (3, 3)
            tgt_trans = (- tgt_rot.T @ torch.from_numpy(sol[2]).float().to(device)).T  # (1, 3)
        except cv2.error as e:
            pass

    max_t_error = 1e6
    max_r_error = 60
    trans_error = (best_trans.cpu() - tgt_trans.cpu().squeeze()).norm().item()
    rot_error = torch.matmul(torch.transpose(best_rot, dim0=-2, dim1=-1), tgt_rot.unsqueeze(0))
    rot_error = torch.diagonal(rot_error, dim1=-2, dim2=-1).sum(-1)
    if rot_error < -1:
        rot_error = -2 - rot_error
    elif rot_error > 3:
        rot_error = 6 - rot_error
    rot_error = torch.rad2deg(torch.abs(torch.arccos((rot_error - 1) / 2))).item()

    if trans_error > max_t_error or rot_error > max_r_error:
        tgt_trans = best_trans.clone().detach().unsqueeze(0)
        tgt_rot = best_rot.clone().detach()

    return tgt_trans, tgt_rot


def get_matcher(cfg, device='cpu'):
    # Return appropriate matching module
    match_model_type = getattr(cfg, 'match_model_type', 'SuperGlue')
    if match_model_type == 'SuperGlue':
        superglue_cfg = {
                'superpoint': {
                    'nms_radius': getattr(cfg, 'nms_radius', 4),
                    'keypoint_threshold': getattr(cfg, 'keypoint_threshold', 0.005),
                    'max_keypoints': getattr(cfg, 'max_keypoints', -1)
                },
                'superglue': {
                    'weights': getattr(cfg, 'weights', 'indoor'),
                    'sinkhorn_iterations': getattr(cfg, 'sinkhorn_iterations', 20),
                    'match_threshold': getattr(cfg, 'match_threshold', 0.2),
                }
            }
        match_model = Matching(superglue_cfg).eval().to(device)
    elif match_model_type == 'LoFTR':
        match_model = KF.LoFTR(pretrained=getattr(cfg, 'weights', 'indoor')).to(device)
    else:
        raise ValueError("Invalid model type")

    return match_model
