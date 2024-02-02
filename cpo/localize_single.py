import torch
import cv2
import os
import time
from typing import NamedTuple
import numpy as np
from log_utils import PoseLogger, save_logger
import data_utils
from dict_utils import get_init_dict_cpo
from color_utils import color_match, color_mod
from utils import (
    out_of_room,
    generate_trans_points,
    generate_rot_points,
    make_score_map_2d,
    process_score_map_2d,
    make_score_map_3d,
    histogram_pose_search,
    make_pano
)
from cpo.sampling_loss import refine_pose_sampling_loss


def localize(cfg: NamedTuple, log_dir: str, query_img_path: str, color_pcd_path: str):
    """
    Main function for performing localization against a single 3D map and image.

    Args:
        cfg: Config file
        log_dir: Directory in which logs will be saved
        query_img_path: Path to query image
        color_pcd_path: Path to colored point cloud, saved in .txt format with colors in [0...255]
    
    Returns:
        None
    """

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    dataset = cfg.dataset
    supported_datasets = ['omniscenes', 'stanford']  # Currently supported datasets

    if dataset not in supported_datasets:
        raise ValueError("Invalid dataset")

    # Algorithm configs
    sample_rate = getattr(cfg, 'sample_rate', 1)
    top_k_candidate = getattr(cfg, 'top_k_candidate', 5)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Optionally resize image
    init_downsample_h = getattr(cfg, 'init_downsample_h', 1)
    init_downsample_w = getattr(cfg, 'init_downsample_w', 1)
    main_downsample_h = getattr(cfg, 'main_downsample_h', 1)
    main_downsample_w = getattr(cfg, 'main_downsample_w', 1)

    # Load point cloud
    xyz_np, rgb_np = data_utils.read_txt_pcd(color_pcd_path, sample_rate=sample_rate)
    xyz = torch.from_numpy(xyz_np).float()
    rgb = torch.from_numpy(rgb_np).float()

    xyz = xyz.to(device)
    rgb = rgb.to(device)

    # Read image
    orig_img = cv2.cvtColor(cv2.imread(query_img_path), cv2.COLOR_BGR2RGB)
    orig_img = cv2.resize(orig_img, (2048, 1024))  # CPO assumes images of this size

    # Optionally sharpen or match color histograms between point cloud and query image
    sharpen_color = getattr(cfg, 'sharpen_color', False)
    match_color = getattr(cfg, 'match_color', False)
    num_bins = getattr(cfg, 'num_bins', 256)

    mod_img = torch.from_numpy(orig_img).float() / 255.
    mod_img = mod_img.to(device)

    if sharpen_color or match_color:
        if match_color:
            new_img = color_match(mod_img, rgb)
            orig_img = (255 * new_img.cpu().numpy()).astype(np.uint8)
        if sharpen_color:
            new_img, rgb = color_mod(mod_img, rgb, num_bins)
            orig_img = (255 * new_img.cpu().numpy()).astype(np.uint8)

    img = cv2.resize(orig_img, (orig_img.shape[1] // init_downsample_w, orig_img.shape[0] // init_downsample_h))
    img = torch.from_numpy(img).float() / 255.
    img = img.to(device)

    # Set input point cloud
    input_xyz = xyz

    init_dict = get_init_dict_cpo(cfg)

    # Attributes used for inlier filtering
    inlier_init_dict = dict(init_dict)
    inlier_init_dict['is_inlier_dict'] = True
    inlier_init_dict['num_trans'] = getattr(cfg, 'inlier_num_trans', init_dict['num_trans'])
    inlier_init_dict['num_yaw'] = getattr(cfg, 'inlier_num_yaw', 4)
    inlier_init_dict['num_pitch'] = getattr(cfg, 'inlier_num_pitch', 4)
    inlier_init_dict['num_roll'] = getattr(cfg, 'inlier_num_roll', 4)
    inlier_init_dict['trans_init_mode'] = getattr(cfg, 'inlier_trans_init_mode', 'quantile')

    inlier_test_trans = generate_trans_points(input_xyz, inlier_init_dict, device=input_xyz.device)
    inlier_test_rot = generate_rot_points(inlier_init_dict, device=input_xyz.device)
    inlier_num_split_h = getattr(cfg, 'inlier_num_split_h', 8)
    inlier_num_split_w = getattr(cfg, 'inlier_num_split_w', 16)
    margin = inlier_num_split_h // 8

    # 2D score map generation
    score_map_2d = make_score_map_2d(img, input_xyz, rgb, inlier_test_trans, inlier_test_rot, inlier_num_split_h, inlier_num_split_w, margin)
    score_map_2d_search = process_score_map_2d(torch.zeros(cfg.num_split_h, cfg.num_split_w, device=xyz.device), score_map_2d, 'preserve', 0.0)  # Score map for pose search
    score_map_2d_refine = process_score_map_2d(torch.from_numpy(orig_img).to(xyz.device), score_map_2d, 'preserve', 0.0).unsqueeze(-1)  # Score map for refinement

    # 3D score map generation
    pcd_weight = None
    init_input_xyz = input_xyz
    init_input_rgb = rgb
    score_map_3d = make_score_map_3d(img, xyz, rgb, inlier_test_trans, inlier_test_rot, inlier_num_split_h, inlier_num_split_w, 
        margin, match_rgb=match_color)
    
    pcd_weight = score_map_3d

    # Pose search
    rot = generate_rot_points(init_dict, device=img.device)
    trans = generate_trans_points(xyz, init_dict, device=img.device)

    input_trans, input_rot = histogram_pose_search(img, init_input_xyz, init_input_rgb, trans, rot, top_k_candidate,
        init_dict['num_split_h'], init_dict['num_split_w'], score_map_2d_search, init_dict['sin_hist'])

    # Pose refinement
    main_input_xyz = init_input_xyz
    main_input_rgb = init_input_rgb

    img = cv2.resize(orig_img, (orig_img.shape[1] // main_downsample_w, orig_img.shape[0] // main_downsample_h))
    img = torch.from_numpy(img).float() / 255.
    img = img.to(device)

    result = []
    for i in range(top_k_candidate):
        result.append(refine_pose_sampling_loss(img, main_input_xyz, main_input_rgb, input_trans, input_rot,
            i, cfg, img_weight=score_map_2d_refine, pcd_weight=pcd_weight))

    # Measure metrics
    with torch.no_grad():
        result = np.asarray(result, dtype=object)
        min_ind = result[:, 2].argmin()
        t = (result[:, 0])[min_ind]
        r = (result[:, 1])[min_ind]
        refined_trans = t.reshape(1, 3).to(device)
        refined_rot = r.to(device)

    # Visualize localization
    vis_img = cv2.resize(orig_img, (400, 200))
    result_img = make_pano((xyz - refined_trans) @ refined_rot.T, rgb)
    margin_img = np.zeros_like(vis_img[:, :100, :])
    final_img = np.concatenate([vis_img, margin_img, result_img], axis=1)
    final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(log_dir, 'result.png'), final_img)

    print(f"Saved localization visualization in {os.path.join(log_dir, 'result.png')}")
