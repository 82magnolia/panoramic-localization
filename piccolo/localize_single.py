import torch
import numpy as np
import random
import cv2
import os
import time
import csv
from torch.utils.tensorboard import SummaryWriter
from glob import glob
from typing import NamedTuple
from collections import defaultdict
from color_utils import color_mod, color_match
from utils import generate_trans_points, generate_rot_points, sampling_histogram_pose_search, out_of_room, make_pano
import data_utils
from log_utils import PoseLogger, save_logger
from piccolo.sampling_loss import refine_pose_sampling_loss, refine_pose_sampling_loss_batch
from dict_utils import get_init_dict_piccolo


def localize(cfg: NamedTuple, log_dir: str, query_img_path: str, color_pcd_path: str):
    """
    Main function for performing localization in Stanford2D-3D-S dataset.

    Args:
        cfg: Config file
        log_dir: Directory in which logs will be saved
        query_img_path: Path to query image
        color_pcd_path: Path to colored point cloud, saved in .txt format with colors in [0...255]
    
    Returns:
        None
    """
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # Algorithm configs
    sample_rate = getattr(cfg, 'sample_rate', 1)
    top_k_candidate = getattr(cfg, 'top_k_candidate', 5)
    num_intermediate = getattr(cfg, 'num_intermediate', 20)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load point cloud
    xyz_np, rgb_np = data_utils.read_txt_pcd(color_pcd_path, sample_rate=sample_rate)
    xyz = torch.from_numpy(xyz_np).float()
    rgb = torch.from_numpy(rgb_np).float()

    xyz = xyz.to(device)
    rgb = rgb.to(device)

    # Optionally resize image
    init_downsample_h = getattr(cfg, 'init_downsample_h', 1)
    init_downsample_w = getattr(cfg, 'init_downsample_w', 1)
    main_downsample_h = getattr(cfg, 'main_downsample_h', 1)
    main_downsample_w = getattr(cfg, 'main_downsample_w', 1)

    orig_img = cv2.cvtColor(cv2.imread(query_img_path), cv2.COLOR_BGR2RGB)
    orig_img = cv2.resize(orig_img, (2048, 1024))

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

    # Set image for pose search
    img = cv2.resize(orig_img, (orig_img.shape[1] // init_downsample_w, orig_img.shape[0] // init_downsample_h))
    img = torch.from_numpy(img).float() / 255.
    img = img.to(device)

    init_dict = get_init_dict_piccolo(cfg)

    # Point cloud inlier filtering for initialization
    init_input_xyz = xyz
    init_input_rgb = rgb

    # Pose search
    rot = generate_rot_points(init_dict, device=img.device)
    trans = generate_trans_points(xyz, init_dict, device=img.device)

    input_trans, input_rot = sampling_histogram_pose_search(img, init_input_xyz, init_input_rgb,
        trans, rot, top_k_candidate, init_dict['num_split_h'], init_dict['num_split_w'], num_intermediate)

    # Set image for refinement
    img = cv2.resize(orig_img, (orig_img.shape[1] // main_downsample_w, orig_img.shape[0] // main_downsample_h))
    img = torch.from_numpy(img).float() / 255.
    img = img.to(device)

    # Pose refinement
    result = []

    if getattr(cfg, 'refine_parallel', False):
        result.append(refine_pose_sampling_loss_batch(img, xyz, rgb, input_trans, input_rot, cfg))
    else:
        for i in range(top_k_candidate):
            result.append(refine_pose_sampling_loss(img, xyz, rgb, input_trans, input_rot, i, cfg))

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
