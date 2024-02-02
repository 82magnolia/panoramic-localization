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
from utils import generate_trans_points, generate_rot_points, sampling_histogram_pose_search, out_of_room
import data_utils
from log_utils import PoseLogger, save_logger
from piccolo.sampling_loss import refine_pose_sampling_loss, refine_pose_sampling_loss_batch
from dict_utils import get_init_dict_piccolo


def localize(cfg: NamedTuple, log_dir: str):
    """
    Main function for performing localization in Stanford2D-3D-S dataset.

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
    sample_rate = getattr(cfg, 'sample_rate', 1)
    top_k_candidate = getattr(cfg, 'top_k_candidate', 5)
    num_intermediate = getattr(cfg, 'num_intermediate', 20)

    logger = PoseLogger(log_dir)

    filenames = data_utils.get_filename(dataset, split_name=split_name)

    if room_name is not None:
        if isinstance(room_name, str):
            filenames = [file_name for file_name in filenames if room_name in file_name]
        elif isinstance(room_name, list):
            filenames = [file_name for file_name in filenames if any([rm in file_name for rm in room_name])]
    if scene_number is not None:
        filenames = [file_name for file_name in filenames if "scene_{}".format(scene_number) in file_name]

    well_posed = 0
    valid_trial = 0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    past_pcd_name = ""

    # Optionally resize image
    init_downsample_h = getattr(cfg, 'init_downsample_h', 1)
    init_downsample_w = getattr(cfg, 'init_downsample_w', 1)
    main_downsample_h = getattr(cfg, 'main_downsample_h', 1)
    main_downsample_w = getattr(cfg, 'main_downsample_w', 1)

    # Main localization loop
    for trial, filename in enumerate(filenames):
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

        # Load point cloud
        if past_pcd_name != pcd_name:
            xyz_np, rgb_np = data_utils.read_pcd(dataset, pcd_name=pcd_name, sample_rate=sample_rate)
            xyz = torch.from_numpy(xyz_np).float()
            rgb = torch.from_numpy(rgb_np).float()

            xyz = xyz.to(device)
            rgb = rgb.to(device)

        orig_img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
        orig_img = cv2.resize(orig_img, (2048, 1024))

        # Optionally sharpen or match color histograms between point cloud and query image
        sharpen_color = getattr(cfg, 'sharpen_color', False)
        match_color = getattr(cfg, 'match_color', False)
        num_bins = getattr(cfg, 'num_bins', 256)

        mod_img = torch.from_numpy(orig_img).float() / 255.
        mod_img = mod_img.to(device)

        if sharpen_color or match_color:
            if past_pcd_name == pcd_name:
                rgb = torch.from_numpy(rgb_np).float()
                rgb = rgb.to(device)
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

        init_dict = get_init_dict_piccolo(cfg)

        # Point cloud inlier filtering for initialization
        init_input_xyz = xyz
        init_input_rgb = rgb

        # Change pcd_name, scene_no to new name
        past_pcd_name = pcd_name

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
            gt_trans = gt_trans.cpu().numpy()
            gt_rot = gt_rot.cpu().numpy()
            
            min_ind = result[:, 2].argmin()
            t = (result[:, 0])[min_ind]
            r = (result[:, 1])[min_ind]

            t_error = np.linalg.norm(gt_trans - np.array(t.clone().detach().cpu()))
            r_error = np.trace(np.matmul(np.transpose(r.clone().detach().cpu()), gt_rot))
            if r_error < -1:
                r_error = -2 - r_error
            elif r_error > 3:
                r_error = 6 - r_error
            r_error = np.rad2deg(np.abs(np.arccos((r_error - 1) / 2)))

        logger.add_error(t_error, r_error, filename, f"{room_type}_{room_no}", split_type)
        logger.add_estimate(filename, t.squeeze(), r, f"{room_type}_{room_no}", split_type)

        print("=============== CURRENT RESULTS ===============")
        print("t-error: ", t_error)
        print("r-error: ", r_error)

        if (t_error < 0.1) and (r_error < 5):
            well_posed += 1
        print("Accuracy: ", well_posed / valid_trial)
        print("===============================================")

    # Calculate statistics and save logger
    logger.calc_statistics('room')
    logger.calc_statistics('split')
    logger.calc_statistics('room_split')
    logger.calc_statistics('total')
    save_logger(os.path.join(log_dir, getattr(cfg, 'log_name', 'result.pkl')), logger)
