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
    histogram_pose_search
)
from cpo.omniloc import refine_pose_sampling_loss


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

        # Read image
        orig_img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
        orig_img = cv2.resize(orig_img, (2048, 1024))  # CPO assumes images of this size
        
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
        if getattr(cfg, 'recycle_score_pcd', True):
            if past_pcd_name != pcd_name:  # Only conditionally make inlier map
                num_query = getattr(cfg, 'num_query', 1)
                score_map_3d = make_score_map_3d(img, xyz, rgb, inlier_test_trans, inlier_test_rot, inlier_num_split_h, inlier_num_split_w, 
                    margin, filename=filename, num_query=num_query, match_rgb=match_color)
        else:
            score_map_3d = make_score_map_3d(img, xyz, rgb, inlier_test_trans, inlier_test_rot, inlier_num_split_h, inlier_num_split_w, 
                margin, match_rgb=match_color)
        
        pcd_weight = score_map_3d

        # Pose search
        rot = generate_rot_points(init_dict, device=img.device)
        trans = generate_trans_points(xyz, init_dict, device=img.device)

        input_trans, input_rot = histogram_pose_search(img, init_input_xyz, init_input_rgb, trans, rot, top_k_candidate,
            init_dict['num_split_h'], init_dict['num_split_w'], score_map_2d_search, init_dict['sin_hist'])

        # Update past_pcd_name
        past_pcd_name = pcd_name

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
