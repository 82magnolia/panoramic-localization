from typing import NamedTuple
from matplotlib.pyplot import get
import numpy as np


def get_init_dict_ldl(cfg: NamedTuple):
    xy_only = getattr(cfg, 'xy_only', False)
    num_trans = getattr(cfg, 'num_trans', 50)
    yaw_only = getattr(cfg, 'yaw_only', False)
    num_yaw = getattr(cfg, 'num_yaw', 4)
    num_pitch = getattr(cfg, 'num_pitch', 0)
    num_roll = getattr(cfg, 'num_roll', 0)

    max_yaw = getattr(cfg, 'max_yaw', 2 * np.pi)
    min_yaw = getattr(cfg, 'min_yaw', 0)
    max_pitch = getattr(cfg, 'max_pitch', 2 * np.pi)
    min_pitch = getattr(cfg, 'min_pitch', 0)
    max_roll = getattr(cfg, 'max_roll', 2 * np.pi)
    min_roll = getattr(cfg, 'min_roll', 0)

    x_max = getattr(cfg, 'x_max', None)
    x_min = getattr(cfg, 'x_min', None)
    y_max = getattr(cfg, 'y_max', None)
    y_min = getattr(cfg, 'y_min', None)
    z_max = getattr(cfg, 'z_max', None)
    z_min = getattr(cfg, 'z_min', None)

    z_prior = getattr(cfg, 'z_prior', None)
    dataset = cfg.dataset
    trans_init_mode = getattr(cfg, 'trans_init_mode', 'quantile')

    init_dict = {'xy_only': xy_only,
        'num_trans': num_trans,
        'yaw_only': yaw_only,
        'num_yaw': num_yaw,
        'num_pitch': num_pitch,
        'num_roll': num_roll,
        'max_yaw': max_yaw,
        'min_yaw': min_yaw,
        'max_pitch': max_pitch,
        'min_pitch': min_pitch,
        'max_roll': max_roll,
        'min_roll': min_roll,
        'z_prior': z_prior,
        'dataset': dataset,
        'trans_init_mode': trans_init_mode,
        'x_max': x_max,
        'x_min': x_min,
        'y_max': y_max,
        'y_min': y_min,
        'z_max': z_max,
        'z_min': z_min,
    }

    return init_dict


def get_init_dict_cpo(cfg: NamedTuple):
    xy_only = getattr(cfg, 'xy_only', False)
    num_trans = getattr(cfg, 'num_trans', 50)
    yaw_only = getattr(cfg, 'yaw_only', False)
    num_yaw = getattr(cfg, 'num_yaw', 8)
    num_pitch = getattr(cfg, 'num_pitch', 8)
    num_roll = getattr(cfg, 'num_roll', 8)

    max_yaw = getattr(cfg, 'max_yaw', 2 * np.pi)
    min_yaw = getattr(cfg, 'min_yaw', 0)
    max_pitch = getattr(cfg, 'max_pitch', 2 * np.pi)
    min_pitch = getattr(cfg, 'min_pitch', 0)
    max_roll = getattr(cfg, 'max_roll', 2 * np.pi)
    min_roll = getattr(cfg, 'min_roll', 0)

    x_max = getattr(cfg, 'x_max', None)
    x_min = getattr(cfg, 'x_min', None)
    y_max = getattr(cfg, 'y_max', None)
    y_min = getattr(cfg, 'y_min', None)
    z_max = getattr(cfg, 'z_max', None)
    z_min = getattr(cfg, 'z_min', None)

    z_prior = getattr(cfg, 'z_prior', None)
    dataset = cfg.dataset
    sample_rate_for_init = getattr(cfg, 'sample_rate_for_init', None)
    trans_init_mode = getattr(cfg, 'trans_init_mode', 'quantile')

    num_split_h = getattr(cfg, 'num_split_h', 2)
    num_split_w = getattr(cfg, 'num_split_w', 4)

    hist_stat = getattr(cfg, 'hist_stat', 'mean')
    sin_hist = getattr(cfg, 'sin_hist', False)  # If True, uses sin weights for histogram intersection
    benchmark_grid = getattr(cfg, 'benchmark_grid', False)  # If True, fixes grid numbers to octree numbers
    is_inlier_dict = False

    height_enforce = False  # Enforce height value found from estimation
    height_uncertainty = False  # Add uncertainty values to enforced height value

    renew_hist = False  # If True, renew hist for fast initialization
    
    init_dict = {'xy_only': xy_only,
        'num_trans': num_trans,
        'yaw_only': yaw_only,
        'num_yaw': num_yaw,
        'num_pitch': num_pitch,
        'num_roll': num_roll,
        'max_yaw': max_yaw,
        'min_yaw': min_yaw,
        'max_pitch': max_pitch,
        'min_pitch': min_pitch,
        'max_roll': max_roll,
        'min_roll': min_roll,
        'z_prior': z_prior,
        'dataset': dataset,
        'sample_rate_for_init': sample_rate_for_init,
        'trans_init_mode': trans_init_mode,
        'x_max': x_max,
        'x_min': x_min,
        'y_max': y_max,
        'y_min': y_min,
        'z_max': z_max,
        'z_min': z_min,
        'num_split_h': num_split_h,
        'num_split_w': num_split_w,
        'hist_stat': hist_stat,
        'height_enforce': height_enforce,
        'height_uncertainty': height_uncertainty,
        'sin_hist': sin_hist,
        'benchmark_grid': benchmark_grid,
        'is_inlier_dict': is_inlier_dict,
        'renew_hist': renew_hist}

    return init_dict


def get_init_dict_piccolo(cfg: NamedTuple):
    xy_only = getattr(cfg, 'xy_only', True)
    num_trans = getattr(cfg, 'num_trans', 50)
    yaw_only = getattr(cfg, 'yaw_only', True)
    num_yaw = getattr(cfg, 'num_yaw', 4)
    num_pitch = getattr(cfg, 'num_pitch', 0)
    num_roll = getattr(cfg, 'num_roll', 0)

    max_yaw = getattr(cfg, 'max_yaw', 2 * np.pi)
    min_yaw = getattr(cfg, 'min_yaw', 0)
    max_pitch = getattr(cfg, 'max_pitch', 2 * np.pi)
    min_pitch = getattr(cfg, 'min_pitch', 0)
    max_roll = getattr(cfg, 'max_roll', 2 * np.pi)
    min_roll = getattr(cfg, 'min_roll', 0)

    x_max = getattr(cfg, 'x_max', None)
    x_min = getattr(cfg, 'x_min', None)
    y_max = getattr(cfg, 'y_max', None)
    y_min = getattr(cfg, 'y_min', None)
    z_max = getattr(cfg, 'z_max', None)
    z_min = getattr(cfg, 'z_min', None)

    z_prior = getattr(cfg, 'z_prior', None)
    dataset = cfg.dataset
    sample_rate_for_init = getattr(cfg, 'sample_rate_for_init', None)
    trans_init_mode = getattr(cfg, 'trans_init_mode', 'quantile')

    num_split_h = getattr(cfg, 'num_split_h', 2)
    num_split_w = getattr(cfg, 'num_split_w', 4)

    init_dict = {'xy_only': xy_only,
        'num_trans': num_trans,
        'yaw_only': yaw_only,
        'num_yaw': num_yaw,
        'num_pitch': num_pitch,
        'num_roll': num_roll,
        'max_yaw': max_yaw,
        'min_yaw': min_yaw,
        'max_pitch': max_pitch,
        'min_pitch': min_pitch,
        'max_roll': max_roll,
        'min_roll': min_roll,
        'z_prior': z_prior,
        'dataset': dataset,
        'sample_rate_for_init': sample_rate_for_init,
        'trans_init_mode': trans_init_mode,
        'x_max': x_max,
        'x_min': x_min,
        'y_max': y_max,
        'y_min': y_min,
        'z_max': z_max,
        'z_min': z_min,
        'num_split_h': num_split_h,
        'num_split_w': num_split_w}

    return init_dict


def get_init_dict_fgpl(cfg: NamedTuple):
    xy_only = getattr(cfg, 'xy_only', False)
    num_trans = getattr(cfg, 'num_trans', 50)
    yaw_only = getattr(cfg, 'yaw_only', False)
    num_yaw = getattr(cfg, 'num_yaw', 4)
    num_pitch = getattr(cfg, 'num_pitch', 4)
    num_roll = getattr(cfg, 'num_roll', 4)

    max_yaw = getattr(cfg, 'max_yaw', 2 * np.pi)
    min_yaw = getattr(cfg, 'min_yaw', 0)
    max_pitch = getattr(cfg, 'max_pitch', 2 * np.pi)
    min_pitch = getattr(cfg, 'min_pitch', 0)
    max_roll = getattr(cfg, 'max_roll', 2 * np.pi)
    min_roll = getattr(cfg, 'min_roll', 0)
    include_end_points = getattr(cfg, 'include_end_points', False)  # Optionally include end points for rotation

    x_max = getattr(cfg, 'x_max', None)
    x_min = getattr(cfg, 'x_min', None)
    y_max = getattr(cfg, 'y_max', None)
    y_min = getattr(cfg, 'y_min', None)
    z_max = getattr(cfg, 'z_max', None)
    z_min = getattr(cfg, 'z_min', None)

    z_prior = getattr(cfg, 'z_prior', None)
    dataset = cfg.dataset
    sample_rate_for_init = getattr(cfg, 'sample_rate_for_init', None)
    trans_init_mode = getattr(cfg, 'trans_init_mode', 'quantile')

    num_split_h = getattr(cfg, 'num_split_h', 2)
    num_split_w = getattr(cfg, 'num_split_w', 4)

    hist_stat = getattr(cfg, 'hist_stat', 'mean')
    sin_hist = getattr(cfg, 'sin_hist', False)  # If True, uses sin weights for histogram intersection
    benchmark_grid = getattr(cfg, 'benchmark_grid', False)  # If True, fixes grid numbers to octree numbers
    is_inlier_dict = False

    height_enforce = False  # Enforce height value found from estimation
    height_uncertainty = False  # Add uncertainty values to enforced height value

    renew_hist = False  # If True, renew hist for fast initialization
    nmin = getattr(cfg, 'nmin', 3)  # Parameter determining size of octree
    voxel_size = getattr(cfg, 'voxel_size', 0.5)  # Size of voxel for initialization
    quantile_thres = getattr(cfg, 'quantile_thres', 0.05)  # Quantile threshold for generating voxels

    init_dict = {'xy_only': xy_only,
        'num_trans': num_trans,
        'yaw_only': yaw_only,
        'num_yaw': num_yaw,
        'num_pitch': num_pitch,
        'num_roll': num_roll,
        'max_yaw': max_yaw,
        'min_yaw': min_yaw,
        'max_pitch': max_pitch,
        'min_pitch': min_pitch,
        'max_roll': max_roll,
        'min_roll': min_roll,
        'z_prior': z_prior,
        'dataset': dataset,
        'sample_rate_for_init': sample_rate_for_init,
        'trans_init_mode': trans_init_mode,
        'x_max': x_max,
        'x_min': x_min,
        'y_max': y_max,
        'y_min': y_min,
        'z_max': z_max,
        'z_min': z_min,
        'num_split_h': num_split_h,
        'num_split_w': num_split_w,
        'hist_stat': hist_stat,
        'height_enforce': height_enforce,
        'height_uncertainty': height_uncertainty,
        'sin_hist': sin_hist,
        'benchmark_grid': benchmark_grid,
        'is_inlier_dict': is_inlier_dict,
        'renew_hist': renew_hist,
        'nmin': nmin,
        'voxel_size': voxel_size,
        'quantile_thres': quantile_thres,
        'include_end_points': include_end_points}

    return init_dict
