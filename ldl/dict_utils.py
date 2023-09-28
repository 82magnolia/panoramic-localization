from typing import NamedTuple
from matplotlib.pyplot import get
import numpy as np


def get_init_dict(cfg: NamedTuple):
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
