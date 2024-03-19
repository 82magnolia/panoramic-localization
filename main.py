from parse_utils import parse_ini, parse_value
import argparse
import os
from collections import namedtuple
import configparser
import torch
import numpy as np
import random
import importlib


if __name__ == '__main__':
    # General config parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Config file to use for running experiments", default=None, type=str)
    parser.add_argument("--log", help="Log directory for logging accuracy", default="./log", type=str)
    parser.add_argument('--override', default=None, help='Arguments for overriding config')
    parser.add_argument('--method', required=True, help='Localization method to use')
    
    # Single image & point cloud query configs
    parser.add_argument("--single", action='store_true', help='If True, perform localization on single sample')
    parser.add_argument("--query_img", default=None, help="Path to query image to localize for single sample case")
    parser.add_argument("--color_pcd", default=None, help="Path to colored point cloud to use as the map for single sample case")
    parser.add_argument("--line_pcd", default=None, help="Path to line cloud to use as the map for single sample case")
    parser.add_argument("--crop_up_down", action='store_true', help="If True, crops panorama's upper and lower regions during localization")
    args = parser.parse_args()
    cfg = parse_ini(args.config)

    log_dir = args.log
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if args.override is not None:
        equality_split = args.override.split('=')
        num_equality = len(equality_split)
        assert num_equality > 0
        if num_equality == 2:
            override_dict = {equality_split[0]: parse_value(equality_split[1])}
        else:
            keys = [equality_split[0]]  # First key
            keys += [equality.split(',')[-1] for equality in equality_split[1:-1]]  # Other keys
            values = [equality.replace(',' + key, '') for equality, key in zip(equality_split[1:-1], keys[1:])]  # Get values other than last field
            values.append(equality_split[-1])  # Get last value
            values = [value.replace('[', '').replace(']', '') for value in values]

            override_dict = {key: parse_value(value) for key, value in zip(keys, values)}

        cfg_dict = cfg._asdict()

        Config = namedtuple('Config', tuple(set(cfg._fields + tuple(override_dict.keys()))))
        
        cfg_dict.update(override_dict)

        cfg = Config(**cfg_dict)

    config = configparser.ConfigParser()
    config.add_section('Default')

    cfg_dict = cfg._asdict()

    for key in cfg_dict:
        if key != 'name':
            config['Default'][key] = str(cfg_dict[key]).replace('[', '').replace(']', '')
        else:
            config['Default'][key] = str(cfg_dict[key])

    with open(os.path.join(args.log, 'config.ini'), 'w') as configfile:
        config.write(configfile)

    # Fix seed
    seed = getattr(cfg, 'seed', 0)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Branch on dataset
    if args.method in ['ldl', 'cpo', 'piccolo', 'fgpl']:
        if args.single:
            if args.method in ['ldl']:  # Line-based methods
                importlib.import_module(f"{args.method}.localize_single").localize(cfg, log_dir, args.query_img, args.color_pcd, args.line_pcd, args.crop_up_down)
            elif args.method in ['fgpl']:  # Fully geometric methods
                importlib.import_module(f"{args.method}.localize_single").localize(cfg, log_dir, args.query_img, args.line_pcd, args.crop_up_down)
            else:  # Color-based methods
                importlib.import_module(f"{args.method}.localize_single").localize(cfg, log_dir, args.query_img, args.color_pcd)
        else:
            importlib.import_module(f"{args.method}.localize").localize(cfg, log_dir)
    else:
        raise NotImplementedError("Other pipelines currently not supported")
