import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Union
try:
    from torch_scatter import scatter_min
except ImportError:
    pass
from tqdm import tqdm
from collections import defaultdict
from math import ceil
import matplotlib.pyplot as plt
from glob import glob
from typing import Optional
import math
from torch.nn.functional import cosine_similarity
import os
from scipy.ndimage import map_coordinates


def cloud2idx(xyz: torch.Tensor, batched: bool = False) -> torch.Tensor:
    """
    Change 3d coordinates to image coordinates ranged in [-1, 1].

    Args:
        xyz: (N, 3) torch tensor containing xyz values of the point cloud data
        batched: If True, performs batched operation with xyz considered as shape (B, N, 3)

    Returns:
        coord_arr: (N, 2) torch tensor containing transformed image coordinates
    """
    if batched:
        # first project 3d coordinates to a unit sphere and obtain vertical/horizontal angle

        # vertical angle
        theta = torch.unsqueeze(torch.atan2((torch.norm(xyz[..., :2], dim=-1)), xyz[..., 2] + 1e-6), -1)  # (B, N, 1)

        # horizontal angle
        phi = torch.atan2(xyz[..., 1:2], xyz[..., 0:1] + 1e-6)  # (B, N, 1)
        phi += np.pi

        sphere_cloud_arr = torch.cat([phi, theta], dim=-1)  # (B, N, 2)

        # image coordinates ranged in [0, 1]
        coord_arr = torch.stack([1.0 - sphere_cloud_arr[..., 0] / (np.pi * 2), sphere_cloud_arr[..., 1] / np.pi], dim=-1)
        # Rearrange so that the range is in [-1, 1]
        coord_arr = (2 * coord_arr - 1)  # (B, N, 2)

    else:
        # first project 3d coordinates to a unit sphere and obtain vertical/horizontal angle

        # vertical angle
        theta = torch.unsqueeze(torch.atan2((torch.norm(xyz[:, :2], dim=-1)), xyz[:, 2] + 1e-6), 1)

        # horizontal angle
        phi = torch.atan2(xyz[:, 1:2], xyz[:, 0:1] + 1e-6)
        phi += np.pi

        sphere_cloud_arr = torch.cat([phi, theta], dim=-1)

        # image coordinates ranged in [0, 1]
        coord_arr = torch.stack([1.0 - sphere_cloud_arr[:, 0] / (np.pi * 2), sphere_cloud_arr[:, 1] / np.pi], dim=-1)
        # Rearrange so that the range is in [-1, 1]
        coord_arr = (2 * coord_arr - 1)

    return coord_arr


def inv_cloud2idx(coord_arr: torch.Tensor):
    # Inversion of cloud2idx: given a (N, 2) coord_arr, returns a set of (N, 3) 3D points on a sphere.
    sphere_cloud_arr = (coord_arr + 1.) / 2.
    sphere_cloud_arr[:, 0] = (1.0 - sphere_cloud_arr[:, 0]) * (2 * np.pi)
    sphere_cloud_arr[:, 1] = np.pi * sphere_cloud_arr[:, 1]  # Contains [phi, theta] of sphere

    sphere_cloud_arr[:, 0] -= np.pi  # Subtraction to accomodate for cloud2idx

    sphere_xyz = torch.zeros(sphere_cloud_arr.shape[0], 3, device=coord_arr.device)
    sphere_xyz[:, 0] = torch.sin(sphere_cloud_arr[:, 1]) * torch.cos(sphere_cloud_arr[:, 0])
    sphere_xyz[:, 1] = torch.sin(sphere_cloud_arr[:, 1]) * torch.sin(sphere_cloud_arr[:, 0])
    sphere_xyz[:, 2] = torch.cos(sphere_cloud_arr[:, 1])

    return sphere_xyz


def sample_from_img(img: torch.Tensor, coord_arr: torch.Tensor, padding='zeros', mode='bilinear', batched=False) -> torch.Tensor:
    """
    Image sampling function
    Use coord_arr as a grid for sampling from img

    Args:
        img: (H, W, 3) torch tensor containing image RGB values
        coord_arr: (N, 2) torch tensor containing image coordinates, ranged in [-1, 1], converted from 3d coordinates
        padding: Padding mode to use for grid_sample
        mode: How to sample from grid
        batched: If True, assumes an additional batch dimension for coord_arr

    Returns:
        sample_rgb: (N, 3) torch tensor containing sampled RGB values
    """
    if batched:
        img = img.permute(2, 0, 1)
        img = torch.unsqueeze(img, 0)

        # sampling from img
        sample_arr = coord_arr.reshape(coord_arr.shape[0], coord_arr.shape[1], 1, 2)
        sample_arr = torch.clip(sample_arr, min=-0.99, max=0.99)
        sample_rgb = F.grid_sample(img.expand(coord_arr.shape[0], -1, -1, -1), sample_arr, mode=mode, align_corners=False, padding_mode=padding)

        sample_rgb = torch.squeeze(sample_rgb)  # (B, 3, N)
        sample_rgb = torch.transpose(sample_rgb, 1, 2)  # (B, N, 3)  

    else:
        img = img.permute(2, 0, 1)
        img = torch.unsqueeze(img, 0)

        # sampling from img
        sample_arr = coord_arr.reshape(1, -1, 1, 2)
        sample_arr = torch.clip(sample_arr, min=-0.99, max=0.99)
        sample_rgb = F.grid_sample(img, sample_arr, mode=mode, align_corners=False, padding_mode=padding)

        sample_rgb = torch.squeeze(torch.squeeze(sample_rgb, 0), 2)
        sample_rgb = torch.transpose(sample_rgb, 0, 1)

    return sample_rgb


def warp_from_img(img: torch.Tensor, coord_arr: torch.Tensor, padding='zeros', mode='bilinear') -> torch.Tensor:
    """
    Image warping function
    Use coord_arr as a grid for warping from img

    Args:
        img: (H, W, C) torch tensor containing image RGB values
        coord_arr: (H, W, 2) torch tensor containing image coordinates, ranged in [-1, 1], converted from 3d coordinates
        padding: Padding mode to use for grid_sample
        mode: How to sample from grid

    Returns:
        sample_rgb: (H, W, C) torch tensor containing sampled RGB values
    """

    img = img.permute(2, 0, 1)  # (C, H, W)
    img = torch.unsqueeze(img, 0)  # (1, C, H, W)

    # sampling from img
    sample_arr = coord_arr.unsqueeze(0)  # (1, H, W, 2)
    sample_arr = torch.clip(sample_arr, min=-0.99, max=0.99)
    sample_rgb = F.grid_sample(img, sample_arr, align_corners=False, padding_mode=padding, mode=mode)  # (1, C, H, W)

    sample_rgb = sample_rgb.squeeze(0).permute(1, 2, 0)  # (H, W, C)

    return sample_rgb


def ij2coord(ij_values, resolution):
    # Convert (N, 2) image ij-coordinates to 3D spherical coordinates
    coord_idx = torch.flip(ij_values.float(), [-1])
    coord_idx[:, 0] /= (resolution[1] - 1)
    coord_idx[:, 1] /= (resolution[0] - 1)

    coord_idx = 2. * coord_idx - 1.

    sphere_xyz = inv_cloud2idx(coord_idx)  # Points on sphere
    return sphere_xyz


def make_pano(xyz: torch.Tensor, rgb: torch.Tensor, resolution: Tuple[int, int] = (200, 400), 
        return_torch: bool = False, return_coord: bool = False, return_norm_coord: bool = False, default_white=False) -> Union[torch.Tensor, np.array]:
    """
    Make panorama image from xyz and rgb tensors

    Args:
        xyz: (N, 3) torch tensor containing xyz coordinates
        rgb: (N, 3) torch tensor containing rgb values, ranged in [0, 1]
        resolution: Tuple size of 2, returning panorama image of size resolution
        return_torch: if True, return image as torch.Tensor
                      if False, return image as numpy.array
        return_coord: If True, return coordinate in long format
        return_norm_coord: If True, return coordinate in normalized float format
        default_white: If True, defaults the color values to white

    Returns:
        image: (H, W, 3) torch.Tensor or numpy.array
    """

    with torch.no_grad():

        # project farther points first
        dist = torch.norm(xyz, dim=-1)
        mod_idx = torch.argsort(dist)
        mod_idx = torch.flip(mod_idx, dims=[0])
        mod_xyz = xyz.clone().detach()[mod_idx]
        mod_rgb = rgb.clone().detach()[mod_idx]

        orig_coord_idx = cloud2idx(mod_xyz)
        coord_idx = (orig_coord_idx + 1.0) / 2.0
        # coord_idx[:, 0] is x coordinate, coord_idx[:, 1] is y coordinate
        coord_idx[:, 0] *= (resolution[1] - 1)
        coord_idx[:, 1] *= (resolution[0] - 1)

        coord_idx = torch.flip(coord_idx, [-1])
        coord_idx = coord_idx.long()
        save_coord_idx = coord_idx.clone().detach()
        coord_idx = tuple(coord_idx.t())

        if default_white:
            image = torch.ones([resolution[0], resolution[1], 3], dtype=torch.float, device=xyz.device)
        else:
            image = torch.zeros([resolution[0], resolution[1], 3], dtype=torch.float, device=xyz.device)

        # color the image
        # pad by 1
        temp = torch.ones_like(coord_idx[0], device=xyz.device)
        coord_idx1 = (torch.clamp(coord_idx[0] + temp, max=resolution[0] - 1),
                      torch.clamp(coord_idx[1] + temp, max=resolution[1] - 1))
        coord_idx2 = (torch.clamp(coord_idx[0] + temp, max=resolution[0] - 1),
                      coord_idx[1])
        coord_idx3 = (torch.clamp(coord_idx[0] + temp, max=resolution[0] - 1),
                      torch.clamp(coord_idx[1] - temp, min=0))
        coord_idx4 = (torch.clamp(coord_idx[0] - temp, min=0),
                      torch.clamp(coord_idx[1] + temp, max=resolution[1] - 1))
        coord_idx5 = (torch.clamp(coord_idx[0] - temp, min=0),
                      coord_idx[1])
        coord_idx6 = (torch.clamp(coord_idx[0] - temp, min=0),
                      torch.clamp(coord_idx[1] - temp, min=0))
        coord_idx7 = (coord_idx[0],
                      torch.clamp(coord_idx[1] + temp, max=resolution[1] - 1))
        coord_idx8 = (coord_idx[0],
                      torch.clamp(coord_idx[1] - temp, min=0))

        image.index_put_(coord_idx8, mod_rgb, accumulate=False)
        image.index_put_(coord_idx7, mod_rgb, accumulate=False)
        image.index_put_(coord_idx6, mod_rgb, accumulate=False)
        image.index_put_(coord_idx5, mod_rgb, accumulate=False)
        image.index_put_(coord_idx4, mod_rgb, accumulate=False)
        image.index_put_(coord_idx3, mod_rgb, accumulate=False)
        image.index_put_(coord_idx2, mod_rgb, accumulate=False)
        image.index_put_(coord_idx1, mod_rgb, accumulate=False)
        image.index_put_(coord_idx, mod_rgb, accumulate=False)

        image = image * 255

        if not return_torch:
            image = image.cpu().numpy().astype(np.uint8)
    if return_coord:
        # mod_idx is in (i, j) format, not (x, y) format
        inv_mod_idx = torch.argsort(mod_idx)
        return image, save_coord_idx[inv_mod_idx]
    elif return_norm_coord:
        inv_mod_idx = torch.argsort(mod_idx)
        return image, orig_coord_idx[inv_mod_idx]
    else:
        return image


def quantile(x: torch.Tensor, q: float) -> Tuple[Union[int, float], Union[int, float]]:
    """
    Obtain q quantile value and (1 - q) quantile value from x

    Args:
        x: 1-dim torch tensor
        q: q value for quantile

    Returns:
        result_1: q quantile value of x
        result_2: (1 - q) quantile value of x
    """

    with torch.no_grad():
        inds = torch.argsort(x)
        val_1 = int(len(x) * q)
        val_2 = int(len(x) * (1 - q))

        result_1 = x[inds[val_1]]
        result_2 = x[inds[val_2]]

    return result_1, result_2


def out_of_room(xyz: torch.Tensor, trans: torch.Tensor, out_quantile: float = 0.05) -> bool:
    """
    Check if translation is out of xyz coordinates

    Args:
        xyz: (N, 3) torch tensor containing xyz coordinates
        trans: (3, 1) torch tensor containing xyz translation

    Returns:
        False if translation is not out of room
        True if translation is out of room
    """

    with torch.no_grad():
        # rejecting outliers
        x_min, x_max = quantile(xyz[:, 0], out_quantile)
        y_min, y_max = quantile(xyz[:, 1], out_quantile)
        z_min, z_max = quantile(xyz[:, 2], out_quantile)

        if x_min < trans[0][0] < x_max and y_min < trans[1][0] < y_max and z_min < trans[2][0] < z_max:
            return False
        else:
            return True


def get_bound(xyz: torch.Tensor, cfg, return_brute=False):
    # Obtain bounds for use in bayesian optimization
    out_quantile = getattr(cfg, 'out_of_room_quantile', 0.05)

    with torch.no_grad():
        # rejecting outliers
        x_min, x_max = quantile(xyz[:, 0], out_quantile)
        y_min, y_max = quantile(xyz[:, 1], out_quantile)
        z_min, z_max = quantile(xyz[:, 2], out_quantile)

    max_yaw = getattr(cfg, 'max_yaw', 2 * np.pi)
    min_yaw = getattr(cfg, 'min_yaw', 0)
    max_pitch = getattr(cfg, 'max_pitch', np.pi)
    min_pitch = getattr(cfg, 'min_pitch', 0)
    max_roll = getattr(cfg, 'max_roll', 2 *  np.pi)
    min_roll = getattr(cfg, 'min_roll', 0)

    if return_brute:
        return (slice(x_min.item(), x_max.item()), slice(y_min.item(), y_max.item()), slice(z_min.item(), z_max.item()),
            slice(min_yaw, max_yaw), slice(min_pitch, max_pitch), slice(min_roll, max_roll))
    else:
        return {'x': (x_min.item(), x_max.item()), 'y': (y_min.item(), y_max.item()), 'z': (z_min.item(), z_max.item()),
            'yaw': (min_yaw, max_yaw), 'pitch': (min_pitch, max_pitch), 'roll': (min_roll, max_roll)}


def adaptive_trans_num(xyz: torch.Tensor, max_trans_num: int, xy_only: bool = False) -> Tuple[int, int]:
    """
    Make the number of translation x, y coordinate candidates

    Args:
        xyz: (N, 3) torch tensor containing xyz coordinates of point cloud data
        max_trans_num: maximum number of translation candidates
        xy_only: If True, initialize only on x, y

    Returns:
        num_start_trans_x: number of x coordinate translation candidates
        num_start_trans_y: number of y coordinate translation candidates
        num_start_trans_z: number of z coordinate translation candidates, only returned when xy_only is False
    """

    xyz_max = torch.quantile(xyz, dim=0, q=0.90)
    xyz_min = torch.quantile(xyz, dim=0, q=0.10)
    xyz_length = xyz_max - xyz_min

    if xy_only:
        num_start_trans_x = ceil((xyz_length[0] * max_trans_num / xyz_length[1]) ** (1 / 2))
        num_start_trans_y = ceil((xyz_length[1] * max_trans_num / xyz_length[0]) ** (1 / 2))

        return num_start_trans_x, num_start_trans_y
    else:
        num_start_trans_x = ceil((xyz_length[0] ** 2 * max_trans_num / (xyz_length[1] * xyz_length[2])) ** (1 / 3))
        num_start_trans_y = ceil((xyz_length[1] ** 2 * max_trans_num / (xyz_length[0] * xyz_length[2])) ** (1 / 3))
        num_start_trans_z = ceil((xyz_length[2] ** 2 * max_trans_num / (xyz_length[0] * xyz_length[1])) ** (1 / 3))

        if num_start_trans_x % 2 == 0:
            num_start_trans_x -= 1
        if num_start_trans_y % 2 == 0:
            num_start_trans_y -= 1
        if num_start_trans_z % 2 == 0:
            num_start_trans_z -= 1

        return num_start_trans_x, num_start_trans_y, num_start_trans_z


def generate_rot_points(init_dict=None, device='cpu'):
    """
    Generate rotation starting points

    Args:
        init_dict: Dictionary containing details of initialization
        device: Device in which rotation starting points will be saved

    Returns:
        rot_arr: (N, 3) array containing (yaw, pitch, roll) starting points
    """

    if init_dict['yaw_only']:
        rot_arr = torch.zeros(init_dict['num_yaw'], 3, device=device)
        rot = torch.arange(init_dict['num_yaw'], dtype=torch.float, device=device)
        rot = rot * 2 * np.pi / init_dict['num_yaw']
        rot_arr[:, 0] = rot

    else:
        # Perform 3 DoF initialization
        rot_coords = torch.meshgrid(torch.arange(init_dict['num_yaw'], device=device).float() / init_dict['num_yaw'],
            torch.arange(init_dict['num_pitch'], device=device).float() / init_dict['num_pitch'],
            torch.arange(init_dict['num_roll'], device=device).float() / init_dict['num_roll'])

        rot_arr = torch.stack([rot_coords[0].reshape(-1), rot_coords[1].reshape(-1), rot_coords[2].reshape(-1)], dim=0).t()

        rot_arr[:, 0] = (rot_arr[:, 0] * (init_dict['max_yaw'] - init_dict['min_yaw'])) + init_dict['min_yaw']
        rot_arr[:, 1] = (rot_arr[:, 1] * (init_dict['max_pitch'] - init_dict['min_pitch'])) + init_dict['min_pitch']
        rot_arr[:, 2] = (rot_arr[:, 2] * (init_dict['max_roll'] - init_dict['min_roll'])) + init_dict['min_roll']

        # Initialize grid sample locations
        grid_list = [compute_sampling_grid(ypr, init_dict['num_yaw'], init_dict['num_pitch']) for ypr in rot_arr]

        # Filter out overlapping rotations
        round_digit = 3
        rot_list = [str(np.around(grid.cpu().numpy(), round_digit)) for grid in grid_list]
        valid_rot_idx = [rot_list.index(rot_mtx) for rot_mtx in sorted(set(rot_list))]  # sorted added to make things deterministic
        rot_arr = torch.stack([rot_arr[idx] for idx in valid_rot_idx], dim=0)
        
        # Put identity at front
        zero_idx = torch.where(rot_arr.sum(-1) == 0.)[0].item()
        rot_arr[[0, zero_idx]] = rot_arr[[zero_idx, 0]]

    return rot_arr


def generate_trans_points(xyz, init_dict=None, device='cpu'):
    """
    Generate translation starting points

    Args:
        xyz: Point cloud coordinates
        init_dict: Dictionary containing details of initialization
        device: Device in which translation starting points will be saved

    Returns:
        trans_arr: (N, 3) array containing (x, y, z) starting points
    """
    def get_starting_points(num_trans_x, num_trans_y, num_trans_z=None):
        if init_dict['trans_init_mode'] == 'uniform':
            x_points = (torch.arange(num_trans_x, device=device) + 1) / (num_trans_x + 1) * (xyz[:, 0].max() - xyz[:, 0].min()) + xyz[:, 0].min()
            y_points = (torch.arange(num_trans_y, device=device) + 1) / (num_trans_y + 1) * (xyz[:, 1].max() - xyz[:, 1].min()) + xyz[:, 1].min()
            if num_trans_z is not None:
                z_points = (torch.arange(num_trans_z, device=device) + 1) / (num_trans_z + 1) * (xyz[:, 2].max() - xyz[:, 2].min()) + xyz[:, 2].min()
        elif init_dict['trans_init_mode'] == 'manual':
            x_points = (torch.arange(num_trans_x, device=device)) / (num_trans_x - 1) * (init_dict['x_max'] - init_dict['x_min']) + init_dict['x_min']
            y_points = (torch.arange(num_trans_y, device=device)) / (num_trans_y - 1) * (init_dict['y_max'] - init_dict['y_min']) + init_dict['y_min']
            if num_trans_z is not None:
                z_points = (torch.arange(num_trans_z, device=device)) / (num_trans_z - 1) * (init_dict['z_max'] - init_dict['z_min']) + init_dict['z_min']
        else:  # Default is quantile
            split_x = (torch.arange(num_trans_x, device=device) + 1) / (num_trans_x + 1) if 1 / (num_trans_x + 1) > 0.1 else torch.linspace(0.1, 0.9, num_trans_x, device=device)
            split_y = (torch.arange(num_trans_y, device=device) + 1) / (num_trans_y + 1) if 1 / (num_trans_y + 1) > 0.1 else torch.linspace(0.1, 0.9, num_trans_y, device=device)
            x_points = torch.quantile(xyz[:, 0], split_x)
            y_points = torch.quantile(xyz[:, 1], split_y)
            if num_trans_z is not None:
                split_z = (torch.arange(num_trans_z, device=device) + 1) / (num_trans_z + 1) if 1 / (num_trans_z + 1) > 0.1 else torch.linspace(0.1, 0.9, num_trans_z, device=device)
                z_points = torch.quantile(xyz[:, 2], split_z)

        if num_trans_z is not None:
            return x_points, y_points, z_points
        else:
            return x_points, y_points

    if init_dict['xy_only']:
        if init_dict['trans_init_mode'] == 'octree':
            trans_arr = generate_octree_2d(xyz, init_dict['z_prior'], device)
        else:
            if init_dict['benchmark_grid'] and not init_dict['is_inlier_dict']:
                tot_trans_count = len(generate_octree_2d(xyz, init_dict['z_prior'], device))
            else:
                tot_trans_count = init_dict['num_trans']
            num_trans_x, num_trans_y = adaptive_trans_num(xyz, tot_trans_count, xy_only=True)
            trans_arr = torch.zeros(num_trans_x * num_trans_y, 3, device=device)

            x_points, y_points = get_starting_points(num_trans_x, num_trans_y)
            trans_coords = torch.meshgrid(x_points, y_points)
            trans_arr[:, :2] = torch.stack([trans_coords[0].reshape(-1), trans_coords[1].reshape(-1)], dim=0).t()
            if init_dict['z_prior'] is not None:
                trans_arr[:, 2] = init_dict['z_prior']
            else:
                trans_arr[:, 2] = xyz[:, 2].mean()

    else:
        nmin = init_dict.get('nmin', 2)
        if init_dict['trans_init_mode'] == 'octree':
            trans_arr = generate_octree(xyz, device, nmin)
        elif init_dict['trans_init_mode'] == 'grid_octree':
            # Generate octree
            octree_trans = generate_octree(xyz, device, nmin)
            
            # Generate grid
            num_grid_points = len(octree_trans)  # Hard-coded grid points
            num_trans_x, num_trans_y, num_trans_z = adaptive_trans_num(xyz, num_grid_points, xy_only=False)
            x_points, y_points, z_points = get_starting_points(num_trans_x, num_trans_y, num_trans_z)

            trans_coords = torch.meshgrid(x_points, y_points, z_points)
            grid_trans = torch.stack([trans_coords[0].reshape(-1), trans_coords[1].reshape(-1), trans_coords[2].reshape(-1)], dim=0).t()

            # Merge octree and grid
            mutual_dist = (grid_trans.unsqueeze(1) - octree_trans.unsqueeze(0)).norm(dim=-1)  # (Num of grid, num of octree)
            valid_idx = mutual_dist.min(dim=-1).values > 1.5  # Choose grid points that are far away from octree points
            trans_arr = torch.cat([grid_trans[valid_idx], octree_trans], dim=0)
        elif init_dict['trans_init_mode'] == 'quantile':
            num_trans = init_dict['num_trans']
            num_trans_x, num_trans_y, num_trans_z = adaptive_trans_num(xyz, num_trans, xy_only=False)
            x_points, y_points, z_points = get_starting_points(num_trans_x, num_trans_y, num_trans_z)

            trans_coords = torch.meshgrid(x_points, y_points, z_points)
            trans_arr = torch.stack([trans_coords[0].reshape(-1), trans_coords[1].reshape(-1), trans_coords[2].reshape(-1)], dim=0).t()
        elif init_dict['trans_init_mode'] == 'voxel':
            voxel_size = init_dict['voxel_size']
            quantile_thres = min(init_dict['quantile_thres'], 1 - init_dict['quantile_thres']) 
            xyz_min = xyz.quantile(quantile_thres, dim=0)
            xyz_max = xyz.quantile(1 - quantile_thres, dim=0)
            
            # Generate x, y, z points with the designated voxel size
            num_x, num_y, num_z = ((xyz_max - xyz_min) / voxel_size).long()
            x_points = torch.linspace(xyz_min[0], xyz_max[0], num_x, device=xyz.device)
            y_points = torch.linspace(xyz_min[1], xyz_max[1], num_y, device=xyz.device)
            z_points = torch.linspace(xyz_min[2], xyz_max[2], num_z, device=xyz.device)
            trans_coords = torch.meshgrid(x_points, y_points, z_points)
            trans_arr = torch.stack([trans_coords[0].reshape(-1), trans_coords[1].reshape(-1), trans_coords[2].reshape(-1)], dim=0).t()
        else:
            if init_dict['benchmark_grid'] and not init_dict['is_inlier_dict']:
                tot_trans_count = len(generate_octree(xyz, device, nmin))
            else:
                tot_trans_count = init_dict['num_trans']
            num_trans_x, num_trans_y, num_trans_z = adaptive_trans_num(xyz, tot_trans_count, xy_only=False)
            x_points, y_points, z_points = get_starting_points(num_trans_x, num_trans_y, num_trans_z)

            trans_coords = torch.meshgrid(x_points, y_points, z_points)
            trans_arr = torch.stack([trans_coords[0].reshape(-1), trans_coords[1].reshape(-1), trans_coords[2].reshape(-1)], dim=0).t()

    return trans_arr


def generate_trans_center(trans_tensor, range, num_split):
    """
    Generate translation starting points centered around a given translation starting point

    Args:
        trans_tensor: (1, 3) torch tensor used as center
        range: Size of each size in translation domain
        num_split: Number of splits to make along each translation axis

    Returns:
        trans_arr: (N, 3) array containing (x, y, z) starting points
    """
    min_x, min_y, min_z = trans_tensor - range / 2.
    max_x, max_y, max_z = trans_tensor + range / 2.
    x_points = torch.linspace(start=min_x, end=max_x, steps=num_split, device=trans_tensor.device)
    y_points = torch.linspace(start=min_y, end=max_y, steps=num_split, device=trans_tensor.device)
    z_points = torch.linspace(start=min_z, end=max_z, steps=num_split, device=trans_tensor.device)

    trans_coords = torch.meshgrid(x_points, y_points, z_points)
    trans_arr = torch.stack([trans_coords[0].reshape(-1), trans_coords[1].reshape(-1), trans_coords[2].reshape(-1)], dim=0).t()

    return trans_arr


def rot_from_ypr(ypr_array):
    def _ypr2mtx(ypr):
        # ypr is assumed to have a shape of [3, ]
        yaw, pitch, roll = ypr
        yaw = yaw.unsqueeze(0)
        pitch = pitch.unsqueeze(0)
        roll = roll.unsqueeze(0)

        tensor_0 = torch.zeros(1, device=yaw.device)
        tensor_1 = torch.ones(1, device=yaw.device)

        RX = torch.stack([
                        torch.stack([tensor_1, tensor_0, tensor_0]),
                        torch.stack([tensor_0, torch.cos(roll), -torch.sin(roll)]),
                        torch.stack([tensor_0, torch.sin(roll), torch.cos(roll)])]).reshape(3, 3)

        RY = torch.stack([
                        torch.stack([torch.cos(pitch), tensor_0, torch.sin(pitch)]),
                        torch.stack([tensor_0, tensor_1, tensor_0]),
                        torch.stack([-torch.sin(pitch), tensor_0, torch.cos(pitch)])]).reshape(3, 3)

        RZ = torch.stack([
                        torch.stack([torch.cos(yaw), -torch.sin(yaw), tensor_0]),
                        torch.stack([torch.sin(yaw), torch.cos(yaw), tensor_0]),
                        torch.stack([tensor_0, tensor_0, tensor_1])]).reshape(3, 3)

        R = torch.mm(RZ, RY)
        R = torch.mm(R, RX)

        return R
    
    if len(ypr_array.shape) == 1:
        return _ypr2mtx(ypr_array)
    else:
        tot_mtx = []
        for ypr in ypr_array:
            tot_mtx.append(_ypr2mtx(ypr))
        return torch.stack(tot_mtx)


def reshape_img_tensor(img: torch.Tensor, size: Tuple):
    # Note that size is (X, Y)
    cv_img = (img.cpu().numpy() * 255).astype(np.uint8)
    cv_img = cv2.resize(cv_img, size)
    cv_img = cv_img / 255.

    return torch.from_numpy(cv_img).float().to(img.device)


def debug_visualize(tgt_tensor):
    """
    Visualize target tensor. If batch dimension exists, visualizes the first instance. Multi-channel inputs are shown as 'slices'.
    If number of channels is 3, displayed in RGB. Otherwise results are shown as single channel images.
    For inputs that are float, we assume that the tgt_tensor values are normalized within [0, 1].
    For inputs that are int, we assume that the tgt_tensor values are normalized within [0, 255].

    Args:
        tgt_tensor: torch.tensor with one of the following shapes: (H, W), (H, W, C), (B, H, W, C)

    Returns:
        None
    """
    if "torch" in str(type(tgt_tensor)):
        vis_tgt = tgt_tensor.cpu().float().numpy()
    elif "numpy" in str(type(tgt_tensor)):
        vis_tgt = tgt_tensor.astype(np.float)
    else:
        raise ValueError("Invalid input!")

    if vis_tgt.max() > 2.0:  # If tgt_tensor is in range greater than 2.0, we assume it is an RGB image
        vis_tgt /= 255.

    if len(vis_tgt.shape) == 2:
        H, W = vis_tgt.shape
        plt.imshow(vis_tgt, cmap='gray', vmin=vis_tgt.min(), vmax=vis_tgt.max())
        plt.show()

    elif len(vis_tgt.shape) == 3:
        H, W, C = vis_tgt.shape

        if C > 3 or C == 2:
            fig = plt.figure(figsize=(50, 50))
            for i in range(C):
                fig.add_subplot(C // 2, 2, i + 1)
                plt.imshow(vis_tgt[..., i], cmap='gray', vmin=vis_tgt[..., i].min(), vmax=vis_tgt[..., i].max())
        elif C == 3:  # Display as RGB
            plt.imshow(vis_tgt)
        elif C == 1:
            plt.imshow(vis_tgt, cmap='gray', vmin=vis_tgt.min(), vmax=vis_tgt.max())

        plt.show()

    elif len(vis_tgt.shape) == 4:
        B, H, W, C = vis_tgt.shape
        vis_tgt = vis_tgt[0]

        if C > 3 or C == 2:
            fig = plt.figure(figsize=(50, 50))
            for i in range(C):
                fig.add_subplot(C // 2, 2, i + 1)
                plt.imshow(vis_tgt[..., i], cmap='gray', vmin=vis_tgt[..., i].min(), vmax=vis_tgt[..., i].max())
        elif C == 3:  # Display as RGB
            plt.imshow(vis_tgt)
        elif C == 1:
            plt.imshow(vis_tgt, cmap='gray', vmin=vis_tgt.min(), vmax=vis_tgt.max())

        plt.show()


def generate_octree(xyz, device, nmin=2):

    xyz_min = torch.min(xyz, dim=0)[0]
    xyz_max = torch.max(xyz, dim=0)[0]
    xyz_med = (xyz_min + xyz_max) / 2
    new_xyz = xyz - xyz_med
    xyz_length = abs(xyz_max - xyz_med)

    adaptive_xyz = torch.sqrt(xyz_length)
    adaptive_xyz /= adaptive_xyz.min()
    # adaptive_xyz[adaptive_xyz > 2.5] = 2.5
    adaptive_nmax = torch.ceil(nmin * adaptive_xyz - 0.2).long()

    if 2 ** adaptive_nmax.sum() < 256:
        adaptive_nmax += 1

    scaler = (2 ** adaptive_nmax) / xyz_length

    nmin = adaptive_nmax.min().item()
    nmax = adaptive_nmax.max().item()
    nmed = adaptive_nmax.median().item()


    def check_location(point, depth, plane):
        valid_axis = depth < adaptive_nmax

        location = torch.zeros((len(point)), dtype=torch.long).to(device)
        for count, i in enumerate(torch.nonzero(valid_axis, as_tuple=False).reshape(-1)):
            location[point[:, i] >= plane[:, i]] += (2 ** count)

        return location


    def update_planes(code, depth, plane):
        res = code[:, depth]

        valid_axis = depth < adaptive_nmax
        loc = torch.zeros((len(res), valid_axis.sum())).to(device)

        for i in range(valid_axis.sum()):
            if i == 0:
                loc[:, i] = (res % 4) % 2
            elif i == 1:
                loc[:, i] = (res % 4) // 2
            else:
                loc[:, i] = res // 4
        loc = loc * 2 - 1

        plane[:, valid_axis] += loc * (2 ** (adaptive_nmax[valid_axis] - depth - 1)) / scaler[valid_axis]

        return plane


    def merge_code(code):

        merged_code = code.clone()

        for i, j in enumerate(range(nmax - 1, 0, -1)):
            output, inverse_inds, counts = torch.unique_consecutive(merged_code[:, :j], True, True, dim=0)
            if j < nmin:
                merge_count = 2 ** (nmax - nmed) * 4 ** (nmed - nmin) * 8 ** (i + nmin - nmax + 1)
            elif j < nmed:
                merge_count = 2 ** (nmax - nmed) * 4 ** (i + nmed - nmax + 1)
            else:
                merge_count = 2 ** (i + 1)

            empty_ind = counts >= merge_count
            merged_code[empty_ind[inverse_inds], j:] = -1

        merged_code = torch.unique(merged_code, dim=0)

        return merged_code


    def delete_outer_code(code, empty_code):


        inner_inds = torch.ones((len(empty_code)), dtype=torch.bool).to(empty_code.device)

        for i in range(len(empty_code)):
            curr_empty_code = empty_code[i]
            depth = 0
            same_par_inds = code[:, depth] == curr_empty_code[depth]
            same_parent = code.clone()

            if same_par_inds.sum() == 0:
                inner_inds[i] = False

            while same_par_inds.sum() > 0:
                depth += 1
                same_parent = same_parent[same_par_inds]
                same_par_inds = same_parent[:, depth] == curr_empty_code[depth]

            loc = torch.zeros((len(same_parent), 3)).to(device)
            empty_loc = torch.zeros((3)).to(device)

            for k in range(depth + 1):
                valid_axis = k < adaptive_nmax
                for count, kk in enumerate(torch.nonzero(valid_axis, as_tuple=False).reshape(-1)):
                    if count == 0:
                        loc[:, kk] += (2 ** (nmax - k)) * (2 * ((same_parent[:, k] % 4) % 2) - 1)
                        if curr_empty_code[k] != -1:
                            empty_loc[kk] += (2 ** (nmax - k)) * (2 * ((curr_empty_code[k] % 4) % 2) - 1)
                    elif count == 1:
                        loc[:, kk] += (2 ** (nmax - k)) * (2 * ((same_parent[:, k] % 4) // 2) - 1)
                        if curr_empty_code[k] != -1:
                            empty_loc[kk] += (2 ** (nmax - k)) * (2 * ((curr_empty_code[k] % 4) // 2) - 1)
                    else:
                        loc[:, kk] += (2 ** (nmax - k)) * (2 * (same_parent[:, k] // 4) - 1)
                        if curr_empty_code[k] != -1:
                            empty_loc[kk] += (2 ** (nmax - k)) * (2 * (curr_empty_code[k] // 4) - 1)

            max_loc = loc.max(dim=0)[0]
            min_loc = loc.min(dim=0)[0]

            plus_inds = max_loc > 0
            minus_inds = min_loc < 0

            if (max_loc[plus_inds] < empty_loc[plus_inds]).sum() > 0 or (min_loc[minus_inds] > empty_loc[minus_inds]).sum() > 0:
                inner_inds[i] = False

        return empty_code[inner_inds]


    def code2coords(code):

        xyz_coords = torch.zeros((len(code), 3)).to(device)
        last_merged_ind = (code != -1).sum(dim=1) - 1

        for i in range(nmax - 1, -1, -1):
            curr_code = code[:, i]
            empty_ind = (curr_code == -1)

            valid_axis = i < adaptive_nmax

            loc = torch.zeros((len(curr_code), valid_axis.sum())).to(device)

            for j in range(valid_axis.sum()):
                if j == 0:
                    loc[:, j] = (curr_code % 4) % 2
                elif j == 1:
                    loc[:, j] = (curr_code % 4) // 2
                else:
                    loc[:, j] = curr_code // 4

            loc = loc * 2 - 1
            loc[empty_ind] = 0
            xyz_coords[:, valid_axis] += loc * (2 ** (adaptive_nmax[valid_axis] - i - 1)) / scaler[valid_axis]

        xyz_coords = torch.unique(xyz_coords, dim=0)

        return xyz_coords


    code = torch.zeros((len(new_xyz), nmax), dtype=torch.long).to(device)

    array_shape = [8] * nmin
    array_shape += [4] * (nmed - nmin)
    array_shape += [2] * (nmax - nmed)
    empty_inds = torch.ones(array_shape, dtype=torch.bool).to(device)

    plane = torch.zeros((len(new_xyz), 3)).to(device)

    for depth in range(nmax):
        code[:, depth] = check_location(new_xyz, depth, plane)
        plane = update_planes(code, depth, plane)

    u_code = code.unique(dim=0)
    empty_inds[tuple(u_code.t())] = False
    empty_code = torch.nonzero(empty_inds, as_tuple=False)
    empty_code = merge_code(empty_code)
    empty_code = delete_outer_code(u_code, empty_code)

    empty_coords = code2coords(empty_code)

    return empty_coords + xyz_med


def generate_octree_2d(xyz, height_z, device):

    nmin = 4

    # filter by z-axis
    filtered_xyz = xyz[torch.logical_and(xyz[:, -1] < height_z + 0.2, xyz[:, -1] > height_z - 0.2)]
    xy_min = torch.min(filtered_xyz[:, :2], dim=0)[0]
    xy_max = torch.max(filtered_xyz[:, :2], dim=0)[0]
    xy_med = (xy_min + xy_max) / 2

    new_xy = filtered_xyz[:, :2] - xy_med
    xy_length = abs(xy_max - xy_med)

    adaptive_xy = torch.sqrt(xy_length)
    adaptive_xy /= adaptive_xy.min()
    adaptive_nmax = torch.ceil(nmin * adaptive_xy - 0.2).long()

    adaptive_nmax[adaptive_nmax > 8] = 8

    scaler = (2 ** adaptive_nmax) / xy_length

    nmin = adaptive_nmax.min().item()
    nmax = adaptive_nmax.max().item()
    
    def check_location(point, depth, plane):
        valid_axis = depth < adaptive_nmax

        location = torch.zeros((len(point)), dtype=torch.long).to(device)
        for count, i in enumerate(torch.nonzero(valid_axis, as_tuple=False).reshape(-1)):
            location[point[:, i] >= plane[:, i]] += (2 ** count)

        return location


    def update_planes(code, depth, plane):
        res = code[:, depth]

        valid_axis = depth < adaptive_nmax
        loc = torch.zeros((len(res), valid_axis.sum())).to(device)

        for i in range(valid_axis.sum()):
            if i == 0:
                loc[:, i] = (res % 4) % 2
            elif i == 1:
                loc[:, i] = (res % 4) // 2
            else:
                loc[:, i] = res // 4
        loc = loc * 2 - 1

        plane[:, valid_axis] += loc * (2 ** (adaptive_nmax[valid_axis] - depth - 1)) / scaler[valid_axis]

        return plane


    def merge_code(code):

        merged_code = code.clone()

        for i, j in enumerate(range(nmax - 1, 0, -1)):
            output, inverse_inds, counts = torch.unique_consecutive(merged_code[:, :j], True, True, dim=0)
            if j < nmin:
                merge_count = 2 ** (nmax - nmin) * 4 ** (i + nmin - nmax + 1)
            else:
                merge_count = 2 ** (i + 1)

            empty_ind = counts >= merge_count
            merged_code[empty_ind[inverse_inds], j:] = -1

        merged_code = torch.unique(merged_code, dim=0)

        return merged_code


    def delete_outer_code(code, empty_code):

        inner_inds = torch.ones((len(empty_code)), dtype=torch.bool).to(empty_code.device)

        for i in range(len(empty_code)):
            curr_empty_code = empty_code[i]
            depth = 0
            same_par_inds = code[:, depth] == curr_empty_code[depth]
            same_parent = code.clone()

            if same_par_inds.sum() == 0:
                inner_inds[i] = False

            while same_par_inds.sum() > 0:
                depth += 1
                same_parent = same_parent[same_par_inds]
                same_par_inds = same_parent[:, depth] == curr_empty_code[depth]

            loc = torch.zeros((len(same_parent), 2)).to(device)
            empty_loc = torch.zeros((2)).to(device)

            for k in range(depth + 1):
                valid_axis = k < adaptive_nmax
                for count, kk in enumerate(torch.nonzero(valid_axis, as_tuple=False).reshape(-1)):
                    if count == 0:
                        loc[:, kk] += (2 ** (nmax - k)) * (2 * ((same_parent[:, k] % 4) % 2) - 1)
                        if curr_empty_code[k] != -1:
                            empty_loc[kk] += (2 ** (nmax - k)) * (2 * ((curr_empty_code[k] % 4) % 2) - 1)
                    elif count == 1:
                        loc[:, kk] += (2 ** (nmax - k)) * (2 * ((same_parent[:, k] % 4) // 2) - 1)
                        if curr_empty_code[k] != -1:
                            empty_loc[kk] += (2 ** (nmax - k)) * (2 * ((curr_empty_code[k] % 4) // 2) - 1)
                    else:
                        loc[:, kk] += (2 ** (nmax - k)) * (2 * (same_parent[:, k] // 4) - 1)
                        if curr_empty_code[k] != -1:
                            empty_loc[kk] += (2 ** (nmax - k)) * (2 * (curr_empty_code[k] // 4) - 1)

            max_loc = loc.max(dim=0)[0]
            min_loc = loc.min(dim=0)[0]

            plus_inds = max_loc > 0
            minus_inds = min_loc < 0

            if (max_loc[plus_inds] < empty_loc[plus_inds]).sum() > 0 or (min_loc[minus_inds] > empty_loc[minus_inds]).sum() > 0:
                inner_inds[i] = False

        return empty_code[inner_inds]


    def code2coords(code):

        xy_coords = torch.zeros((len(code), 2)).to(device)
        last_merged_ind = (code != -1).sum(dim=1) - 1

        for i in range(nmax - 1, -1, -1):
            curr_code = code[:, i]
            empty_ind = (curr_code == -1)

            valid_axis = i < adaptive_nmax

            loc = torch.zeros((len(curr_code), valid_axis.sum())).to(device)

            for j in range(valid_axis.sum()):
                if j == 0:
                    loc[:, j] = (curr_code % 4) % 2
                elif j == 1:
                    loc[:, j] = (curr_code % 4) // 2
                else:
                    loc[:, j] = curr_code // 4

            loc = loc * 2 - 1
            loc[empty_ind] = 0
            xy_coords[:, valid_axis] += loc * (2 ** (adaptive_nmax[valid_axis] - i - 1)) / scaler[valid_axis]

        xy_coords = torch.unique(xy_coords, dim=0)
        xyz_coords = torch.cat([xy_coords, torch.ones_like(xy_coords[:, 0:1]) * height_z], dim=-1)

        return xyz_coords


    code = torch.zeros((len(new_xy), nmax), dtype=torch.long).to(device)

    array_shape = [4] * nmin
    array_shape += [2] * (nmax - nmin)

    empty_inds = torch.ones(array_shape, dtype=torch.bool).to(device)

    plane = torch.zeros((len(new_xy), 2)).to(device)

    for depth in range(nmax):
        code[:, depth] = check_location(new_xy, depth, plane)
        plane = update_planes(code, depth, plane)

    u_code = code.unique(dim=0)
    empty_inds[tuple(u_code.t())] = False
    empty_code = torch.nonzero(empty_inds, as_tuple=False)
    empty_code = merge_code(empty_code)
    empty_code = delete_outer_code(u_code, empty_code)

    empty_coords = code2coords(empty_code)
    empty_coords = empty_coords + torch.cat([xy_med, torch.zeros_like(xy_med[0:1])], dim=0)

    return empty_coords


# Code excerpted from https://github.com/haruishi43/equilib
def create_coordinate(h_out: int, w_out: int, device=torch.device('cpu')) -> np.ndarray:
    r"""Create mesh coordinate grid with height and width

    return:
        coordinate: numpy.ndarray
    """
    xs = torch.linspace(0, w_out - 1, w_out, device=device)
    theta = np.pi - xs * 2 * math.pi / w_out
    ys = torch.linspace(0, h_out - 1, h_out, device=device)
    phi = ys * math.pi / h_out
    # NOTE: https://github.com/pytorch/pytorch/issues/15301
    # Torch meshgrid behaves differently than numpy
    phi, theta = torch.meshgrid([phi, theta])
    coord = torch.stack((theta, phi), axis=-1)
    return coord


def create_area_mask(h_out: int, w_out: int, device=torch.device('cpu')) -> np.ndarray:
    """
    Create H x W numpy array containing area size for each location in the gridded sphere: dtheta x dphi x sin(theta)
    """
    ys = torch.linspace(0, h_out - 1, h_out, device=device)
    phi = ys * math.pi / h_out + np.pi / (h_out * 2)
    return torch.stack([phi] * w_out, dim=1)  # h_out * w_out


def compute_sampling_grid(ypr, num_split_h, num_split_w, inverse=False):
    """
    Utility function for computing sampling grid using yaw, pitch, roll
    We assume the equirectangular image to be splitted as follows:

    -------------------------------------
    |   0    |   1    |    2   |    3   |
    |        |        |        |        |
    -------------------------------------
    |   4    |   5    |    6   |    7   |
    |        |        |        |        |
    -------------------------------------

    Indices are assumed to be ordered in compliance to the above convention.
    Args:
        ypr: torch.tensor of shape (3, ) containing yaw, pitch, roll
        num_split_h: Number of horizontal splits
        num_split_w: Number of vertical splits
        inverse: If True, calculates sampling grid with inverted rotation provided from ypr

    Returns:
        grid: Sampling grid for generating rotated images according to yaw, pitch, roll
    """
    if inverse:
        R = rot_from_ypr(ypr)
    else:
        R = rot_from_ypr(ypr).T

    H, W = num_split_h, num_split_w
    a = create_coordinate(H, W, ypr.device)
    a[..., 0] -= np.pi / (num_split_w)  # Add offset to align sampling grid to each pixel center
    a[..., 1] += np.pi / (num_split_h * 2)  # Add offset to align sampling grid to each pixel center
    norm_A = 1
    x = norm_A * torch.sin(a[:, :, 1]) * torch.cos(a[:, :, 0])
    y = norm_A * torch.sin(a[:, :, 1]) * torch.sin(a[:, :, 0])
    z = norm_A * torch.cos(a[:, :, 1])
    A = torch.stack((x, y, z), dim=-1)  # (H, W, 3)
    _B = R @ A.unsqueeze(3)
    _B = _B.squeeze(3)
    grid = cloud2idx(_B.reshape(-1, 3)).reshape(H, W, 2)
    return grid

# Color conversion code excepted from Kornia: https://github.com/kornia/kornia

def rgb_to_grayscale(
    image: torch.Tensor, rgb_weights: torch.Tensor = torch.tensor([0.299, 0.587, 0.114])
) -> torch.Tensor:
    r"""Convert a RGB image to grayscale version of image.

    .. image:: _static/img/rgb_to_grayscale.png

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: RGB image to be converted to grayscale with shape :math:`(*,3,H,W)`.
        rgb_weights: Weights that will be applied on each channel (RGB).
            The sum of the weights should add up to one.
    Returns:
        grayscale version of the image with shape :math:`(*,1,H,W)`.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       color_conversions.html>`__.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> gray = rgb_to_grayscale(input) # 2x1x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    if not isinstance(rgb_weights, torch.Tensor):
        raise TypeError(f"rgb_weights is not a torch.Tensor. Got {type(rgb_weights)}")

    if rgb_weights.shape[-1] != 3:
        raise ValueError(f"rgb_weights must have a shape of (*, 3). Got {rgb_weights.shape}")

    r: torch.Tensor = image[..., 0:1, :, :]
    g: torch.Tensor = image[..., 1:2, :, :]
    b: torch.Tensor = image[..., 2:3, :, :]

    if not torch.is_floating_point(image) and (image.dtype != rgb_weights.dtype):
        raise TypeError(
            f"Input image and rgb_weights should be of same dtype. Got {image.dtype} and {rgb_weights.dtype}"
        )

    w_r, w_g, w_b = rgb_weights.to(image).unbind()
    return w_r * r + w_g * g + w_b * b
