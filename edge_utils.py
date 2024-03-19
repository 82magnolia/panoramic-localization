import torch
from tqdm import tqdm
import cv2
import numpy as np
from utils import (
    make_pano,
)
from scipy.ndimage import map_coordinates
from pylsd import lsd


# General utility functions

def split_by_axes(edge_2d, dirs, principal_3d, rot_mtx, inlier_thres=0.05):
    """
    Split edges in 2D, 3D according to the principal directions.

    Args:
        edge_2d: (N_2D, 9) tensor containing [normals start end]
        dirs: (N_3D, 3) tensor containing 3D edge directions
        principal_3d: (3, 3) tensor containing three principal directions in 3D
        rot_mtx: (3, 3) or (N_r, 3, 3) torch tensor containing rotation matrix
        inlier_thres: Dot product threshold for extracting inliers along principal directions

    Returns:
        edge_2d_mask: (N_2D, 3) or (N_r, N_2D, 3) tensor containing masks for each 2D line along principal direction
        edge_3d_mask: (N_3D, 3) or (N_r, N_3D, 3) tensor containing masks for each 3D line along principal direction  
    """
    if len(rot_mtx.shape) == 2:
        inner_prod_2d = torch.abs(edge_2d[:, :3] @ rot_mtx @ principal_3d.t())
        prod_idx_2d = inner_prod_2d.argmin(-1)
        prod_val_2d = inner_prod_2d.min(-1).values
        edge_2d_mask = torch.stack([(prod_idx_2d == i) & (prod_val_2d < inlier_thres) for i in range(3)], dim=-1)
        
        inner_prod_3d = torch.abs(dirs @ principal_3d.t())
        edge_3d_mask = torch.stack([inner_prod_3d[:, i] > 1 - inlier_thres for i in range(3)], dim=-1)
    elif len(rot_mtx.shape) == 3:
        edge_2d_mask = []
        edge_3d_mask = []
        for rot_arr in rot_mtx:
            inner_prod_2d = torch.abs(edge_2d[:, :3] @ rot_arr @ principal_3d.t())
            prod_idx_2d = inner_prod_2d.argmin(-1)
            prod_val_2d = inner_prod_2d.min(-1).values
            edge_2d_mask.append(torch.stack([(prod_idx_2d == i) & (prod_val_2d < inlier_thres) for i in range(3)], dim=-1))
            
            inner_prod_3d = torch.abs(dirs @ principal_3d.t())
            edge_3d_mask.append(torch.stack([inner_prod_3d[:, i] > 1 - inlier_thres for i in range(3)], dim=-1))
        
        edge_2d_mask = torch.stack(edge_2d_mask, dim=0)
        edge_3d_mask = torch.stack(edge_3d_mask, dim=0)
    else:
        raise ValueError("Invalid rot_mtx shape")

    return edge_2d_mask, edge_3d_mask


def split_2d(edge_2d, principal_2d, inlier_thres=0.05):
    """
    Split edges in 2D according to the principal directions.

    Args:
        edge_2d: (N_2D, 9) tensor containing [normals start end]
        principal_2d: (3, 3) tensor containing three principal directions in 2D
        inlier_thres: Dot product threshold for extracting inliers among principal directions
    
    Returns:
        edge_2d_mask: (N_2D, 3) tensor containing masks for each 2D line along principal direction
    """
    inner_prod_2d = torch.abs(edge_2d[:, :3] @ principal_2d.t())  # (N_2D, 3)
    min_2d_mask = inner_prod_2d.argmin(-1, keepdim=True) == torch.arange(principal_2d.shape[0], device=edge_2d.device).unsqueeze(0).repeat(inner_prod_2d.shape[0], 1)
    edge_2d_mask = (inner_prod_2d < inlier_thres) & min_2d_mask  # (N_2D, 3)

    return edge_2d_mask


def split_3d(dirs, principal_3d, inlier_thres=0.05):
    """
    Split edges in 3D according to the principal directions.

    Args:
        dirs: (N_3D, 3) tensor containing 3D edge directions
        principal_3d: (3, 3) tensor containing three principal directions in 3D
        inlier_thres: Dot product threshold for extracting inliers along principal directions
    
    Returns:
        edge_3d_mask: (N_3D, 3) tensor containing masks for each 3D line along principal direction
    """
    inner_prod_3d = torch.abs(dirs @ principal_3d.t())  # (N_3D, 3)
    edge_3d_mask = inner_prod_3d > 1 - inlier_thres
    return edge_3d_mask


def generate_sphere_pts(level, type='torch', device='cpu'):
    # Generate points on a sphere by sampling from a icosahedron
    sphere_pts, _ = icosahedron2sphere(level)
    if type == 'torch':
        return torch.from_numpy(sphere_pts).float().to(device)
    elif type == 'numpy':
        return sphere_pts


# Extraction functions

def extract_img_line(img, view_size=320, return_edge_img=False, return_full_lines=False, length_thres=None, length_topk=None, length_ratio=None):
    """
    Code excerpted from https://github.com/sunset1995/HorizonNet.
    Extract line segments from an input panorama image.

    Args:
        img: Input image, normalized to range in 0~256
        view_size: Image size of cropped views to perform LSD
        return_edge_img: If True, returns edge image painted with detected edges
        return_full_lines: If True, returns full lines before filtering
        length_thres: If specified, filters lines over length threshold
        length_topk: If specified, only returns lines with top-k length
        length_ratio: If specified, only returns lines with top-ratio length

    Returns:
        coordN_lines: Lines containing [normals starting_coord, ending_coord]
        panoEdge: Image for visualizing line segments
    """
    cut_size = view_size
    fov = np.pi / 3
    xh = np.arange(-np.pi, np.pi*5/6, np.pi/6)
    yh = np.zeros(xh.shape[0])
    xp = np.array([-3/3, -2/3, -1/3, 0/3,  1/3, 2/3, -3/3, -2/3, -1/3,  0/3,  1/3,  2/3]) * np.pi
    yp = np.array([ 1/4,  1/4,  1/4, 1/4,  1/4, 1/4, -1/4, -1/4, -1/4, -1/4, -1/4, -1/4]) * np.pi
    x = np.concatenate([xh, xp, [0, 0]])
    y = np.concatenate([yh, yp, [np.pi/2., -np.pi/2]])

    sepScene = separatePano(img.copy(), fov, x, y, cut_size)
    edge = []
    for scene in sepScene:
        edgeMap, edgeList = lsdWrap(scene['img'])
        edge.append({
            'img': edgeMap,
            'edgeLst': edgeList,
            'vx': scene['vx'],
            'vy': scene['vy'],
            'fov': scene['fov'],
        })
        edge[-1]['panoLst'] = edgeFromImg2Pano(edge[-1])

    lines, coordN_lines = combineEdgesN(edge)
    full_coordN_lines = coordN_lines.copy()
    
    if length_thres is not None:
        coordN_lines, valid_mask = filterEdgeByThres(coordN_lines, length_thres, True)
        lines = lines[valid_mask]
    
    if length_topk is not None:
        coordN_lines, valid_idx = filterEdgeByTopK(coordN_lines, length_topk, True)
        lines = lines[valid_idx]

    if length_ratio is not None:
        coordN_lines, valid_idx = filterEdgeByTopK(coordN_lines, int(coordN_lines.shape[0] * length_ratio), True)
        lines = lines[valid_idx]

    # Transform coordN_lines to PICCOLO coordinate frame
    theta = np.pi / 2
    rot_mtx = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0., 0., 1.]])
    coordN_lines[:, :3] = coordN_lines[:, :3] @ rot_mtx
    coordN_lines[:, 3:6] = coordN_lines[:, 3:6] @ rot_mtx
    coordN_lines[:, 6:9] = coordN_lines[:, 6:9] @ rot_mtx

    full_coordN_lines[:, :3] = full_coordN_lines[:, :3] @ rot_mtx
    full_coordN_lines[:, 3:6] = full_coordN_lines[:, 3:6] @ rot_mtx
    full_coordN_lines[:, 6:9] = full_coordN_lines[:, 6:9] @ rot_mtx

    if return_edge_img:
        panoEdge = paint_line(lines, img.shape[1], img.shape[0])
        if return_full_lines:
            return coordN_lines, panoEdge, full_coordN_lines
        else:
            return coordN_lines, panoEdge
    else:
        if return_full_lines:
            return coordN_lines, full_coordN_lines
        else:
            return coordN_lines


def make_pano_line_2d(edge_2d, resolution=(400, 800), rot_mtx=None, rgb=None):
    """
    Make panorama image from 2D lines

    Args:
        edge_2d: (N, 9) tensor containing [normals start end]
        resolution: Edge image resolution
        rot_mtx: (3, 3) torch tensor containing rotation matrix
        rgb: (N, 3) tensor containing line colors
    
    Returns:
        edge_img: (H, W, 3) numpy array containing edge rederings
    """
    starts = edge_2d[:, 3:6]
    ends = edge_2d[:, 6:]
    dirs = ends - starts

    # Linear samples along line
    line_steps = 300
    line_t = torch.linspace(start=0, end=1, steps=line_steps, device=edge_2d.device).reshape(1, -1, 1)
    tot_pts = (dirs.unsqueeze(1) * line_t + starts.unsqueeze(1)).reshape(-1, 3)

    if rgb is not None:
        line_rgb = rgb.repeat_interleave(line_steps, dim=0)
    else:
        line_rgb = torch.ones_like(tot_pts).float()

    if rot_mtx is None:
        edge_img = make_pano(tot_pts.float(), line_rgb, resolution=resolution, default_white=True)
    else:
        edge_img = make_pano(tot_pts.float() @ rot_mtx.t(), line_rgb, resolution=resolution, default_white=True)
    return edge_img


def make_pano_line_3d(starts, ends, mask=None, resolution=(400, 800), trans_mtx=None, rot_mtx=None, rgb=None):
    """
    Make panorama image from 3D lines

    Args:
        starts: (N, 3) tensor containing starting points of 3D lines
        ends: (N, 3) tensor containing ending points of 3D lines
        mask: (N, 1) boolean tensor designating lines to draw
        resolution: Edge image resolution
        trans_mtx: (1, 3) or (3, ) torch tensor containing translation matrix
        rot_mtx: (3, 3) torch tensor containing rotation matrix
        rgb: (N, 3) tensor containing line colors

    Returns:
        edge_img: (H, W, 3) numpy array containing edge rederings
    """
    if len(trans_mtx.shape) == 1:
        trans_mtx = trans_mtx.unsqueeze(0)
    elif trans_mtx.shape == (3, 1):
        trans_mtx = trans_mtx.t()
    
    if mask is not None:
        line_starts = starts[mask]
        line_ends = ends[mask]
    else:
        line_starts = starts
        line_ends = ends

    dirs = line_ends - line_starts

    # Linear samples along line
    line_steps = 300
    line_t = torch.linspace(start=0, end=1, steps=line_steps, device=dirs.device).reshape(1, -1, 1)
    tot_pts = (dirs.unsqueeze(1) * line_t + line_starts.unsqueeze(1)).reshape(-1, 3)

    if rgb is not None:
        if mask is None:
            line_rgb = rgb.repeat_interleave(line_steps, dim=0)
        else:
            line_rgb = rgb[mask].repeat_interleave(line_steps, dim=0)
    else:
        line_rgb = torch.zeros_like(tot_pts).float()

    if rot_mtx is None:
        edge_img = make_pano((tot_pts.float() - trans_mtx) @ rot_mtx.t(), line_rgb, resolution=resolution, default_white=True)
    else:
        edge_img = make_pano((tot_pts.float() - trans_mtx) @ rot_mtx.t(), line_rgb, resolution=resolution, default_white=True)
    return edge_img


def make_line_cloud(starts, ends, mask=None, rgb=None):
    """
    Make line cloud start and end points

    Args:
        starts: (N, 3) tensor containing starting points of 3D lines
        ends: (N, 3) tensor containing ending points of 3D lines
        mask: (N, 1) boolean tensor designating lines to draw
        resolution: Edge image resolution
        rgb: (N, 3) tensor containing line colors

    Returns:
        colored_line_cloud: (N_l, 3) numpy array containing colored line cloud
    """    
    if mask is not None:
        line_starts = starts[mask]
        line_ends = ends[mask]
    else:
        line_starts = starts
        line_ends = ends

    dirs = line_ends - line_starts

    # Linear samples along line
    line_steps = 300
    line_t = torch.linspace(start=0, end=1, steps=line_steps, device=dirs.device).reshape(1, -1, 1)
    tot_pts = (dirs.unsqueeze(1) * line_t + line_starts.unsqueeze(1)).reshape(-1, 3)

    if rgb is not None:
        if mask is None:
            line_rgb = rgb.repeat_interleave(line_steps, dim=0)
        else:
            line_rgb = rgb[mask].repeat_interleave(line_steps, dim=0)
    else:
        line_rgb = torch.zeros_like(tot_pts).float()

    colored_line_cloud = torch.cat([tot_pts, line_rgb], dim=-1)
    return colored_line_cloud.cpu().numpy()


def make_pano_line_matches(edge_2d, starts, ends, trans_mtx=None, rot_mtx=None):
    """
    Visualize line matches

    Args:
        edge_2d: (N_match, 9) tensor containing [normals start end]
        starts: (N_match, 3) tensor containing starting points of 3D lines
        ends: (N_match, 3) tensor containing ending points of 3D lines
        trans_mtx: (1, 3) or (3, ) torch tensor containing translation matrix
        rot_mtx: (3, 3) torch tensor containing rotation matrix

    Returns:
        match_img: (H, W_match, 3) numpy array containing line matches
    """
    match_rgb = torch.rand_like(starts)  # (N_2D, 3)
    img_2d = make_pano_line_2d(edge_2d, rgb=match_rgb)
    img_3d = make_pano_line_3d(starts, ends, rgb=match_rgb, trans_mtx=trans_mtx, rot_mtx=rot_mtx)
    margin_img = 255 * np.ones([img_2d.shape[0], img_2d.shape[1] // 4, 3], dtype=np.uint8)
    match_img = np.concatenate([img_2d, margin_img, img_3d], axis=1)
    return match_img


# Wrapper for various distance functions where evaluation can be separated across line labels

def split_func_2d(query_points, edge_2d, edge_2d_mask, dist_type='distance'):
    """
    Function that returns distance functions for a set of edge in 2D, split by type specified in edge_2d_mask.
    Note that query points should be normalized.

    Args:
        query_points: (N_q, 3) tensor containing points to query, or length K list of (N, 3) tensors containing
         points to query for each class
        edge_2d: (N_2D, 9) tensor containing 2D edges in [normals starts ends] format
        edge_2d_mask: (N_2D, K) tensor containing masks for each 2D line, which may belong to K edge types
        dist_type: Type of distance metric to evaluate
    
    Returns:
        dist_2d: (N_q, K) tensor or length K list of (N, 3) tensors containing distance to nearest edges separated
         by types provided in edge_2d_mask
    """
    if dist_type == 'distance':
        dist_func = distance_func_2d
    
    if isinstance(query_points, torch.Tensor):
        dist_2d = torch.stack([dist_func(query_points, edge_2d[edge_2d_mask[:, i]]) for i in range(edge_2d_mask.shape[-1])], dim=-1)
    elif isinstance(query_points, list):
        dist_2d = [dist_func(query_points[i], edge_2d[edge_2d_mask[:, i]]) for i in range(edge_2d_mask.shape[-1])]
    else:
        raise ValueError("Invalid query type")

    return dist_2d


def split_func_3d(query_points, starts, ends, trans_mtx, rot_mtx, edge_3d_mask, dist_type='distance'):
    """
    Function that returns distance functions for a set of edge in 3D, split by type specified in edge_3d_mask.
    Note that query points should be normalized.

    Args:
        query_points: (N_q, 3) tensor containing points to query, or length K list of (N, 3) tensors containing
        points to query for each class
        starts: (N_3D, 3) tensor containing 3D edge start points
        ends: (N_3D, 3) tensor containing 3D edge end points
        trans_mtx: (1, 3) tensor containing translation of point cloud
        rot_mtx: (3, 3) tensor containing rotation of point cloud
        edge_3d_mask: (N_3D, K) tensor containing masks for each 3D line, which may belong to K edge types
        dist_type: Type of distance metric to evaluate
    
    Returns:
        dist_3d: (N_q, K) tensor or length K list of (N, 3) tensors containing distance to nearest edges separated
         by types provided in edge_3d_mask
    """
    if dist_type == 'distance':
        dist_func = distance_func_3d

    if isinstance(query_points, torch.Tensor):
        dist_3d = torch.stack([dist_func(query_points, starts[edge_3d_mask[:, i]], ends[edge_3d_mask[:, i]], trans_mtx, rot_mtx)
            for i in range(edge_3d_mask.shape[-1])], dim=-1)
    elif isinstance(query_points, list):
        dist_3d = [dist_func(query_points[i], starts[edge_3d_mask[:, i]], ends[edge_3d_mask[:, i]], trans_mtx, rot_mtx)
            for i in range(edge_3d_mask.shape[-1])]
    else:
        raise ValueError("Invalid query type")

    return dist_3d


def split_func_2d_batch(query_points, edge_2d, edge_2d_mask, rot_mtx=None, perms=None, single_pose_compute=False):
    """
    Function that returns distance functions for a set of edge in 2D, split by type specified in edge_2d_mask.
    Here the edge_2d_mask is applied in a parallel manner. Note that query points should be normalized.

    Args:
        query_points: (N_q, 3) tensor containing points to query
        edge_2d: (N_2D, 9) tensor containing 2D edges in [normals starts ends] format
        edge_2d_mask: (N_2D, K) or (N_r, N_2D, K) tensor containing masks for each 2D line, which may belong to K edge types
        rot_mtx: (N_r, 3, 3) tensor containing rotation candidate poses
        perms: (N_r, 3) tensor containing permutations used for obtaining rotations
        single_pose_compute: If True, compute distance functions for the first pose and obtain distance functions for other views via NN interpolation

    Returns:
        dist_2d: (N_q, K) or (N_r, N_q, K) tensor containing distance to nearest edges separated by types provided in edge_2d_mask
    """
    if single_pose_compute:
        assert rot_mtx is not None and perms is not None
        N_q = query_points.shape[0]
        N_k = edge_2d_mask.shape[-1]
        dist_2d = distance_func_2d(query_points, edge_2d, edge_2d_mask, rot_mtx=None)  # (N_q, N_k)
        rot_query_points = query_points @ rot_mtx.permute(0, 2, 1)  # (N_r, N_q, N_k)
        rot_nn_dist = (rot_query_points.unsqueeze(2) - query_points.reshape(1, 1, N_q, N_k)).norm(dim=-1)  # (N_r, N_q, N_q)
        rot_nn_idx = rot_nn_dist.argmin(-1)  # (N_r, N_q)
        dist_2d = dist_2d[:, perms].permute(1, 0, 2)  # (N_r, N_q, N_k)
        dist_2d = torch.gather(dist_2d, 1, rot_nn_idx.unsqueeze(-1).repeat(1, 1, N_k))  # (N_r, N_q, N_k)
    else:
        dist_2d = distance_func_2d(query_points, edge_2d, edge_2d_mask, rot_mtx=rot_mtx)
    return dist_2d


def split_func_3d_batch(query_points, starts, ends, trans_mtx, rot_mtx, edge_3d_mask):
    """
    Function that returns distance functions for a set of edge in 3D, split by type specified in edge_3d_mask.
    Here the edge_3d_mask is applied in a parallel manner. Note that query points should be normalized.

    Args:
        query_points: (N_q, 3) tensor containing points to query
        starts: (N_3D, 3) tensor containing 3D edge start points
        ends: (N_3D, 3) tensor containing 3D edge end points
        trans_mtx: (N_t, 3) tensor containing translation of point cloud
        rot_mtx: (N_r, 3, 3) tensor containing rotation of point cloud
        edge_3d_mask: (N_3D, K) or (N_r, N_3D, K) tensor containing masks for each 3D line, which may belong to K edge types
        dist_type: Type of distance metric to evaluate
    
    Returns:
        dist_3d: (N_t, N_r, N_q, K) tensor containing distance to nearest edges
    """    
    dist_3d = distance_func_3d_batch(query_points, starts, ends, trans_mtx, rot_mtx, edge_3d_mask)

    return dist_3d

# Distance functions


def distance_func_2d(query_points, edge_2d, mask=None, return_raw=False, rot_mtx=None):
    """
    Function that returns the closest distance to a set of edges in a 2D panorama image.
    Note that query points should be normalized.

    Args:
        query_points: (N_q, 3) tensor containing points to query
        edge_2d: (N_2D, 9) tensor containing 2D edges in [normals starts ends] format
        mask: (N_2D, K) or (N_r, N_2D, K) tensor containing boolean values for lines belonging to one of K classes
        return_raw: If True, return raw (N_q, N_2D) spherical distance matrix
        rot_mtx: (N_r, 3, 3) tensor containing rotation candidate poses, where the rotation inverses will be applied to the lines

    Returns:
        dist_2d: (N_q, ), (N_q, K), or (N_r, N_q, K) tensor containing distance to nearest edges
    """
    if edge_2d.shape[0] == 0:  # Return all infinity if there are no edges
        return torch.ones_like(query_points[:, 0]) * np.inf

    if rot_mtx is None:  # Calculate distance function for fix line
        normals = edge_2d[:, :3]
        starts = edge_2d[:, 3:6]
        ends = edge_2d[:, 6:]
        
        cos_theta = (starts * ends).sum(-1).unsqueeze(0)  # (1, N_2D)
        cos_theta1 = (query_points @ starts.t())  # (N_q, N_2D)
        cos_theta2 = (query_points @ ends.t())  # (N_q, N_2D)

        normal_acute = np.pi / 2 - torch.arccos(torch.abs(query_points @ normals.t()).clip_(min=-1., max=1.))  # Angle between edge point and line normal
        theta1 = torch.arccos(cos_theta1.clip_(min=-1., max=1.))  # Angle between edge point and line start
        theta2 = torch.arccos(cos_theta2.clip_(min=-1., max=1.))  # Angle between edge point and line end

        # Determine if angles of spherical triangle are over 90 degrees
        sign_arc_theta1 = (cos_theta1 - cos_theta * cos_theta2 > 0)  # Positive indicates arc_theta1 is smaller than 90
        sign_arc_theta2 = (cos_theta2 - cos_theta * cos_theta1 > 0)

        sphere_dist = (sign_arc_theta1 & sign_arc_theta2) * normal_acute + \
            torch.bitwise_not(sign_arc_theta1 & sign_arc_theta2) * torch.minimum(theta1, theta2)  # (N_q, N_2D)
        
        if return_raw:
            return sphere_dist

        if mask is None:
            dist_2d = sphere_dist.min(-1).values
        elif len(mask.shape) == 2:
            MAX_LIMIT = np.pi
            dist_2d = sphere_dist.unsqueeze(-1) * mask.unsqueeze(0)  # (N_q, N_2D, K)
            dist_2d += torch.bitwise_not(mask.unsqueeze(0)) * MAX_LIMIT  # Make all other entries not belonging to the class invalid
            dist_2d = dist_2d.min(1).values  # (N_q, K)
        elif len(mask.shape) == 3:
            MAX_LIMIT = np.pi
            dist_2d = sphere_dist.unsqueeze(-1).unsqueeze(0) * mask.unsqueeze(1)  # (N_r, N_q, N_2D, K)
            dist_2d += torch.bitwise_not(mask.unsqueeze(1)) * MAX_LIMIT  # Make all other entries not belonging to the class invalid
            dist_2d = dist_2d.min(2).values  # (N_r, N_q, K)
        else:
            raise ValueError("Invalid mask shape")
    
    else:
        normals = edge_2d[:, :3]
        starts = edge_2d[:, 3:6]
        ends = edge_2d[:, 6:]

        # Apply inverse rotation to the lines
        rot_normals = torch.stack([(normals.unsqueeze(0) * (rot_mtx.permute(0, 2, 1))[:, i:i + 1, :]).sum(-1) 
            for i in range(3)], dim=-1)  # (N_r, N_2D, 3)
        rot_normals = rot_normals / rot_normals.norm(dim=-1, keepdim=True)
        rot_starts = torch.stack([(starts.unsqueeze(0) * (rot_mtx.permute(0, 2, 1))[:, i:i + 1, :]).sum(-1) 
            for i in range(3)], dim=-1)  # (N_r, N_2D, 3)
        rot_starts = rot_starts / rot_starts.norm(dim=-1, keepdim=True)
        rot_ends = torch.stack([(ends.unsqueeze(0) * (rot_mtx.permute(0, 2, 1))[:, i:i + 1, :]).sum(-1) 
            for i in range(3)], dim=-1)  # (N_r, N_2D, 3)
        rot_ends = rot_ends / rot_ends.norm(dim=-1, keepdim=True)

        cos_theta = (rot_starts * rot_ends).sum(-1).unsqueeze(-2)  # (N_r, 1, N_2D)
        cos_theta1 = (rot_starts @ query_points.t()).permute(0, 2, 1)  # (N_r, N_q, N_2D)
        cos_theta2 = (rot_ends @ query_points.t()).permute(0, 2, 1)  # (N_r, N_q, N_2D)

        # Angle between edge point and line normal
        normal_acute = np.pi / 2 - torch.arccos(torch.abs(rot_normals @ query_points.t()).clip_(min=-1., max=1.))  # (N_r, N_2D, N_q)
        normal_acute = normal_acute.permute(0, 2, 1)  # (N_r, N_q, N_2D)
        theta1 = torch.arccos(cos_theta1.clip_(min=-1., max=1.))  # Angle between edge point and line start
        theta2 = torch.arccos(cos_theta2.clip_(min=-1., max=1.))  # Angle between edge point and line end

        # Determine if angles of spherical triangle are over 90 degrees
        sign_arc_theta1 = (cos_theta1 - cos_theta * cos_theta2 > 0)  # Positive indicates arc_theta1 is smaller than 90
        sign_arc_theta2 = (cos_theta2 - cos_theta * cos_theta1 > 0)

        sphere_dist = (sign_arc_theta1 & sign_arc_theta2) * normal_acute + \
            torch.bitwise_not(sign_arc_theta1 & sign_arc_theta2) * torch.minimum(theta1, theta2)  # (N_r, N_q, N_2D)

        if mask is None:
            dist_2d = sphere_dist.min(-1).values
        elif len(mask.shape) == 2:
            MAX_LIMIT = np.pi
            new_mask = mask.reshape(1, 1, mask.shape[0], mask.shape[1])
            dist_2d = sphere_dist.unsqueeze(-1) * new_mask  # (N_r, N_q, N_2D, K)
            dist_2d += torch.bitwise_not(new_mask) * MAX_LIMIT  # Make all other entries not belonging to the class invalid
            dist_2d = dist_2d.min(-2).values  # (N_r, N_q, K)
        elif len(mask.shape) == 3:
            MAX_LIMIT = np.pi
            new_mask = mask.reshape(mask.shape[0], 1, mask.shape[1], mask.shape[2])
            dist_2d = sphere_dist.unsqueeze(-1) * new_mask  # (N_r, N_q, N_3D, K)
            dist_2d += torch.bitwise_not(new_mask) * MAX_LIMIT  # Make all other entries not belonging to the class invalid
            dist_2d = dist_2d.min(-2).values  # (N_r, N_q, K)
        else:
            raise ValueError("Invalid mask shape")

    return dist_2d


def distance_func_3d(query_points, starts, ends, trans_mtx, rot_mtx, mask=None):
    """
    Function that returns the closest distance to a set of edges in a 3D point cloud.
    Note that query points should be normalized.

    Args:
        query_points: (N_q, 3) tensor containing points to query
        starts: (N_3D, 3) tensor containing 3D edge start points
        ends: (N_3D, 3) tensor containing 3D edge end points
        trans_mtx: (1, 3) tensor containing translation of point cloud
        rot_mtx: (3, 3) tensor containing rotation of point cloud
        mask: (N_3D, K) tensor containing boolean values for lines belonging to one of K classes
    
    Returns:
        dist_3d: (N_q, ) or (N_q, K) tensor containing distance to nearest edges
    """
    orig_transform_starts = (starts - trans_mtx) @ rot_mtx.t()
    transform_starts = orig_transform_starts / orig_transform_starts.norm(dim=-1, keepdim=True)
    orig_transform_ends = (ends - trans_mtx) @ rot_mtx.t()
    transform_ends = orig_transform_ends / orig_transform_ends.norm(dim=-1, keepdim=True)
    normals = torch.cross(transform_starts, transform_ends)  # (N_3D, 3)
    normals = normals / normals.norm(dim=-1, keepdim=True)
    
    cos_theta = (transform_starts * transform_ends).sum(-1).unsqueeze(0)  # (1, N_3D)
    cos_theta1 = (query_points @ transform_starts.t())  # (N_q, N_3D)
    cos_theta2 = (query_points @ transform_ends.t())  # (N_q, N_3D)

    normal_acute = np.pi / 2 - torch.arccos(torch.abs(query_points @ normals.t()).clip_(min=-1., max=1.))  # Angle between edge point and line normal
    theta1 = torch.arccos(cos_theta1.clip_(min=-1., max=1.))  # Angle between edge point and line start
    theta2 = torch.arccos(cos_theta2.clip_(min=-1., max=1.))  # Angle between edge point and line end

    # Determine if angles of spherical triangle are over 90 degrees
    sign_arc_theta1 = (cos_theta1 - cos_theta * cos_theta2 > 0)  # Positive indicates arc_theta1 is smaller than 90
    sign_arc_theta2 = (cos_theta2 - cos_theta * cos_theta1 > 0)

    sphere_dist = (sign_arc_theta1 & sign_arc_theta2) * normal_acute + \
        torch.bitwise_not(sign_arc_theta1 & sign_arc_theta2) * torch.minimum(theta1, theta2)  # (N_q, N_3D)

    if mask is None:
        dist_3d = sphere_dist.min(-1).values
    else:
        MAX_LIMIT = np.pi
        dist_3d = sphere_dist.unsqueeze(-1) * mask.unsqueeze(0)  # (N_q, N_3D, K)
        dist_3d += torch.bitwise_not(mask.unsqueeze(0)) * MAX_LIMIT  # Make all other entries not belonging to the class invalid
        dist_3d = dist_3d.min(1).values  # (N_q, K)
    return dist_3d


def distance_func_3d_batch(query_points, starts, ends, trans_mtx, rot_mtx, mask=None):
    """
    Function that returns the closest distance to a set of edges in a 3D point cloud.
    This implementation assumes 'batches' of translations and rotations to be given.
    Note that query points should be normalized.

    Args:
        query_points: (N_q, 3) tensor containing points to query
        starts: (N_3D, 3) tensor containing 3D edge start points
        ends: (N_3D, 3) tensor containing 3D edge end points
        trans_mtx: (N_t, 3) tensor containing translation of point cloud
        rot_mtx: (N_r, 3, 3) tensor containing rotation of point cloud
        mask: (N_3D, K) or (N_r, N_3D, K) tensor containing boolean values for lines belonging to one of K classes
   
    Returns:
        dist_3d: (N_t, N_r, N_q) or (N_t, N_r, N_q, K) tensor containing distance to nearest edges
    """
    orig_transform_starts = (starts.unsqueeze(0) - trans_mtx.unsqueeze(1)).unsqueeze(1)  # (N_t, 1, N_3D, 3) 
    orig_transform_starts = torch.stack([(orig_transform_starts * rot_mtx[:, i:i + 1, :].unsqueeze(0)).sum(-1) 
        for i in range(3)], dim=-1)  # (N_t, N_r, N_3D, 3)
    transform_starts = orig_transform_starts / orig_transform_starts.norm(dim=-1, keepdim=True)
    orig_transform_ends = (ends.unsqueeze(0) - trans_mtx.unsqueeze(1)).unsqueeze(1)  # (N_t, 1, N_3D, 3) 
    orig_transform_ends = torch.stack([(orig_transform_ends * rot_mtx[:, i:i + 1, :].unsqueeze(0)).sum(-1) 
        for i in range(3)], dim=-1)  # (N_t, N_r, N_3D, 3)
    transform_ends = orig_transform_ends / orig_transform_ends.norm(dim=-1, keepdim=True)
    normals = torch.cross(transform_starts, transform_ends, dim=-1)  # (N_t, N_r, N_3D, 3)
    normals = normals / normals.norm(dim=-1, keepdim=True)
    
    cos_theta = (transform_starts * transform_ends).sum(-1).unsqueeze(-2)  # (N_t, N_r, 1, N_3D)
    cos_theta1 = (transform_starts @ query_points.t()).permute(0, 1, 3, 2)  # (N_t, N_r, N_q, N_3D)
    cos_theta2 = (transform_ends @ query_points.t()).permute(0, 1, 3, 2)  # (N_t, N_r, N_q, N_3D)

    # Angle between edge point and line normal
    normal_acute = np.pi / 2 - torch.arccos(torch.abs(normals @ query_points.t()).clip_(min=-1., max=1.))  # (N_t, N_r, N_3D, N_q)
    normal_acute = normal_acute.permute(0, 1, 3, 2)  # (N_t, N_r, N_q, N_3D)
    theta1 = torch.arccos(cos_theta1.clip_(min=-1., max=1.))  # Angle between edge point and line start
    theta2 = torch.arccos(cos_theta2.clip_(min=-1., max=1.))  # Angle between edge point and line end

    # Determine if angles of spherical triangle are over 90 degrees
    sign_arc_theta1 = (cos_theta1 - cos_theta * cos_theta2 > 0)  # Positive indicates arc_theta1 is smaller than 90
    sign_arc_theta2 = (cos_theta2 - cos_theta * cos_theta1 > 0)

    sphere_dist = (sign_arc_theta1 & sign_arc_theta2) * normal_acute + \
        torch.bitwise_not(sign_arc_theta1 & sign_arc_theta2) * torch.minimum(theta1, theta2)  # (N_t, N_r, N_q, N_3D)

    if mask is None:
        dist_3d = sphere_dist.min(-1).values
    elif len(mask.shape) == 2:
        MAX_LIMIT = np.pi
        new_mask = mask.reshape(1, 1, 1, mask.shape[0], mask.shape[1])
        dist_3d = sphere_dist.unsqueeze(-1) * new_mask  # (N_t, N_r, N_q, N_3D, K)
        dist_3d += torch.bitwise_not(new_mask) * MAX_LIMIT  # Make all other entries not belonging to the class invalid
        dist_3d = dist_3d.min(-2).values  # (N_t, N_r, N_q, K)
    elif len(mask.shape) == 3:
        MAX_LIMIT = np.pi
        new_mask = mask.reshape(1, mask.shape[0], 1, mask.shape[1], mask.shape[2])
        dist_3d = sphere_dist.unsqueeze(-1) * new_mask  # (N_t, N_r, N_q, N_3D, K)
        dist_3d += torch.bitwise_not(new_mask) * MAX_LIMIT  # Make all other entries not belonging to the class invalid
        dist_3d = dist_3d.min(-2).values  # (N_t, N_r, N_q, K)
    else:
        raise ValueError("Invalid mask shape")
    return dist_3d


# Visualization functions

def visualize_field_2d(edge_2d, field='distance', mode='pcd', filename='vis_2d', level=5, resolution=(200, 400), return_pano=False):
    """
    Visualize distance field for a set of 2D edges.

    Args:
        edge_2d: (N_2D, 9) tensor containing 2D edges in [normals starts ends] format
        field: Type of distance field to visualize
        mode: If 'pcd', saves visualization as a point cloud, and if 'pano', saves visualization as a panorama image
        filename: Filename to save visualization results suffixes are automatically added
        level: Icosahedron subdivision number used when mode is 'pcd'
        resolution: Resolution of panorama image to generate when mode is 'pano'
        return_pano: If True, returns panorama image and does not save the result

    Returns:
        None
    """
    if mode == 'pcd':
        sphere_pts = generate_sphere_pts(level, type='torch', device=edge_2d.device)  # (N, 3)
        if field == 'distance':
            dist_2d = distance_func_2d(sphere_pts, edge_2d).unsqueeze(-1)  # (N, 1)
        tot_pts = torch.cat([sphere_pts, dist_2d], dim=-1)
        np.savetxt(f"{filename}.txt", tot_pts.cpu())
        
    elif mode == 'pano':
        white_img = torch.ones(resolution, device=edge_2d.device)
        pano_pts = ij2coord(torch.nonzero(white_img), resolution)  # (N, 3)
        if field == 'distance':
            dist_2d = distance_func_2d(pano_pts, edge_2d).unsqueeze(-1)  # (N, 1)
        dist_2d = dist_2d / dist_2d.max()
        pano_img = make_pano(pano_pts, dist_2d.repeat(1, 3), resolution)
        pano_img = pano_img[..., 0].astype(np.uint8)
        if return_pano:
            pano_img = cv2.applyColorMap(pano_img, cv2.COLORMAP_JET)
            return pano_img
        else:
            pano_img = cv2.applyColorMap(pano_img, cv2.COLORMAP_JET)
            cv2.imwrite(f"{filename}.png", pano_img)
    else:
        raise ValueError("Specify a valid mode")


def visualize_field_3d(starts, ends, trans_mtx, rot_mtx, field='distance', mode='pcd', filename='vis_3d', level=5, resolution=(200, 400), return_pano=False):
    """
    Visualize distance field for a set of 2D edges.

    Args:
        starts: (N_3D, 3) tensor containing 3D edge start points
        ends: (N_3D, 3) tensor containing 3D edge end points
        trans_mtx: (1, 3) tensor containing translation of point cloud
        rot_mtx: (3, 3) tensor containing rotation of point cloud
        field: Type of distance field to visualize
        mode: If 'pcd', saves visualization as a point cloud, and if 'pano', saves visualization as a panorama image
        filename: Filename to save visualization results suffixes are automatically added
        level: Icosahedron subdivision number used when mode is 'pcd'
        resolution: Resolution of panorama image to generate when mode is 'pano'
        return_pano: If True, returns panorama image and does not save the result

    Returns:
        None
    """
    if mode == 'pcd':
        sphere_pts = generate_sphere_pts(level, type='torch', device=starts.device)  # (N, 3)
        if field == 'distance':
            dist_3d = distance_func_3d(sphere_pts, starts, ends, trans_mtx, rot_mtx).unsqueeze(-1)  # (N, 1)
        tot_pts = torch.cat([sphere_pts, dist_3d], dim=-1)
        np.savetxt(f"{filename}.txt", tot_pts.cpu())
        
    elif mode == 'pano':
        white_img = torch.ones(resolution, device=starts.device)
        pano_pts = ij2coord(torch.nonzero(white_img), resolution)  # (N, 3)
        if field == 'distance':
            dist_3d = distance_func_3d(pano_pts, starts, ends, trans_mtx, rot_mtx).unsqueeze(-1)  # (N, 1)
        dist_3d = dist_3d / dist_3d.max()
        pano_img = make_pano(pano_pts, dist_3d.repeat(1, 3), resolution)
        pano_img = pano_img[..., 0].astype(np.uint8)
        if return_pano:
            pano_img = cv2.applyColorMap(pano_img, cv2.COLORMAP_JET)
            return pano_img
        else:
            pano_img = cv2.applyColorMap(pano_img, cv2.COLORMAP_JET)
            cv2.imwrite(f"{filename}.png", pano_img)
    else:
        raise ValueError("Specify a valid mode")


# Utility fuctions excerpted from https://github.com/sunset1995/HorizonNet

def filterEdgeByThres(coordN_lines, threshold=0.3, return_mask=False):
    # Filter detected edges over a specified threshold
    length = np.arccos((coordN_lines[:, 3:6] * coordN_lines[:, 6:]).sum(-1))
    if return_mask:
        return coordN_lines[length > threshold], (length > threshold)
    else:
        return coordN_lines[length > threshold]


def filterEdgeByTopK(coordN_lines, k, return_mask=False):
    # Filter detected edges over a specified threshold
    length = np.arccos((coordN_lines[:, 3:6] * coordN_lines[:, 6:]).sum(-1))
    if return_mask:
        return coordN_lines[np.argsort(length)[-k:]], np.argsort(length)[-k:]
    else:
        return coordN_lines[np.argsort(length)[-k:]]


def computeUVN(n, in_, planeID):
    '''
    compute v given u and normal.
    '''
    if planeID == 2:
        n = np.array([n[1], n[2], n[0]])
    elif planeID == 3:
        n = np.array([n[2], n[0], n[1]])
    bc = n[0] * np.sin(in_) + n[1] * np.cos(in_)
    bs = n[2]
    out = np.arctan(-bc / (bs + 1e-9))
    return out


def computeUVN_vec(n, in_, planeID):
    '''
    vectorization version of computeUVN
    @n         N x 3
    @in_      MN x 1
    @planeID   N
    '''
    n = n.copy()
    if (planeID == 2).sum():
        n[planeID == 2] = np.roll(n[planeID == 2], 2, axis=1)
    if (planeID == 3).sum():
        n[planeID == 3] = np.roll(n[planeID == 3], 1, axis=1)
    n = np.repeat(n, in_.shape[0] // n.shape[0], axis=0)
    assert n.shape[0] == in_.shape[0]
    bc = n[:, [0]] * np.sin(in_) + n[:, [1]] * np.cos(in_)
    bs = n[:, [2]]
    out = np.arctan(-bc / (bs + 1e-9))
    return out


def xyz2uvN(xyz, planeID=1):
    ID1 = (int(planeID) - 1 + 0) % 3
    ID2 = (int(planeID) - 1 + 1) % 3
    ID3 = (int(planeID) - 1 + 2) % 3
    normXY = np.sqrt(xyz[:, [ID1]] ** 2 + xyz[:, [ID2]] ** 2)
    normXY[normXY < 0.000001] = 0.000001
    normXYZ = np.sqrt(xyz[:, [ID1]] ** 2 + xyz[:, [ID2]] ** 2 + xyz[:, [ID3]] ** 2)
    v = np.arcsin(xyz[:, [ID3]] / normXYZ)
    u = np.arcsin(xyz[:, [ID1]] / normXY)
    valid = (xyz[:, [ID2]] < 0) & (u >= 0)
    u[valid] = np.pi - u[valid]
    valid = (xyz[:, [ID2]] < 0) & (u <= 0)
    u[valid] = -np.pi - u[valid]
    uv = np.hstack([u, v])
    uv[np.isnan(uv[:, 0]), 0] = 0
    return uv


def uv2xyzN(uv, planeID=1):
    ID1 = (int(planeID) - 1 + 0) % 3
    ID2 = (int(planeID) - 1 + 1) % 3
    ID3 = (int(planeID) - 1 + 2) % 3
    xyz = np.zeros((uv.shape[0], 3))
    xyz[:, ID1] = np.cos(uv[:, 1]) * np.sin(uv[:, 0])
    xyz[:, ID2] = np.cos(uv[:, 1]) * np.cos(uv[:, 0])
    xyz[:, ID3] = np.sin(uv[:, 1])
    return xyz


def uv2xyzN_vec(uv, planeID):
    '''
    vectorization version of uv2xyzN
    @uv       N x 2
    @planeID  N
    '''
    assert (planeID.astype(int) != planeID).sum() == 0
    planeID = planeID.astype(int)
    ID1 = (planeID - 1 + 0) % 3
    ID2 = (planeID - 1 + 1) % 3
    ID3 = (planeID - 1 + 2) % 3
    ID = np.arange(len(uv))
    xyz = np.zeros((len(uv), 3))
    xyz[ID, ID1] = np.cos(uv[:, 1]) * np.sin(uv[:, 0])
    xyz[ID, ID2] = np.cos(uv[:, 1]) * np.cos(uv[:, 0])
    xyz[ID, ID3] = np.sin(uv[:, 1])
    return xyz


def warpImageFast(im, XXdense, YYdense):
    minX = max(1., np.floor(XXdense.min()) - 1)
    minY = max(1., np.floor(YYdense.min()) - 1)

    maxX = min(im.shape[1], np.ceil(XXdense.max()) + 1)
    maxY = min(im.shape[0], np.ceil(YYdense.max()) + 1)

    im = im[int(round(minY-1)):int(round(maxY)),
            int(round(minX-1)):int(round(maxX))]

    assert XXdense.shape == YYdense.shape
    out_shape = XXdense.shape
    coordinates = [
        (YYdense - minY).reshape(-1),
        (XXdense - minX).reshape(-1),
    ]
    im_warp = np.stack([
        map_coordinates(im[..., c], coordinates, order=1).reshape(out_shape)
        for c in range(im.shape[-1])],
        axis=-1)

    return im_warp


def rotatePanorama(img, vp=None, R=None):
    '''
    Rotate panorama
        if R is given, vp (vanishing point) will be overlooked
        otherwise R is computed from vp
    '''
    sphereH, sphereW, C = img.shape

    # new uv coordinates
    TX, TY = np.meshgrid(range(1, sphereW + 1), range(1, sphereH + 1))
    TX = TX.reshape(-1, 1, order='F')
    TY = TY.reshape(-1, 1, order='F')
    ANGx = (TX - sphereW/2 - 0.5) / sphereW * np.pi * 2
    ANGy = -(TY - sphereH/2 - 0.5) / sphereH * np.pi
    uvNew = np.hstack([ANGx, ANGy])
    xyzNew = uv2xyzN(uvNew, 1)

    # rotation matrix
    if R is None:
        R = np.linalg.inv(vp.T)

    xyzOld = np.linalg.solve(R, xyzNew.T).T
    uvOld = xyz2uvN(xyzOld, 1)

    Px = (uvOld[:, 0] + np.pi) / (2*np.pi) * sphereW + 0.5
    Py = (-uvOld[:, 1] + np.pi/2) / np.pi * sphereH + 0.5

    Px = Px.reshape(sphereH, sphereW, order='F')
    Py = Py.reshape(sphereH, sphereW, order='F')

    # boundary
    imgNew = np.zeros((sphereH+2, sphereW+2, C), np.float64)
    imgNew[1:-1, 1:-1, :] = img
    imgNew[1:-1, 0, :] = img[:, -1, :]
    imgNew[1:-1, -1, :] = img[:, 0, :]
    imgNew[0, 1:sphereW//2+1, :] = img[0, sphereW-1:sphereW//2-1:-1, :]
    imgNew[0, sphereW//2+1:-1, :] = img[0, sphereW//2-1::-1, :]
    imgNew[-1, 1:sphereW//2+1, :] = img[-1, sphereW-1:sphereW//2-1:-1, :]
    imgNew[-1, sphereW//2+1:-1, :] = img[0, sphereW//2-1::-1, :]
    imgNew[0, 0, :] = img[0, 0, :]
    imgNew[-1, -1, :] = img[-1, -1, :]
    imgNew[0, -1, :] = img[0, -1, :]
    imgNew[-1, 0, :] = img[-1, 0, :]

    rotImg = warpImageFast(imgNew, Px+1, Py+1)

    return rotImg


def paint_line(parameterLine, width, height):
    lines = parameterLine.copy()
    panoEdgeC = np.zeros((height, width))

    num_sample = max(height, width)
    for i in range(len(lines)):
        n = lines[i, :3]
        sid = lines[i, 4] * 2 * np.pi
        eid = lines[i, 5] * 2 * np.pi
        if eid < sid:
            x = np.linspace(sid, eid + 2 * np.pi, num_sample)
            x = x % (2 * np.pi)
        else:
            x = np.linspace(sid, eid, num_sample)
        u = -np.pi + x.reshape(-1, 1)
        v = computeUVN(n, u, lines[i, 3])
        xyz = uv2xyzN(np.hstack([u, v]), lines[i, 3])
        uv = xyz2uvN(xyz, 1)
        m = np.minimum(np.floor((uv[:,0] + np.pi) / (2 * np.pi) * width) + 1,
            width).astype(np.int32)
        n = np.minimum(np.floor(((np.pi / 2) - uv[:, 1]) / np.pi * height) + 1,
            height).astype(np.int32)
        panoEdgeC[n-1, m-1] = i

    return panoEdgeC


def combineEdgesN(edges):
    '''
    Combine some small line segments, should be very conservative
    OUTPUT
        lines: combined line segments
        line format [nx ny nz projectPlaneID umin umax LSfov score]
        coordN_lines: combined line segments with normal, start coordinate, and end coordinate
    '''
    arcList = []
    for edge in edges:
        panoLst = edge['panoLst']
        if len(panoLst) == 0:
            continue
        arcList.append(panoLst)
    arcList = np.vstack(arcList)

    # ori lines
    numLine = len(arcList)
    ori_lines = np.zeros((numLine, 8))
    ori_coordN_lines = np.zeros((numLine, 9))  # Line containing coordinate and normals

    areaXY = np.abs(arcList[:, 2])
    areaYZ = np.abs(arcList[:, 0])
    areaZX = np.abs(arcList[:, 1])
    planeIDs = np.argmax(np.stack([areaXY, areaYZ, areaZX], -1), 1) + 1  # XY YZ ZX

    for i in range(numLine):
        ori_lines[i, :3] = arcList[i, :3]
        ori_lines[i, 3] = planeIDs[i]
        coord1 = arcList[i, 3:6]
        coord2 = arcList[i, 6:9]
        uv = xyz2uvN(np.stack([coord1, coord2]), planeIDs[i])
        umax = uv[:, 0].max() + np.pi
        umin = uv[:, 0].min() + np.pi
        if umax - umin > np.pi:
            ori_lines[i, 4:6] = np.array([umax, umin]) / 2 / np.pi
        else:
            ori_lines[i, 4:6] = np.array([umin, umax]) / 2 / np.pi
        ori_lines[i, 6] = np.arccos((
            np.dot(coord1, coord2) / (np.linalg.norm(coord1) * np.linalg.norm(coord2))
            ).clip(-1, 1))
        ori_lines[i, 7] = arcList[i, 9]

        # Copy assignments to ori_coordN_lines
        ori_coordN_lines[i, :3] = arcList[i, :3]
        ori_coordN_lines[i, 3:6] = coord1 / np.linalg.norm(coord1, axis=-1)
        ori_coordN_lines[i, 6:9] = coord2 / np.linalg.norm(coord2, axis=-1)

    # additive combination
    lines = ori_lines.copy()
    coordN_lines = ori_coordN_lines.copy()

    for _ in range(3):
        numLine = len(lines)
        valid_line = np.ones(numLine, bool)
        for i in range(numLine):
            if not valid_line[i]:
                continue
            dotProd = (lines[:, :3] * lines[[i], :3]).sum(1)
            valid_curr = np.logical_and((np.abs(dotProd) > np.cos(np.pi / 180)), valid_line)
            valid_curr[i] = False
            for j in np.nonzero(valid_curr)[0]:
                range1 = lines[i, 4:6]
                range2 = lines[j, 4:6]
                valid_rag = _intersection(range1, range2)
                if not valid_rag:
                    continue

                # combine
                I = np.argmax(np.abs(lines[i, :3]))
                if lines[i, I] * lines[j, I] > 0:
                    nc = lines[i, :3] * lines[i, 6] + lines[j, :3] * lines[j, 6]
                else:
                    nc = lines[i, :3] * lines[i, 6] - lines[j, :3] * lines[j, 6]
                nc = nc / np.linalg.norm(nc)

                if _insideRange(range1[0], range2):
                    nrmin = range2[0]
                else:
                    nrmin = range1[0]

                if _insideRange(range1[1], range2):
                    nrmax = range2[1]
                else:
                    nrmax = range1[1]

                u = np.array([[nrmin], [nrmax]]) * 2 * np.pi - np.pi
                v = computeUVN(nc, u, lines[i, 3])
                xyz = uv2xyzN(np.hstack([u, v]), lines[i, 3])
                l = np.arccos(np.dot(xyz[0, :], xyz[1, :]).clip(-1, 1))
                scr = (lines[i,6]*lines[i,7] + lines[j,6]*lines[j,7]) / (lines[i,6]+lines[j,6])

                lines[i] = [*nc, lines[i, 3], nrmin, nrmax, l, scr]
                coordN_lines[i] = [*nc, *xyz[0], *xyz[1]]
                valid_line[j] = False

        lines = lines[valid_line]
        coordN_lines = coordN_lines[valid_line]

    return lines, coordN_lines


def edgeFromImg2Pano(edge):
    edgeList = edge['edgeLst']
    if len(edgeList) == 0:
        return np.array([])

    vx = edge['vx']
    vy = edge['vy']
    fov = edge['fov']
    imH, imW = edge['img'].shape

    R = (imW/2) / np.tan(fov/2)

    # im is the tangent plane, contacting with ball at [x0 y0 z0]
    x0 = R * np.cos(vy) * np.sin(vx)
    y0 = R * np.cos(vy) * np.cos(vx)
    z0 = R * np.sin(vy)
    vecposX = np.array([np.cos(vx), -np.sin(vx), 0])
    vecposY = np.cross(np.array([x0, y0, z0]), vecposX)
    vecposY = vecposY / np.sqrt(vecposY @ vecposY.T)
    vecposX = vecposX.reshape(1, -1)
    vecposY = vecposY.reshape(1, -1)
    Xc = (0 + imW-1) / 2
    Yc = (0 + imH-1) / 2

    vecx1 = edgeList[:, [0]] - Xc
    vecy1 = edgeList[:, [1]] - Yc
    vecx2 = edgeList[:, [2]] - Xc
    vecy2 = edgeList[:, [3]] - Yc

    vec1 = np.tile(vecx1, [1, 3]) * vecposX + np.tile(vecy1, [1, 3]) * vecposY
    vec2 = np.tile(vecx2, [1, 3]) * vecposX + np.tile(vecy2, [1, 3]) * vecposY
    coord1 = [[x0, y0, z0]] + vec1
    coord2 = [[x0, y0, z0]] + vec2

    normal = np.cross(coord1, coord2, axis=1)
    normal = normal / np.linalg.norm(normal, axis=1, keepdims=True)

    panoList = np.hstack([normal, coord1, coord2, edgeList[:, [-1]]])

    return panoList


def _intersection(range1, range2):
    if range1[1] < range1[0]:
        range11 = [range1[0], 1]
        range12 = [0, range1[1]]
    else:
        range11 = range1
        range12 = [0, 0]

    if range2[1] < range2[0]:
        range21 = [range2[0], 1]
        range22 = [0, range2[1]]
    else:
        range21 = range2
        range22 = [0, 0]

    b = max(range11[0], range21[0]) < min(range11[1], range21[1])
    if b:
        return b
    b2 = max(range12[0], range22[0]) < min(range12[1], range22[1])
    b = b or b2
    return b


def _insideRange(pt, range):
    if range[1] > range[0]:
        b = pt >= range[0] and pt <= range[1]
    else:
        b1 = pt >= range[0] and pt <= 1
        b2 = pt >= 0 and pt <= range[1]
        b = b1 or b2
    return b


def lsdWrap(img):
    '''
    Opencv implementation of
    Rafael Grompone von Gioi, Jérémie Jakubowicz, Jean-Michel Morel, and Gregory Randall,
    LSD: a Line Segment Detector, Image Processing On Line, vol. 2012.
    [Rafael12] http://www.ipol.im/pub/art/2012/gjmr-lsd/?utm_source=doi
    @img
        input image
    '''
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    lines = lsd(img, quant=0.7)
    if lines is None:
        return np.zeros_like(img), np.array([])
    edgeMap = np.zeros_like(img)
    for i in range(lines.shape[0]):
        pt1 = (int(lines[i, 0]), int(lines[i, 1]))
        pt2 = (int(lines[i, 2]), int(lines[i, 3]))
        width = lines[i, 4]
        cv2.line(edgeMap, pt1, pt2, 255, int(np.ceil(width / 2)))
    edgeList = np.concatenate([lines, np.ones_like(lines[:, :2])], 1)
    return edgeMap, edgeList


def separatePano(panoImg, fov, x, y, imgSize=320):
    '''cut a panorama image into several separate views'''
    assert x.shape == y.shape
    if not isinstance(fov, np.ndarray):
        fov = fov * np.ones_like(x)

    sepScene = [
        {
            'img': imgLookAt(panoImg.copy(), xi, yi, imgSize, fovi),
            'vx': xi,
            'vy': yi,
            'fov': fovi,
            'sz': imgSize,
        }
        for xi, yi, fovi in zip(x, y, fov)
    ]

    return sepScene


def imgLookAt(im, CENTERx, CENTERy, new_imgH, fov):
    sphereH = im.shape[0]
    sphereW = im.shape[1]
    warped_im = np.zeros((new_imgH, new_imgH, 3))
    TX, TY = np.meshgrid(range(1, new_imgH + 1), range(1, new_imgH + 1))
    TX = TX.reshape(-1, 1, order='F')
    TY = TY.reshape(-1, 1, order='F')
    TX = TX - 0.5 - new_imgH/2
    TY = TY - 0.5 - new_imgH/2
    r = new_imgH / 2 / np.tan(fov/2)

    # convert to 3D
    R = np.sqrt(TY ** 2 + r ** 2)
    ANGy = np.arctan(- TY / r)
    ANGy = ANGy + CENTERy

    X = np.sin(ANGy) * R
    Y = -np.cos(ANGy) * R
    Z = TX

    INDn = np.nonzero(np.abs(ANGy) > np.pi/2)

    # project back to sphere
    ANGx = np.arctan(Z / -Y)
    RZY = np.sqrt(Z ** 2 + Y ** 2)
    ANGy = np.arctan(X / RZY)

    ANGx[INDn] = ANGx[INDn] + np.pi
    ANGx = ANGx + CENTERx

    INDy = np.nonzero(ANGy < -np.pi/2)
    ANGy[INDy] = -np.pi - ANGy[INDy]
    ANGx[INDy] = ANGx[INDy] + np.pi

    INDx = np.nonzero(ANGx <= -np.pi);   ANGx[INDx] = ANGx[INDx] + 2 * np.pi
    INDx = np.nonzero(ANGx >   np.pi);   ANGx[INDx] = ANGx[INDx] - 2 * np.pi
    INDx = np.nonzero(ANGx >   np.pi);   ANGx[INDx] = ANGx[INDx] - 2 * np.pi
    INDx = np.nonzero(ANGx >   np.pi);   ANGx[INDx] = ANGx[INDx] - 2 * np.pi

    Px = (ANGx + np.pi) / (2*np.pi) * sphereW + 0.5
    Py = ((-ANGy) + np.pi/2) / np.pi * sphereH + 0.5

    INDxx = np.nonzero(Px < 1)
    Px[INDxx] = Px[INDxx] + sphereW
    im = np.concatenate([im, im[:, :2]], 1)

    Px = Px.reshape(new_imgH, new_imgH, order='F')
    Py = Py.reshape(new_imgH, new_imgH, order='F')

    warped_im = warpImageFast(im, Px, Py)

    return warped_im


def icosahedron2sphere(level):
    # this function use a icosahedron to sample uniformly on a sphere
    a = 2 / (1 + np.sqrt(5))
    M = np.array([
        0, a, -1, a, 1, 0, -a, 1, 0,
        0, a, 1, -a, 1, 0, a, 1, 0,
        0, a, 1, 0, -a, 1, -1, 0, a,
        0, a, 1, 1, 0, a, 0, -a, 1,
        0, a, -1, 0, -a, -1, 1, 0, -a,
        0, a, -1, -1, 0, -a, 0, -a, -1,
        0, -a, 1, a, -1, 0, -a, -1, 0,
        0, -a, -1, -a, -1, 0, a, -1, 0,
        -a, 1, 0, -1, 0, a, -1, 0, -a,
        -a, -1, 0, -1, 0, -a, -1, 0, a,
        a, 1, 0, 1, 0, -a, 1, 0, a,
        a, -1, 0, 1, 0, a, 1, 0, -a,
        0, a, 1, -1, 0, a, -a, 1, 0,
        0, a, 1, a, 1, 0, 1, 0, a,
        0, a, -1, -a, 1, 0, -1, 0, -a,
        0, a, -1, 1, 0, -a, a, 1, 0,
        0, -a, -1, -1, 0, -a, -a, -1, 0,
        0, -a, -1, a, -1, 0, 1, 0, -a,
        0, -a, 1, -a, -1, 0, -1, 0, a,
        0, -a, 1, 1, 0, a, a, -1, 0])

    coor = M.T.reshape(3, 60, order='F').T
    coor, idx = np.unique(coor, return_inverse=True, axis=0)
    tri = idx.reshape(3, 20, order='F').T

    # extrude
    coor = list(coor / np.tile(np.linalg.norm(coor, axis=1, keepdims=True), (1, 3)))

    for _ in range(level):
        triN = []
        for t in range(len(tri)):
            n = len(coor)
            coor.append((coor[tri[t, 0]] + coor[tri[t, 1]]) / 2)
            coor.append((coor[tri[t, 1]] + coor[tri[t, 2]]) / 2)
            coor.append((coor[tri[t, 2]] + coor[tri[t, 0]]) / 2)

            triN.append([n, tri[t, 0], n+2])
            triN.append([n, tri[t, 1], n+1])
            triN.append([n+1, tri[t, 2], n+2])
            triN.append([n, n+1, n+2])
        tri = np.array(triN)

        # uniquefy
        coor, idx = np.unique(coor, return_inverse=True, axis=0)
        tri = idx[tri]

        # extrude
        coor = list(coor / np.tile(np.sqrt(np.sum(coor * coor, 1, keepdims=True)), (1, 3)))

    return np.array(coor), np.array(tri)


# Point distance functions
def distance_func_point_2d(query_points, kpts_2d, mask=None):
    """
    Function that returns the closest distance to a set of line intersections in a 2D panorama image.
    Note that query points should be normalized.

    Args:
        query_points: (N_q, 3) tensor containing points to query
        kpts_2d: (N_2D, 3) tensor containing 2D keypoints in sphere format
        mask: (N_2D, K) or (N_r, N_2D, K) tensor containing boolean values for points belonging to one of K classes
    
    Returns:
        dist_2d: (N_q, ), (N_q, K), or (N_r, N_q, K) tensor containing distance to nearest edges
    """
    cos_theta = (query_points @ kpts_2d.t())  # (N_q, N_2D)
    sphere_dist = torch.arccos(cos_theta.clip_(min=-1., max=1.))  # Angle between edge point and line start

    if mask is None:
        dist_2d = sphere_dist.min(-1).values
    elif len(mask.shape) == 2:
        MAX_LIMIT = np.pi
        dist_2d = sphere_dist.unsqueeze(-1) * mask.unsqueeze(0)  # (N_q, N_2D, K)
        dist_2d += torch.bitwise_not(mask.unsqueeze(0)) * MAX_LIMIT  # Make all other entries not belonging to the class invalid
        dist_2d = dist_2d.min(1).values  # (N_q, K)
    elif len(mask.shape) == 3:
        MAX_LIMIT = np.pi
        dist_2d = sphere_dist.unsqueeze(-1).unsqueeze(0) * mask.unsqueeze(1)  # (N_r, N_q, N_2D, K)
        dist_2d += torch.bitwise_not(mask.unsqueeze(1)) * MAX_LIMIT  # Make all other entries not belonging to the class invalid
        dist_2d = dist_2d.min(2).values  # (N_r, N_q, K)
    else:
        raise ValueError("Invalid mask shape")
    return dist_2d


def distance_func_point_2d_batch(query_points, kpts_2d, mask=None, rot_mtx=None):
    """
    Function that returns the closest distance to a set of line intersections in a 2D panorama image.
    Note that query points should be normalized.

    Args:
        query_points: (N_q, 3) tensor containing points to query
        kpts_2d: (N_r, N_2D, 3) tensor containing 2D keypoints in sphere format
        mask: (N_r, N_2D, K) tensor containing boolean values for points belonging to one of K classes
        rot_mtx: (N_r, 3, 3) tensor containing rotation candidate poses, where the rotation inverses will be applied to the points
    
    Returns:
        dist_2d: (N_r, N_q, K) tensor containing distance to nearest edges
    """
    if rot_mtx is None:
        cos_theta = (kpts_2d @ query_points.t()).permute(0, 2, 1)  # (N_r, N_q, N_2D)
    else:
        rot_kpts_2d = torch.stack([(kpts_2d * (rot_mtx.permute(0, 2, 1))[:, i:i + 1, :]).sum(-1)
            for i in range(3)], dim=-1)  # (N_r, N_2D, 3)
        cos_theta = (rot_kpts_2d @ query_points.t()).permute(0, 2, 1)  # (N_r, N_q, N_2D)
    sphere_dist = torch.arccos(cos_theta.clip_(min=-1., max=1.))  # Angle between edge point and line start

    if mask is None:
        dist_2d = sphere_dist.min(-1).values
    else:
        MAX_LIMIT = np.pi
        dist_2d = sphere_dist.unsqueeze(-1) * mask.unsqueeze(1)  # (N_r, N_q, N_2D, K)
        dist_2d += torch.bitwise_not(mask.unsqueeze(1)) * MAX_LIMIT  # Make all other entries not belonging to the class invalid
        dist_2d = dist_2d.min(2).values  # (N_r, N_q, K)
    return dist_2d


def distance_func_point_2d_single_compute(query_points, kpts_2d, mask, rot_mtx, perms):
    """
    Function that returns the closest distance to a set of line intersections in a 2D panorama image, where a single pose value is calculated and used for interpolation.
    Note that query points should be normalized.

    Args:
        query_points: (N_q, 3) tensor containing points to query
        kpts_2d: (N_2D, 3) tensor containing 2D keypoints in sphere format
        mask: (N_2D, K) tensor containing boolean values for points belonging to one of K classes
        rot_mtx: (N_r, 3, 3) tensor containing rotation candidate poses, where the rotation inverses will be applied to the points
        perms: (N_r, 3) tensor containing permutations used for obtaining rotations

    Returns:
        dist_2d: (N_r, N_q, K) tensor containing distance to nearest edges
    """
    N_q = query_points.shape[0]
    N_k = mask.shape[-1]
    dist_2d = distance_func_point_2d(query_points, kpts_2d, mask)  # (N_q, N_K)
    rot_query_points = query_points @ rot_mtx.permute(0, 2, 1)  # (N_r, N_q, N_k)
    rot_nn_dist = (rot_query_points.unsqueeze(2) - query_points.reshape(1, 1, N_q, N_k)).norm(dim=-1)  # (N_r, N_q, N_q)
    rot_nn_idx = rot_nn_dist.argmin(-1)  # (N_r, N_q)
    dist_2d = dist_2d[:, perms].permute(1, 0, 2)  # (N_r, N_q, N_k)
    dist_2d = torch.gather(dist_2d, 1, rot_nn_idx.unsqueeze(-1).repeat(1, 1, N_k))  # (N_r, N_q, N_k)
    return dist_2d


def distance_func_point_3d_batch(query_points, kpts_3d, trans_mtx, rot_mtx, mask=None):
    """
    Function that returns the closest distance to a set of line intersections in a 3D point cloud.
    Note that query points should be normalized.

    Args:
        query_points: (N_q, 3) tensor containing points to query
        kpts_3d: (N_3D, 3) tensor containing 3D keypoints
        trans_mtx: (N_t, 3) tensor containing translation of point cloud
        rot_mtx: (N_r, 3, 3) tensor containing rotation of point cloud
        mask: (N_3D, K) tensor containing boolean values for lines belonging to one of K classes

    Returns:
        dist_3d: (N_t, N_r, N_q) or (N_t, N_r, N_q, K) tensor containing distance to nearest edges
    """
    orig_transform_kpts = (kpts_3d.unsqueeze(0) - trans_mtx.unsqueeze(1)).unsqueeze(1)  # (N_t, 1, N_3D, 3) 
    orig_transform_kpts = torch.stack([(orig_transform_kpts * rot_mtx[:, i:i + 1, :].unsqueeze(0)).sum(-1) 
        for i in range(3)], dim=-1)  # (N_t, N_r, N_3D, 3)

    transform_kpts = orig_transform_kpts / orig_transform_kpts.norm(dim=-1, keepdim=True)
    cos_theta = (transform_kpts @ query_points.t()).permute(0, 1, 3, 2)  # (N_t, N_r, N_q, N_3D)
    sphere_dist = torch.arccos(cos_theta.clip_(min=-1., max=1.))  # Angle between edge point and line start

    if mask is None:
        dist_3d = sphere_dist.min(-1).values  # (N_t, N_r, N_q)
    else:
        MAX_LIMIT = np.pi
        new_mask = mask.reshape(1, 1, 1, mask.shape[0], mask.shape[1])
        dist_3d = sphere_dist.unsqueeze(-1) * new_mask  # (N_t, N_r, N_q, N_3D, K)
        dist_3d += torch.bitwise_not(new_mask) * MAX_LIMIT  # Make all other entries not belonging to the class invalid
        dist_3d = dist_3d.min(-2).values  # (N_t, N_r, N_q, K)

    return dist_3d


def extract_principal_2d(edge_lines, vote_sphere_pts=None):
    tot_directions = []
    counts_2d = []
    vote_edge_lines = edge_lines.clone().detach()
    max_search_iter = 20
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # Sphere for voting principal directions
    if vote_sphere_pts is None:
        vote_sphere_pts = generate_sphere_pts(5, device=device)
        vote_sphere_pts = vote_sphere_pts[:vote_sphere_pts.shape[0] // 2]

    for idx in range(max_search_iter):
        if len(vote_edge_lines) == 0:
            break
        vote_2d = torch.where(torch.abs(vote_edge_lines[:, :3] @ vote_sphere_pts.t()) < 0.05)[1].bincount(minlength=vote_sphere_pts.shape[0])
        max_idx = vote_2d.argmax()
        counts_2d.append(vote_2d.max().item())
        cand_direction = vote_sphere_pts[max_idx]
        tot_directions.append(cand_direction)
        outlier_idx = (torch.abs(vote_edge_lines[:, :3] @ vote_sphere_pts[max_idx: max_idx + 1].t()) > 0.05).squeeze()
        vote_edge_lines = vote_edge_lines[outlier_idx]

    tot_directions = torch.stack(tot_directions, dim=0)

    # Search through all combinations to find a match with principal_3d
    combs = torch.combinations(torch.arange(tot_directions.shape[0]), r=3)

    comb_directions = tot_directions[combs]
    comb_dots = torch.stack([(comb_directions[:, i % 3] * comb_directions[:, (i + 1) % 3]).sum(-1).abs()
        for i in range(3)], dim=-1)
    valid_comb_idx = (comb_dots < 0.1).sum(-1) == 3  # Assume perpendicular 3D, but can be extended

    if valid_comb_idx.sum() == 0:  # Failed to find from total directions, get 2 directions and obtain rest from cross
        if (comb_dots < 0.15).sum() != 0:  # At least one pair is perpendicular
            new_valid_comb_idx = torch.where(comb_dots < 0.15)  # Assume perpendicular 3D, but can be extended
            vec_0 = comb_directions[new_valid_comb_idx[0][0], new_valid_comb_idx[1][0]]
            vec_1 = comb_directions[new_valid_comb_idx[0][0], (new_valid_comb_idx[1][0] + 1) % 3]
            two_vec = torch.stack([vec_0, vec_1], dim=0)
            third_vec = torch.cross(two_vec[0], two_vec[1]).unsqueeze(0)
            principal_2d = torch.cat([two_vec, third_vec])
        else:  # Worst case: pick two most frequent directions
            two_vec = tot_directions[:2]
            third_vec = torch.cross(two_vec[0], two_vec[1]).unsqueeze(0)
            principal_2d = torch.cat([two_vec, third_vec])
    else:
        extracted_comb = torch.where(valid_comb_idx)[0][0]
        principal_2d = comb_directions[extracted_comb]

    if torch.det(principal_2d) < 0:
        principal_2d[-1, :] *= -1
    return principal_2d
