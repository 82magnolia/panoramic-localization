import torch
import cv2
import numpy as np
import os
import time
from parse_utils import parse_ini
from edge_utils import split_2d, split_3d


# Helper function for intersection indexing
def intersections_idx(idx0, idx1):
    """
    Return the intersection index given two principal direction indices.
    
    Note:
        For three principal directions, intersection indices are given as follows.
            
            Principal Directions        Index
            (0, 1)                      0
            (1, 2)                      1
            (2, 0)                      2
        
        This can be similarly extended to more than three directions
    """
    res_idx0 = idx0 % 3
    res_idx1 = idx1 % 3

    if (res_idx0, res_idx1) in [(0, 1), (1, 0)]:
        return 0
    
    elif (res_idx0, res_idx1) in [(1, 2), (2, 1)]:
        return 1
    
    else:
        return 2


def intersections_2d(edge_2d, principal_2d, inlier_thres=0.3, intersect_thres=0.1, return_idx=False):
    """
    Find the intersections of 2D lines considering their principal directions

    Args:
        edge_2d: (N_2D, 9) tensor containing [normals start end]
        principal_2d: (3, 3) tensor containing three principal directions in 2D
        inlier_thres: Dot product threshold for extracting inliers among principal directions
        intersect_thres: Threshold value used to determine if two 2D lines intersect
        return_idx: If True, returns a line index tensor containing indices of line segments producing each intersection
    
    Returns:
        total_intersections_2d: List containing three (N_intersections, 3) tensors. Each tensor's shape may be different from one another
                                total_intersections_2d[i] contains intersections between {lines along principal direction i} and {lines along principal direction (i+1)%3}  
                                total_intersections_2d[i] is None if there are no intersections between two groups of lines
        total_intersections_ids_2d: List containing long tensors of shape (N_intersection, 2) containing line segment indices for each intersection.
    """
    pi = torch.acos(torch.zeros(1)).item() * 2

    # Classify each 2D edge using principal directions
    edge_2d_mask = split_2d(edge_2d, principal_2d, inlier_thres)
    edge_2d_p = [edge_2d[edge_2d_mask[:, i]] for i in range(edge_2d_mask.shape[-1])]
    edge_2d_num = [edge_2d_p[i].shape[0] for i in range(len(edge_2d_p))]    # number of 2D lines along each principal direction. [N_0, N_1, N_2]
    
    # Compute indices of each line for later use in indexing
    full_range = torch.arange(edge_2d.shape[0], device=edge_2d.device)
    ids_p0 = [full_range[edge_2d_mask[:, i], None] for i in range(edge_2d_mask.shape[-1])]
    ids_p1 = [full_range[edge_2d_mask[:, i], None] for i in range(edge_2d_mask.shape[-1])]

    # Calculate arc length
    arc_length = [0]*3
    for i in range(len(edge_2d_p)):
        arc_length[i] = torch.acos((edge_2d_p[i][:, 3:6] * edge_2d_p[i][:, 6:]).sum(dim=1))   # (N_i, )

    # Find intersection candidates (by having cross product of two lines' norms)
    intersect_cand = [0] * 3
    for i in range(len(edge_2d_p)):
        j = (i + 1) % 3
        ni = edge_2d_num[i]
        nj = edge_2d_num[j]
        edge_rep0 = edge_2d_p[i][:,:3].repeat_interleave(nj, dim=0)
        edge_rep1 = edge_2d_p[j][:,:3].repeat(ni, 1)
        intersect_cand[i] = torch.cross(edge_rep0, edge_rep1, dim=-1).reshape(ni, nj, 3)    # (N_i, N_(i+1), 3)

        # Set indices accordingly
        ids_p0[i] = ids_p0[i][:,:3].repeat_interleave(nj, dim=0).reshape(ni, nj)
        ids_p1[j] = ids_p1[j][:,:3].repeat(ni, 1).reshape(ni, nj)

        # Normalize each intersection candidate
        intersect_cand_norm = torch.norm(intersect_cand[i], dim=-1)
        intersect_cand[i] = intersect_cand[i] / (intersect_cand_norm.repeat_interleave(3, dim=1).reshape(ni, nj, 3))

    # Check if intersection candidate is on both arcs
        # Assume d, s, e are all 3-vectors. d is an intersection candidate, s is start point of the arc, and e is end point of the arc
        # Three points are all on the same great circle
        # d is on the arc if len(ds)+len(de)=len(se). len() indicates (absolute value of) spherical distance
    total_intersections_2d = []
    total_intersections_ids_2d = []
    for i in range(len(intersect_cand)):
        j = (i + 1) % 3
        ni = edge_2d_num[i]
        nj = edge_2d_num[j]

        if ni > 0 and nj > 0:
            # Calculate spherical distance between intersection candidate and start point
            start0 = edge_2d_p[i][:, 3:6].repeat_interleave(nj, dim=0).reshape(ni, nj, 3)
            ds0_length = torch.acos((intersect_cand[i] * start0).sum(dim=-1))     # (N_i, N_(i+1))
            start1 = edge_2d_p[j][:, 3:6].repeat(ni, 1).reshape(ni, nj, 3)
            ds1_length = torch.acos((intersect_cand[i] * start1).sum(dim=-1))     # (N_i, N_(i+1))

            # Calculate spherical distance between intersection candidate and end point
            end0 = edge_2d_p[i][:, 6:].repeat_interleave(nj, dim=0).reshape(ni, nj, 3)
            de0_length = torch.acos((intersect_cand[i] * end0).sum(dim=-1))     # (N_i, N_(i+1))
            end1 = edge_2d_p[j][:, 6:].repeat(ni, 1).reshape(ni, nj, 3)
            de1_length = torch.acos((intersect_cand[i] * end1).sum(dim=-1))     # (N_i, N_(i+1))

            # 1) Check if intersection candidate itself (denoted as up) is on both arcs
            check_up_0 = ds0_length + de0_length - arc_length[i].repeat_interleave(nj).reshape(ni, nj) # (N_i, N_(i+1))
            check_up_1 = ds1_length + de1_length - arc_length[j]    # (N_i, N_(i+1))
            bool_up_0 = check_up_0 < intersect_thres                # (N_i, N_(i+1))
            bool_up_1 = check_up_1 < intersect_thres                # (N_i, N_(i+1))
            up_on_arc = bool_up_0 * bool_up_1
            up_intersects = intersect_cand[i][up_on_arc, :]         # (number of valid intersections among 'up intersection candidates', 3)
            up_ids_0 = ids_p0[i][up_on_arc]
            up_ids_1 = ids_p1[j][up_on_arc]
            up_ids = torch.stack([up_ids_0, up_ids_1], dim=-1)

            # 2) Check if the opposite of intersection candidate (denoted as down) is on both arcs
            check_down_0 = (pi-ds0_length) + (pi-de0_length) - arc_length[i].repeat_interleave(nj).reshape(ni, nj) # (N_i, N_(i+1))
            check_down_1 = (pi-ds1_length) + (pi-de1_length) - arc_length[j]    # (N_i, N_(i+1))
            bool_down_0 = check_down_0 < intersect_thres                        # (N_i, N_(i+1))
            bool_down_1 = check_down_1 < intersect_thres                        # (N_i, N_(i+1))
            down_on_arc = bool_down_0 * bool_down_1
            down_intersects = -intersect_cand[i][down_on_arc, :]                # (number of valid intersections among 'down intersection candidates', 3)
            down_ids_0 = ids_p0[i][down_on_arc]
            down_ids_1 = ids_p1[j][down_on_arc]
            down_ids = torch.stack([down_ids_0, down_ids_1], dim=-1)

            intersects = torch.cat((up_intersects, down_intersects), 0)
            intersects_ids = torch.cat((up_ids, down_ids), 0)
            if intersects.shape[0] > 0:
                total_intersections_2d.append(intersects)
                total_intersections_ids_2d.append(intersects_ids)
            else:
                total_intersections_2d.append(torch.zeros((0, 3), device=edge_2d.device))
                total_intersections_ids_2d.append(torch.zeros((0, 2), device=edge_2d.device, dtype=int))
        else:
            total_intersections_2d.append(torch.zeros((0, 3), device=edge_2d.device))
            total_intersections_ids_2d.append(torch.zeros((0, 2), device=edge_2d.device, dtype=int))

    if return_idx:
        return total_intersections_2d, total_intersections_ids_2d
    else:
        return total_intersections_2d


def intersections_3d(dirs, starts, ends, principal_3d, inlier_thres=0.05, intersect_thres=0.15, return_idx=False):
    """
    Find the intersections of 3D lines considering their principal directions

    Args:
        dirs: (N_3D, 3) tensor containing 3D line directions
        starts: (N_3D, 3) tensor containing start points in each 3D line
        ends: (N_3D, 3) tensor containing end points in each 3D line
        principal_3d: (3, 3) tensor containing three principal directions in 3D
        inlier_thres: Dot product threshold for extracting inliers among principal directions
        intersect_thres: Threshold for determining intersection of two lines
        return_idx: If True, returns a line index tensor containing indices of line segments producing each intersection
    
    Returns:
        total_intersections_3d: List containing three (N_intersections, 3) tensors. Each tensor's shape may be different from one another
                                total_intersections_3d[i] contains intersections between {lines along principal direction i} and {lines along principal direction (i+1)%3}  
                                total_intersections_3d[i] is None if there are no intersections between two groups of lines
        total_intersections_ids_3d: List containing long tensors of shape (N_intersection, 2) containing line segment indices for each intersection.
    """

    # Classify each 3D edge using principal directions
    edge_3d_mask = split_3d(dirs, principal_3d, inlier_thres)
    starts_p = [starts[edge_3d_mask[:, i]] for i in range(edge_3d_mask.shape[-1])]
    ends_p = [ends[edge_3d_mask[:, i]] for i in range(edge_3d_mask.shape[-1])]

    total_intersections_3d = []
    total_intersections_ids_3d = []

    # Compute indices of each line for later use in indexing
    full_range = torch.arange(dirs.shape[0], device=dirs.device)
    ids_p = [full_range[edge_3d_mask[:, i]] for i in range(edge_3d_mask.shape[-1])]

    # Find intersections using two lines that have different principal directions
    for i in range(len(starts_p)):
        starts_0 = starts_p[i]
        ends_0 = ends_p[i]
        ids_0 = ids_p[i]

        starts_1 = starts_p[(i + 1) % 3]
        ends_1 = ends_p[(i + 1) % 3]
        ids_1 = ids_p[(i + 1) % 3]
        num_0 = starts_0.shape[0]
        num_1 = starts_1.shape[0]

        if num_0 > 0 and num_1 > 0:
            index_0 = torch.arange(num_0)
            index_1 = torch.arange(num_1)
            grid_x, grid_y = torch.meshgrid(index_0, index_1)
            final_idx = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)

            starts_0_expand = starts_0[final_idx[:, 0]]  # (num_0 X num_1, 3)
            ends_0_expand = ends_0[final_idx[:, 0]]
            starts_1_expand = starts_1[final_idx[:, 1]]
            ends_1_expand = ends_1[final_idx[:, 1]]
            ids_0_expand = ids_0[final_idx[:, 0]]
            ids_1_expand = ids_1[final_idx[:, 1]]

            dirs_0 = ends_0_expand - starts_0_expand    # (num_0 X num_1, 3)
            dirs_1 = ends_1_expand - starts_1_expand
            dirs_cross = torch.cross(dirs_0, dirs_1, dim=-1)
            dirs_cross_norm = torch.norm(dirs_cross, dim=-1)    # (num_0 X num_1,)

            starts_diff = starts_1_expand - starts_0_expand
            sd1 = torch.cross(starts_diff, dirs_1, dim=-1)
            sd0 = torch.cross(starts_diff, dirs_0, dim=-1)
            u = (sd1 * dirs_cross).sum(dim=-1) / torch.square(dirs_cross_norm)    # (num_0 X num_1, )
            v = (sd0 * dirs_cross).sum(dim=-1) / torch.square(dirs_cross_norm)
            
            dirs_0_length = torch.norm(dirs_0, dim=1)
            dirs_1_length = torch.norm(dirs_1, dim=1)

            # Check if intersection candidates are on each line
            on_line0 = (u > -intersect_thres / dirs_0_length) * (u < 1 + intersect_thres / dirs_0_length)   # (num_0 X num_1, )
            on_line1 = (v > -intersect_thres / dirs_1_length) * (v < 1 + intersect_thres / dirs_1_length)
            on_line = on_line0 * on_line1
            
            point0 = starts_0_expand[on_line, :] + u[on_line].reshape(-1, 1) * dirs_0[on_line, :]
            point1 = starts_1_expand[on_line, :] + v[on_line].reshape(-1, 1) * dirs_1[on_line, :]
            valid_intersect = torch.norm(point0 - point1, dim=-1) < intersect_thres
            intersects = point0[valid_intersect]
            intersects_ids_0 = ids_0_expand[on_line][valid_intersect]
            intersects_ids_1 = ids_1_expand[on_line][valid_intersect]
            intersect_ids = torch.stack([intersects_ids_0, intersects_ids_1], dim=-1)
            
            if intersects.shape[0] == 0:
                intersects = None
            total_intersections_3d.append(intersects)
            total_intersections_ids_3d.append(intersect_ids)

    if return_idx:
        return total_intersections_3d, total_intersections_ids_3d
    else:
        return total_intersections_3d
