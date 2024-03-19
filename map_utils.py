import torch
import time
import data_utils
from dict_utils import get_init_dict_fgpl
from fgpl.line_intersection import intersections_3d
from utils import generate_trans_points
from edge_utils import generate_sphere_pts


def generate_line_map(cfg, room_list):  # Generate line maps for a list of rooms
    map_dict = {room: {} for room in room_list}
    topk_ratio_list = []  # List containing top k ratios for each room (percentage of 2D lines to keep)
    sparse_topk_ratio_list = []  # List containing sparse top k ratios for each room (percentage of 2D lines to keep for rotation estimation)
    dataset = cfg.dataset
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    num_principal = 3
    sample_rate = getattr(cfg, 'sample_rate', 1)
    max_edge_count = getattr(cfg, 'max_edge_count', 1000)  # Max edge count to prevent GPU overload
    intersect_thres_3d = getattr(cfg, 'intersect_thres_3d', 0.2)
    inlier_thres_3d = getattr(cfg, 'inlier_thres_3d', 0.1)
    
    # Sphere for voting principal directions
    vote_sphere_pts = generate_sphere_pts(5, device=device)
    vote_sphere_pts = vote_sphere_pts[:vote_sphere_pts.shape[0] // 2]
    for room in map_dict.keys():
        start = time.time()
        print(f"Preparing map for {room}...")
        if dataset == 'omniscenes':
            room_type = room.split('_')[0]
            room_no = room.split('_')[1]
            pcd_name = data_utils.get_pcd_name(dataset, room_type=room_type, room_no=room_no)
        elif dataset == 'stanford':
            area_name = room.split('/')[0]
            room_type = (room.split('/')[1]).split('_')[0]
            room_no = (room.split('/')[1]).split('_')[1]
            pcd_name = data_utils.get_pcd_name(dataset, area_name=area_name, room_type=room_type, room_no=room_no)
        xyz_np, pcd_rgb_np = data_utils.read_pcd(dataset, pcd_name=pcd_name, sample_rate=sample_rate)
        xyz = torch.from_numpy(xyz_np).float().to(device)
        rgb = torch.from_numpy(pcd_rgb_np).float().to(device)

        # Extract lines used for LDF computation
        mean_size = (xyz_np.max(0) - xyz_np.min(0)).mean()
        sparse_length_thres = mean_size * 0.10  # Follow heuristic of LDL when selecting principal 2D directions
        dirs_np, starts_np, ends_np, num_unq = data_utils.read_line(pcd_name, sparse_length_thres, max_edge_count=max_edge_count)
        dirs = torch.from_numpy(dirs_np).float().to(device)
        starts = torch.from_numpy(starts_np).float().to(device)
        ends = torch.from_numpy(ends_np).float().to(device)
        sparse_topk_ratio = starts_np.shape[0] / num_unq  # Top-k ratio to select from 2D lines for rotation estimation
        sparse_topk_ratio_list.append(sparse_topk_ratio)

        # Extract lines used for intersection extraction
        dense_length_thres = getattr(cfg, 'dense_line_thres', 0.2)  # Fixed length thres
        dense_dirs_np, dense_starts_np, dense_ends_np, num_unq = data_utils.read_line(pcd_name, dense_length_thres, max_edge_count=max_edge_count)
        dense_dirs = torch.from_numpy(dense_dirs_np).float().to(device)
        dense_starts = torch.from_numpy(dense_starts_np).float().to(device)
        dense_ends = torch.from_numpy(dense_ends_np).float().to(device)
        topk_ratio = dense_starts_np.shape[0] / num_unq  # Top-k ratio to select from 2D lines
        topk_ratio_list.append(topk_ratio)

        # Set 3D principal directions
        principal_3d = []
        counts_3d = []
        vote_dirs = dirs.clone().detach()
        for _ in range(num_principal):
            vote_3d = torch.abs(vote_dirs[:, :3] @ vote_sphere_pts.t()).argmax(-1).bincount(minlength=vote_sphere_pts.shape[0])
            max_idx = vote_3d.argmax()
            counts_3d.append(vote_3d.max().item())
            principal_3d.append(vote_sphere_pts[max_idx])
            outlier_idx = (torch.abs(vote_dirs[:, :3] @ vote_sphere_pts[max_idx: max_idx + 1].t()) < 0.95).squeeze()
            vote_dirs = vote_dirs[outlier_idx]
        principal_3d = torch.stack(principal_3d, dim=0)

        if torch.det(principal_3d) < 0:
            principal_3d[-1, :] *= -1

        # Set translation start points
        init_dict = get_init_dict_fgpl(cfg)
        trans_tensors = generate_trans_points(xyz, init_dict, device=device)

        # Compute chamfer distance to only keep translation points far away from the point cloud
        sample_xyz = xyz[torch.randperm(xyz.shape[0])[:xyz.shape[0] // 100]]
        sample_cmf = (trans_tensors.unsqueeze(1) - sample_xyz.unsqueeze(0)).norm(dim=-1).min(dim=-1).values
        trans_tensors = trans_tensors[sample_cmf > 0.3]

        # extract 3D intersections
        inter_3d, inter_3d_idx = intersections_3d(dense_dirs, dense_starts, dense_ends, principal_3d, inlier_thres=inlier_thres_3d, intersect_thres=intersect_thres_3d, return_idx=True)
        inter_3d_mask = []
        
        for k in range(len(inter_3d)):
            if inter_3d[k] is not None:
                # inter_3d[k]: intersection of kth principal direction line and (k+1)th principal direction line
                mask_temp = torch.zeros_like(inter_3d[k])
                mask_temp[:, k] = 1
                mask_temp[:, (k + 1) % 3] = 1
                inter_3d_mask.append(mask_temp.bool())
            else:
                inter_3d[k] = torch.zeros((0, 3), device=device)
                inter_3d_idx[k] = torch.zeros((0, 2), device=device, dtype=int)
        inter_3d_mask = torch.cat(inter_3d_mask, dim=0)
        inter_3d = torch.cat(inter_3d, dim=0)
        inter_3d_idx = torch.cat(inter_3d_idx, dim=0)

        map_dict[room]['dirs'] = dirs
        map_dict[room]['starts'] = starts
        map_dict[room]['ends'] = ends
        map_dict[room]['dense_dirs'] = dense_dirs
        map_dict[room]['dense_starts'] = dense_starts
        map_dict[room]['dense_ends'] = dense_ends
        map_dict[room]['xyz'] = xyz
        map_dict[room]['rgb'] = rgb
        map_dict[room]['principal_3d'] = principal_3d
        map_dict[room]['trans_tensors'] = trans_tensors
        map_dict[room]['inter_3d'] = inter_3d
        map_dict[room]['inter_3d_mask'] = inter_3d_mask
        map_dict[room]['inter_3d_idx'] = inter_3d_idx

        elapsed = time.time() - start
        print(f"Finished in {elapsed:.5f}s \n")

    topk_ratio = sorted(topk_ratio_list)[len(topk_ratio_list) // 2]  # Pick the median topk_ratio
    sparse_topk_ratio = sorted(sparse_topk_ratio_list)[len(sparse_topk_ratio_list) // 2]  # Pick the median topk_ratio

    return map_dict, topk_ratio, sparse_topk_ratio


def generate_line_map_single(cfg, line_pcd_path, room_list):  # Generate line maps for a list of rooms
    map_dict = {room: {} for room in room_list}
    topk_ratio_list = []  # List containing top k ratios for each room (percentage of 2D lines to keep)
    sparse_topk_ratio_list = []  # List containing sparse top k ratios for each room (percentage of 2D lines to keep for rotation estimation)
    dataset = cfg.dataset
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    num_principal = 3
    sample_rate = getattr(cfg, 'sample_rate', 1)
    max_edge_count = getattr(cfg, 'max_edge_count', 1000)  # Max edge count to prevent GPU overload
    intersect_thres_3d = getattr(cfg, 'intersect_thres_3d', 0.2)
    inlier_thres_3d = getattr(cfg, 'inlier_thres_3d', 0.1)
    
    # Sphere for voting principal directions
    vote_sphere_pts = generate_sphere_pts(5, device=device)
    vote_sphere_pts = vote_sphere_pts[:vote_sphere_pts.shape[0] // 2]

    room = room_list[0]
    start = time.time()
    print(f"Preparing map for {room}...")

    # Extract lines used for LDF computation
    _, starts_np, ends_np, _ = data_utils.read_line(line_pcd_path)  # Initial read for determining map size
    xyz_np = (starts_np + ends_np) / 2.
    xyz = torch.from_numpy(xyz_np).float().to(device)
    mean_size = (xyz_np.max(0) - xyz_np.min(0)).mean()
    sparse_length_thres = mean_size * 0.10  # Follow heuristic of LDL when selecting principal 2D directions
    dirs_np, starts_np, ends_np, num_unq = data_utils.read_line(line_pcd_path, sparse_length_thres, max_edge_count=max_edge_count)
    dirs = torch.from_numpy(dirs_np).float().to(device)
    starts = torch.from_numpy(starts_np).float().to(device)
    ends = torch.from_numpy(ends_np).float().to(device)
    sparse_topk_ratio = starts_np.shape[0] / num_unq  # Top-k ratio to select from 2D lines for rotation estimation
    sparse_topk_ratio_list.append(sparse_topk_ratio)

    # Extract lines used for intersection extraction
    dense_length_thres = getattr(cfg, 'dense_line_thres', 0.2)  # Fixed length thres
    dense_dirs_np, dense_starts_np, dense_ends_np, num_unq = data_utils.read_line(line_pcd_path, dense_length_thres, max_edge_count=max_edge_count)
    dense_dirs = torch.from_numpy(dense_dirs_np).float().to(device)
    dense_starts = torch.from_numpy(dense_starts_np).float().to(device)
    dense_ends = torch.from_numpy(dense_ends_np).float().to(device)
    topk_ratio = dense_starts_np.shape[0] / num_unq  # Top-k ratio to select from 2D lines
    topk_ratio_list.append(topk_ratio)

    # Set 3D principal directions
    principal_3d = []
    counts_3d = []
    vote_dirs = dirs.clone().detach()
    for _ in range(num_principal):
        vote_3d = torch.abs(vote_dirs[:, :3] @ vote_sphere_pts.t()).argmax(-1).bincount(minlength=vote_sphere_pts.shape[0])
        max_idx = vote_3d.argmax()
        counts_3d.append(vote_3d.max().item())
        principal_3d.append(vote_sphere_pts[max_idx])
        outlier_idx = (torch.abs(vote_dirs[:, :3] @ vote_sphere_pts[max_idx: max_idx + 1].t()) < 0.95).squeeze()
        vote_dirs = vote_dirs[outlier_idx]
    principal_3d = torch.stack(principal_3d, dim=0)

    if torch.det(principal_3d) < 0:
        principal_3d[-1, :] *= -1

    # Set translation start points
    init_dict = get_init_dict_fgpl(cfg)
    trans_tensors = generate_trans_points(xyz, init_dict, device=device)

    # Compute chamfer distance to only keep translation points far away from the point cloud
    sample_xyz = xyz[torch.randperm(xyz.shape[0])[:xyz.shape[0] // 100]]
    sample_cmf = (trans_tensors.unsqueeze(1) - sample_xyz.unsqueeze(0)).norm(dim=-1).min(dim=-1).values
    trans_tensors = trans_tensors[sample_cmf > 0.3]

    # extract 3D intersections
    inter_3d, inter_3d_idx = intersections_3d(dense_dirs, dense_starts, dense_ends, principal_3d, inlier_thres=inlier_thres_3d, intersect_thres=intersect_thres_3d, return_idx=True)
    inter_3d_mask = []
    
    for k in range(len(inter_3d)):
        if inter_3d[k] is not None:
            # inter_3d[k]: intersection of kth principal direction line and (k+1)th principal direction line
            mask_temp = torch.zeros_like(inter_3d[k])
            mask_temp[:, k] = 1
            mask_temp[:, (k + 1) % 3] = 1
            inter_3d_mask.append(mask_temp.bool())
        else:
            inter_3d[k] = torch.zeros((0, 3), device=device)
            inter_3d_idx[k] = torch.zeros((0, 2), device=device, dtype=int)
    inter_3d_mask = torch.cat(inter_3d_mask, dim=0)
    inter_3d = torch.cat(inter_3d, dim=0)
    inter_3d_idx = torch.cat(inter_3d_idx, dim=0)

    map_dict[room]['dirs'] = dirs
    map_dict[room]['starts'] = starts
    map_dict[room]['ends'] = ends
    map_dict[room]['dense_dirs'] = dense_dirs
    map_dict[room]['dense_starts'] = dense_starts
    map_dict[room]['dense_ends'] = dense_ends
    map_dict[room]['xyz'] = xyz
    map_dict[room]['principal_3d'] = principal_3d
    map_dict[room]['trans_tensors'] = trans_tensors
    map_dict[room]['inter_3d'] = inter_3d
    map_dict[room]['inter_3d_mask'] = inter_3d_mask
    map_dict[room]['inter_3d_idx'] = inter_3d_idx

    elapsed = time.time() - start
    print(f"Finished in {elapsed:.5f}s \n")

    topk_ratio = sorted(topk_ratio_list)[len(topk_ratio_list) // 2]  # Pick the median topk_ratio
    sparse_topk_ratio = sorted(sparse_topk_ratio_list)[len(sparse_topk_ratio_list) // 2]  # Pick the median topk_ratio

    return map_dict, topk_ratio, sparse_topk_ratio
