import torch
from edge_utils import (
    split_func_3d_batch,
    generate_sphere_pts,
    split_by_axes,
    split_func_2d_batch,
    distance_func_point_2d_batch,
    distance_func_point_3d_batch,
    distance_func_point_2d_single_compute
)
from itertools import permutations


class XDFCanonicalPrecompute:  # Precompute LDFs, PDFs, etc. (note all outputs are cached to CPU to save GPU memory)
    def __init__(self, cfg, log_dir, map_dict):
        self.cfg = cfg
        self.log_dir = log_dir
        self.map_dict = map_dict
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.query_points_level = getattr(self.cfg, 'query_points_level', 3)
        self.query_points = generate_sphere_pts(level=self.query_points_level, type='torch', device=self.device)
        self.room_list = [key for key in map_dict.keys()]
        
        # Prepare permutations for canonical rotation estimation
        num_principal = 3
        perms = list(permutations(range(num_principal)))
        perms = torch.tensor(perms, device=self.device, dtype=torch.long)
        self.bin_mask = torch.ones([len(perms) * 4, 3, 1], device=self.device)  # Ambiguity in permutation and sign

        for perm_idx, perm in enumerate(perms):
            for idx in range(4):
                self.bin_mask[perm_idx * 4 + idx, 0, 0] = (-1) ** (idx // 2)
                self.bin_mask[perm_idx * 4 + idx, 1, 0] = (-1) ** (idx % 2)
                self.bin_mask[perm_idx * 4 + idx, 2, 0] = (-1) ** (idx // 2 + idx % 2)
                if perm_idx in [1, 2, 5]:
                    self.bin_mask[perm_idx * 4 + idx, 2, 0] *= -1

        self.perms = torch.repeat_interleave(perms, repeats=torch.tensor([4] * len(perms), dtype=torch.long, device=self.device), dim=0)
        self.canonical_rot_3d = {room: None for room in self.room_list}

    def generate_ldf_3d(self):
        precomputed_dist_3d = {room: None for room in self.room_list}
        precomputed_mask_3d = {room: None for room in self.room_list}

        query_points = self.query_points

        for room_idx, room in enumerate(self.room_list):
            # Prepare inputs
            starts = self.map_dict[room]['starts'].clone().detach()
            ends = self.map_dict[room]['ends'].clone().detach()
            dirs = self.map_dict[room]['dirs'].clone().detach()
            principal_3d = self.map_dict[room]['principal_3d'].clone().detach()
            trans_tensors = self.map_dict[room]['trans_tensors'].clone().detach()

            # Prevent GPU memory overflow
            if trans_tensors.shape[0] > 200:
                step = 200
                num_loop = trans_tensors.shape[0] // step + 1 if trans_tensors.shape[0] % step != 0 else trans_tensors.shape[0] // step
            else:
                step = trans_tensors.shape[0]
                num_loop = 1

            # Transform according to canonical rotation
            self.canonical_rot_3d[room] = principal_3d
            starts = starts @ self.canonical_rot_3d[room].T
            ends = ends @ self.canonical_rot_3d[room].T
            dirs = dirs @ self.canonical_rot_3d[room].T
            principal_3d = principal_3d @ self.canonical_rot_3d[room].T
            trans_tensors = trans_tensors @ self.canonical_rot_3d[room].T

            inner_prod_3d = torch.abs(dirs @ principal_3d.t())
            inlier_thres = 0.05
            edge_3d_mask = torch.stack([inner_prod_3d[:, i] > 1 - inlier_thres for i in range(3)], dim=-1)
            precomputed_mask_3d[room] = edge_3d_mask.cpu()  # (N_3D, 3)

            precomputed_dist_3d[room] = []

            for i in range(num_loop):
                precomputed_dist_3d[room].append(split_func_3d_batch(
                    query_points,
                    starts,
                    ends,
                    trans_tensors[step * i: step * (i + 1)],
                    torch.eye(3, device=self.device).reshape(-1, 3, 3),
                    edge_3d_mask
                ).squeeze(1).cpu())  # (N_t, N_q, K)
            precomputed_dist_3d[room] = torch.cat(precomputed_dist_3d[room], dim=0)

        return precomputed_dist_3d, precomputed_mask_3d

    def generate_ldf_2d(self, principal_2d, edge_lines, perms, bin_mask, single_pose_compute=False):
        query_points = self.query_points
        N_perms = perms.shape[0]
        canonical_principal_3d = torch.eye(3, device=self.device).reshape(-1, 3, 3)  # (1, 3, 3)
        pts_2d = principal_2d[perms]  # (N_perms, 3, 3)
        H = (canonical_principal_3d.permute(0, 2, 1) @ (bin_mask * pts_2d))  # (N_perms, 3, 3)
        U, S, V = torch.svd(H)  # (N_perms, 3, 3)
        U_t = torch.transpose(U, 1, 2)
        d = torch.sign(torch.det(V @ U_t))  # (N_perms, )
        diag_mtx = torch.eye(3, device=self.device)[None, ...].repeat(N_perms, 1, 1)  # (N_perms, 3, 3)
        diag_mtx[:, 2, 2] = d
        canonical_estim_rot = V @ diag_mtx @ U_t  # (N_perms, 3, 3)

        # Compute 2D mask
        inlier_thres = 0.05
        inner_prod_2d = torch.abs(edge_lines[None, :, None, :3] @ canonical_estim_rot[:, None] @ canonical_principal_3d.reshape(1, 1, 3, 3).transpose(2, 3)).squeeze(-2)  # (N_perms, N_2D, 3)
        prod_idx_2d = inner_prod_2d.argmin(-1)  # (N_perms, N_2D)
        prod_val_2d = inner_prod_2d.min(-1).values  # (N_perms, N_2D)
        canonical_batch_mask_2d = torch.stack([(prod_idx_2d == i) & (prod_val_2d < inlier_thres) for i in range(3)], dim=-1)  # (N_perms, N_2D, 3)

        # Compute 2D distance function
        if single_pose_compute:  # Compute 2D LDF for single view and sample values at other rotations
            dist_2d = split_func_2d_batch(query_points, edge_lines, canonical_batch_mask_2d[0], rot_mtx=canonical_estim_rot, perms=perms, single_pose_compute=True)  # (N_perms, N_q, N_K)
        else:  # Directly compute 2D LDFs for all views
            dist_2d = split_func_2d_batch(query_points, edge_lines, canonical_batch_mask_2d, rot_mtx=canonical_estim_rot)  # (N_perms, N_q, N_K)
        
        canonical_estim_rot = canonical_estim_rot.cpu()
        dist_2d = dist_2d.cpu()
        canonical_batch_mask_2d = canonical_batch_mask_2d.cpu()
        
        precomputed_rot = {room: canonical_estim_rot for room in self.room_list}
        precomputed_dist_2d = {room: dist_2d for room in self.room_list}
        precomputed_mask_2d = {room: canonical_batch_mask_2d for room in self.room_list}

        return precomputed_rot, precomputed_mask_2d, precomputed_dist_2d
    
    def generate_pdf_3d(self):
        precomputed_point_dist_3d = {room: None for room in self.room_list}
        assert all([self.canonical_rot_3d[room] is not None for room in self.room_list])

        query_points = self.query_points

        for room in self.room_list:
            trans_tensors = self.map_dict[room]['trans_tensors'].clone().detach()
            trans_tensors = trans_tensors @ self.canonical_rot_3d[room].T

            # Prevent GPU memory overflow
            if trans_tensors.shape[0] > 200:
                step = 200
                num_loop = trans_tensors.shape[0] // step + 1 if trans_tensors.shape[0] % step != 0 else trans_tensors.shape[0] // step
            else:
                step = trans_tensors.shape[0]
                num_loop = 1

            inter_3d = self.map_dict[room]['inter_3d'].clone().detach()
            inter_3d_mask = self.map_dict[room]['inter_3d_mask']

            inter_3d = inter_3d @ self.canonical_rot_3d[room].T

            precomputed_point_dist_3d[room] = []

            for i in range(num_loop):
                if getattr(self.cfg, 'split_pdf', True):  # TODO: Remove this part after ablation study
                    precomputed_point_dist_3d[room].append(distance_func_point_3d_batch(
                        query_points,
                        inter_3d,
                        trans_tensors[step * i: step * (i + 1)],
                        torch.eye(3, device=self.device).reshape(-1, 3, 3),
                        inter_3d_mask
                    ).squeeze(1).cpu())  # (N_t, N_q, K)
                else:
                    precomputed_point_dist_3d[room].append(distance_func_point_3d_batch(
                        query_points,
                        inter_3d,
                        trans_tensors[step * i: step * (i + 1)],
                        torch.eye(3, device=self.device).reshape(-1, 3, 3),
                        None
                    ).squeeze(1).unsqueeze(-1).cpu())  # (N_t, N_q, K)
            precomputed_point_dist_3d[room] = torch.cat(precomputed_point_dist_3d[room], dim=0)

        return precomputed_point_dist_3d

    def generate_pdf_2d(self, inter_2d, inter_2d_mask, precomputed_rot, single_pose_compute=False):
        query_points = self.query_points

        # Use arbitrary rotation as this is constant for all rooms
        rot_mtx = precomputed_rot[self.room_list[0]].to(query_points.device)
        if single_pose_compute:  # Compute 2D PDF for single view and sample values at other rotations
            dist_2d = distance_func_point_2d_single_compute(
                query_points,
                inter_2d[0],
                mask=inter_2d_mask[0],
                rot_mtx=rot_mtx,
                perms=self.perms,
            ).cpu()
        else:  # Directly compute 2D PDFs for all views
            if getattr(self.cfg, 'split_pdf', True):
                dist_2d = distance_func_point_2d_batch(
                    query_points,
                    inter_2d,
                    mask=inter_2d_mask,
                    rot_mtx=rot_mtx
                ).cpu()
            else:
                dist_2d = distance_func_point_2d_batch(
                    query_points,
                    inter_2d,
                    mask=None,
                    rot_mtx=rot_mtx
                ).unsqueeze(-1).cpu()
        precomputed_point_dist_2d = {room: dist_2d for room in self.room_list}

        return precomputed_point_dist_2d

    def infer_room(self, precomputed_dist_2d, precomputed_dist_3d, num_room=1, method='inlier', sample_per_room=10):
        total_cost = []
        for room_idx, room in enumerate(self.room_list):
            dist_2d = precomputed_dist_2d[room].to(self.device)  # (N_r, N_q, K)
            dist_3d = precomputed_dist_3d[room].to(self.device)  # (N_t, N_q, K)
            cost = torch.abs(dist_2d.unsqueeze(0) - dist_3d.unsqueeze(1))  # (N_t, N_r, N_q, K)

            if method == 'mean':
                cost = cost.mean(-2)  # (N_t, N_r, K)
                cost = cost.mean(-1)  # (N_t, N_r)
            elif method == 'median':
                cost = cost.median(-2).values  # (N_t, N_r, K)
                cost = cost.mean(-1)  # (N_t, N_r)
            elif method == 'inlier':
                cost = (cost < 0.1).sum(-2).sum(-1)  # (N_t, N_r)
                cost = -cost

            cost = cost.reshape(-1)
            cost = cost[cost.argsort()[:sample_per_room]].float().mean()
            total_cost.append(cost)
        total_cost = torch.stack(total_cost)
        room_idx_list = torch.argsort(total_cost)[:num_room].tolist()

        return room_idx_list
