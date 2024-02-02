import torch
import numpy as np
from torch import sin, cos
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import cloud2idx, sample_from_img, quantile, make_pano
from tqdm import tqdm
from PIL import Image
import torch.nn as nn


def refine_pose_sampling_loss(img, xyz, rgb, input_trans, input_rot, starting_point, cfg):
    # xyz, rgb are (N, 3) input point clouds
    # img is (H, W, 3) image tensor
    
    translation = input_trans[starting_point].unsqueeze(0).t().requires_grad_()
    yaw, pitch, roll = input_rot[starting_point]
    yaw = yaw.unsqueeze(0).requires_grad_()
    roll = roll.unsqueeze(0).requires_grad_()
    pitch = pitch.unsqueeze(0).requires_grad_()

    tensor_0 = torch.zeros(1, device=xyz.device)
    tensor_1 = torch.ones(1, device=xyz.device)

    # Get configs
    lr = getattr(cfg, 'lr', 0.1)
    num_iter = getattr(cfg, 'num_iter', 100)
    patience = getattr(cfg, 'patience', 5)
    factor = getattr(cfg, 'factor', 0.9)
    vis = getattr(cfg, 'visualize', False)
    out_quantile = getattr(cfg, 'out_of_room_quantile', 0.05)
    in_xyz = xyz.detach().clone()
    
    final_optimizer = torch.optim.Adam([translation, yaw, roll, pitch], lr=lr)

    loss = 0.0

    final_scheduler = ReduceLROnPlateau(final_optimizer, mode='min', patience=patience, factor=factor)
    
    frames = []
    in_rgb = rgb

    loss_func = SamplingLoss(in_xyz, in_rgb, img, xyz.device, cfg)

    for iter in tqdm(range(num_iter), desc="Starting point {}".format(starting_point)):
        final_optimizer.zero_grad()
        loss = loss_func(translation, yaw, pitch, roll)
        loss.backward()

        final_optimizer.step()
        final_scheduler.step(loss)

        with torch.no_grad():
            x_min, x_max = quantile(xyz[:, 0], out_quantile)
            y_min, y_max = quantile(xyz[:, 1], out_quantile)
            z_min, z_max = quantile(xyz[:, 2], out_quantile)
            translation[0] = torch.clamp(translation[0], min=x_min, max=x_max)
            translation[1] = torch.clamp(translation[1], min=y_min, max=y_max)
            translation[2] = torch.clamp(translation[2], min=z_min, max=z_max)

    RX = torch.stack([
                    torch.stack([tensor_1, tensor_0, tensor_0]),
                    torch.stack([tensor_0, cos(roll), -sin(roll)]),
                    torch.stack([tensor_0, sin(roll), cos(roll)])]).reshape(3, 3)

    RY = torch.stack([
                    torch.stack([cos(pitch), tensor_0, sin(pitch)]),
                    torch.stack([tensor_0, tensor_1, tensor_0]),
                    torch.stack([-sin(pitch), tensor_0, cos(pitch)])]).reshape(3, 3)

    RZ = torch.stack([
                    torch.stack([cos(yaw), -sin(yaw), tensor_0]),
                    torch.stack([sin(yaw), cos(yaw), tensor_0]),
                    torch.stack([tensor_0, tensor_0, tensor_1])]).reshape(3, 3)

    R = torch.mm(RZ, RY)
    R = torch.mm(R, RX)

    return [translation.cpu(), R.cpu(), loss.cpu()]


def refine_pose_sampling_loss_batch(img, xyz, rgb, input_trans, input_rot, cfg):
    # xyz, rgb are (N, 3) input point clouds
    # img is (H, W, 3) image tensor
    assert cfg.top_k_candidate > 1

    batch_size = input_trans.shape[0]
    translation = input_trans.unsqueeze(-1)  # (B, 3, 1)
    yaw = input_rot[..., 0:1]  # (B, 1)
    pitch = input_rot[..., 1:2]
    roll = input_rot[..., 2:3]

    translation_list = torch.chunk(translation, batch_size)
    yaw_list = torch.chunk(yaw, batch_size)
    pitch_list = torch.chunk(pitch, batch_size)
    roll_list = torch.chunk(roll, batch_size)

    tensor_0 = torch.zeros(1, device=xyz.device)
    tensor_1 = torch.ones(1, device=xyz.device)

    # Get configs
    lr = getattr(cfg, 'lr', 0.1)
    num_iter = getattr(cfg, 'num_iter', 100)
    patience = getattr(cfg, 'patience', 5)
    factor = getattr(cfg, 'factor', 0.9)
    out_quantile = getattr(cfg, 'out_of_room_quantile', 0.05)

    in_xyz = xyz.detach().clone()
    in_rgb = rgb
    loss_func = BatchSamplingLoss(in_xyz, in_rgb, img, xyz.device, cfg)

    optimizer_list = [torch.optim.Adam([translation_list[idx].requires_grad_(), yaw_list[idx].requires_grad_(), 
        roll_list[idx].requires_grad_(), pitch_list[idx].requires_grad_()], lr=lr) for idx in range(batch_size)]
    scheduler_list = [ReduceLROnPlateau(optimizer_list[idx], mode='min', patience=patience, factor=factor) for idx in range(batch_size)]

    translation = torch.cat(translation_list)
    yaw = torch.cat(yaw_list)
    pitch = torch.cat(pitch_list)
    roll = torch.cat(roll_list)

    with torch.no_grad():
        x_min, x_max = quantile(xyz[:, 0], out_quantile)
        y_min, y_max = quantile(xyz[:, 1], out_quantile)
        z_min, z_max = quantile(xyz[:, 2], out_quantile)

    for iteration in tqdm(range(num_iter), desc="Global Step"):
        for idx in range(batch_size):
            optimizer_list[idx].zero_grad()
        
        loss, loss_list = loss_func(translation, yaw, pitch, roll)  # scalar and tensor of shape (B, )
        loss.backward()

        for idx in range(batch_size):
            optimizer_list[idx].step()
            scheduler_list[idx].step(loss_list[idx])

        translation = torch.cat(translation_list)
        yaw = torch.cat(yaw_list)
        pitch = torch.cat(pitch_list)
        roll = torch.cat(roll_list)

        with torch.no_grad():
            for idx in range(batch_size):
                translation_list[idx][0, 0, 0] = torch.clamp(translation_list[idx][0, 0, 0], min=x_min, max=x_max)
                translation_list[idx][0, 1, 0] = torch.clamp(translation_list[idx][0, 1, 0], min=y_min, max=y_max)
                translation_list[idx][0, 2, 0] = torch.clamp(translation_list[idx][0, 2, 0], min=z_min, max=z_max)

    min_idx = loss_list.argmin().item()
    translation = translation[min_idx]  # (3, 1)
    yaw = yaw[min_idx]
    pitch = pitch[min_idx]
    roll = roll[min_idx]
    loss = loss_list[min_idx]

    RX = torch.stack([
                    torch.stack([tensor_1, tensor_0, tensor_0]),
                    torch.stack([tensor_0, cos(roll), -sin(roll)]),
                    torch.stack([tensor_0, sin(roll), cos(roll)])]).reshape(3, 3)

    RY = torch.stack([
                    torch.stack([cos(pitch), tensor_0, sin(pitch)]),
                    torch.stack([tensor_0, tensor_1, tensor_0]),
                    torch.stack([-sin(pitch), tensor_0, cos(pitch)])]).reshape(3, 3)

    RZ = torch.stack([
                    torch.stack([cos(yaw), -sin(yaw), tensor_0]),
                    torch.stack([sin(yaw), cos(yaw), tensor_0]),
                    torch.stack([tensor_0, tensor_0, tensor_1])]).reshape(3, 3)

    R = torch.mm(RZ, RY)
    R = torch.mm(R, RX)

    return [translation.cpu(), R.cpu(), loss.cpu()]


class SamplingLoss(nn.Module):
    def __init__(self, xyz: torch.tensor, rgb: torch.tensor, img: torch.tensor, device: torch.device, cfg):
        super(SamplingLoss, self).__init__()
        self.xyz = xyz
        self.rgb = rgb
        self.img = img
        self.cfg = cfg

        self.tensor_0 = torch.zeros(1, device=xyz.device)
        self.tensor_1 = torch.ones(1, device=xyz.device)

    def forward(self, translation, yaw, pitch, roll):
        RX = torch.stack([
                        torch.stack([self.tensor_1, self.tensor_0, self.tensor_0]),
                        torch.stack([self.tensor_0, cos(roll), -sin(roll)]),
                        torch.stack([self.tensor_0, sin(roll), cos(roll)])]).reshape(3, 3)

        RY = torch.stack([
                        torch.stack([cos(pitch), self.tensor_0, sin(pitch)]),
                        torch.stack([self.tensor_0, self.tensor_1, self.tensor_0]),
                        torch.stack([-sin(pitch), self.tensor_0, cos(pitch)])]).reshape(3, 3)

        RZ = torch.stack([
                        torch.stack([cos(yaw), -sin(yaw), self.tensor_0]),
                        torch.stack([sin(yaw), cos(yaw), self.tensor_0]),
                        torch.stack([self.tensor_0, self.tensor_0, self.tensor_1])]).reshape(3, 3)

        R = torch.mm(RZ, RY)
        R = torch.mm(R, RX)

        new_xyz = torch.transpose(self.xyz, 0, 1) - translation
        new_xyz = torch.transpose(torch.matmul(R, new_xyz), 0, 1)

        coord_arr = cloud2idx(new_xyz)

        refined_rgb = self.rgb

        sample_rgb = sample_from_img(self.img, coord_arr)
        mask = torch.sum(sample_rgb == 0, dim=1) != 3

        rgb_loss = torch.norm(sample_rgb[mask] - refined_rgb[mask], dim=-1).mean()

        return rgb_loss


class BatchSamplingLoss(nn.Module):
    def __init__(self, xyz: torch.tensor, rgb: torch.tensor, img: torch.tensor, device: torch.device, cfg):
        super(BatchSamplingLoss, self).__init__()
        self.xyz = xyz
        self.rgb = rgb

        self.img = img
        self.cfg = cfg
        self.num_input = cfg.top_k_candidate
        self.tensor_0 = torch.zeros(self.num_input, 1, device=xyz.device)
        self.tensor_1 = torch.ones(self.num_input, 1, device=xyz.device)

    def forward(self, translation, yaw, pitch, roll):
        # translation has shape (N, 3, 1)
        # yaw, pitch, roll has shape (N, 1)

        RX = torch.cat([
                        torch.stack([self.tensor_1, self.tensor_0, self.tensor_0], dim=-1),
                        torch.stack([self.tensor_0, cos(roll), -sin(roll)], dim=-1),
                        torch.stack([self.tensor_0, sin(roll), cos(roll)], dim=-1)], dim=1)
        RY = torch.cat([
                        torch.stack([cos(pitch), self.tensor_0, sin(pitch)], dim=-1),
                        torch.stack([self.tensor_0, self.tensor_1, self.tensor_0], dim=-1),
                        torch.stack([-sin(pitch), self.tensor_0, cos(pitch)], dim=-1)], dim=1)
        RZ = torch.cat([
                        torch.stack([cos(yaw), -sin(yaw), self.tensor_0], dim=-1),
                        torch.stack([sin(yaw), cos(yaw), self.tensor_0], dim=-1),
                        torch.stack([self.tensor_0, self.tensor_0, self.tensor_1], dim=-1)], dim=1)

        # RX, RY, RZ: (N, 3, 3)
        R = torch.bmm(RZ, RY)
        R = torch.bmm(R, RX)

        new_xyz = self.xyz - torch.transpose(translation, 1, 2)
        
        # Faster way to bmm
        tmp_xyz = torch.zeros_like(new_xyz, device=new_xyz.device)
        tmp_xyz[..., 0] = (new_xyz * R[:, 0:1, :]).sum(-1)
        tmp_xyz[..., 1] = (new_xyz * R[:, 1:2, :]).sum(-1)
        tmp_xyz[..., 2] = (new_xyz * R[:, 2:3, :]).sum(-1)

        new_xyz = tmp_xyz

        coord_arr = cloud2idx(new_xyz, batched=True)  # (B, N, 2)

        refined_rgb = self.rgb  # (B, N, 3)

        sample_rgb = sample_from_img(self.img, coord_arr, batched=True)  # (B, N, 3)
        mask = torch.sum(sample_rgb == 0, dim=-1) != 3  # (B, N)

        rgb_loss_list = torch.norm(sample_rgb - refined_rgb, dim=-1) * mask  # (B, N, )
        rgb_loss_list = rgb_loss_list.sum(-1)  # (B, )

        mask_count = mask.sum(-1)  # (B, )
        rgb_loss_list /= mask_count

        rgb_loss = rgb_loss_list.sum()
        return rgb_loss, rgb_loss_list
