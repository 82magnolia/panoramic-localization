import torch
import numpy as np
from torch import sin, cos
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import cloud2idx, refine_sampling_coords, sample_from_img, quantile, make_pano
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
import os
import cv2


def refine_pose_sampling_loss(img, xyz, rgb, input_trans, input_rot, starting_point, cfg, 
        img_weight=None, pcd_weight=None):
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
    filter_dist_thres = getattr(cfg, 'filter_dist_thres', None)  # Filter points only within distance threshold

    if filter_dist_thres is not None:
        dist = torch.norm(xyz - translation.t(), dim=-1)
        in_xyz = xyz[dist < filter_dist_thres].detach().clone()
    else:
        in_xyz = xyz.detach().clone()
    
    final_optimizer = torch.optim.Adam([translation, yaw, roll, pitch], lr=lr)

    loss = 0.0

    final_scheduler = ReduceLROnPlateau(final_optimizer, mode='min', patience=patience, factor=factor)
    
    frames = []

    in_rgb = rgb

    loss_func = SamplingLoss(in_xyz, in_rgb, img, xyz.device, cfg, img_weight, pcd_weight)

    for iteration in tqdm(range(num_iter), desc="Starting point {}".format(starting_point)):
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

        if vis:
            with torch.no_grad():
                tmp_roll = roll.clone().detach()
                tmp_pitch = pitch.clone().detach()
                tmp_yaw = yaw.clone().detach()
                tmp_trans = translation.clone().detach()
                tmp_xyz = in_xyz.clone().detach()

                RX = torch.stack([
                                torch.stack([tensor_1, tensor_0, tensor_0]),
                                torch.stack([tensor_0, cos(tmp_roll), -sin(tmp_roll)]),
                                torch.stack([tensor_0, sin(tmp_roll), cos(tmp_roll)])]).reshape(3, 3)

                RY = torch.stack([
                                torch.stack([cos(tmp_pitch), tensor_0, sin(tmp_pitch)]),
                                torch.stack([tensor_0, tensor_1, tensor_0]),
                                torch.stack([-sin(tmp_pitch), tensor_0, cos(tmp_pitch)])]).reshape(3, 3)

                RZ = torch.stack([
                                torch.stack([cos(tmp_yaw), -sin(tmp_yaw), tensor_0]),
                                torch.stack([sin(tmp_yaw), cos(tmp_yaw), tensor_0]),
                                torch.stack([tensor_0, tensor_0, tensor_1])]).reshape(3, 3)

                R = torch.mm(RZ, RY)
                R = torch.mm(R, RX)

                new_xyz = torch.transpose(tmp_xyz, 0, 1) - tmp_trans
                new_xyz = torch.transpose(torch.matmul(R, new_xyz), 0, 1)

                image_factor = getattr(cfg, 'image_factor', 2)
                cur_img = Image.fromarray(make_pano(new_xyz.clone().detach(), in_rgb.clone().detach(), resolution=(img.shape[0] // image_factor, img.shape[1] // image_factor)))
                gt_img = Image.fromarray(np.uint8(img.detach().cpu().numpy() * 255), 'RGB').resize((cur_img.width, cur_img.height))
                    
                vis_list = getattr(cfg, 'visualize_list', None)
                if vis_list is None:
                    new_frame = Image.new('RGB', (cur_img.width, 2 * cur_img.height))
                    new_frame.paste(gt_img, (0, 0))
                    new_frame.paste(cur_img, (0, cur_img.height))

                else:
                    new_frame = Image.new('RGB', (cur_img.width, len(vis_list) * cur_img.height))
                    curr_idx = 0
                    
                    if 'gt_img' in vis_list:
                        new_frame.paste(gt_img, (0, curr_idx))
                        curr_idx += cur_img.height
                    
                    if 'cur_img' in vis_list:
                        new_frame.paste(cur_img, (0, curr_idx))
                        curr_idx += cur_img.height
                    
                if iteration == 0:
                    for i in range(4):
                        frames.append(new_frame)
                frames.append(new_frame)

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


def sampling_loss(img, xyz, rgb, input_trans, input_rot, starting_point, cfg, return_list=True):
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
    filter_dist_thres = getattr(cfg, 'filter_dist_thres', None)  # Filter points only within distance threshold

    if filter_dist_thres is not None:
        dist = torch.norm(xyz - translation.t(), dim=-1)
        in_xyz = xyz[dist < filter_dist_thres].detach().clone()
    else:
        in_xyz = xyz.detach().clone()

    in_rgb = rgb

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

    new_xyz = torch.transpose(in_xyz, 0, 1) - translation
    new_xyz = torch.transpose(torch.matmul(R, new_xyz), 0, 1)

    coord_arr = cloud2idx(new_xyz)

    filter_factor = getattr(cfg, 'filter_factor', 1)
    filtered_idx = refine_sampling_coords(coord_arr, torch.norm(new_xyz, dim=-1), rgb, quantization=(img.shape[0] // filter_factor, img.shape[1] // filter_factor))

    coord_arr = coord_arr[filtered_idx]
    refined_rgb = in_rgb[filtered_idx]

    sample_rgb = sample_from_img(img, coord_arr)
    mask = torch.sum(sample_rgb == 0, dim=1) != 3

    rgb_loss = torch.norm(sample_rgb[mask] - refined_rgb[mask], dim=-1).mean()

    loss = rgb_loss

    if return_list:
        return [translation.cpu(), R.cpu(), loss.cpu()]
    else:
        return loss.cpu()


class SamplingLoss(nn.Module):
    def __init__(self, xyz: torch.tensor, rgb: torch.tensor, img: torch.tensor, device: torch.device, cfg,
            img_weight: torch.tensor, pcd_weight: torch.tensor):
        super(SamplingLoss, self).__init__()
        self.xyz = xyz
        self.rgb = rgb
        self.img = img
        self.cfg = cfg
        self.tensor_0 = torch.zeros(1, device=xyz.device)
        self.tensor_1 = torch.ones(1, device=xyz.device)
        self.img_weight = img_weight
        self.pcd_weight = pcd_weight

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

        in_rgb = self.rgb

        filter_factor = getattr(self.cfg, 'filter_factor', 1)
        filtered_mask = refine_sampling_coords(coord_arr, torch.norm(new_xyz, dim=-1), in_rgb, quantization=(self.img.shape[0] // filter_factor, 
            self.img.shape[1] // filter_factor), return_valid_mask=True)
        
        sample_rgb = sample_from_img(self.img, coord_arr)
        raw_loss = torch.norm(sample_rgb - in_rgb, dim=-1)
        
        # Add weight to loss
        loss_weight_list = []
        if self.img_weight is not None:
            with torch.no_grad():
                loss_img_weight = sample_from_img(self.img_weight, coord_arr)
            loss_weight_list.append(loss_img_weight.squeeze())
        if self.pcd_weight is not None:
            loss_weight_list.append(self.pcd_weight.squeeze())

        if len(loss_weight_list) == 1:
            raw_loss = loss_weight_list[0] * raw_loss
        elif len(loss_weight_list) == 2:
            raw_loss = 0.5 * (loss_weight_list[0] + loss_weight_list[1]) * raw_loss

        if getattr(self.cfg, 'filter_idx', False):
            mask = (torch.sum(sample_rgb == 0, dim=1) != 3) & filtered_mask
            rgb_loss = (raw_loss * mask).sum() / mask.sum().float()
        else:
            mask = torch.sum(sample_rgb == 0, dim=1) != 3
            rgb_loss = (raw_loss * mask).sum() / mask.sum().float()

        return rgb_loss
