#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


# 正交投影矩阵
def getOrthographicProjectionMatrix(left, right, bottom, top, znear, zfar):
    P = torch.zeros(4, 4)
    P[0, 0] = 2.0 / (right - left)
    P[1, 1] = 2.0 / (top - bottom)
    # P[2, 2] = -2.0 / (zfar - znear)
    P[2, 2] = 2.0 / (zfar - znear)
    P[0, 3] = -(right + left) / (right - left)
    P[1, 3] = -(top + bottom) / (top - bottom)
    P[2, 3] = -(zfar + znear) / (zfar - znear)
    P[3, 3] = 1.0
    return P


# 渲染用
#'''
class OrthoCamera(nn.Module):
    def __init__(self, colmap_id, R, T, R_ortho, T_ortho, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 ortho_left=-1.0, ortho_right=1.0, ortho_bottom=-1.0, ortho_top=1.0):
        super(OrthoCamera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.R_ortho = R_ortho
        self.T_ortho = T_ortho
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            # self.original_image *= gt_alpha_mask.to(self.data_device)
            self.gt_alpha_mask = gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            self.gt_alpha_mask = None
        

        self.zfar = 100.0
        self.znear = 1

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R_ortho, T_ortho, trans, scale)).transpose(0, 1).to(self.data_device)
        self.world_view_transform_non_ortho = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).to(self.data_device)
        # 确定动态范围
        aspect_ratio = self.image_width / self.image_height
        # 使用针孔相机模型计算每单位长度对应的像素数
        # pixels_per_unit = calculate_pixels_per_unit(intrinsics, (self.image_width, self.image_height), self.znear)
        # print(pixels_per_unit) 
        ortho_width = self.image_width / 25 # 每?个像素代表1个单位(21,,19可见边界，145) 400  #原像素级：20
        ortho_height = ortho_width / aspect_ratio
        left = -ortho_width / 2
        right = ortho_width / 2
        bottom = -ortho_height / 2
        top = ortho_height / 2
        # 使用正交投影矩阵
        self.projection_matrix = getOrthographicProjectionMatrix(left, right, bottom, top, self.znear, self.zfar).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
#'''

#'''
class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            # self.original_image *= gt_alpha_mask.to(self.data_device)
            self.gt_alpha_mask = gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            self.gt_alpha_mask = None
        
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
#'''
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

