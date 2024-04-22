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
from torch import Tensor
import numpy as np
from numpy import ndarray

from modules.camera import load_camera
from modules.utils.graphics_utils import getWorld2View2, getProjectionMatrix


class Camera:

    def __init__(self, uid:int, colmap_id:int, R:ndarray, T:ndarray, FoVx:float, FoVy:float, 
                 image:Tensor, mask:Tensor=None, image_name:str=None, trans:ndarray=np.zeros([3]), scale:float=1.0,
                 data_device:str='cuda'):
        self.uid = uid
        self.colmap_id = colmap_id

        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.zfar = 100.0
        self.znear = 0.01
        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        img = image.clamp_(0.0, 1.0).to(data_device)
        if mask is not None:
            img *= mask.to(device=img)

        if 'add artificial occlusion':
            C, H, W = image.shape
            h = int(H * 0.2)    # ~4% area
            w = int(W * 0.2)
            y = np.random.randint(0, H-h)
            x = np.random.randint(0, W-w)
            img[:, y:y+h, x:x+w] = np.random.random()   # pure-color patch

        self.image = img
        self.image_name = image_name
        self.image_height = self.image.shape[1]
        self.image_width = self.image.shape[2]
