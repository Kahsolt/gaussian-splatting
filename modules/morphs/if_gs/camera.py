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

from typing import Tuple, List

import torch
from torch import Tensor
import numpy as np
from numpy import ndarray
from PIL.Image import Image as PILImage

from modules.data import CameraInfo
from modules.utils.graphics_utils import getWorld2View2, getProjectionMatrix

from .hparam import HyperParams
from .image_utils import split_freqs, pil_to_np

WARNED = False


class Camera:

    def __init__(self, uid:int, colmap_id:int, R:ndarray, T:ndarray, FoVx:float, FoVy:float, 
                 images:List[Tensor], mask:Tensor=None, image_name:str=None, trans:ndarray=np.zeros([3]), scale:float=1.0,
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

        imgs = [img.clamp_(0.0, 1.0).to(data_device) for img in images]
        if mask is not None:
            mask = mask.to(imgs[0].device)
            for idx in range(len(imgs)):
                imgs[idx] *= mask
        self.images = imgs
        self.image_name = image_name
        self.image_height = self.images[0].shape[1]
        self.image_width = self.images[0].shape[2]


def PILtoTorch(img:PILImage, resize:Tuple[int, int]=None, hp:HyperParams=None) -> List[Tensor]:
    assert hp is not None
    if resize: img = img.resize(resize)
    n_freq_img = split_freqs(pil_to_np(img), n_bands=hp.L_freq, scale_w=hp.scale_w, kind=hp.split_kind)
    Xs: Tensor = [torch.from_numpy(np.array(img)) / 255.0 for img in n_freq_img]
    if img.mode == 'L': Xs = [X.unsqueeze(dim=-1) for X in Xs]
    return [X.permute(2, 0, 1) for X in Xs]


def load_camera(hp:HyperParams, id:int, cam_info:CameraInfo, resolution_scale:float) -> Camera:
    global WARNED

    resolution = hp.resolution
    orig_w, orig_h = cam_info.image.size
    if resolution in [1, 2, 4, 8]:
        resolution = round(orig_w / (resolution_scale * resolution)), round(orig_h / (resolution_scale * resolution))
    else:  # should be a type that converts to float
        if resolution == -1:
            if orig_w > 1600:
                if not WARNED:
                    print('[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.')
                    print('If this is not desired, please explicitly specify "--resolution/-r" as 1')
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution, hp)
    gt_images = [rgb[:3, ...] for rgb in resized_image_rgb]
    loaded_mask = None
    if resized_image_rgb[0].shape[1] == 4:
        loaded_mask = resized_image_rgb[0][3:4, ...]

    return Camera(
        uid=id, colmap_id=cam_info.uid, 
        R=cam_info.R, T=cam_info.T, FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
        images=gt_images, mask=loaded_mask, image_name=cam_info.image_name, 
    )
