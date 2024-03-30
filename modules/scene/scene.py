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

import os
import json
import random
from typing import Dict, Tuple

import torch
from torch import Tensor
import numpy as np
from PIL.Image import Image as PILImage

from modules.arguments import ModelParams
from modules.data import SCENE_DATA_LOADERS, CameraInfo
from modules.scene import Camera, GaussianModel

WARNED = False


def PILtoTorch(img:PILImage, resize:Tuple[int, int]=None) -> Tensor:
    if resize: img = img.resize(resize)
    X: Tensor = torch.from_numpy(np.array(img)) / 255.0
    if img.mode == 'L': X = X.unsqueeze(dim=-1)
    return X.permute(2, 0, 1)


def load_camera(args:ModelParams, id:int, cam_info:CameraInfo, resolution_scale:float) -> Camera:
    global WARNED

    orig_w, orig_h = cam_info.image.size
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                if not WARNED:
                    print('[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.')
                    print('If this is not desired, please explicitly specify "--resolution/-r" as 1')
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(
        colmap_id=cam_info.uid, 
        R=cam_info.R, T=cam_info.T, 
        FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
        image=gt_image, gt_alpha_mask=loaded_mask,
        image_name=cam_info.image_name, uid=id, 
        data_device=args.data_device,
    )


class Scene:

    def __init__(self, args:ModelParams, gaussians:GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = max([int(fname.split('_')[-1]) for fname in os.listdir(os.path.join(self.model_path, 'point_cloud'))])
            else:
                self.loaded_iter = load_iteration
            print(f'Loading trained model at iteration {self.loaded_iter}')

        self.train_cameras: Dict[int, Camera] = {}
        self.test_cameras:  Dict[int, Camera] = {}

        if os.path.exists(os.path.join(args.source_path, 'sparse')):
            scene_info = SCENE_DATA_LOADERS['Colmap'](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, 'transforms_train.json')):
            print('Found transforms_train.json file, assuming Blender data set!')
            scene_info = SCENE_DATA_LOADERS['Blender'](args.source_path, args.white_background, args.eval)
        else:
            assert False, 'Could not recognize scene type!'

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as fh_in, open(os.path.join(self.model_path, 'input.ply') , 'wb') as fh_out:
                fh_out.write(fh_in.read())
            json_cams = [cam.to_json(id) for id, cam in enumerate(scene_info.test_cameras + scene_info.train_cameras)]
            with open(os.path.join(self.model_path, 'cameras.json'), 'w') as fh:
                json.dump(json_cams, fh, indent=2, ensure_ascii=False)

        if shuffle:
            # Multi-res consistent random shuffling
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        self.cameras_extent: float = scene_info.nerf_normalization['radius']

        for resolution_scale in resolution_scales:
            print('Loading Training Cameras')
            self.train_cameras[resolution_scale] = [load_camera(args, id, cam, resolution_scale) for id, cam in enumerate(scene_info.train_cameras)]
            print('Loading Test Cameras')
            self.test_cameras[resolution_scale] = [load_camera(args, id, cam, resolution_scale) for id, cam in enumerate(scene_info.test_cameras)]

        if self.loaded_iter:
            print('>> [init] via load_ply')
            self.gaussians.load_ply(os.path.join(self.model_path, 'point_cloud', 'iteration_' + str(self.loaded_iter), 'point_cloud.ply'))
        else:
            print('>> [init] via create_from_pcd')
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def getTrainCameras(self, scale:float=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale:float=1.0):
        return self.test_cameras[scale]

    def save(self, steps:int):
        self.gaussians.save_ply(os.path.join(self.model_path, 'point_cloud', f'iteration_{steps}', 'point_cloud.ply'))
