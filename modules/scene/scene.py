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
import shutil
from typing import List, Dict, Tuple, Any

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
        uid=id, colmap_id=cam_info.uid, 
        R=cam_info.R, T=cam_info.T, FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
        image=gt_image, mask=loaded_mask, image_name=cam_info.image_name, 
    )


class Scene:

    def __init__(self, args:ModelParams, load_iter:int=None, resolution_scales:List[float]=[1.0]):
        self.args = args
        self.train_cameras: Dict[int, Camera] = {}
        self.test_cameras:  Dict[int, Camera] = {}
        self.gaussians = GaussianModel(args)
        self.background = (torch.ones if args.white_background else torch.zeros)([3], dtype=torch.float, device='cuda')

        if os.path.exists(os.path.join(args.source_path, 'sparse')):
            scene_info = SCENE_DATA_LOADERS['Colmap'](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, 'dense', 'sparse')):
            scene_info = SCENE_DATA_LOADERS['ColmapExt'](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, 'transforms_train.json')):
            print('Found transforms_train.json file, assuming Blender data set!')
            scene_info = SCENE_DATA_LOADERS['Blender'](args.source_path, args.white_background, args.eval)
        else:
            raise TypeError(f'Could not recognize scene type for dataset {args.source_path}')

        self.cameras_extent: float = scene_info.nerf_normalization['radius']
        print('>> cameras_extent:', self.cameras_extent)
        self.gaussians.spatial_lr_scale = self.cameras_extent   # FIXME: not elegant

        for res in resolution_scales:
            print('Loading Train Cameras')
            self.train_cameras[res] = [load_camera(args, id, cam, res) for id, cam in enumerate(scene_info.train_cameras)]
            print('Loading Test Cameras')
            self.test_cameras[res] = [load_camera(args, id, cam, res) for id, cam in enumerate(scene_info.test_cameras)]

        if load_iter is not None and load_iter < 0:
            try:
                load_iter = max([int(fn.split('_')[-1]) for fn in os.listdir(os.path.join(self.model_path, 'point_cloud'))])
            except:
                print('>> not found saved point_cloud.ply')
                load_iter = None
        self.load_iter = load_iter
        if load_iter is None:
            shutil.copyfile(scene_info.ply_path, os.path.join(self.model_path, 'input.ply'))
            cam_infos = [cam.to_json(id) for id, cam in enumerate(scene_info.train_cameras + scene_info.test_cameras)]
            with open(os.path.join(self.model_path, 'cameras.json'), 'w') as fh:
                json.dump(cam_infos, fh, indent=2, ensure_ascii=False)
            print('>> [gaussian] init via from_pcd')
            self.gaussians.from_pcd(scene_info.point_cloud, args.sanitize_init_pcd)
        else:
            print(f'>> [gaussian] init via load_ply at iteration-{load_iter}')
            self.load_gaussian(load_iter)

    @property
    def model_path(self) -> str: return self.args.model_path

    @classmethod
    def random_background(cls) -> Tensor:
        return torch.rand([3], dtype=torch.float, device='cuda')

    def get_train_cameras(self, scale:float=1.0):
        return self.train_cameras[scale]

    def get_test_cameras(self, scale:float=1.0):
        return self.test_cameras[scale]

    def save_gaussian(self, steps:int):
        self.gaussians.save_ply(os.path.join(self.model_path, 'point_cloud', f'iteration_{steps}', 'point_cloud.ply'))

    def load_gaussian(self, steps:int):
        self.gaussians.load_ply(os.path.join(self.model_path, 'point_cloud', f'iteration_{steps}', 'point_cloud.ply'), self.args.sanitize_load_guass)

    def save_checkpoint(self, steps:int):
        state_dict = self.gaussians.state_dict()
        state_dict['steps'] = steps
        torch.save(state_dict, os.path.join(self.model_path, f'ckpt-{steps}.pth'))

    def load_checkpoint(self, path:str) -> int:
        state_dict: Dict[str, Any] = torch.load(path)
        steps = state_dict.get('steps', 0)
        self.gaussians.load_state_dict(state_dict)
        return steps
