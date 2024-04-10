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
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch import Tensor
from tqdm import tqdm

from modules.data import SCENE_DATA_LOADERS
from .camera import Camera, load_camera
from .model import MutilFreqGaussianModel, SingleFreqGaussianModel

try: from modules.hparam import HyperParams
except: pass


class Scene:

    def __init__(self, hp:HyperParams, resolution_scales:List[float]=[1.0]):
        self.hp: HyperParams = hp
        self.train_cameras: Dict[float, List[Camera]] = {}
        self.test_cameras:  Dict[float, List[Camera]] = {}
        self.gaussians: SingleFreqGaussianModel = None      # current active GaussianModel
        self.multifreq_gaussians = MutilFreqGaussianModel(hp)
        self.background = (torch.ones if hp.white_background else torch.zeros)([3], dtype=torch.float, device='cuda')

        if os.path.exists(os.path.join(hp.source_path, 'sparse')):
            scene_info = SCENE_DATA_LOADERS['Colmap'](hp.source_path, hp.images, hp.eval)
        elif os.path.exists(os.path.join(hp.source_path, 'dense', 'sparse')):
            scene_info = SCENE_DATA_LOADERS['ColmapExt'](hp.source_path, hp.images, hp.eval)
        elif os.path.exists(os.path.join(hp.source_path, 'transforms_train.json')):
            print('Found transforms_train.json file, assuming Blender data set!')
            scene_info = SCENE_DATA_LOADERS['Blender'](hp.source_path, hp.white_background, hp.eval)
        else:
            raise TypeError(f'Could not recognize scene type for dataset {hp.source_path}')

        self.cameras_extent: float = scene_info.nerf_normalization['radius']
        print('>> cameras_extent:', self.cameras_extent)
        self.multifreq_gaussians.spatial_lr_scale = self.cameras_extent  # FIXME: monkey patch

        for res in resolution_scales:
            print('Loading Train Cameras')
            self.train_cameras[res] = [load_camera(hp, id, cam, res) for id, cam in enumerate(tqdm(scene_info.train_cameras[:10]))]
            print('Loading Test Cameras')
            self.test_cameras[res] = [load_camera(hp, id, cam, res) for id, cam in enumerate(tqdm(scene_info.test_cameras))]

        load_iter = hp.load_iter
        if load_iter is not None and load_iter < 0:
            try:
                load_iter = max([int(dn.split('_')[-1]) for dn in os.listdir(os.path.join(self.model_path, 'point_cloud'))])
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
            self.multifreq_gaussians.from_pcd(scene_info.point_cloud)
        else:
            print(f'>> [gaussian] init via load_ply at iteration-{load_iter}')
            self.load_gaussian(load_iter)

    @property
    def model_path(self) -> Path: return Path(self.hp.model_path)

    @classmethod
    def random_background(cls) -> Tensor:
        return torch.rand([3], dtype=torch.float, device='cuda')

    def get_train_cameras(self, scale:float=1.0) -> List[Camera]:
        return self.train_cameras[scale]

    def get_test_cameras(self, scale:float=1.0) -> List[Camera]:
        return self.test_cameras[scale]

    def activate_gaussian(self, idx:int=0) -> SingleFreqGaussianModel:
        self.gaussians = self.multifreq_gaussians.activate_gaussian(idx)
        return self.gaussians

    def save_gaussian(self, steps:int):
        base_dir = os.path.join(self.model_path, 'point_cloud', f'iteration_{steps}')
        for idx, gaussians in self.multifreq_gaussians.gaussians.items():
            gaussians.save_ply(os.path.join(base_dir, f'point_cloud_{idx}.ply'))

    def load_gaussian(self, steps:int):
        base_dir = os.path.join(self.model_path, 'point_cloud', f'iteration_{steps}')
        for idx, gaussians in self.multifreq_gaussians.gaussians.items():
            gaussians.load_ply(os.path.join(base_dir, f'point_cloud_{idx}.ply'))

    def save_checkpoint(self, steps:int):
        state_dict = {'steps': steps}
        for idx, gaussians in self.multifreq_gaussians.gaussians.items():
            try: state_dict[idx] = gaussians.state_dict()
            except: print(f'>> {idx} is not availbale')
        torch.save(state_dict, os.path.join(self.model_path, f'ckpt-{steps}.pth'))


    def load_checkpoint(self, path:str) -> int:
        state_dict: Dict[str, Any] = torch.load(path)
        steps = state_dict.get('steps', 0)
        for idx, gaussians in self.multifreq_gaussians.gaussians.items():
            try: gaussians.load_state_dict(state_dict[idx])
            except: print(f'>> {idx} is not availbale')
        return steps
