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
from importlib import import_module
from traceback import print_exc
from typing import List, Dict, Any

import torch
from torch import Tensor

from modules.data import SCENE_DATA_LOADERS
from modules.camera import Camera, load_camera

try:  # only for annotation use, avoid cyclic import error :(
    from modules.hparam import HyperParams
    from modules.morphs import GaussianModel
except ImportError: pass


class Scene:

    def __init__(self, hp:HyperParams, resolution_scales:List[float]=[1.0]):
        # resolve GaussianModel class
        try:
            mod = import_module(f'modules.morphs.{hp.morph}.model')
            GaussianModel_cls = getattr(mod, 'GaussianModel')
        except:
            print_exc()
            print('>> should implement model.py script')

        self.hp: HyperParams = hp
        self.train_cameras: Dict[float, List[Camera]] = {}
        self.test_cameras:  Dict[float, List[Camera]] = {}
        self.gaussians: GaussianModel = GaussianModel_cls(hp)
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

        # FIXME: monkey patch for neural GaussianModel_cls
        if hasattr(self.gaussians, 'init_embeddings'):
            self.gaussians.init_embeddings(len(scene_info.train_cameras))

        # FIXME: monkey patch init spatial_lr_scale
        self.cameras_extent: float = scene_info.nerf_normalization['radius']
        print('>> cameras_extent:', self.cameras_extent)
        self.gaussians.spatial_lr_scale = self.cameras_extent

        for res in resolution_scales:
            print('Loading Train Cameras')
            self.train_cameras[res] = [load_camera(hp.resolution, id, cam, res) for id, cam in enumerate(scene_info.train_cameras)]
            print('Loading Test Cameras')
            self.test_cameras[res] = [load_camera(hp.resolution, id, cam, res) for id, cam in enumerate(scene_info.test_cameras)]

        load_iter = hp.load_iter
        if load_iter is not None and load_iter < 0:
            try:
                load_iter = max([int(dp.stem.split('_')[-1]) for dp in (self.model_path / 'point_cloud').iterdir()])
            except:
                print('>> not found saved point_cloud.ply')
                load_iter = None
        self.load_iter = load_iter
        if load_iter is None:
            shutil.copyfile(scene_info.ply_path, self.model_path / 'input.ply')
            cam_infos = [cam.to_json(id) for id, cam in enumerate(scene_info.train_cameras + scene_info.test_cameras)]
            with open(self.model_path / 'cameras.json', 'w') as fh:
                json.dump(cam_infos, fh, indent=2, ensure_ascii=False)
            print('>> [gaussian] init via from_pcd')
            self.gaussians.from_pcd(scene_info.point_cloud)
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

    def save_gaussian(self, steps:int):
        base_dir = self.model_path / 'point_cloud' / f'iteration_{steps}'
        self.gaussians.save_ply(base_dir / 'point_cloud.ply')
        # FIXME: monkey patch for neural GaussianModel_cls
        if hasattr(self.gaussians, 'save_pth'):
            self.gaussians.save_pth(base_dir / 'model.pth')

    def load_gaussian(self, steps:int):
        base_dir = self.model_path / 'point_cloud' / f'iteration_{steps}'
        self.gaussians.load_ply(base_dir / 'point_cloud.ply')
        # FIXME: monkey patch for neural GaussianModel_cls
        if hasattr(self.gaussians, 'load_pth'):
            self.gaussians.load_pth(base_dir / 'model.pth')

    def save_checkpoint(self, steps:int):
        state_dict = self.gaussians.state_dict()
        state_dict['steps'] = steps
        torch.save(state_dict, self.model_path / f'ckpt-{steps}.pth')

    def load_checkpoint(self, path:str) -> int:
        state_dict: Dict[str, Any] = torch.load(path)
        steps = state_dict.get('steps', 0)
        self.gaussians.load_state_dict(state_dict)
        return steps
