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
from typing import List, Dict, Any

import torch
from tqdm import tqdm

from modules.scene import Scene as SceneBase
from modules.model import GaussianModel_Neural
from modules.utils.general_utils import mkdir

from .model import MutilFreqGaussianModel, SingleFreqGaussianModel
from .camera import Camera, load_camera
from .hparam import HyperParams


class Scene(SceneBase):

    def __init__(self, hp:HyperParams, resolution_scales:List[float]=[1.0]):
        super().__init__(hp, resolution_scales)

        self.hp: HyperParams
        self.gaussians: MutilFreqGaussianModel
        self.train_cameras: Dict[float, List[Camera]]
        self.test_cameras:  Dict[float, List[Camera]]

    def process_cameras(self):
        limit = self.hp.limit
        for res in self.resolution_scales:
            print('Loading Train Cameras')
            self.train_cameras[res] = [load_camera(self.hp, id, cam, res) for id, cam in enumerate(tqdm(self.scene_info.train_cameras)) if id < limit]
            print('Loading Test Cameras')
            self.test_cameras[res] = [load_camera(self.hp, id, cam, res) for id, cam in enumerate(tqdm(self.scene_info.test_cameras)) if id < limit]

    @property
    def cur_gaussian(self) -> SingleFreqGaussianModel:
        return self.gaussians.cur_gaussian

    @property
    def all_gaussians(self) -> Dict[int, SingleFreqGaussianModel]:
        return self.gaussians.gaussians

    def activate_gaussian(self, idx:int=0, swap:bool=False) -> SingleFreqGaussianModel:
        self.gaussians.activate_gaussian(idx, swap)
        return self.cur_gaussian

    def save_gaussian(self, steps:int):
        base_dir = mkdir(self.model_path / 'point_cloud' / f'iteration_{steps}', parents=True)
        # only save the current training one
        idx = self.gaussians.cur_idx
        gaussians = self.gaussians.cur_gaussian
        super()._save_ply(base_dir / f'point_cloud_{idx}.ply', *gaussians.save_ply())
        if isinstance(gaussians, GaussianModel_Neural):
            gaussians.save_pth(base_dir / f'model-{idx}.pth')

    def load_gaussian(self, steps:int):
        base_dir = self.model_path / 'point_cloud' / f'iteration_{steps}'
        for idx, gaussians in self.all_gaussians.items():
            gaussians.load_ply(super()._load_ply(base_dir / f'point_cloud_{idx}.ply'))
            if isinstance(gaussians, GaussianModel_Neural):
                gaussians.load_pth(base_dir / f'model-{idx}.pth')

    def save_checkpoint(self, steps:int):
        state_dict = {'steps': steps}
        for idx, gaussians in self.all_gaussians.items():
            try: state_dict[idx] = gaussians.state_dict()
            except AttributeError: print(f'>> freq-{idx} is not availbale')
        torch.save(state_dict, os.path.join(self.model_path, f'ckpt-{steps}.pth'))

    def load_checkpoint(self, path:str) -> int:
        state_dict: Dict[str, Any] = torch.load(path)
        steps = state_dict.get('steps', 0)
        for idx, gaussians in self.all_gaussians.items():
            try: gaussians.load_state_dict(state_dict[idx])
            except AttributeError: print(f'>> freq-{idx} is not availbale')
        return steps
