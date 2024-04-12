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

from pathlib import Path
from typing import  Dict, Any

import torch
import torch.nn as nn
from torch import Tensor

from modules.camera import Camera
from modules.layers import ColorMLP
from modules.model import GaussianModel_Neural
from modules.utils.modeling_utils import *
from modules.utils.training_utils import *

from .hparam import HyperParams


class GaussianModel(GaussianModel_Neural):

    ''' mlp-gs from bhy '''

    def __init__(self, hp:HyperParams):
        super().__init__(hp)

        self.hp: HyperParams

        # networks
        in_dim = self.hp.feat_dim + (4 if self.hp.add_view else 0)
        self.color_mlp = ColorMLP(in_dim, self.hp.feat_dim)

    def feature_encoder(self, camera:Camera, visible_mask:Tensor=None):
        if visible_mask is None: visible_mask = slice(None)
        view_feat = self.features[visible_mask]
        
        if self.hp.add_view:
            pts = self.xyz[visible_mask]
            ob_view = pts - camera.camera_center            # [N, 3]
            ob_dist = ob_view.norm(dim=1, keepdim=True)     # [N, 1]
            ob_view = ob_view / ob_dist                     # [N, 3]
            ob_dist = torch.log(ob_dist)
            view_feat = torch.cat([view_feat, ob_view, ob_dist], dim=1)

        return self.color_mlp(view_feat)

    def state_dict(self) -> Dict[str, Any]:
        state_dict = super().state_dict()
        state_dict.update({
            'color_mlp': self.color_mlp.state_dict(),
        })
        return state_dict

    def load_state_dict(self, state_dict:Dict[str, Any]):
        self.color_mlp.load_state_dict(state_dict['color_mlp'])
        super().load_state_dict(state_dict)

    def save_pth(self, fp:Path):
        fp.parent.mkdir(exist_ok=True, parents=True)
        ckpt = {
            'color_mlp': self.color_mlp.state_dict(),
        }
        torch.save(ckpt, fp)

    def load_pth(self, fp:Path):
        ckpt = torch.load(fp)
        self.color_mlp.load_state_dict(ckpt['color_mlp'])

    ''' optimize '''

    def setup_training(self):
        hp = self.hp
        param_groups = [
            {'name': 'xyz',       'params': [self._xyz],      'lr': hp.position_lr_init * self.spatial_lr_scale},
            {'name': 'scaling',   'params': [self._scaling],  'lr': hp.scaling_lr},
            {'name': 'rotation',  'params': [self._rotation], 'lr': hp.rotation_lr},
            {'name': 'features',  'params': [self._features], 'lr': hp.feature_lr},
            {'name': 'opacity',   'params': [self._opacity],  'lr': hp.opacity_lr},
            {'name': 'mlp_color', 'params': self.color_mlp.parameters(), 'lr': hp.mlp_color_lr_init},
        ]
        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
        self.xyz_scheduler = get_expon_lr_func(
            lr_init=hp.position_lr_init * self.spatial_lr_scale,
            lr_final=hp.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=hp.position_lr_delay_mult,
            max_steps=hp.position_lr_max_steps,
        )
        self.mlp_color_scheduler_args = get_expon_lr_func(**make_expon_lr_func_args(hp, 'mlp_color'))
        self.xyz_grad_accum = torch.zeros((self.xyz.shape[0], 1), dtype=torch.float, device='cuda')
        self.xyz_grad_count = torch.zeros((self.xyz.shape[0], 1), dtype=torch.int,   device='cuda')
        self.max_radii2D    = torch.zeros((self.xyz.shape[0]),    dtype=torch.int,   device='cuda')
        self.percent_dense = hp.percent_dense

    def update_learning_rate(self, steps:int):
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'xyz':
                lr = self.xyz_scheduler(steps)
            elif param_group['name'] == 'mlp_color':
                lr = self.mlp_color_scheduler_args(steps)
            else: continue  # skip
            param_group['lr'] = lr

    def cat_tensors_to_optimizer(self, tensors_dict:Dict[str, Tensor]) -> Dict[str, Tensor]:
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group['name'].startswith('mlp'): continue
            assert len(group['params']) == 1
            extension_tensor = tensors_dict[group['name']]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state['exp_avg'] = torch.cat((stored_state['exp_avg'], torch.zeros_like(extension_tensor)), dim=0)
                stored_state['exp_avg_sq'] = torch.cat((stored_state['exp_avg_sq'], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group['params'][0] = nn.Parameter(torch.cat((group['params'][0], extension_tensor), dim=0), requires_grad=True)
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group['name']] = group['params'][0]
            else:
                group['params'][0] = nn.Parameter(torch.cat((group['params'][0], extension_tensor), dim=0), requires_grad=True)
                optimizable_tensors[group['name']] = group['params'][0]
        return optimizable_tensors

    def prune_optimizer(self, mask:Tensor) -> Dict[str, Tensor]:
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group['name'].startswith('mlp'): continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state['exp_avg'] = stored_state['exp_avg'][mask]
                stored_state['exp_avg_sq'] = stored_state['exp_avg_sq'][mask]

                del self.optimizer.state[group['params'][0]]
                group['params'][0] = nn.Parameter(group['params'][0][mask], requires_grad=True)
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group['name']] = group['params'][0]
            else:
                group['params'][0] = nn.Parameter(group['params'][0][mask], requires_grad=True)
                optimizable_tensors[group['name']] = group['params'][0]
        return optimizable_tensors
