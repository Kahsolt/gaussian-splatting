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
import torch.nn.functional as F
from torch import Tensor

from modules.camera import Camera
from modules.layers import ColorMLP, Embedding
from modules.model import GaussianModel_Neural
from modules.utils.modeling_utils import *
from modules.utils.training_utils import *

from .hparam import HyperParams


class GaussianModel(GaussianModel_Neural):

    ''' cd-gs from bhy '''

    def __init__(self, hp:HyperParams):
        super().__init__(hp)

        self.hp: HyperParams

        # networks
        self.sun_vis_embedding: Embedding = nn.Identity()
        self.sun_shade_embedding: Embedding = nn.Identity()
        self.sky_shade_embedding: Embedding = nn.Identity()
        self.reflectance_embedding: Embedding = nn.Identity()

        view_dim = 4 if hp.add_view else 0
        sun_vis_dim     = hp.sun_vis_embedding_dim     + (hp.per_feat_dim + view_dim)
        sun_shade_dim   = hp.sun_shade_embedding_dim   + (hp.per_feat_dim + view_dim)
        sky_shade_dim   = hp.sky_shade_embedding_dim   + (hp.per_feat_dim + view_dim)
        reflectance_dim = hp.reflectance_embedding_dim + (hp.per_feat_dim + view_dim)
        self.sun_vis_color_mlp     = ColorMLP(sun_vis_dim,     hp.hidden_dim)
        self.sun_shade_color_mlp   = ColorMLP(sun_shade_dim,   hp.hidden_dim)
        self.sky_shade_color_mlp   = ColorMLP(sky_shade_dim,   hp.hidden_dim)
        self.reflectance_color_mlp = ColorMLP(reflectance_dim, hp.hidden_dim)

    def feature_encoder(self, camera:Camera, visible_mask=None):
        if visible_mask is None: visible_mask = slice(None)
        feats = self.features[visible_mask]

        hp = self.hp
        if hp.add_view:
            pts = self.xyz[visible_mask]
            ob_view = pts - camera.camera_center            # (N, 3)
            ob_dist = ob_view.norm(dim=1, keepdim=True)     # (N, 1)
            ob_view = ob_view / ob_dist                     # (N, 3)
            ob_dist = torch.log(ob_dist)

        colors = []
        mlps = self.mlps
        embeddings = self.embeddings
        camera_indicies = torch.ones_like(feats[:, 0], dtype=torch.long) * camera.uid
        for i in range(len(mlps)):
            feat = feats[:, hp.per_feat_dim*i: hp.per_feat_dim*(i+1)]
            if hp.add_view:
                feat = torch.cat([feat, ob_view, ob_dist], -1)
            if embeddings[i] is not None:   
                embedding = embeddings[i](camera_indicies)
                feat = torch.cat([feat, embedding], -1)
            colors.append(mlps[i](feat))

        return colors

    def setup_transform_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.rotation_activation = F.normalize
        self.opacity_activation = torch.sigmoid
        self.opacity_inverse_activation = inverse_sigmoid
        self.covariance_activation = build_covariance_from_scaling_rotation

    def init_embeddings(self, num_cameras:int):
        hp = self.hp
        if hp.sun_vis_embedding_dim > 0:
            self.sun_vis_embedding = Embedding(num_cameras, hp.sun_vis_embedding_dim)
        if hp.sun_shade_embedding_dim > 0:
            self.sun_shade_embedding = Embedding(num_cameras, hp.sun_shade_embedding_dim)
        if hp.sky_shade_embedding_dim > 0:
            self.sky_shade_embedding = Embedding(num_cameras, hp.sky_shade_embedding_dim)
        if hp.reflectance_embedding_dim > 0:
            self.reflectance_embedding = Embedding(num_cameras, hp.reflectance_embedding_dim)

    @property
    def mlps(self):
        return [
            self.sun_vis_color_mlp, 
            self.sun_shade_color_mlp, 
            self.sky_shade_color_mlp,
            self.reflectance_color_mlp,
        ]

    @property
    def embeddings(self):
        return [
            self.sun_vis_embedding,
            self.sun_shade_embedding,
            self.sky_shade_embedding,
            self.reflectance_embedding,
        ]

    def state_dict(self) -> Dict[str, Any]:
        state_dict = super().state_dict()
        neural_layers = {}
        for name in ['sun_vis', 'sun_shade', 'sky_shade', 'reflectance']:
            key = f'{name}_color_mlp'
            neural_layers[key] = getattr(self, key).state_dict()
            key = f'{name}_embedding'
            neural_layers[key] = getattr(self, key).state_dict()
        state_dict.update(neural_layers)
        return state_dict

    def load_state_dict(self, state_dict:Dict[str, Any]):
        for name in ['sun_vis', 'sun_shade', 'sky_shade', 'reflectance']:
            key = f'{name}_color_mlp'
            mod: ColorMLP = getattr(self, key)
            mod.load_state_dict(state_dict[key])
            key = f'{name}_embedding'
            mod: Embedding = getattr(self, key)
            mod.load_state_dict(state_dict[key])
        super().load_state_dict(state_dict)

    def save_pth(self, fp:Path):
        fp.parent.mkdir(exist_ok=True, parents=True)
        state_dict = {}
        for name in ['sun_vis', 'sun_shade', 'sky_shade', 'reflectance']:
            key = f'{name}_color_mlp'
            mod: ColorMLP = getattr(self, key)
            state_dict[key] = mod.state_dict()
            key = f'{name}_embedding'
            mod: Embedding = getattr(self, key)
            state_dict[key] = mod.state_dict()
        torch.save(state_dict, fp)

    def load_pth(self, fp:Path):
        state_dict = torch.load(fp)
        for name in ['sun_vis', 'sun_shade', 'sky_shade', 'reflectance']:
            key = f'{name}_color_mlp'
            mod: ColorMLP = getattr(self, key)
            mod.load_state_dict(state_dict[key])
            key = f'{name}_embedding'
            mod: Embedding = getattr(self, key)
            mod.load_state_dict(state_dict[key])

    ''' optimize '''

    def setup_training(self):
        hp = self.hp
        param_groups = [
            {'name': 'xyz',      'params': [self._xyz],      'lr': hp.position_lr_init * self.spatial_lr_scale},
            {'name': 'scaling',  'params': [self._scaling],  'lr': hp.scaling_lr},
            {'name': 'rotation', 'params': [self._rotation], 'lr': hp.rotation_lr},
            {'name': 'features', 'params': [self._features], 'lr': hp.feature_lr},
            {'name': 'opacity',  'params': [self._opacity],  'lr': hp.opacity_lr},
            {'name': 'sun_vis_color_mlp',     'params': self.sun_vis_color_mlp.    parameters(), 'lr': hp.mlp_color_lr_init},
            {'name': 'sun_shade_color_mlp',   'params': self.sun_shade_color_mlp.  parameters(), 'lr': hp.mlp_color_lr_init},
            {'name': 'sky_shade_color_mlp',   'params': self.sky_shade_color_mlp.  parameters(), 'lr': hp.mlp_color_lr_init},
            {'name': 'reflectance_color_mlp', 'params': self.reflectance_color_mlp.parameters(), 'lr': hp.mlp_color_lr_init},
            {'name': 'sun_vis_embedding',     'params': self.sun_vis_embedding.    parameters(), 'lr': hp.embedding_lr_init},
            {'name': 'sun_shade_embedding',   'params': self.sun_shade_embedding.  parameters(), 'lr': hp.embedding_lr_init},
            {'name': 'sky_shade_embedding',   'params': self.sky_shade_embedding.  parameters(), 'lr': hp.embedding_lr_init},
            {'name': 'reflectance_embedding', 'params': self.reflectance_embedding.parameters(), 'lr': hp.embedding_lr_init},
        ]
        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
        self.xyz_scheduler = get_expon_lr_func(
            lr_init=hp.position_lr_init * self.spatial_lr_scale,
            lr_final=hp.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=hp.position_lr_delay_mult,
            max_steps=hp.position_lr_max_steps,
        )
        self.mlp_color_scheduler_args = get_expon_lr_func(**make_expon_lr_func_args(hp, 'mlp_color'))
        self.embedding_scheduler_args = get_expon_lr_func(**make_expon_lr_func_args(hp, 'embedding'))
        self.xyz_grad_accum = torch.zeros((self.xyz.shape[0], 1), dtype=torch.float, device='cuda')
        self.xyz_grad_count = torch.zeros((self.xyz.shape[0], 1), dtype=torch.int,   device='cuda')
        self.max_radii2D    = torch.zeros((self.xyz.shape[0]),    dtype=torch.int,   device='cuda')
        self.percent_dense = hp.percent_dense

    def update_learning_rate(self, steps:int):
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'xyz':
                lr = self.xyz_scheduler(steps)
            elif param_group['name'].endswith('color_mlp'):
                lr = self.mlp_color_scheduler_args(steps)
            elif param_group['name'].endswith('embedding'):
                lr = self.embedding_scheduler_args(steps)
            else: continue  # skip
            param_group['lr'] = lr

    def cat_tensors_to_optimizer(self, tensors_dict:Dict[str, Tensor]) -> Dict[str, Tensor]:
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group['name'].endswith('color_mlp') or group['name'].endswith('embedding'): continue
            assert len(group["params"]) == 1
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
            if group['name'].endswith('color_mlp') or group['name'].endswith('embedding'): continue
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
