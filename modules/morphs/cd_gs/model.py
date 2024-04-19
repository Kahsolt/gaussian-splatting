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
from typing import List, Dict, Any

import torch
import torch.nn as nn
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

    def feature_encoder(self, camera:Camera, visible_mask=slice(None)):
        feats = self.features[visible_mask]

        hp = self.hp
        if hp.add_view:
            pts = self.xyz[visible_mask]
            ob_view = pts - camera.camera_center            # (N, 3)
            ob_dist = ob_view.norm(dim=1, keepdim=True)     # (N, 1)
            ob_view = ob_view / ob_dist                     # (N, 3)
            ob_dist = torch.log(ob_dist)

        colors: List[Tensor] = []
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

    def make_param_group(self) -> List[Dict[str, Any]]:
        hp = self.hp
        param_group = super().make_param_group()
        param_group.extend([
            {'name': 'sun_vis_color_mlp',     'params': self.sun_vis_color_mlp.    parameters(), 'lr': hp.mlp_color_lr_init},
            {'name': 'sun_shade_color_mlp',   'params': self.sun_shade_color_mlp.  parameters(), 'lr': hp.mlp_color_lr_init},
            {'name': 'sky_shade_color_mlp',   'params': self.sky_shade_color_mlp.  parameters(), 'lr': hp.mlp_color_lr_init},
            {'name': 'reflectance_color_mlp', 'params': self.reflectance_color_mlp.parameters(), 'lr': hp.mlp_color_lr_init},
            {'name': 'sun_vis_embedding',     'params': self.sun_vis_embedding.    parameters(), 'lr': hp.embedding_lr_init},
            {'name': 'sun_shade_embedding',   'params': self.sun_shade_embedding.  parameters(), 'lr': hp.embedding_lr_init},
            {'name': 'sky_shade_embedding',   'params': self.sky_shade_embedding.  parameters(), 'lr': hp.embedding_lr_init},
            {'name': 'reflectance_embedding', 'params': self.reflectance_embedding.parameters(), 'lr': hp.embedding_lr_init},
        ])
        return param_group

    def setup_training(self):
        super().setup_training()
        self.mlp_color_scheduler_args = get_expon_lr_func(**make_expon_lr_func_args(self.hp, 'mlp_color'))
        self.embedding_scheduler_args = get_expon_lr_func(**make_expon_lr_func_args(self.hp, 'embedding'))

    def update_learning_rate(self, steps:int):
        super().update_learning_rate(steps)
        for param_group in self.optimizer.param_groups:
            if param_group['name'].endswith('_color_mlp'):
                lr = self.mlp_color_scheduler_args(steps)
            elif param_group['name'].endswith('_embedding'):
                lr = self.embedding_scheduler_args(steps)
            else: continue  # skip
            param_group['lr'] = lr

    def cat_tensors_to_optimizer(self, tensors_dict:Dict[str, Tensor], excludes:List[str]=[]) -> Dict[str, Tensor]:
        return super().cat_tensors_to_optimizer(tensors_dict, excludes + ['*_color_mlp', '*_embedding'])

    def prune_optimizer(self, mask:Tensor, excludes:List[str]=[]) -> Dict[str, Tensor]:
        return super().prune_optimizer(mask, excludes + ['*_color_mlp', '*_embedding'])
