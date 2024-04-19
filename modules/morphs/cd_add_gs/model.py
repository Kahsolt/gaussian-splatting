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
from typing import Tuple, List, Dict, Any

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

    ''' cd-gs from bhy, modified version '''

    def __init__(self, hp:HyperParams):
        super().__init__(hp)

        self.hp: HyperParams

        # networks
        self.view_embedding: Embedding = nn.Identity()

        in_dim = (hp.feat_dim if hp.use_feat else 0)
        self.center_mlp = ColorMLP(hp.feat_dim,                        hp.hidden_dim)
        self.shift_mlp  = ColorMLP(in_dim + hp.view_embedding_dim + 4, hp.hidden_dim)

    def feature_encoder(self, camera:Camera, visible_mask=slice(None)) -> Tuple[Tensor, Tensor]:
        feats = self.features[visible_mask]

        hp = self.hp
        if 'view feats':
            pts = self.xyz[visible_mask]
            ob_view = pts - camera.camera_center            # (N, 3)
            ob_dist = ob_view.norm(dim=1, keepdim=True)     # (N, 1)
            ob_view = ob_view / ob_dist                     # (N, 3)
            ob_dist = torch.log(ob_dist)
            view_feat = torch.cat([ob_view, ob_dist], -1)

            cam_idx = torch.ones_like(feats[:, 0], dtype=torch.long) * camera.uid
            view_embed = self.view_embedding(cam_idx)

        center = self.center_mlp(feats)
        shift = self.shift_mlp(torch.cat(([feats] if hp.use_feat else []) + [view_embed, view_feat], -1))
        return center, shift

    def init_embeddings(self, num_cameras:int):
        hp = self.hp
        if hp.view_embedding_dim > 0:
            self.view_embedding = Embedding(num_cameras, hp.view_embedding_dim)

    def state_dict(self) -> Dict[str, Any]:
        state_dict = super().state_dict()
        state_dict.update({
            'view_embedding': self.view_embedding.state_dict(),
            'center_mlp': self.center_mlp.state_dict(),
            'shift_mlp': self.shift_mlp.state_dict(),
        })
        return state_dict

    def load_state_dict(self, state_dict:Dict[str, Any]):
        self.view_embedding.load_state_dict(state_dict['view_embedding'])
        self.center_mlp.load_state_dict(state_dict['center_mlp'])
        self.shift_mlp.load_state_dict(state_dict['shift_mlp'])
        super().load_state_dict(state_dict)

    def save_pth(self, fp:Path):
        fp.parent.mkdir(exist_ok=True, parents=True)
        state_dict = {
            'view_embedding': self.view_embedding.state_dict(),
            'center_mlp': self.center_mlp.state_dict(),
            'shift_mlp': self.shift_mlp.state_dict(),
        }
        torch.save(state_dict, fp)

    def load_pth(self, fp:Path):
        state_dict = torch.load(fp)
        self.view_embedding.load_state_dict(state_dict['view_embedding'])
        self.center_mlp.load_state_dict(state_dict['center_mlp'])
        self.shift_mlp.load_state_dict(state_dict['shift_mlp'])

    def make_param_group(self) -> List[Dict[str, Any]]:
        hp = self.hp
        param_group = super().make_param_group()
        param_group.extend([
            {'name': 'center_mlp',     'params': self.center_mlp.    parameters(), 'lr': hp.mlp_lr_init},
            {'name': 'shift_mlp',      'params': self.shift_mlp.     parameters(), 'lr': hp.mlp_lr_init},
            {'name': 'view_embedding', 'params': self.view_embedding.parameters(), 'lr': hp.embedding_lr_init},
        ])
        return param_group

    def setup_training(self):
        super().setup_training()
        self.center_mlp_scheduler_args = get_expon_lr_func(**make_expon_lr_func_args(self.hp, 'mlp'))
        self.shift_mlp_scheduler_args = get_expon_lr_func(**make_expon_lr_func_args(self.hp, 'mlp'))
        self.embedding_scheduler_args = get_expon_lr_func(**make_expon_lr_func_args(self.hp, 'embedding'))

    def update_learning_rate(self, steps:int):
        super().update_learning_rate(steps)
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'center_mlp':
                lr = self.center_mlp_scheduler_args(steps)
            elif param_group['name'] == 'shift_mlp':
                lr = self.shift_mlp_scheduler_args(steps)
            elif param_group['name'].endswith('_embedding'):
                lr = self.embedding_scheduler_args(steps)
            else: continue  # skip
            param_group['lr'] = lr

    def cat_tensors_to_optimizer(self, tensors_dict:Dict[str, Tensor], excludes:List[str]=[]) -> Dict[str, Tensor]:
        return super().cat_tensors_to_optimizer(tensors_dict, excludes + ['*_mlp', '*_embedding'])

    def prune_optimizer(self, mask:Tensor, excludes:List[str]=[]) -> Dict[str, Tensor]:
        return super().prune_optimizer(mask, excludes + ['*_mlp', '*_embedding'])
