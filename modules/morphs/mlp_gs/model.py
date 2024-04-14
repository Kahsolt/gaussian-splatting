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

    def make_param_group(self) -> List[Dict[str, Any]]:
        hp = self.hp
        param_group = super().make_param_group()
        param_group.extend([
            {'name': 'mlp_color', 'params': self.color_mlp.parameters(), 'lr': hp.mlp_color_lr_init},
        ])
        return param_group

    def setup_training(self):
        super().setup_training()
        self.mlp_color_scheduler_args = get_expon_lr_func(**make_expon_lr_func_args(self.hp, 'mlp_color'))

    def update_learning_rate(self, steps:int):
        super().update_learning_rate(steps)
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'mlp_color':
                lr = self.mlp_color_scheduler_args(steps)
                param_group['lr'] = lr

    def cat_tensors_to_optimizer(self, tensors_dict:Dict[str, Tensor], excludes:List[str]=[]) -> Dict[str, Tensor]:
        return super().cat_tensors_to_optimizer(tensors_dict, excludes + ['mlp_color'])

    def prune_optimizer(self, mask:Tensor, excludes:List[str]=[]) -> Dict[str, Tensor]:
        return super().prune_optimizer(mask, excludes + ['mlp_color'])
