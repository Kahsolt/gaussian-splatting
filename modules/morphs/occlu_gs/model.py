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

from modules.morphs.mlp_gs.model import GaussianModel as GaussianModelBase
from modules.utils.modeling_utils import *
from modules.utils.training_utils import *

from .hparam import HyperParams
from .camera import Camera


class GaussianModel(GaussianModelBase):

    ''' occlu-gs '''

    def __init__(self, hp:HyperParams):
        super().__init__(hp)

        self.hp: HyperParams

        # networks
        self.view_spec_embed = nn.Identity()

    def init_embeddings(self, num_cameras:int):
        super().init_embeddings(num_cameras)
        weight = torch.empty([num_cameras, self.hp.view_embed_dim]).normal_(mean=0, std=0.2)
        self.view_spec_embed = nn.Embedding(num_cameras, self.hp.view_embed_dim, _weight=weight)

    def feature_encoder(self, camera:Camera, visible_mask:Tensor=slice(None)):
        view_feat = self.features[visible_mask]

        if self.hp.add_view:
            pts = self.xyz[visible_mask]
            ob_view = pts - camera.camera_center            # [N, 3]
            ob_dist = ob_view.norm(dim=1, keepdim=True)     # [N, 1]
            ob_view = ob_view / ob_dist                     # [N, 3]
            ob_dist = torch.log(ob_dist)
            view_feat = torch.cat([view_feat, ob_view, ob_dist], dim=1)

        return self.color_mlp(view_feat)

    @property
    def is_train(self):
        return self.view_spec_embed.training

    def post_process(self, camera:Camera, rendered:Tensor) -> Tensor:
        C, H, W = rendered.shape
        embed = self.view_spec_embed(torch.tensor([camera.uid], device=rendered.device))
        residual = embed.reshape([self.hp.view_embed_ch, H, W])
        return rendered + residual

    def state_dict(self) -> Dict[str, Any]:
        state_dict = super().state_dict()
        state_dict.update({
            'view_spec_embed': self.view_spec_embed.state_dict(),
        })
        return state_dict

    def load_state_dict(self, state_dict:Dict[str, Any]):
        self.view_spec_embed.load_state_dict(state_dict['view_spec_embed'])
        super().load_state_dict(state_dict)

    def save_pth(self, fp:Path):
        fp.parent.mkdir(exist_ok=True, parents=True)
        ckpt = {
            'view_spec_embed': self.view_spec_embed.state_dict(),
        }
        torch.save(ckpt, fp)

    def load_pth(self, fp:Path):
        ckpt = torch.load(fp)
        self.view_spec_embed.load_state_dict(ckpt['view_spec_embed'])

    def make_param_group(self) -> List[Dict[str, Any]]:
        hp = self.hp
        param_group = super().make_param_group()
        param_group.extend([
            {'name': 'view_spec_embed', 'params': self.view_spec_embed.parameters(), 'lr': hp.view_spec_embed_lr_init},
        ])
        return param_group

    def setup_training(self):
        super().setup_training()
        self.view_spec_embed_scheduler_args = get_expon_lr_func(**make_expon_lr_func_args(self.hp, 'view_spec_embed'))

    def update_learning_rate(self, steps:int):
        super().update_learning_rate(steps)
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'view_spec_embed':
                lr = self.view_spec_embed_scheduler_args(steps)
                param_group['lr'] = lr

    def cat_tensors_to_optimizer(self, tensors_dict:Dict[str, Tensor], excludes:List[str]=[]) -> Dict[str, Tensor]:
        return super().cat_tensors_to_optimizer(tensors_dict, excludes + ['view_spec_embed'])

    def prune_optimizer(self, mask:Tensor, excludes:List[str]=[]) -> Dict[str, Tensor]:
        return super().prune_optimizer(mask, excludes + ['view_spec_embed'])
