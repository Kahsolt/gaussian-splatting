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

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from modules.camera import Camera
from modules.layers import ColorMLP, Embedding
from modules.morphs.cd_mul_gs.model import GaussianModel as GaussianModelBase
from modules.utils.modeling_utils import *
from modules.utils.training_utils import *

from .hparam import HyperParams


class GaussianModel(GaussianModelBase):

    ''' modified cd_mul_gs, seperated feature bank '''

    def __init__(self, hp:HyperParams):
        super().__init__(hp)

        self.hp: HyperParams

        # networks
        self.view_embedding: Embedding = nn.Identity()

        self.rgb_mlp  = ColorMLP(hp.rgb_dim,                              hp.hidden_dim)
        self.gate_mlp = ColorMLP(hp.gate_dim + hp.view_embedding_dim + 4, hp.hidden_dim, out_dim=1)
        print(self.rgb_mlp)
        print(self.gate_mlp)

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

        rgb = self.rgb_mlp(feats[:, :hp.rgb_dim])
        gate = self.gate_mlp(torch.cat([feats[:, -hp.gate_dim:], view_embed, view_feat], -1))
        gate_ex = torch.cat([gate, torch.zeros_like(gate), torch.zeros_like(gate)], dim=-1)  # make 3-ch pseudo-color
        return rgb, gate_ex
