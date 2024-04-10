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

from argparse import Namespace
from modules.hparam import HyperParams as HyperParamsBase


class HyperParams(HyperParamsBase):

    def __init__(self):
        super().__init__()

        ''' Model '''
        self.per_feat_dim = 16
        self.hidden_dim = 32
        self.add_view = True
        self.sun_vis_embedding_dim = 16
        self.sun_shade_embedding_dim = 16
        self.sky_shade_embedding_dim = 16
        self.reflectance_embedding_dim = 16

        ''' Optimizer '''
        self.feature_lr = 0.0075

        self.mlp_color_lr_init = 0.008
        self.mlp_color_lr_final = 0.00005
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = 30_000

        self.embedding_lr_init = 0.05
        self.embedding_lr_final = 0.0005
        self.embedding_lr_delay_mult = 0.01
        self.embedding_lr_max_steps = 30_000

    def extract_from(self, args:Namespace):
        super().extract_from(args)
        self.feat_dim = self.per_feat_dim * 4
