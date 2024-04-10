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

from modules.hparam import HyperParams as HyperParamsBase


class HyperParams(HyperParamsBase):

    def __init__(self):
        super().__init__()

        ''' Model '''
        self.feat_dim = 48
        self.add_view = True

        ''' Optimizer '''
        self.feature_lr = 0.0075

        self.mlp_color_lr_init = 0.008
        self.mlp_color_lr_final = 0.00005
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = 30_000
