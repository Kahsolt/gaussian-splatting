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
from modules.hparam import HyperParams_SH


class HyperParams(HyperParams_SH):

    def __init__(self):
        super().__init__()

        ''' Model '''
        self.sanitize_init_pcd = False
        self.sanitize_load_gauss = False

        ''' Optimizer '''
        self.importance_lr = 0.005

        self.m_loss_weight = False
        self.m_loss_depth = False
        self.m_loss_depth_reverse = False
        self.m_loss_importance = False
        self.nerfw_loss = False

        ''' Pipeline '''
        self.rasterizer = 'ours'
        self.limit_n_contrib = -1

    def extract_from(self, args: Namespace):
        super().extract_from(args)
        if self.nerfw_loss:
            assert self.rasterizer == 'ours', '--nerfw_loss only works with --rasterizer ours'
