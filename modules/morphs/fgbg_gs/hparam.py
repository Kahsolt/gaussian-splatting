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

from modules.hparam import HyperParams_Neural


class HyperParams(HyperParams_Neural):

    def __init__(self):
        super().__init__()

        ''' Model '''
        self.feat_dim = 48
        self.hidden_dim = 32
        self.appear_embedding_dim = 16
        self.occlus_embedding_dim = 16

        ''' Optimizer '''
        self.feature_lr = 0.0075
        self.importance_lr = 0.05

        self.mlp_lr_init = 0.008
        self.mlp_lr_final = 0.00005
        self.mlp_lr_delay_mult = 0.01
        self.mlp_lr_max_steps = 30_000

        self.embedding_lr_init = 0.05
        self.embedding_lr_final = 0.0005
        self.embedding_lr_delay_mult = 0.01
        self.embedding_lr_max_steps = 30_000

        ''' Pipeline '''
        self.rasterizer = 'ours'
