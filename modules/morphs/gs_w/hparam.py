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
        self.add_view = False
        self.use_view_emb = False           # 使用视角编码MLP
        self.view_emb_dim = 16              # 视角编码深度
        self.appearance_dim = 0             # 外观嵌入深度
        self.occlusion_dim = 0              # 遮挡嵌入深度
        self.add_view_emb_to_color = False  # 在外观编码MLP中引入视角编码
        self.add_view_emb_to_occlu = False  # 在遮挡编码MLP中引入视角编码

        ''' Optimizer '''
        self.feature_lr = 0.0075
        self.importance_lr = 0.002

        self.mlp_view_lr_init = 0.008
        self.mlp_view_lr_final = 0.00005
        self.mlp_view_lr_delay_mult = 0.01
        self.mlp_view_lr_max_steps = 30_000
        
        self.mlp_color_lr_init = 0.008
        self.mlp_color_lr_final = 0.00005
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = 30_000
        
        self.mlp_occlu_lr_init = 0.008
        self.mlp_occlu_lr_final = 0.00005
        self.mlp_occlu_lr_delay_mult = 0.01
        self.mlp_occlu_lr_max_steps = 30_000
        
        self.appearance_lr_init = 0.05
        self.appearance_lr_final = 0.0005
        self.appearance_lr_delay_mult = 0.01
        self.appearance_lr_max_steps = 30_000
        
        self.occlusion_lr_init = 0.05
        self.occlusion_lr_final = 0.0005
        self.occlusion_lr_delay_mult = 0.01
        self.occlusion_lr_max_steps = 30_000

        ''' Pipeline '''
        self.rasterizer = 'ours'
