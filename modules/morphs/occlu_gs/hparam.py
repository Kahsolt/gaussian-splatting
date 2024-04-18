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

from modules.morphs.mlp_gs.hparam import HyperParams as HyperParamsBase


class HyperParams(HyperParamsBase):
  
  def __init__(self):
    super().__init__()

    ''' Model '''
    # FIXME: =H*W, this is stupid large! 
    # use a generative model to generative it from view_feats [P, C=3]?
    self.view_embed_ch = 1      # 1 or 3
    self.view_embed_dim = 545 * 980 * self.view_embed_ch

    ''' Optimizer '''
    self.view_spec_embed_lr_init = 0.0075
    self.view_spec_embed_lr_final = 0.0001
    self.view_spec_embed_lr_delay_mult = 0.01
    self.view_spec_embed_lr_max_steps = 30_000
