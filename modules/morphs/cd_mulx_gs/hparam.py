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
from modules.morphs.cd_mul_gs.hparam import HyperParams as HyperParamsBase


class HyperParams(HyperParamsBase):

    def __init__(self):
        super().__init__()

        ''' Model '''
        self.feat_dim = 48
        self.rgb_dim = 40
        self.gate_dim = 8

    def extract_from(self, args:Namespace):
        super().extract_from(args)
        assert self.feat_dim == self.rgb_dim + self.gate_dim
