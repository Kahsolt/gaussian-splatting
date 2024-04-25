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
from modules.morphs.mlp_gs.hparam import HyperParams as HyperParamsBase


class HyperParams(HyperParamsBase):

    ''' based on mlp-gs '''

    def __init__(self):
        super().__init__()

        ''' Model '''
        self.split_method = 'fft'
        self.blur_r = 5         # for unsharp_mask
        self.blur_s = 1.1       # for unsharp_mask
        self.n_freqs = 2        # for fft, svd
        self.scale_w = 0.05     # for fft, svd
        self.wavlet = 'db3'     # for dwt
        self.padding = 'zero'   # for dwt

        self.sub_gauss_init = 'copy'

    def get_split_freqs_kwargs(self):
        return {
            'n_freqs': self.n_freqs,
            'scale_w': self.scale_w,
            'r': self.blur_r,
            's': self.blur_s,
            'wavlet': self.wavlet,
            'padding': self.padding,
        }

    def extract_from(self, args: Namespace):
        super().extract_from(args)
        assert self.split_method in ['unsharp_mask', 'fft', 'svd', 'dwt']
        assert self.sub_gauss_init in ['copy', 'uniform_sparse']

        if self.split_method == 'unsharp_mask':
            assert self.n_freqs == 2
