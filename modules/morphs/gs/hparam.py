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
        self.sh_degree = 3

        ''' Pipeline '''
        self.convert_SHs_python = False
