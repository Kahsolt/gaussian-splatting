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

import sys
import random
from datetime import datetime

import torch
import numpy as np


def safe_state(silent:bool):
    _stdout = sys.stdout

    class F:
        def __init__(self, silent):
            self.silent = silent
        def write(self, x:str):
            if not self.silent:
                if x.endswith('\n'):
                    _stdout.write(x.replace('\n', f" [{datetime.now().strftime('%d/%m %H:%M:%S')}]\n"))
                else:
                    _stdout.write(x)
        def flush(self):
            _stdout.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device('cuda:0'))
