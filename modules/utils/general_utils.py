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
from pathlib import Path
from datetime import datetime
from typing import Tuple

import torch
from torch import Tensor
import numpy as np

BASE_PATH = Path(__file__).parent.parent.parent
DATA_PATH = BASE_PATH / 'data'

RASTERIZER_PROVIDERS = [
    'original',
    'depth',
    'ours',
    'ours-dev',
]


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


class ImageState:

    def __init__(self, buffer:Tensor, size:Tuple[int, int], align:int=128):
        H, W = size
        N = H * W
        offset = 0
        buffer = buffer.cpu().numpy()

        def next_offset() -> int:
            nonlocal offset
            while offset % align:
                offset += 1

        next_offset()
        final_T = torch.frombuffer(memoryview(buffer[offset:offset+4*N]), dtype=torch.float32).reshape((H, W))
        next_offset()
        n_contrib = torch.frombuffer(memoryview(buffer[offset:offset+4*N]), dtype=torch.int32).reshape((H, W))
        next_offset()
        ranges = torch.frombuffer(memoryview(buffer[offset:offset+8*N]), dtype=torch.int32).reshape((H, W, 2))

        self._final_T = final_T      # float, 4 bytes
        self._n_contrib = n_contrib  # uint32_t, 4 bytes
        self._ranges = ranges        # uint2, 8 bytes

    @property
    def final_T(self): return self._final_T
    @property
    def n_contrib(self): return self._n_contrib
    @property
    def ranges(self): return self._ranges
