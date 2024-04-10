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

import os
import sys
import json
from datetime import datetime
from argparse import Namespace

import numpy as np

from modules.hparam import HyperParams

TENSORBOARD_FOUND = False
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    try:
        from tensorboardX import SummaryWriter
        TENSORBOARD_FOUND = True
    except ImportError: pass
except ImportError: pass
if not TENSORBOARD_FOUND:
    print('>> [Warn] Tensorboard not available, ignore SummaryWriter progress')


class NoSummaryWriter:
    def __init__(self, *args, **kwargs): pass
    def add_scalar(self, *args, **kwargs): pass
    def add_scalars(self, *args, **kwargs): pass
    def add_tensor(self, *args, **kwargs): pass
    def add_histogram(self, *args, **kwargs): pass
    def add_image(self, *args, **kwargs): pass
    def add_images(self, *args, **kwargs): pass
    def add_figure(self, *args, **kwargs): pass


def init_log(args:Namespace, hp:HyperParams) -> SummaryWriter:
    if args.nolog: return NoSummaryWriter()

    print(f'>> Output folder: {hp.model_path}')
    os.makedirs(hp.model_path, exist_ok=True)
    with open(os.path.join(hp.model_path, 'cfg_args'), 'w') as fh:      # SIBR_gaussianViewer_app need this, DO NOT touch!!
        fh.write(str(Namespace(**vars(hp))))
    with open(os.path.join(hp.model_path, 'config.json'), 'w') as fh:   # we save all hparams in another file :)
        json.dump({
            'cmd': ' '.join(sys.argv),
            'ts': str(datetime.now()),
            'args': vars(args),
            'hp': vars(hp),
        }, fh, indent=2, ensure_ascii=False)

    # Create Tensorboard writer
    return SummaryWriter(hp.model_path) if TENSORBOARD_FOUND else NoSummaryWriter()


def make_expon_lr_func_args(hp:HyperParams, prefix:str):
    kwargs = {
        'lr_init':        getattr(hp, f'{prefix}_lr_init'),
        'lr_final':       getattr(hp, f'{prefix}_lr_final'),
        'lr_delay_steps': getattr(hp, f'{prefix}_lr_delay_steps', None),
        'lr_delay_mult':  getattr(hp, f'{prefix}_lr_delay_mult', None),
        'max_steps':      getattr(hp, f'{prefix}_lr_max_steps', None),
    }
    return {k: v for k, v in kwargs.items() if v is not None}

def get_expon_lr_func(lr_init:float, lr_final:float, lr_delay_steps:int=0, lr_delay_mult:float=1.0, max_steps:int=1000000):
    '''
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    '''

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper
