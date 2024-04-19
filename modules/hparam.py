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
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser, Namespace
from typing import Callable, Any

WARN_GETATTR = False
WARN_SETATTR = False


class HyperParams:

    # initila letter must be unique ↓↓↓
    SHORT_HANDS = [
        'source_path',
        'model_path',
        #'resolution'
        'images',
        'white_background',
        'rasterizer',
    ]

    def __init__(self):
        ''' Meta '''
        self.morph = 'gs'

        ''' Data '''
        self.source_path = 'data/tandt/train'
        self.images = 'images'
        self.limit = 99999          # limit sample count
        self.eval = False           # split train/test
        self.resolution = -1

        ''' Optimizer '''
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002

        ''' Checkpoint '''
        self.model_path = ''
        self.test_iterations = [7_000, 30_000]
        self.save_iterations = [7_000, 30_000]
        self.checkpoint_iterations = [30_000]
        self.load = ''          # resume from checkpoint *.pth
        self.load_iter = -1     # load guassian from <steps>.ply

        ''' Pipeline '''
        self.rasterizer = 'original'
        self.white_background = False
        self.random_background = False
        self.compute_cov3D_python = False
        self.debug = False      # debug rasterizer

    def __getattr__(self, name:str) -> Any:
        global WARN_GETATTR
        if name not in dir(self):
            if not WARN_GETATTR:
                WARN_GETATTR = True
                print('>> HyperParams.__getattr__ is deprecated and going to be removed')
            print(f'>> calling HyperParams.__getattr__({name})')
        return self.__dict__.get(name)

    def __setattr__(self, name:str, value:Any):
        global WARN_SETATTR
        if name not in dir(self):
            if not WARN_SETATTR:
                WARN_SETATTR = False
                print('>> HyperParams.__setattr__ is deprecated and going to be removed')
            print(f'>> calling HyperParams.__setattr__({name}, {value})')
        self.__dict__[name] = value

    def send_to(self, parser:ArgumentParser):
        for key, val in vars(self).items():
            if isinstance(val, Callable): continue
            if key in ['morph']: continue
            args = (f'--{key}',) + ((f'-{key[0]}',) if key in self.SHORT_HANDS else tuple())
            kwargs = {'default': val}
            tval = type(val)
            if tval == bool:
                kwargs.update({'action': 'store_true'})
            elif tval in [list, tuple]:
                kwargs.update({'nargs': '+', 'type': type(val[0])})
            else:
                kwargs.update({'type': tval})
            parser.add_argument(*args, **kwargs)

    def extract_from(self, args:Namespace):
        # allow override by cmd_args
        self_vars = vars(self)
        for key, val in vars(args).items():
            if isinstance(val, Callable): continue
            if key in self_vars:
                setattr(self, key, val)
        # sanity check
        from modules.morphs import MODEL_MORPHS
        assert self.morph in MODEL_MORPHS, (self.morph, MODEL_MORPHS)
        from modules.utils.general_utils import RASTERIZER_PROVIDERS
        assert self.rasterizer in RASTERIZER_PROVIDERS, (self.rasterizer, RASTERIZER_PROVIDERS)
        assert os.path.exists(self.source_path)
        # postfix
        self.source_path = os.path.abspath(self.source_path)
        self.checkpoint_iterations.append(self.iterations)
        if not self.model_path:
            exp_name = os.getenv('EXP_NAME', os.getenv('OAR_JOB_ID'))
            if exp_name is None:
                time_str = str(datetime.now()).replace(' ', 'T').replace(':', '-').split('.')[0]    # 2024-04-11T17-22-33
                dataset_str = Path(self.source_path).stem
                exp_name = f'{time_str}_{dataset_str}_M={self.morph}'
            self.model_path = os.path.join('output', exp_name)


class HyperParams_SH(HyperParams):

    def __init__(self):
        super().__init__()

        ''' Model '''
        self.sh_degree = 3

        ''' Pipeline '''
        self.convert_SHs_python = False


class HyperParams_Neural(HyperParams):

    def __init__(self):
        super().__init__()

        ''' Model '''
        self.feat_dim = 48


# priority: cmd_args > cfg_args > default_hp
def get_combined_args(cmd_args:ArgumentParser, default_hp:HyperParams) -> Namespace:
    cfg_fp = os.path.join(cmd_args.model_path, 'cfg_args')
    try:
        with open(cfg_fp) as fh:
            cfgfile_string = fh.read()
        print(f'Load config from: {cfg_fp}')
    except:
        cfgfile_string = 'Namespace()'
        print(f'[Warn] missing config file at {cfg_fp}')
    cfg_args = eval(cfgfile_string)

    merged_dict = vars(default_hp).copy()
    # cfg_args > default_hp
    for k, v in vars(cfg_args).items():
        if v is not None:
            merged_dict[k] = v
    # cmd_args > cfg_args
    for k, v in vars(cmd_args).items():
        if v != getattr(default_hp, k):
            merged_dict[k] = v
    return Namespace(**merged_dict)
