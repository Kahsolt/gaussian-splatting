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
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser, Namespace
from typing import Callable, Any


# priority: cmdline args > saved/reloaded hparams.json > defaults
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
        return self.__dict__.get(name)

    def __setattr__(self, name:str, value:Any):
        self.__dict__[name] = value

    def send_to(self, parser:ArgumentParser, fill_default:bool=True):
        for key, val in vars(self).items():
            if isinstance(val, Callable): continue
            if key in ['morph']: continue
            args = (f'--{key}',) + ((f'-{key[0]}',) if key in self.SHORT_HANDS else tuple())
            kwargs = {'default': val} if fill_default else {}
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


def get_combined_args(parser:ArgumentParser) -> Namespace:
    cmdline_string = sys.argv[1:]
    cfgfile_string = 'Namespace()'
    args_cmdline, _ = parser.parse_known_args(cmdline_string)

    try:
        cfg_fp = os.path.join(args_cmdline.model_path, 'cfg_args')
        print('Looking for config file in', cfg_fp)
        with open(cfg_fp) as cfg_file:
            print(f'Config file found: {cfg_fp}')
            cfgfile_string = cfg_file.read()
    except TypeError:
        print('Config file not found at')
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
