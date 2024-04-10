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

import warnings ; warnings.filterwarnings(action='ignore', category=UserWarning)

from pprint import pprint
from argparse import ArgumentParser
from importlib import import_module
from traceback import print_exc

import torch
from modules.hparam import HyperParams
from modules.morphs import MODEL_MORPHS
from modules.utils.general_utils import safe_state


if __name__ == '__main__':
    parser = ArgumentParser(description='Training script parameters')
    parser.add_argument('-M', '--morph', default='gs', choices=MODEL_MORPHS)
    parser.add_argument('--network_gui', action='store_true')
    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--nolog', action='store_true', help='no tensorboard logs')
    args, _ = parser.parse_known_args()

    # Initialize system state (RNG)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # Resolve real implemetations
    try:
        try:
            mod = import_module(f'modules.morphs.{args.morph}.hparam')
            HyperParams_cls = getattr(mod, 'HyperParams')
        except AttributeError:
            HyperParams_cls = HyperParams
            print('>> no overrided HyperParams class found, use default')
        try:
            mod = import_module(f'modules.morphs.{args.morph}.train')
            train = getattr(mod, 'train')
        except Exception as e:
            raise NotImplementedError('missing implementation of train.py script') from e
    except: print_exc()

    # Go on routine
    hp = HyperParams_cls()
    hp.send_to(parser)
    args = parser.parse_args()
    hp.extract_from(args)
    print('Training:', hp.model_path)

    print('Hparams:')
    pprint(vars(hp))
    train(args, hp)

    # Done
    print()
    print('Training complete.')
