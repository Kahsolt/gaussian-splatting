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

from argparse import ArgumentParser
from importlib import import_module
from traceback import print_exc

from modules.hparam import HyperParams, get_combined_args
from modules.scene import Scene
from modules.utils.general_utils import safe_state


def get_ckpt_morph() -> str:
    parser = ArgumentParser()
    hp = HyperParams()
    hp.send_to(parser)
    cmd_args, _ = parser.parse_known_args()
    args = get_combined_args(cmd_args, hp)
    hp.extract_from(args)
    return hp.morph


if __name__ == '__main__':
    parser = ArgumentParser(description='Testing script parameters')
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--skip_test', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--debug_render_set', action='store_true')
    args, _ = parser.parse_known_args()

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Recover -M/--morph at training
    morph = get_ckpt_morph()
    print('>> morph:', morph)

    # Resolve real implemetations
    try:
        try:
            mod = import_module(f'modules.morphs.{morph}.hparam')
            HyperParams_cls = getattr(mod, 'HyperParams')
        except AttributeError:
            HyperParams_cls = HyperParams
            print('>> no overrided HyperParams class found, use default')
        try:
            mod = import_module(f'modules.morphs.{morph}.scene')
            Scene_cls = getattr(mod, 'Scene')
        except (ModuleNotFoundError, AttributeError):
            Scene_cls = Scene
            print('>> no overrided Scene class found, use default')
        try:
            mod = import_module(f'modules.morphs.{morph}.render')
            render_set = getattr(mod, 'render_set_debug' if args.debug_render_set else 'render_set')
        except Exception as e:
            raise NotImplementedError('missing implementation of render.py script') from e
    except: print_exc()

    # Go on routine
    hp = HyperParams_cls()
    hp.send_to(parser)
    cmd_args, _ = parser.parse_known_args()
    args = get_combined_args(cmd_args, hp)
    hp.extract_from(args)
    print('Rendering:', hp.model_path)

    scene = Scene_cls(hp)
    if not args.skip_train: render_set(scene, 'train')
    if not args.skip_test:  render_set(scene, 'test')

    # Done
    print()
    print('Rendering complete.')
