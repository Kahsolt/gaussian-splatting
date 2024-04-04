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
from pathlib import Path
from random import shuffle
from datetime import datetime
from argparse import ArgumentParser, Namespace
from typing import List, Tuple, Callable

import torch
from torch import Tensor
from tqdm import tqdm

from modules.arguments import ModelParams, PipelineParams, OptimizationParams
from modules.scene import Scene, Camera, GaussianModel, render
from modules import network_gui
from modules.utils.loss_utils import l1_loss, ssim, psnr, nerfw_loss
from modules.utils.general_utils import safe_state

TENSORBOARD_FOUND = False
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    try:
        from tensorboardX import SummaryWriter
        TENSORBOARD_FOUND = True
    except ImportError: pass
except ImportError:
    class SummaryWriter:
        def __init__(self, *args, **kwargs): pass
        def add_scalar(self, *args, **kwargs): pass
        def add_scalars(self, *args, **kwargs): pass
        def add_tensor(self, *args, **kwargs): pass
        def add_histogram(self, *args, **kwargs): pass
        def add_image(self, *args, **kwargs): pass
        def add_images(self, *args, **kwargs): pass
        def add_figure(self, *args, **kwargs): pass


def init_log_summary(args:Namespace, mp:ModelParams, pp:PipelineParams, op:OptimizationParams) -> SummaryWriter:
    # Set up output folder
    if not mp.model_path:
        exp_name = os.getenv('OAR_JOB_ID')
        if exp_name is None:
            exp_name = str(datetime.now()).replace(' ', 'T').replace(':', '-') + '_' +  Path(mp.source_path).stem
        mp.model_path = os.path.join('output', exp_name)

    print(f'>> Output folder: {mp.model_path}')
    os.makedirs(mp.model_path, exist_ok=True)
    with open(os.path.join(mp.model_path, 'cfg_args'), 'w') as fh:      # SIBR_gaussianViewer_app need this, DO NOT touch!!
        fh.write(str(Namespace(**vars(mp))))
    with open(os.path.join(mp.model_path, 'config.json'), 'w') as fh:   # we save all hparams in another file :)
        json.dump({
            'cmd': ' '.join(sys.argv),
            'ts': str(datetime.now()),
            'args': vars(args),
            'mp': vars(mp),
            'pp': vars(pp),
            'op': vars(op),
        }, fh, indent=2, ensure_ascii=False)

    # Create Tensorboard writer
    if not TENSORBOARD_FOUND:
        print('>> [warn] Tensorboard not available, ignore SummaryWriter progress')
    return SummaryWriter(mp.model_path)


def add_log_summary_test_step(sw:SummaryWriter, steps:int, test_steps:List[int], loss_fn:Callable, scene:Scene, renderFunc:Callable, renderArgs:Tuple):
    sw.add_histogram('scene/opacity_histogram', scene.gaussians.opacity, global_step=steps)
    sw.add_scalar('total_points', scene.gaussians.xyz.shape[0], global_step=steps)

    validation_configs = (
        {'name': 'test',  'cameras': scene.get_test_cameras()},
        {'name': 'train', 'cameras': [scene.get_train_cameras()[idx % len(scene.get_train_cameras())] for idx in range(5, 30, 5)]},
    )
    for config in validation_configs:
        if not config['cameras'] or not len(config['cameras']): continue

        l1_test, psnr_test, total = 0.0, 0.0, 0
        for idx, viewpoint in enumerate(config['cameras']):
            viewpoint: Camera
            render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
            render = torch.clamp(render_pkg['render'], 0.0, 1.0)
            gt = torch.clamp(viewpoint.image.to('cuda'), 0.0, 1.0)
            if idx < 5:
                sw.add_images(config['name'] + f'_view_{viewpoint.image_name}/render', render, global_step=steps, dataformats='CHW')
                if steps == test_steps[0]:
                    sw.add_images(config['name'] + f'_view_{viewpoint.image_name}/gt', gt, global_step=steps, dataformats='CHW')
            l1_test += loss_fn(render, gt).mean()
            psnr_test += psnr(render, gt).mean()
            total += 1
        l1_test /= total
        psnr_test /= total
        print(f"\n[ITER {steps}] Evaluating {config['name']}: L1 {l1_test}, PSNR {psnr_test}")

        sw.add_scalar(config['name'] + 'l1', l1_test, global_step=steps)
        sw.add_scalar(config['name'] + 'psnr', psnr_test, global_step=steps)


def render_network_gui(pipe:PipelineParams, opt:OptimizationParams, scene:Scene, steps:int):
    if network_gui.conn == None:
        network_gui.try_connect()
    while network_gui.conn != None:
        try:
            net_image_bytes = None
            custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
            if custom_cam != None:
                net_image = render(custom_cam, scene.gaussians, pipe, scene.background, scaling_modifer)['render']
                net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
            network_gui.send(net_image_bytes, args.source_path)
            if do_training and (steps < int(opt.iterations) or not keep_alive):
                break
        except Exception as e:
            network_gui.conn = None


def train(args:Namespace, mp:ModelParams, pp:PipelineParams, op:OptimizationParams):
    ''' Log & Bookkeep '''
    sw = init_log_summary(args, mp, pp, op)
    start_steps = 0
    loss_ema_for_log: float = 0.0
    ts_start = torch.cuda.Event(enable_timing=True)
    ts_end = torch.cuda.Event(enable_timing=True)

    ''' Model '''
    scene = Scene(mp)
    gaussians: GaussianModel = scene.gaussians
    gaussians.setup_training(op)
    if args.start_checkpoint:
        start_steps = scene.load_checkpoint(args.start_checkpoint)

    ''' Train '''
    viewpoint_stack: List[Camera] = None
    start_steps += 1
    pbar = tqdm(range(start_steps, op.iterations + 1), desc='Training progress')
    for steps in range(start_steps, op.iterations + 1):
        # Debug
        render_network_gui(pp, op, scene, steps)
        if steps == args.debug_from: pp.debug = True

        ts_start.record()

        # Decay learning rate
        gaussians.update_learning_rate(steps)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if steps % 1000 == 0:
            gaussians.oneup_SH_degree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.get_train_cameras().copy()
            shuffle(viewpoint_stack)
        viewpoint_cam: Camera = viewpoint_stack.pop()

        # Render
        render_pkg = render(viewpoint_cam, gaussians, pp, scene.random_background() if op.random_background else scene.background)
        # [C=3, H=545, W=980]
        image = render_pkg['render']
        # [P=182686, pos=3]
        viewspace_point_tensor = render_pkg['viewspace_points']
        # [P=182686]
        visibility_filter = render_pkg['visibility_filter']
        # [P=182686], int32
        radii = render_pkg['radii']

        # Loss mask
        loss_mask = 1.0
        if op.m_loss_weight and render_pkg.get('weight_map') is not None:
            weight_map = render_pkg['weight_map']
            loss_mask = weight_map.unsqueeze(0).expand(image.shape[0], -1, -1)
        if (op.m_loss_depth or op.m_loss_depth_reverse) and render_pkg.get('depth_map') is not None:
            depth_map = render_pkg['depth_map']
            depth_map_act = torch.sigmoid(depth_map - depth_map.mean())
            if op.m_loss_depth:            # punish more on near things
                depth_map_act = 1 - depth_map_act
            if op.m_loss_depth_reverse:    # punish more on far things
                pass
            loss_mask = depth_map_act.unsqueeze(0).expand(image.shape[0], -1, -1)
        if op.m_loss_importance and render_pkg.get('importance_map') is not None:
            importance_map = render_pkg['importance_map']
            loss_mask = torch.exp2(importance_map).unsqueeze(0).expand(image.shape[0], -1, -1)

        # Loss
        gt_image = viewpoint_cam.image.cuda()
        if op.nerfw_loss:
            loss = nerfw_loss(image, gt_image, loss_mask)
        else:
            Ll1 = l1_loss(image, gt_image, reduction='none') * loss_mask
            Lssim = ssim(image, gt_image, reduction='none') * loss_mask
            loss: Tensor = (1.0 - op.lambda_dssim) * Ll1.mean() + op.lambda_dssim * (1.0 - Lssim.mean())
        loss.backward()

        ts_end.record()

        with torch.no_grad():
            # Progress bar
            loss_ema_for_log = 0.4 * loss.item() + 0.6 * loss_ema_for_log
            if steps % 10 == 0:
                pbar.set_postfix({'Loss': f'{loss_ema_for_log:.{7}f}', 'n_pts': gaussians.xyz.shape[0]})
                pbar.update(10)
            if steps == op.iterations:
                pbar.close()

            # Log and save
            sw.add_scalar('train_loss_patches/l1_loss', Ll1.mean().item(), global_step=steps)
            sw.add_scalar('train_loss_patches/total_loss', loss.item(), global_step=steps)
            sw.add_scalar('iter_time', ts_start.elapsed_time(ts_end), global_step=steps)
            if steps in args.test_iterations:
                torch.cuda.empty_cache()
                add_log_summary_test_step(sw, steps, args.test_iterations, l1_loss, scene, render, (pp, scene.background))
                torch.cuda.empty_cache()

            # Densification
            if steps < op.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if steps > op.densify_from_iter and steps % op.densification_interval == 0:
                    size_threshold = 20 if steps > op.opacity_reset_interval else None
                    gaussians.densify_and_prune(op.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if steps % op.opacity_reset_interval == 0 or (mp.white_background and steps == op.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if steps < op.iterations:
                gaussians.optimizer_step()

            # Save
            if steps in args.save_iterations:
                print(f'[ITER {steps}] Saving Gaussians')
                scene.save_gaussian(steps)
            if steps in args.checkpoint_iterations:
                print(f'[ITER {steps}] Saving Checkpoint')
                scene.save_checkpoint(steps)


if __name__ == '__main__':
    parser = ArgumentParser(description='Training script parameters')
    mp = ModelParams(parser)
    pp = PipelineParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--test_iterations', nargs='+', type=int, default=[7_000, 30_000])
    parser.add_argument('--save_iterations', nargs='+', type=int, default=[7_000, 30_000])
    parser.add_argument('--checkpoint_iterations', nargs='+', type=int, default=[7_000, 30_000])
    parser.add_argument('--start_checkpoint', type=str)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print('Hparams:', vars(args))

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    train(args, mp.extract(args), pp.extract(args), op.extract(args))

    # All done
    print('\nTraining complete.')
