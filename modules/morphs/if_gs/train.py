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

from pathlib import Path
from random import shuffle
from argparse import Namespace
from typing import List, Dict, Callable

import torch
from torch import Tensor
from torchvision.utils import save_image
from tqdm import tqdm

from modules import network_gui
from modules.utils.loss_utils import l1_loss, ssim, psnr
from modules.utils.training_utils import init_log

from .hparam import HyperParams
from .scene import Scene
from .camera import Camera
from .model import SingleFreqGaussianModel
from .render import render


def network_gui_handle(render_func:Callable, scene:Scene, steps:int):
    if network_gui.conn == None:
        network_gui.try_connect()
    while network_gui.conn != None:
        try:
            hp = scene.hp
            net_image_bytes = None
            custom_cam, do_training, _, hp.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
            if custom_cam != None:
                rendered = render_func(scene.cur_gaussian, custom_cam, scene.background, scaling_modifer)['render']
                net_image_bytes = memoryview((torch.clamp(rendered, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
            network_gui.send(net_image_bytes, hp.source_path)
            if do_training and (steps < int(hp.iterations) or not keep_alive):
                break
        except Exception as e:
            network_gui.conn = None


def train(args:Namespace, hp:HyperParams):
    ''' Log & Bookkeep '''
    if args.network_gui: network_gui.init(args.ip, args.port)
    sw = init_log(args, hp)
    start_steps = 0
    loss_ema_for_log: float = 0.0
    ts_start = torch.cuda.Event(enable_timing=True)
    ts_end = torch.cuda.Event(enable_timing=True)

    ''' Model '''
    scene = Scene(hp)
    for freq_idx in scene.all_gaussians.keys():
        scene.activate_gaussian(freq_idx)
        gaussians: SingleFreqGaussianModel = scene.cur_gaussian
        gaussians.setup_training()
        if hp.load: start_steps = scene.load_checkpoint(hp.load)

        ''' Train '''
        viewpoint_stack: List[Camera] = None
        start_steps += 1
        pbar = tqdm(range(start_steps, hp.iterations + 1), desc='Training progress')
        for steps in range(start_steps, hp.iterations + 1):
            # Debug
            if steps == args.debug_from: hp.debug = True
            if args.network_gui: network_gui_handle(render, scene, steps)

            ts_start.record()

            # Decay learning rate
            gaussians.update_learning_rate(steps)

            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.get_train_cameras().copy()
                shuffle(viewpoint_stack)
            viewpoint_cam = viewpoint_stack.pop()

            # Render
            render_pkg = render(gaussians, viewpoint_cam, scene.random_background() if hp.random_background else scene.background)
            image = render_pkg['render']                        # [C=3, H=545, W=980]
            viewspace_points = render_pkg['viewspace_points']   # [P=182686, pos=3]
            visibility_filter = render_pkg['visibility_filter'] # [P=182686]
            radii = render_pkg['radii']                         # [P=182686], int32

            # Loss
            gt_image = viewpoint_cam.image(freq_idx).cuda()
            Ll1 = l1_loss(image, gt_image)
            Lssim = ssim(image, gt_image)
            loss: Tensor = (1.0 - hp.lambda_dssim) * Ll1 + hp.lambda_dssim * (1.0 - Lssim)
            loss.backward()

            ts_end.record()

            with torch.no_grad():
                # Progress bar
                loss_ema_for_log = 0.4 * loss.item() + 0.6 * loss_ema_for_log
                if steps % 10 == 0:
                    pbar.set_postfix({'loss': f'{loss_ema_for_log:.7f}', 'n_pts': gaussians.xyz.shape[0]})
                    pbar.update(10)
                if steps >= hp.iterations:
                    pbar.close()

                # Peep middle results
                if steps % 100 == 0:
                    save_dir = mkdir(Path(scene.model_path) / 'look_up' / f'freq_{freq_idx}', parents=True)
                    rendered_cat = torch.cat([image, gt_image], -1)
                    save_image(rendered_cat, save_dir / f'{steps:05d}-{viewpoint_cam.uid}.png')

                # Log and save
                sw.add_scalar('train_loss_patches/l1_loss', Ll1.mean().item(), global_step=steps)
                sw.add_scalar('train_loss_patches/total_loss', loss.item(), global_step=steps)
                sw.add_scalar('iter_time', ts_start.elapsed_time(ts_end), global_step=steps)
                sw.add_scalar('n_points', gaussians.n_points, global_step=steps)
                if steps in hp.test_iterations:
                    sw.add_histogram('scene/opacity_histogram', gaussians.opacity, global_step=steps)

                    validation_configs: Dict[str, List[Camera]] = {
                        'test': scene.get_test_cameras(),
                        'train': [scene.get_train_cameras()[idx % len(scene.get_train_cameras())] for idx in range(5, 30, 5)],
                    }

                    torch.cuda.empty_cache()
                    for split, cameras in validation_configs.items():
                        if not cameras or not len(cameras): continue

                        l1_test, psnr_test, total = 0.0, 0.0, 0
                        for idx, viewpoint in enumerate(cameras):
                            render_pkg = render(gaussians, viewpoint, scene.background)
                            rendered = torch.clamp(render_pkg['render'], 0.0, 1.0)
                            gt = viewpoint.image(freq_idx).cuda()
                            if idx < 5:
                                sw.add_images(f'{split}_view_{viewpoint.image_name}/render', rendered, global_step=steps, dataformats='CHW')
                                if steps == hp.test_iterations[0]:
                                    sw.add_images(f'{split}_view_{viewpoint.image_name}/gt', gt, global_step=steps, dataformats='CHW')
                            l1_test += l1_loss(rendered, gt).mean()
                            psnr_test += psnr(rendered, gt).mean()
                            total += 1
                        l1_test /= total
                        psnr_test /= total
                        print(f'[ITER {steps}] Evaluating {split}: L1 {l1_test}, PSNR {psnr_test}')

                        sw.add_scalar(f'{split}/l1', l1_test, global_step=steps)
                        sw.add_scalar(f'{split}/psnr', psnr_test, global_step=steps)

                    torch.cuda.empty_cache()

                # Densification
                if steps < hp.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_points, visibility_filter)

                    if steps > hp.densify_from_iter and steps % hp.densification_interval == 0:
                        size_threshold = 20 if steps > hp.opacity_reset_interval else None
                        gaussians.densify_and_prune(hp.densify_grad_threshold, 0.05, scene.cameras_extent, size_threshold)

                    if steps % hp.opacity_reset_interval == 0 or (hp.white_background and steps == hp.densify_from_iter):
                        gaussians.reset_opacity()

                # Optimizer step
                if steps < hp.iterations:
                    gaussians.optimizer_step()

                # Save
                if steps in hp.save_iterations:
                    print(f'[ITER {steps}] Saving Gaussians')
                    scene.save_gaussian(steps)
                if steps in hp.checkpoint_iterations:
                    print(f'[ITER {steps}] Saving Checkpoint')
                    scene.save_checkpoint(steps)
