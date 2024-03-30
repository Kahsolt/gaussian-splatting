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
from tqdm import tqdm
from argparse import ArgumentParser

import torch
from torchvision.utils import save_image
import seaborn as sns
import matplotlib.pyplot as plt

from modules.arguments import ModelParams, PipelineParams, get_combined_args
from modules.scene import Scene, GaussianModel, render, DGR_PROVIDER, ImageState
from modules.utils.general_utils import safe_state


def render_set(model_path, name, iteration, views, gaussians:GaussianModel, pipeline, background):
    base_path = os.path.join(model_path, name, f'ours_{iteration}')

    render_path = os.path.join(base_path, 'renders')
    gts_path = os.path.join(base_path, 'gt')
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)

    if DGR_PROVIDER == 'ours':
        importance_path = os.path.join(base_path, 'importance')
        os.makedirs(importance_path, exist_ok=True)
    elif DGR_PROVIDER == 'ours-dev':
        finalT_path = os.path.join(base_path, 'finalT')
        n_contrib_path = os.path.join(base_path, 'n_contrib')
        os.makedirs(finalT_path, exist_ok=True)
        os.makedirs(n_contrib_path, exist_ok=True)
    elif DGR_PROVIDER == 'depth':
        depth_path = os.path.join(base_path, 'depth')
        weight_path = os.path.join(base_path, 'weight')
        os.makedirs(depth_path, exist_ok=True)
        os.makedirs(weight_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc='Rendering progress')):
        render_results = render(view, gaussians, pipeline, background)
        rendering = render_results['render']
        gt = view.original_image[0:3, :, :]
        save_image(rendering, os.path.join(render_path, f'{idx:05d}.png'))
        save_image(gt, os.path.join(gts_path, f'{idx:05d}.png'))

        if DGR_PROVIDER == 'ours':
            save_image(gaussians.importance_activation(render_results['importance_map']), os.path.join(importance_path, f'{idx:05d}.png'), normalize=True)
        if DGR_PROVIDER == 'ours-dev':
            img_state: ImageState = render_results['img_state']
            save_image(img_state.final_T, os.path.join(finalT_path, f'{idx:05d}.png'))
            plt.clf()
            sns.heatmap(render_results['n_contrib'].cpu().numpy(), cbar=True, vmin=0, vmax=400)     # FIXME: magic vrng limit
            plt.savefig(os.path.join(n_contrib_path, f'{idx:05d}.png'), dpi=600)
            plt.close()
        elif DGR_PROVIDER == 'depth':
            save_image(render_results['depth_map'] / render_results['depth_map'].max(), os.path.join(depth_path, f'{idx:05d}.png'))
            save_image(render_results['weight_map'], os.path.join(weight_path, f'{idx:05d}.png'))


@torch.no_grad()
def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')

    if not skip_train:
        render_set(dataset.model_path, 'train', scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
    if not skip_test:
        render_set(dataset.model_path, 'test', scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)


if __name__ == '__main__':
    parser = ArgumentParser(description='Testing script parameters')
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument('--iteration', default=-1, type=int)
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--skip_test', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    args = get_combined_args(parser)
    print('Rendering ' + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
