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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, DGR_PROVIDER, ImageState
import torchvision
import seaborn as sns
import matplotlib.pyplot as plt
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

def render_set(model_path, name, iteration, views, gaussians:GaussianModel, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    if DGR_PROVIDER == 'ours':
        importance_path = os.path.join(model_path, name, "ours_{}".format(iteration), "importance")
        makedirs(importance_path, exist_ok=True)
    elif DGR_PROVIDER == 'ours-dev':
        finalT_path = os.path.join(model_path, name, "ours_{}".format(iteration), "finalT")
        n_contrib_path = os.path.join(model_path, name, "ours_{}".format(iteration), "n_contrib")
        makedirs(finalT_path, exist_ok=True)
        makedirs(n_contrib_path, exist_ok=True)
    elif DGR_PROVIDER == 'depth':
        depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
        weight_path = os.path.join(model_path, name, "ours_{}".format(iteration), "weight")
        makedirs(depth_path, exist_ok=True)
        makedirs(weight_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_results = render(view, gaussians, pipeline, background)
        rendering = render_results["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        if DGR_PROVIDER == 'ours':
            torchvision.utils.save_image(gaussians.importance_activation(render_results['importance_map']), os.path.join(importance_path, '{0:05d}'.format(idx) + ".png"), normalize=True)
        if DGR_PROVIDER == 'ours-dev':
            img_state: ImageState = render_results["img_state"]
            torchvision.utils.save_image(img_state.final_T, os.path.join(finalT_path, '{0:05d}'.format(idx) + ".png"))
            plt.clf()
            sns.heatmap(render_results["n_contrib"].cpu().numpy(), cbar=True, vmin=0, vmax=400)     # FIXME: magic vrng limit
            plt.savefig(os.path.join(n_contrib_path, '{0:05d}'.format(idx) + ".png"), dpi=600)
            plt.close()
        elif DGR_PROVIDER == 'depth':
            torchvision.utils.save_image(render_results['depth_map'] / render_results['depth_map'].max(), os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(render_results['weight_map'], os.path.join(weight_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)