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
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    finalT_path = os.path.join(model_path, name, "ours_{}".format(iteration), "finalT")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    weight_path = os.path.join(model_path, name, "ours_{}".format(iteration), "weight")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(finalT_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(weight_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_results = render(view, gaussians, pipeline, background)
        rendering = render_results["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        if 'finalT' and render_results.get('imgBuffer') is not None:
            imgBuffer_byte_tensor = render_results["imgBuffer"]     # imgBuffer is the struct ImageState
            C, H, W = gt.shape
            finalT_byte_tensor = imgBuffer_byte_tensor[:4*H*W]
            finalT = torch.frombuffer(memoryview(finalT_byte_tensor.cpu().numpy()), dtype=torch.float32).reshape((H, W))
            torchvision.utils.save_image(finalT, os.path.join(finalT_path, '{0:05d}'.format(idx) + ".png"))
        if 'depth':
            if render_results.get('depth_map') is not None:
                torchvision.utils.save_image(render_results['depth_map'] / render_results['depth_map'].max(), os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
            if render_results.get('weight_map') is not None:
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