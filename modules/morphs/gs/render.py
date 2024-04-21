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

import math
from pathlib import Path
from typing import List, Dict

import torch
from torch import Tensor
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from modules.scene import Scene, Camera
from modules.utils.sh_utils import eval_sh
from modules.utils.general_utils import ImageState

from .hparam import HyperParams
from .model import GaussianModel


def render(pc:GaussianModel, vp_cam:Camera, bg_color:Tensor, scaling_modifier:float=1.0, override_color:Tensor=None) -> Dict[str, Tensor]:
    hp = pc.hp
    if hp.rasterizer == 'original':
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    elif hp.rasterizer == 'depth':
        from diff_gaussian_rasterization_depth import GaussianRasterizationSettings, GaussianRasterizer
    else:
        raise ValueError(f'>> unknown supported rasterizer engine: {hp.rasterizer} for this model')

    # Set up rasterization configuration
    raster_settings = GaussianRasterizationSettings(
        image_height=vp_cam.image_height,
        image_width=vp_cam.image_width,
        tanfovx=math.tan(vp_cam.FoVx * 0.5),
        tanfovy=math.tan(vp_cam.FoVy * 0.5),
        bg=bg_color.cuda(),     # Background tensor (bg_color) must be on GPU!
        scale_modifier=scaling_modifier,
        viewmatrix=vp_cam.world_view_transform,
        projmatrix=vp_cam.full_proj_transform,
        sh_degree=pc.cur_sh_degree,
        campos=vp_cam.camera_center,
        prefiltered=False,
        debug=hp.debug,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.xyz, dtype=pc.xyz.dtype, requires_grad=True, device='cuda')
    try: screenspace_points.retain_grad()
    except: pass

    means3D = pc.xyz
    means2D = screenspace_points
    opacity = pc.opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if hp.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.scaling
        rotations = pc.rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if hp.convert_SHs_python:
            shs_view = pc.features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.xyz - vp_cam.camera_center.repeat(pc.features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.cur_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, *extra_data = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    if hp.rasterizer == 'depth':
        depth_map = extra_data[0]
        weight_map = extra_data[1]
    elif hp.rasterizer == 'ours':
        imgBuffer = extra_data[0]
        n_contrib = extra_data[1]
        img_state = extra_data[2]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        'render': rendered_image,
        'viewspace_points': screenspace_points,
        'visibility_filter': radii > 0,
        'radii': radii,
        'depth_map': locals().get('depth_map'),
        'weight_map': locals().get('weight_map'),
        'img_state': locals().get('img_state'),
        'n_contrib': locals().get('n_contrib'),
    }


@torch.inference_mode()
def render_set(scene:Scene, split:str):
    base_path = Path(scene.model_path) / split / f'ours_{scene.load_iter}'
    base_path.mkdir(exist_ok=True, parents=True)

    render_path = base_path / 'renders'
    gts_path = base_path / 'gt'
    render_path.mkdir(exist_ok=True)
    gts_path.mkdir(exist_ok=True)

    hp = scene.hp
    if hp.rasterizer == 'depth':
        depth_path = base_path / 'depth'
        weight_path = base_path / 'weight'
        depth_path.mkdir(exist_ok=True)
        weight_path.mkdir(exist_ok=True)
        finalT_path = base_path / 'finalT'
    elif hp.rasterizer == 'ours':
        n_contrib_path = base_path / 'n_contrib'
        finalT_path.mkdir(exist_ok=True)
        n_contrib_path.mkdir(exist_ok=True)

    gaussians: GaussianModel = scene.gaussians
    views: List[Camera] = getattr(scene, f'get_{split}_cameras')()
    for idx, view in enumerate(tqdm(views, desc='Rendering progress')):
        render_pkg = render(gaussians, view, scene.background)
        rendered = render_pkg['render']
        gt = view.image[0:3, ...].cuda()
        save_image(rendered, render_path / f'{idx:05d}.png')
        save_image(gt, gts_path / f'{idx:05d}.png')

        if hp.rasterizer == 'depth':
            save_image(render_pkg['depth_map'] / render_pkg['depth_map'].max(), depth_path / f'{idx:05d}.png')
            save_image(render_pkg['weight_map'], weight_path / f'{idx:05d}.png')
        elif hp.rasterizer == 'ours':
            img_state: ImageState = render_pkg['img_state']
            save_image(img_state.final_T, finalT_path / f'{idx:05d}.png')
            plt.clf()
            sns.heatmap(render_pkg['n_contrib'].cpu().numpy(), cbar=True, vmin=0, vmax=400)     # FIXME: magic vrng limit
            plt.savefig(n_contrib_path / f'{idx:05d}.png', dpi=600)
            plt.close()
