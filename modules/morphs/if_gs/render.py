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

from .scene import Scene, Camera
from .model import SingleFreqGaussianModel, MutilFreqGaussianModel


def render(pc:SingleFreqGaussianModel, vp_cam:Camera, bg_color:Tensor, scaling_modifier:float=1.0, override_color:Tensor=None) -> Dict[str, Tensor]:
    hp = pc.hp
    if hp.rasterizer == 'original':
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
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
        sh_degree=0,
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
    if override_color is None:
        visible_mask = rasterizer.markVisible(positions=means3D)
        colors_precomp = pc.feature_encoder(vp_cam, visible_mask)
    else:
        visible_mask = slice(None)
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, *extra_data = rasterizer(
        means3D=means3D[visible_mask],
        means2D=means2D[visible_mask],
        shs=None,
        colors_precomp=colors_precomp,
        opacities=opacity[visible_mask],
        scales=scales[visible_mask],
        rotations=rotations[visible_mask],
        cov3D_precomp=cov3D_precomp,
    )

    radii_expand = torch.zeros_like(visible_mask, dtype=radii.dtype, device=visible_mask.device)
    radii_expand[visible_mask] = radii
    visibility_filter = torch.logical_and(visible_mask, radii_expand > 0)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        'render': rendered_image,
        'viewspace_points': screenspace_points,
        'visibility_filter': visibility_filter,
        'radii': radii_expand,
    }


@torch.inference_mode()
def render_set(scene:Scene, split:str):
    base_path = Path(scene.model_path) / split / f'ours_{scene.load_iter}'
    base_path.mkdir(exist_ok=True, parents=True)

    render_path = base_path / 'renders'
    gts_path = base_path / 'gt'
    render_path.mkdir(exist_ok=True)
    gts_path.mkdir(exist_ok=True)

    multifreq_gaussians: MutilFreqGaussianModel = scene.multifreq_gaussians
    views: List[Camera] = getattr(scene, f'get_{split}_cameras')()
    for img_idx, view in enumerate(tqdm(views, desc='Rendering progress')):
        for idx in range(multifreq_gaussians.n_gaussians):
            gaussians = scene.activate_gaussian(idx)
            render_pkg = render(gaussians, view, scene.background)
            rendered = render_pkg['render']
            gt = view.images[idx+1][0:3, ...].cuda()

            render_freq_path = render_path / f'freq_{idx}'
            gts_freq_path = gts_path / f'freq_{idx}'
            render_freq_path.mkdir(exist_ok=True)
            gts_freq_path.mkdir(exist_ok=True)
            save_image(rendered, render_freq_path / f'{img_idx:05d}.png')
            save_image(gt, gts_freq_path / f'{img_idx:05d}.png')
