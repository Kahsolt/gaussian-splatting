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

from modules.utils.general_utils import mkdir

from .camera import Camera
from .scene import Scene
from .model import GaussianModel


def render(pc:GaussianModel, vp_cam:Camera, bg_color:Tensor, scaling_modifier:float=1.0, override_color:Tensor=None, steps:int=None, model_path:Path=None) -> Dict[str, Tensor]:
    assert override_color is None, 'not supported'

    hp = pc.hp
    if hp.rasterizer == 'ours':
        from diff_gaussian_rasterization_ks import GaussianRasterizationSettings, GaussianRasterizer
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
    importance = pc.importance

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
    visible_mask = rasterizer.markVisible(positions=means3D)
    colors_alphas = pc.feature_encoder(vp_cam, visible_mask)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_set = []     # image, occlu
    occlu_map: Tensor = None
    for color, alpha in colors_alphas:
        rendered_image, radii, *extra_data = rasterizer(
            means3D=means3D[visible_mask],
            means2D=means2D[visible_mask],
            shs=None,
            colors_precomp=color,
            opacities=pc.opacity[visible_mask],
            importances=importance[visible_mask],
            scales=scales[visible_mask],
            rotations=rotations[visible_mask],
            cov3D_precomp=cov3D_precomp,
        )
        rendered_set.append(rendered_image)
        occlu_map = extra_data[0]      # use occlu_map from occlu pass
    image, occlu = rendered_set

    # blend background & foreground
    method = 'weight'
    if method == 'binary':
        mask = occlu_map > 0.5
        rendered = mask * occlu + ~mask * image
    elif method == 'weight':
        mask = occlu_map
        rendered = mask * occlu + (1 - mask) * image

    # Peep middle results
    if steps and model_path and steps % 100 == 0:
        mask_ex = mask.unsqueeze(0).expand(3, -1, -1)
        save_dir = mkdir(model_path / 'look_up', parents=True)
        rendered_cat = torch.cat([torch.cat([image, occlu, mask_ex], -1), rendered, vp_cam.image.cuda()], -1)
        save_image(rendered_cat, save_dir / f'{steps:05d}-{vp_cam.uid}.png')

    radii_expand = torch.zeros_like(visible_mask, dtype=radii.dtype, device=visible_mask.device)
    radii_expand[visible_mask] = radii
    visibility_filter = torch.logical_and(visible_mask, radii_expand > 0)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        'render': rendered,
        'viewspace_points': screenspace_points,
        'visibility_filter': visibility_filter,
        'radii': radii_expand,
    }


@torch.inference_mode()
def render_set(scene:Scene, split:str):
    base_path = mkdir(Path(scene.model_path) / split / f'ours_{scene.load_iter}', parents=True)

    render_path = mkdir(base_path / 'renders')
    gts_path = mkdir(base_path / 'gt')

    gaussians: GaussianModel = scene.gaussians
    gaussians.cuda()
    views: List[Camera] = getattr(scene, f'get_{split}_cameras')()
    for idx, view in enumerate(tqdm(views, desc='Rendering progress')):
        render_pkg = render(gaussians, view, scene.background)
        rendered = render_pkg['render']
        gt = view.image[:3]
        save_image(rendered, render_path / f'{idx:05d}.png')
        save_image(gt, gts_path / f'{idx:05d}.png')
