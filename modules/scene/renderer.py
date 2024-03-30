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

DGR_PROVIDER = 'ours'

import math
from typing import Tuple

import torch
from torch import Tensor

if DGR_PROVIDER == 'original':
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
elif DGR_PROVIDER == 'depth':
    from diff_gaussian_rasterization_depth import GaussianRasterizationSettings, GaussianRasterizer
elif DGR_PROVIDER == 'ours':
    from diff_gaussian_rasterization_ks import GaussianRasterizationSettings, GaussianRasterizer
elif DGR_PROVIDER == 'ours-dev':
    from diff_gaussian_rasterization_ks import GaussianRasterizationSettings, GaussianRasterizer
print('>> DGR_PROVIDER:', DGR_PROVIDER)

from modules.arguments import PipelineParams
from modules.scene import Camera, GaussianModel
from modules.utils.sh_utils import eval_sh


class ImageState:

    def __init__(self, buffer:Tensor, size:Tuple[int, int], align:int=128):
        H, W = size
        N = H * W
        offset = 0
        buffer = buffer.cpu().numpy()

        def next_offset() -> int:
            nonlocal offset
            while offset % align:
                offset += 1

        next_offset()
        final_T = torch.frombuffer(memoryview(buffer[offset:offset+4*N]), dtype=torch.float32).reshape((H, W))
        next_offset()
        n_contrib = torch.frombuffer(memoryview(buffer[offset:offset+4*N]), dtype=torch.int32).reshape((H, W))
        next_offset()
        ranges = torch.frombuffer(memoryview(buffer[offset:offset+8*N]), dtype=torch.int32).reshape((H, W, 2))

        self._final_T = final_T      # float, 4 bytes
        self._n_contrib = n_contrib  # uint32_t, 4 bytes
        self._ranges = ranges        # uint2, 8 bytes

    @property
    def final_T(self): return self._final_T
    @property
    def n_contrib(self): return self._n_contrib
    @property
    def ranges(self): return self._ranges


def render(vp_cam:Camera, pc:GaussianModel, pipe:PipelineParams, bg_color:Tensor, scaling_modifier:float=1.0, override_color=None):
    ''' Background tensor (bg_color) must be on GPU! '''

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.xyz, dtype=pc.xyz.dtype, requires_grad=True, device='cuda') + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(vp_cam.FoVx * 0.5)
    tanfovy = math.tan(vp_cam.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(vp_cam.image_height),
        image_width=int(vp_cam.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=vp_cam.world_view_transform,
        projmatrix=vp_cam.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=vp_cam.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.xyz
    means2D = screenspace_points
    opacity = pc.opacity
    importance = pc.importance

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.scaling
        rotations = pc.rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.xyz - vp_cam.camera_center.repeat(pc.features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.features
    else:
        colors_precomp = override_color

    extra_kwargs = {}
    if DGR_PROVIDER == 'ours':
        extra_kwargs = {
            'importances': importance,
        }

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, *extra_data = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        **extra_kwargs,
    )

    if DGR_PROVIDER == 'ours':
        tmp = extra_data[0]
        importance_map = radii
        radii = tmp
    elif DGR_PROVIDER == 'ours-dev':
        imgBuffer = extra_data[0]
        img_state = ImageState(imgBuffer, (raster_settings.image_height, raster_settings.image_width))
        n_contrib = extra_data[1]
    elif DGR_PROVIDER == 'depth':
        depth_map = extra_data[0]
        weight_map = extra_data[1]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        'render': rendered_image,
        'viewspace_points': screenspace_points,
        'visibility_filter': radii > 0,
        'radii': radii,
        'img_state': locals().get('img_state'),
        'n_contrib': locals().get('n_contrib'),
        'importance_map': locals().get('importance_map'),
        'depth_map': locals().get('depth_map'),
        'weight_map': locals().get('weight_map'),
    }
