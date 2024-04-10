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
import math
from argparse import Namespace
from typing import List, Dict, Callable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch.optim import Optimizer
from plyfile import PlyData, PlyElement, PlyProperty
import numpy as np
from simple_knn._C import distCUDA2

from modules.data import BasicPointCloud
from modules.camera import Camera
from modules.utils.sh_utils import RGB2SH, eval_sh
from modules.utils.general_utils import ImageState, RASTERIZER_PROVIDERS
from modules.utils.modeling_utils import *
from modules.utils.training_utils import *

from .hparam import HyperParams


class GaussianModel(nn.Module):

    def __init__(self, hp:HyperParams):
        super().__init__()


        # ↓↓↓ new add neural decoder (from bhy)
        self.embedding_appearance: nn.Embedding = nn.Identity()    # dummy
        self.embedding_occlusion: nn.Embedding = nn.Identity()

        if hp.use_view_emb:
            self.mlp_view = nn.Sequential(
                nn.Linear(4, hp.view_emb_dim), 
            )
            view_emb_dim = hp.view_emb_dim
        else:
            self.mlp_view = nn.Identity()
            view_emb_dim = 4
        view_emb_dim = 0        # tmp force ignore

        self.color_mlp_in_dim = hp.sh_feat_dim + view_emb_dim + (hp.appearance_dim if hp.add_view_emb_to_color else 0)
        self.mlp_color = nn.Sequential(
            nn.Linear(self.color_mlp_in_dim, hp.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hp.feat_dim, 3),
            nn.Sigmoid(),
        )

        self.occlu_mlp_in_dim = hp.sh_feat_dim + view_emb_dim + (hp.occlusion_dim if hp.add_view_emb_to_occlu else 0)
        self.mlp_occlu = nn.Sequential(
            nn.Linear(self.occlu_mlp_in_dim, hp.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hp.feat_dim, 1),
            nn.Softplus(),
        )

    def feature_encode(self, vp_cam:Camera, visible_mask:Tensor=None):
        ## view frustum filtering for acceleration    
        if visible_mask is None: visible_mask = slice(None)

        pts = self.xyz[visible_mask]                        # (N, 3)
        feat = self.features[visible_mask].flatten(1)       # (N, 16*3)

        if not 'bind view':
            ob_view = pts - vp_cam.camera_center                # (N, 3)
            ob_dist = ob_view.norm(dim=1, keepdim=True)         # (N, 1)
            ob_view = ob_view / ob_dist                         # (N, 3)
            ob_dist = torch.log(ob_dist)
            cat_view = torch.cat([ob_view, ob_dist], dim=1)     # (N, 4)

        # encode view
        mp = self.mp
        if mp.use_view_emb:
            cat_view = self.mlp_view(cat_view)      # (N, 16), vrng R

        # predict colors
        #cat_color_feat = torch.cat([feat, cat_view], -1)    # (N, 48)
        cat_color_feat = feat
        if mp.add_view_emb_to_color:
            camera_indicies = torch.ones_like(cat_view[:, 0], dtype=torch.long) * vp_cam.uid
            appearance = self.embedding_appearance(camera_indicies)
            cat_color_feat = torch.cat([cat_color_feat, appearance], -1)    # (N, 80)
        colors = self.mlp_color(cat_color_feat)   # (N, 3), vrng [0, 1]

        # predict occlus
        #cat_occlu_feat = torch.cat([feat, cat_view], -1)    # (N, 16)
        cat_occlu_feat = feat
        if mp.add_view_emb_to_occlu:
            camera_indicies = torch.ones_like(cat_view[:, 0], dtype=torch.long) * vp_cam.uid
            occlusion = self.embedding_occlusion(camera_indicies)
            cat_occlu_feat = torch.cat([cat_occlu_feat, occlusion], -1)    # (N, 80)
        occlus = self.mlp_occlu(cat_occlu_feat)   # (N, 1), vrng [0, inf]

        return colors, occlus

    def setup_transform_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.rotation_activation = F.normalize
        self.opacity_activation = torch.sigmoid
        self.opacity_inverse_activation = inverse_sigmoid
        self.importance_activation = torch.tanh
        self.importance_inverse_activation = torch.atanh
        self.covariance_activation = build_covariance_from_scaling_rotation

    @property
    def importance(self):
        return self.importance_activation(self._importance)

    def from_pcd(self, pcd:BasicPointCloud, sanitize:bool=False):
        points = torch.from_numpy(np.asarray(pcd.points)).to(dtype=torch.float, device='cuda')
        colors = torch.from_numpy(np.asarray(pcd.colors)).to(dtype=torch.float, device='cuda')

        if sanitize:
            print('Number of points loaded:', points.shape[0])

            # 每个点到最近三个邻居的平均距离的平方
            dist2 = torch.clamp_min(distCUDA2(points), 1e-8)

            if os.getenv('DEBUG_DIST'):
                import matplotlib.pyplot as plt
                plt.hist(dist2.sqrt().log().flatten().cpu().numpy(), bins=100)
                plt.show()

            # 删掉离群点
            mask = dist2.sqrt().log() < -4
            points = points[mask]
            colors = colors[mask]

        n_pts = points.shape[0]
        print('Number of points initialized:', n_pts)

        dists = torch.clamp_min(distCUDA2(points), 0.0000001)
        scales = torch.log(torch.sqrt(dists))[...,None].repeat(1, 3)
        rots = torch.zeros((n_pts, 4), device='cuda')
        rots[:, 0] = 1
        features = torch.zeros((n_pts, 3, (self.max_sh_degree + 1) ** 2), dtype=torch.float, device='cuda')
        features[:, :3, 0 ] = RGB2SH(colors)  # dc
        features[:, 3:, 1:] = 0.0            # rest
        opacities = inverse_sigmoid(0.1 * torch.ones((n_pts, 1), dtype=torch.float, device='cuda'))
        importances = self.importance_inverse_activation(torch.zeros((n_pts, 1), dtype=torch.float, device='cuda'))

        self._xyz           = nn.Parameter(points, requires_grad=True)
        self._scaling       = nn.Parameter(scales, requires_grad=True)
        self._rotation      = nn.Parameter(rots  , requires_grad=True)
        self._features_dc   = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous(), requires_grad=True)
        self._features_rest = nn.Parameter(features[:, :, 1: ].transpose(1, 2).contiguous(), requires_grad=True)
        self._opacity       = nn.Parameter(opacities,   requires_grad=True)
        self._importance    = nn.Parameter(importances, requires_grad=True)

    def load_ply(self, path:str, sanitize:bool=False):
        plydata = PlyData.read(path)
        elem: PlyElement = plydata.elements[0]
        properties: List[PlyProperty] = elem.properties
        sort_fn = lambda x: int(x.split('_')[-1])

        if sanitize:
            importances = np.asarray(elem['importance'])

            if os.getenv('DEBUG_IMPORTANCE'):
                import matplotlib.pyplot as plt
                plt.hist(importances.flatten(), bins=100)
                plt.show()
                plt.close()

            # 负重要性的是不确定度高的点，因为它难学，所以正向学习时会倾向于压低 loss 权重
            mask = importances > 0
            print(f'>> reduce: {len(mask)} => {mask.sum()}')
            elem_fake = {}
            for prop in properties:
                elem_fake[prop.name] = elem[prop.name][mask]
            elem = elem_fake
            importances = importances[mask]

        xyz = np.stack((np.asarray(elem['x']), np.asarray(elem['y']), np.asarray(elem['z'])), axis=1)
        scale_names = sorted([p.name for p in properties if p.name.startswith('scale_')], key=sort_fn)
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, prop in enumerate(scale_names):
            scales[:, idx] = np.asarray(elem[prop])
        rot_names = sorted([p.name for p in properties if p.name.startswith('rot_')], key=sort_fn)
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, prop in enumerate(rot_names):
            rots[:, idx] = np.asarray(elem[prop])
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        for i in range(3):
            features_dc[:, i, 0] = np.asarray(elem[f'f_dc_{i}'])
        extra_f_names = sorted([p.name for p in properties if p.name.startswith('f_rest_')], key=sort_fn)
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, prop in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(elem[prop])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        opacities = np.asarray(elem['opacity'])[..., np.newaxis]
        importances = np.asarray(elem['importance'])[..., np.newaxis]

        self._xyz           = nn.Parameter(torch.tensor(xyz,            dtype=torch.float, device='cuda'), requires_grad=True)
        self._scaling       = nn.Parameter(torch.tensor(scales,         dtype=torch.float, device='cuda'), requires_grad=True)
        self._rotation      = nn.Parameter(torch.tensor(rots,           dtype=torch.float, device='cuda'), requires_grad=True)
        self._features_dc   = nn.Parameter(torch.tensor(features_dc,    dtype=torch.float, device='cuda').transpose(1, 2).contiguous(), requires_grad=True)
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device='cuda').transpose(1, 2).contiguous(), requires_grad=True)
        self._opacity       = nn.Parameter(torch.tensor(opacities,      dtype=torch.float, device='cuda'), requires_grad=True)
        self._importance    = nn.Parameter(torch.tensor(importances,    dtype=torch.float, device='cuda'), requires_grad=True)

        self.cur_sh_degree = self.max_sh_degree     # assume optimization completed

    def make_save_ply_data(self, path:str):
        property_data = [
            self._xyz          .detach().cpu().numpy(),
            self._scaling      .detach().cpu().numpy(),
            self._rotation     .detach().cpu().numpy(),
            self._features_dc  .detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy(),
            self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy(),
            self._opacity      .detach().cpu().numpy(),
            self._importance   .detach().cpu().numpy(),
        ]
        property_names =  [
            'x', 'y', 'z',
            # All channels except the 3 DC
            *[f'scale_{i}' for i in range(self._scaling.shape[1])],
            *[f'rot_{i}' for i in range(self._rotation.shape[1])],
            *[f'f_dc_{i}' for i in range(self._features_dc.shape[1]*self._features_dc.shape[2])],
            *[f'f_rest_{i}' for i in range(self._features_rest.shape[1]*self._features_rest.shape[2])],
            'opacity',
            'importance',
        ]
        return property_data, property_names

    ''' optimize '''

    def update_learning_rate(self, steps:int):
        super().update_learning_rate(steps)
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == 'mlp_color':
                lr = self.color_scheduler_args(steps)
                param_group['lr'] = lr

    def prune_points(self, mask:Tensor):
        valid_points_mask = ~mask
        optimizable_tensors = self.prune_optimizer(valid_points_mask)

        self._xyz           = optimizable_tensors['xyz']
        self._scaling       = optimizable_tensors['scaling']
        self._rotation      = optimizable_tensors['rotation']
        self._features_dc   = optimizable_tensors['f_dc']
        self._features_rest = optimizable_tensors['f_rest']
        self._opacity       = optimizable_tensors['opacity']
        self._importance    = optimizable_tensors['importance']

        self.xyz_grad_accum = self.xyz_grad_accum[valid_points_mask]
        self.xyz_grad_count = self.xyz_grad_count[valid_points_mask]
        self.max_radii2D    = self.max_radii2D   [valid_points_mask]

    def densification_postfix(self, new_xyz:Tensor, new_scaling:Tensor, new_rotation:Tensor, new_features_dc:Tensor, new_features_rest:Tensor, new_opacities:Tensor, new_importances:Tensor):
        states = {
            'xyz':        new_xyz,
            'scaling':    new_scaling,
            'rotation':   new_rotation,
            'f_dc':       new_features_dc,
            'f_rest':     new_features_rest,
            'opacity':    new_opacities,
            'importance': new_importances,
        }
        optimizable_tensors = self.cat_tensors_to_optimizer(states)

        self._xyz           = optimizable_tensors['xyz']
        self._scaling       = optimizable_tensors['scaling']
        self._rotation      = optimizable_tensors['rotation']
        self._features_dc   = optimizable_tensors['f_dc']
        self._features_rest = optimizable_tensors['f_rest']
        self._opacity       = optimizable_tensors['opacity']
        self._importance    = optimizable_tensors['importance']

        self.xyz_grad_accum = torch.zeros((self.xyz.shape[0], 1), device='cuda')
        self.xyz_grad_count = torch.zeros((self.xyz.shape[0], 1), device='cuda')
        self.max_radii2D    = torch.zeros((self.xyz.shape[0]),    device='cuda')

    def densify_and_split(self, grads:Tensor, grad_threshold:float, scene_extent:float, N:int=2):
        n_init_points = self.xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device='cuda')
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(self.scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device='cuda')
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz           = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.xyz[selected_pts_mask].repeat(N, 1)
        new_scaling       = self.scaling_inverse_activation(self.scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation      = self._rotation     [selected_pts_mask].repeat(N, 1)
        new_features_dc   = self._features_dc  [selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacities     = self._opacity      [selected_pts_mask].repeat(N, 1)
        new_importances   = self._importance   [selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_scaling, new_rotation, new_features_dc, new_features_rest, new_opacities, new_importances)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device='cuda', dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads:Tensor, grad_threshold:float, scene_extent:float):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(self.scaling, dim=1).values <= self.percent_dense*scene_extent)

        new_xyz           = self._xyz          [selected_pts_mask]
        new_scaling       = self._scaling      [selected_pts_mask]
        new_rotation      = self._rotation     [selected_pts_mask]
        new_features_dc   = self._features_dc  [selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities     = self._opacity      [selected_pts_mask]
        new_importances   = self._importance   [selected_pts_mask]

        self.densification_postfix(new_xyz, new_scaling, new_rotation, new_features_dc, new_features_rest, new_opacities, new_importances)

    def oneup_SH_degree(self):
        if self.cur_sh_degree < self.max_sh_degree:
            self.cur_sh_degree += 1

    ''' render '''

    def render(self, vp_cam:Camera, bg_color:Tensor, scaling_modifier:float=1.0, override_color:Tensor=None):
        hp = self.hp
        if hp.rasterizer == 'original':
            from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        elif hp.rasterizer == 'depth':
            from diff_gaussian_rasterization_depth import GaussianRasterizationSettings, GaussianRasterizer
        elif hp.rasterizer == 'ours':
            from diff_gaussian_rasterization_ks import GaussianRasterizationSettings, GaussianRasterizer
        elif hp.rasterizer == 'ours-dev':
            from diff_gaussian_rasterization_ks import GaussianRasterizationSettings, GaussianRasterizer
        else:
            raise ValueError(f'>> unknown unsupported rasterizer engine: {hp.rasterizer} for this model')

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(self.xyz, dtype=self.xyz.dtype, requires_grad=True, device='cuda')
        try:
            screenspace_points.retain_grad()
        except:
            pass

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
            sh_degree=self.cur_sh_degree,
            campos=vp_cam.camera_center,
            prefiltered=False,
            debug=hp.debug,
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = self.xyz
        means2D = screenspace_points
        opacity = self.opacity
        importance = self.importance

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if hp.compute_cov3D_python:
            cov3D_precomp = self.get_covariance(scaling_modifier)
        else:
            scales = self.scaling
            rotations = self.rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if self.mp.use_neural_decoder:
            #visible_mask = rasterizer.markVisible(positions=means3D)
            visible_mask = None
            colors, occlusions = self.feature_encode(vp_cam, visible_mask)
            colors_precomp = colors
        else:
            if override_color is None:
                if hp.convert_SHs_python:
                    shs_view = self.features.transpose(1, 2).view(-1, 3, (self.max_sh_degree+1)**2)
                    dir_pp = (self.xyz - vp_cam.camera_center.repeat(self.features.shape[0], 1))
                    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                    sh2rgb = eval_sh(self.cur_sh_degree, shs_view, dir_pp_normalized)
                    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
                else:
                    shs = self.features
            else:
                colors_precomp = override_color

        extra_kwargs = {}
        if hp.rasterizer == 'ours':
            extra_kwargs = {
                'importances': importance,
            }

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
            **extra_kwargs,
        )

        if hp.rasterizer == 'ours':
            tmp = extra_data[0]
            importance_map = radii
            radii = tmp
        elif hp.rasterizer == 'ours-dev':
            imgBuffer = extra_data[0]
            img_state = ImageState(imgBuffer, (raster_settings.image_height, raster_settings.image_width))
            n_contrib = extra_data[1]
        elif hp.rasterizer == 'depth':
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
