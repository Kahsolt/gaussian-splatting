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
from modules.utils.sh_utils import RGB2SH
from modules.utils.modeling_utils import *
from modules.utils.training_utils import *

from .hparam import HyperParams


class GaussianModel:

    ''' original 3d-gs gaussain model '''

    def __init__(self, hp:HyperParams):
        super().__init__()

        # props
        self._xyz:           Parameter = None
        self._scaling:       Parameter = None
        self._rotation:      Parameter = None
        self._features_dc:   Parameter = None
        self._features_rest: Parameter = None
        self._opacity:       Parameter = None
        # optim
        self.optimizer:        Optimizer = None
        self.xyz_scheduler:    Callable  = None
        self.xyz_grad_accum:   Tensor    = None
        self.xyz_grad_count:   Tensor    = None
        self.max_radii2D:      Tensor    = None
        self.max_sh_degree:    int       = hp.sh_degree
        self.cur_sh_degree:    int       = 0
        self.percent_dense:    float     = 0.01
        self.spatial_lr_scale: float     = 1.0
        # consts
        self.hp = hp
        self.setup_transform_functions()

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
        self.covariance_activation = build_covariance_from_scaling_rotation

    @property
    def n_points(self):
        return self.xyz.shape[0]

    @property
    def xyz(self):
        return self._xyz

    @property
    def scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def features(self):  # colors, [N, D=16, C=3] for SH_deg=3
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier:float=1):
        return self.covariance_activation(self.scaling, scaling_modifier, self._rotation)

    def state_dict(self) -> Dict[str, Any]:
        return {
            # props
            '_xyz':             self._xyz,
            '_scaling':         self._scaling,
            '_rotation':        self._rotation,
            '_features_dc':     self._features_dc,
            '_features_rest':   self._features_rest,
            '_opacity':         self._opacity,
            # optim
            'optimizer':        self.optimizer.state_dict(),
            'xyz_grad_accum':   self.xyz_grad_accum,
            'xyz_grad_count':   self.xyz_grad_count,
            'max_radii2D':      self.max_radii2D,
            'max_sh_degree':    self.max_sh_degree,
            'cur_sh_degree':    self.cur_sh_degree,
            'percent_dense':    self.percent_dense,
            'spatial_lr_scale': self.spatial_lr_scale,
        }

    def load_state_dict(self, state_dict:Dict[str, Any]):
        # load data first
        self._xyz           = state_dict['_xyz']
        self._scaling       = state_dict['_scaling']
        self._rotation      = state_dict['_rotation']
        self._features_dc   = state_dict['_features_dc']
        self._features_rest = state_dict['_features_rest']
        self._opacity       = state_dict['_opacity']
        # then recover optim state
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.xyz_grad_accum   = state_dict['xyz_grad_accum']
        self.xyz_grad_count   = state_dict['xyz_grad_count']
        self.max_radii2D      = state_dict['max_radii2D']
        self.max_sh_degree    = state_dict['max_sh_degree']
        self.cur_sh_degree    = state_dict['cur_sh_degree']
        self.percent_dense    = state_dict['percent_dense']
        self.spatial_lr_scale = state_dict['spatial_lr_scale']

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

        self._xyz           = nn.Parameter(points, requires_grad=True)
        self._scaling       = nn.Parameter(scales, requires_grad=True)
        self._rotation      = nn.Parameter(rots  , requires_grad=True)
        self._features_dc   = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous(), requires_grad=True)
        self._features_rest = nn.Parameter(features[:, :, 1: ].transpose(1, 2).contiguous(), requires_grad=True)
        self._opacity       = nn.Parameter(opacities,   requires_grad=True)

    def load_ply(self, path:str, sanitize:bool=False):
        plydata = PlyData.read(path)
        elem: PlyElement = plydata.elements[0]
        properties: List[PlyProperty] = elem.properties
        sort_fn = lambda x: int(x.split('_')[-1])

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

        self._xyz           = nn.Parameter(torch.tensor(xyz,            dtype=torch.float, device='cuda'), requires_grad=True)
        self._scaling       = nn.Parameter(torch.tensor(scales,         dtype=torch.float, device='cuda'), requires_grad=True)
        self._rotation      = nn.Parameter(torch.tensor(rots,           dtype=torch.float, device='cuda'), requires_grad=True)
        self._features_dc   = nn.Parameter(torch.tensor(features_dc,    dtype=torch.float, device='cuda').transpose(1, 2).contiguous(), requires_grad=True)
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device='cuda').transpose(1, 2).contiguous(), requires_grad=True)
        self._opacity       = nn.Parameter(torch.tensor(opacities,      dtype=torch.float, device='cuda'), requires_grad=True)

        self.cur_sh_degree = self.max_sh_degree     # assume optimization completed

    def save_ply(self, path:str):
        xyz          = self._xyz          .detach().cpu().numpy()
        scaling      = self._scaling      .detach().cpu().numpy()
        rotation     = self._rotation     .detach().cpu().numpy()
        feature_dc   = self._features_dc  .detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        feature_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities    = self._opacity      .detach().cpu().numpy()

        property_names =  [
            'x', 'y', 'z',
            # All channels except the 3 DC
            *[f'scale_{i}' for i in range(self._scaling.shape[1])],
            *[f'rot_{i}' for i in range(self._rotation.shape[1])],
            *[f'f_dc_{i}' for i in range(self._features_dc.shape[1]*self._features_dc.shape[2])],
            *[f'f_rest_{i}' for i in range(self._features_rest.shape[1]*self._features_rest.shape[2])],
            'opacity',
        ]
        vertexes = np.empty(xyz.shape[0], dtype=[(prop, 'f4') for prop in property_names])
        properties = np.concatenate((xyz, scaling, rotation, feature_dc, feature_rest, opacities), axis=1)
        vertexes[:] = list(map(tuple, properties))
        elem = PlyElement.describe(vertexes, 'vertex')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        PlyData([elem]).write(path)

    ''' optimize '''

    def setup_training(self):
        hp = self.hp
        param_groups = [
            {'name': 'xyz',      'params': [self._xyz],           'lr': hp.position_lr_init * self.spatial_lr_scale},
            {'name': 'scaling',  'params': [self._scaling],       'lr': hp.scaling_lr},
            {'name': 'rotation', 'params': [self._rotation],      'lr': hp.rotation_lr},
            {'name': 'f_dc',     'params': [self._features_dc],   'lr': hp.feature_lr},
            {'name': 'f_rest',   'params': [self._features_rest], 'lr': hp.feature_lr / 20.0},
            {'name': 'opacity',  'params': [self._opacity],       'lr': hp.opacity_lr},
        ]
        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
        self.xyz_scheduler = get_expon_lr_func(
            lr_init=hp.position_lr_init * self.spatial_lr_scale,
            lr_final=hp.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=hp.position_lr_delay_mult,
            max_steps=hp.position_lr_max_steps,
        )
        self.xyz_grad_accum = torch.zeros((self.xyz.shape[0], 1), dtype=torch.float, device='cuda')
        self.xyz_grad_count = torch.zeros((self.xyz.shape[0], 1), dtype=torch.int,   device='cuda')
        self.max_radii2D    = torch.zeros((self.xyz.shape[0]),    dtype=torch.int,   device='cuda')
        self.percent_dense = hp.percent_dense

    def optimizer_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def update_learning_rate(self, steps:int):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'xyz':
                lr = self.xyz_scheduler(steps)
                param_group['lr'] = lr

    def replace_tensor_to_optimizer(self, tensor:Tensor, name:str) -> Dict[str, Tensor]:
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group['name'] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state['exp_avg'] = torch.zeros_like(tensor)
                stored_state['exp_avg_sq'] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group['params'][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group['name']] = group['params'][0]
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict:Dict[str, Tensor]) -> Dict[str, Tensor]:
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group['params']) == 1
            extension_tensor = tensors_dict[group['name']]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state['exp_avg'] = torch.cat((stored_state['exp_avg'], torch.zeros_like(extension_tensor)), dim=0)
                stored_state['exp_avg_sq'] = torch.cat((stored_state['exp_avg_sq'], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group['params'][0] = nn.Parameter(torch.cat((group['params'][0], extension_tensor), dim=0), requires_grad=True)
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group['name']] = group['params'][0]
            else:
                group['params'][0] = nn.Parameter(torch.cat((group['params'][0], extension_tensor), dim=0), requires_grad=True)
                optimizable_tensors[group['name']] = group['params'][0]
        return optimizable_tensors

    def prune_optimizer(self, mask:Tensor) -> Dict[str, Tensor]:
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state['exp_avg'] = stored_state['exp_avg'][mask]
                stored_state['exp_avg_sq'] = stored_state['exp_avg_sq'][mask]

                del self.optimizer.state[group['params'][0]]
                group['params'][0] = nn.Parameter(group['params'][0][mask], requires_grad=True)
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group['name']] = group['params'][0]
            else:
                group['params'][0] = nn.Parameter(group['params'][0][mask], requires_grad=True)
                optimizable_tensors[group['name']] = group['params'][0]
        return optimizable_tensors

    def prune_points(self, mask:Tensor):
        valid_points_mask = ~mask
        optimizable_tensors = self.prune_optimizer(valid_points_mask)

        self._xyz           = optimizable_tensors['xyz']
        self._scaling       = optimizable_tensors['scaling']
        self._rotation      = optimizable_tensors['rotation']
        self._features_dc   = optimizable_tensors['f_dc']
        self._features_rest = optimizable_tensors['f_rest']
        self._opacity       = optimizable_tensors['opacity']

        self.xyz_grad_accum = self.xyz_grad_accum[valid_points_mask]
        self.xyz_grad_count = self.xyz_grad_count[valid_points_mask]
        self.max_radii2D    = self.max_radii2D   [valid_points_mask]

    def densification_postfix(self, new_xyz:Tensor, new_scaling:Tensor, new_rotation:Tensor, new_features_dc:Tensor, new_features_rest:Tensor, new_opacities:Tensor):
        states = {
            'xyz':        new_xyz,
            'scaling':    new_scaling,
            'rotation':   new_rotation,
            'f_dc':       new_features_dc,
            'f_rest':     new_features_rest,
            'opacity':    new_opacities,
        }
        optimizable_tensors = self.cat_tensors_to_optimizer(states)

        self._xyz           = optimizable_tensors['xyz']
        self._scaling       = optimizable_tensors['scaling']
        self._rotation      = optimizable_tensors['rotation']
        self._features_dc   = optimizable_tensors['f_dc']
        self._features_rest = optimizable_tensors['f_rest']
        self._opacity       = optimizable_tensors['opacity']

        self.xyz_grad_accum = torch.zeros((self.xyz.shape[0], 1), dtype=torch.float, device='cuda')
        self.xyz_grad_count = torch.zeros((self.xyz.shape[0], 1), dtype=torch.int,   device='cuda')
        self.max_radii2D    = torch.zeros((self.xyz.shape[0]),    dtype=torch.int,   device='cuda')

    def densify_and_split(self, grads:Tensor, grad_threshold:float, scene_extent:float, N:int=2):
        n_init_points = self.xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device='cuda')
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(self.scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3),device='cuda')
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz           = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.xyz[selected_pts_mask].repeat(N, 1)
        new_scaling       = self.scaling_inverse_activation(self.scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation      = self._rotation     [selected_pts_mask].repeat(N, 1)
        new_features_dc   = self._features_dc  [selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacities     = self._opacity      [selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_scaling, new_rotation, new_features_dc, new_features_rest, new_opacities)

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

        self.densification_postfix(new_xyz, new_scaling, new_rotation, new_features_dc, new_features_rest, new_opacities)

    def densify_and_prune(self, max_grad:float, min_opacity:float, extent:float, max_screen_size:int):
        grads = self.xyz_grad_accum / self.xyz_grad_count
        if os.getenv('DEBUG_GRAD'):
            with torch.no_grad():
                has_grad = ~grads.isnan()
                has_grad_cnt = has_grad.sum()
                fixed_grads = grads.clone()
                fixed_grads[~has_grad] = 0.0
                print(f'[has_grad] {has_grad_cnt} / {grads.numel()} = {has_grad_cnt / grads.numel()}')
                print(f'[abs(grad)] max: {fixed_grads.max()}, mean: {fixed_grads.sum() / has_grad_cnt}')
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor:Tensor, update_filter):
        self.xyz_grad_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.xyz_grad_count[update_filter] += 1

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.opacity, torch.ones_like(self.opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, 'opacity')
        self._opacity = optimizable_tensors['opacity']

    def oneup_SH_degree(self):
        if self.cur_sh_degree < self.max_sh_degree:
            self.cur_sh_degree += 1
