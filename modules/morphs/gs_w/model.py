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
from pathlib import Path
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
from modules.layers import ColorMLP, OccluMLP, Embedding
from modules.utils.modeling_utils import *
from modules.utils.training_utils import *

from .hparam import HyperParams


class GaussianModel(nn.Module):

    ''' gs-w from bhy '''

    def __init__(self, hp:HyperParams):
        super().__init__()

        # props
        self._xyz:        Parameter = None
        self._scaling:    Parameter = None
        self._rotation:   Parameter = None
        self._features:   Parameter = None
        self._opacity:    Parameter = None
        self._importance: Parameter = None
        # optim
        self.optimizer:        Optimizer = None
        self.xyz_scheduler:    Callable  = None
        self.xyz_grad_accum:   Tensor    = None
        self.xyz_grad_count:   Tensor    = None
        self.max_radii2D:      Tensor    = None
        self.percent_dense:    float     = 0.01
        self.spatial_lr_scale: float     = 1.0
        # consts
        self.hp = hp
        self.setup_transform_functions()

        # networks
        self.embedding_appearance: Embedding = nn.Identity()
        self.embedding_occlusion: Embedding = nn.Identity()

        if hp.use_view_emb:
            self.mlp_view = nn.Sequential(
                nn.Linear(4, self.view_emb_dim), 
            )
            self.view_emb_dim = self.view_emb_dim
        else:
            self.mlp_view = nn.Identity()
            self.view_emb_dim = 4
        
        self.color_mlp_in_dim = hp.feat_dim + hp.appearance_dim + (self.view_emb_dim if hp.add_view_emb_to_color else 0)
        self.mlp_color = ColorMLP(self.color_mlp_in_dim, hp.feat_dim)
        self.occlu_mlp_in_dim = 1 + hp.occlusion_dim + (self.view_emb_dim if hp.add_view_emb_to_occlu else 0)
        self.mlp_occlu = OccluMLP(self.occlu_mlp_in_dim, hp.feat_dim // 2)

    def feature_encoder(self, vp_cam:Camera, visible_mask:Tensor=None):
        if visible_mask is None: visible_mask = slice(None)
        feat = self.features[visible_mask]              # (N, 32)
        importance = self.importance[visible_mask]

        hp = self.hp
        if hp.add_view_emb_to_color or hp.add_view_emb_to_occlu:
            pts = self.xyz[visible_mask]                        # (N, 3)
            ob_view = pts - vp_cam.camera_center                # (N, 3)
            ob_dist = ob_view.norm(dim=1, keepdim=True)         # (N, 1)
            ob_view = ob_view / ob_dist                         # (N, 3)
            ob_dist = torch.log(ob_dist)
            cat_view = torch.cat([ob_view, ob_dist], dim=1)     # (N, 4)

            # encode view
            if hp.use_view_emb:
                cat_view = self.mlp_view(cat_view)      # (N, 16), vrng R
            # predict colors
            cat_color_feat = torch.cat([feat, cat_view], -1)    # (N, 48)
            cat_occlu_feat = torch.cat([importance, cat_view], -1)
        
        if hp.appearance_dim > 0:
            camera_indicies = torch.ones_like(cat_view[:, 0], dtype=torch.long) * vp_cam.uid
            appearance = self.embedding_appearance(camera_indicies)
            if hp.add_view_emb_to_color:
                cat_color_feat = torch.cat([cat_color_feat, appearance], -1)    # (N, 80)
            else:
                cat_color_feat = torch.cat([feat, appearance], -1)
        else:
            cat_color_feat = feat
        colors = self.mlp_color(cat_color_feat)   # (N, 3), vrng [0, 1]

        # predict occlus
        if hp.occlusion_dim > 0:
            camera_indicies = torch.ones_like(cat_view[:, 0], dtype=torch.long) * vp_cam.uid
            occlusion = self.embedding_occlusion(camera_indicies)
            if hp.add_view_emb_to_occlu:
                cat_occlu_feat = torch.cat([cat_occlu_feat, occlusion], -1)    # (N, 80)
            else:
                cat_occlu_feat = torch.cat([importance, occlusion], -1)
        else:
            cat_occlu_feat = importance
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
        self.covariance_activation = build_covariance_from_scaling_rotation

    @property
    def is_train(self):
        return self.mlp_color.training

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
    def features(self):
        return self._features

    @property
    def opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def importance(self):
        return self._importance

    def get_covariance(self, scaling_modifier:float=1):
        return self.covariance_activation(self.scaling, scaling_modifier, self._rotation)

    def init_embeddings(self, num_cameras:int):
        hp = self.hp
        if hp.appearance_dim > 0:
            self.embedding_appearance = Embedding(num_cameras, hp.appearance_dim)
        if hp.occlusion_dim > 0:
            self.embedding_occlusion = Embedding(num_cameras, hp.occlusion_dim)

    def state_dict(self) -> Dict[str, Any]:
        return {
            # props
            '_xyz':             self._xyz,
            '_scaling':         self._scaling,
            '_rotation':        self._rotation,
            '_features':        self._features,
            '_opacity':         self._opacity,
            '_importance':      self._importance,
            'color_mlp':        self.mlp_color.state_dict(),
            'occlu_mlp':        self.mlp_occlu.state_dict(),
            'view_mlp':         self.mlp_view.state_dict(),
            'appearance':       self.embedding_appearance.state_dict(),
            'occlusion':        self.embedding_occlusion.state_dict(),
            # optim
            'optimizer':        self.optimizer.state_dict(),
            'xyz_grad_accum':   self.xyz_grad_accum,
            'xyz_grad_count':   self.xyz_grad_count,
            'max_radii2D':      self.max_radii2D,
            'percent_dense':    self.percent_dense,
            'spatial_lr_scale': self.spatial_lr_scale,
        }

    def load_state_dict(self, state_dict:Dict[str, Any]):
        # load data first
        self._xyz        = state_dict['_xyz']
        self._scaling    = state_dict['_scaling']
        self._rotation   = state_dict['_rotation']
        self._features   = state_dict['_features']
        self._opacity    = state_dict['_opacity']
        self._importance = state_dict['_importance']
        self.mlp_color.load_state_dict(state_dict['mlp_color'])
        self.mlp_occlu.load_state_dict(state_dict['mlp_occlu'])
        self.mlp_view.load_state_dict(state_dict['mlp_view'])
        self.embedding_appearance.load_state_dict(state_dict['embedding_appearance'])
        self.embedding_occlusion.load_state_dict(state_dict['embedding_occlusion'])
        # then recover optim state
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.xyz_grad_accum   = state_dict['xyz_grad_accum']
        self.xyz_grad_count   = state_dict['xyz_grad_count']
        self.max_radii2D      = state_dict['max_radii2D']
        self.percent_dense    = state_dict['percent_dense']
        self.spatial_lr_scale = state_dict['spatial_lr_scale']

    def save_pth(self, fp:Path):
        fp.parent.mkdir(exist_ok=True, parents=True)
        ckpt = {
            'color_mlp':  self.mlp_color.state_dict(),
            'occlu_mlp':  self.mlp_occlu.state_dict(),
            'view_mlp':   self.mlp_view.state_dict(),
            'appearance': self.embedding_appearance.state_dict(),
            'occlusion':  self.embedding_occlusion.state_dict(),
        }
        torch.save(ckpt, fp)

    def load_pth(self, fp:Path):
        ckpt = torch.load(fp)
        self.mlp_view.load_state_dict(ckpt['view_mlp'])
        self.mlp_color.load_state_dict(ckpt['color_mlp'])
        self.mlp_occlu.load_state_dict(ckpt['occlu_mlp'])
        self.embedding_appearance.load_state_dict(ckpt['appearance'])
        self.embedding_occlusion.load_state_dict(ckpt['occlusion'])

    def from_pcd(self, pcd:BasicPointCloud):
        points = torch.from_numpy(np.asarray(pcd.points)).to(dtype=torch.float, device='cuda')
        n_pts = points.shape[0]
        print('Number of points initialized:', n_pts)

        dists = torch.clamp_min(distCUDA2(points), 0.0000001)
        scales = torch.log(torch.sqrt(dists))[...,None].repeat(1, 3)
        rots = torch.zeros((n_pts, 4), device='cuda')
        rots[:, 0] = 1
        features = torch.zeros((n_pts, self.hp.feat_dim), dtype=torch.float, device='cuda')
        opacities = inverse_sigmoid(0.1 * torch.ones((n_pts, 1), dtype=torch.float, device='cuda'))
        importances = torch.ones_like(opacities)

        self._xyz        = nn.Parameter(points,    requires_grad=True)
        self._scaling    = nn.Parameter(scales,    requires_grad=True)
        self._rotation   = nn.Parameter(rots,      requires_grad=True)
        self._features   = nn.Parameter(features,  requires_grad=True)
        self._opacity    = nn.Parameter(opacities, requires_grad=True)
        self._importance = nn.Parameter(importances, requires_grad=True)

    def load_ply(self, path:str):
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
        features = np.zeros((xyz.shape[0], self.hp.feat_dim))
        feat_names = sorted([p.name for p in properties if p.name.startswith('feat_')], key=sort_fn)
        assert len(feat_names) == self.hp.feat_dim
        for idx, prop in enumerate(feat_names):
            features[:, idx] = np.asarray(elem[prop])
        opacities = np.asarray(elem['opacity'])[..., np.newaxis]
        importances = np.asarray(elem["importance"])[..., np.newaxis]

        self._xyz        = nn.Parameter(torch.tensor(xyz,         dtype=torch.float, device='cuda'), requires_grad=True)
        self._scaling    = nn.Parameter(torch.tensor(scales,      dtype=torch.float, device='cuda'), requires_grad=True)
        self._rotation   = nn.Parameter(torch.tensor(rots,        dtype=torch.float, device='cuda'), requires_grad=True)
        self._features   = nn.Parameter(torch.tensor(features,    dtype=torch.float, device='cuda'), requires_grad=True)
        self._opacity    = nn.Parameter(torch.tensor(opacities,   dtype=torch.float, device='cuda'), requires_grad=True)
        self._importance = nn.Parameter(torch.tensor(importances, dtype=torch.float, device="cuda"), requires_grad=True)

    def save_ply(self, path:str):
        xyz         = self._xyz       .detach().cpu().numpy()
        scaling     = self._scaling   .detach().cpu().numpy()
        rotation    = self._rotation  .detach().cpu().numpy()
        features    = self._features  .detach().cpu().numpy()
        opacities   = self._opacity   .detach().cpu().numpy()
        importances = self._importance.detach().cpu().numpy()

        property_names =  [
            'x', 'y', 'z',
            # All channels except the 3 DC
            *[f'scale_{i}' for i in range(self._scaling.shape[1])],
            *[f'rot_{i}'   for i in range(self._rotation.shape[1])],
            *[f'feat_{i}'  for i in range(self._features.shape[1])],
            'opacity',
            'importance',
        ]
        vertexes = np.empty(xyz.shape[0], dtype=[(prop, 'f4') for prop in property_names])
        properties = np.concatenate((xyz, scaling, rotation, features, opacities, importances), axis=1)
        vertexes[:] = list(map(tuple, properties))
        elem = PlyElement.describe(vertexes, 'vertex')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        PlyData([elem]).write(path)

    ''' optimize '''

    def setup_training(self):
        hp = self.hp
        param_groups = [
            {'name': 'xyz',        'params': [self._xyz],       'lr': hp.position_lr_init * self.spatial_lr_scale},
            {'name': 'scaling',    'params': [self._scaling],   'lr': hp.scaling_lr},
            {'name': 'rotation',   'params': [self._rotation],  'lr': hp.rotation_lr},
            {'name': 'features',   'params': [self._features],  'lr': hp.feature_lr},
            {'name': 'opacity',    'params': [self._opacity],   'lr': hp.opacity_lr},
            {'name': 'importance', 'params': [self._importance], 'lr': hp.importance_lr},
            {'name': 'mlp_view',             'params': self.mlp_view            .parameters(), 'lr': hp.mlp_view_lr_init}   if hp.use_view_emb else None,
            {'name': 'mlp_color',            'params': self.mlp_color           .parameters(), 'lr': hp.mlp_color_lr_init},
            {'name': 'mlp_occlu',            'params': self.mlp_occlu           .parameters(), 'lr': hp.mlp_occlu_lr_init},
            {'name': 'embedding_appearance', 'params': self.embedding_appearance.parameters(), 'lr': hp.appearance_lr_init} if hp.appearance_dim > 0 else None,
            {'name': 'embedding_occlusion',  'params': self.embedding_occlusion .parameters(), 'lr': hp.occlusion_lr_init}  if hp.occlusion_dim  > 0 else None,
        ]
        param_groups = [it for it in param_groups if it is not None]
        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
        self.xyz_scheduler = get_expon_lr_func(
            lr_init=hp.position_lr_init * self.spatial_lr_scale,
            lr_final=hp.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=hp.position_lr_delay_mult,
            max_steps=hp.position_lr_max_steps,
        )
        self.view_scheduler_args       = get_expon_lr_func(**make_expon_lr_func_args(hp, 'mlp_view'))
        self.color_scheduler_args      = get_expon_lr_func(**make_expon_lr_func_args(hp, 'mlp_color'))
        self.occlu_scheduler_args      = get_expon_lr_func(**make_expon_lr_func_args(hp, 'mlp_occlu'))
        self.appearance_scheduler_args = get_expon_lr_func(**make_expon_lr_func_args(hp, 'appearance'))
        self.occlusion_scheduler_args  = get_expon_lr_func(**make_expon_lr_func_args(hp, 'occlusion'))
        self.xyz_grad_accum = torch.zeros((self.xyz.shape[0], 1), dtype=torch.float, device='cuda')
        self.xyz_grad_count = torch.zeros((self.xyz.shape[0], 1), dtype=torch.int,   device='cuda')
        self.max_radii2D    = torch.zeros((self.xyz.shape[0]),    dtype=torch.int,   device='cuda')
        self.percent_dense = hp.percent_dense

    def optimizer_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def update_learning_rate(self, steps:int):
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'xyz':
                lr = self.xyz_scheduler(steps)
            elif param_group['name'] == 'mlp_color':
                lr = self.color_scheduler_args(steps)
            elif param_group['name'] == 'mlp_occlu':
                lr = self.occlu_scheduler_args(steps)
            elif param_group['name'] == 'mlp_view':
                lr = self.view_scheduler_args(steps)
            elif param_group['name'] == 'embedding_appearance':
                lr = self.appearance_scheduler_args(steps)
            elif param_group['name'] == 'embedding_occlusion':
                lr = self.occlusion_scheduler_args(steps)
            else: continue  # skip
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
            if group['name'].startswith('mlp') or group['name'].startswith('embedding'): continue
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
            if group['name'].startswith('mlp') or group['name'].startswith('embedding'): continue
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

        self._xyz        = optimizable_tensors['xyz']
        self._scaling    = optimizable_tensors['scaling']
        self._rotation   = optimizable_tensors['rotation']
        self._features   = optimizable_tensors['features']
        self._opacity    = optimizable_tensors['opacity']
        self._importance = optimizable_tensors['importance']

        self.xyz_grad_accum = self.xyz_grad_accum[valid_points_mask]
        self.xyz_grad_count = self.xyz_grad_count[valid_points_mask]
        self.max_radii2D    = self.max_radii2D   [valid_points_mask]

    def densification_postfix(self, new_xyz:Tensor, new_scaling:Tensor, new_rotation:Tensor, new_features:Tensor, new_opacities:Tensor, new_importances:Tensor):
        states = {
            'xyz':        new_xyz,
            'scaling':    new_scaling,
            'rotation':   new_rotation,
            'features':   new_features,
            'opacity':    new_opacities,
            'importance': new_importances,
        }
        optimizable_tensors = self.cat_tensors_to_optimizer(states)

        self._xyz        = optimizable_tensors['xyz']
        self._scaling    = optimizable_tensors['scaling']
        self._rotation   = optimizable_tensors['rotation']
        self._features   = optimizable_tensors['features']
        self._opacity    = optimizable_tensors['opacity']
        self._importance = optimizable_tensors['importance']

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

        stds = self.scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device='cuda')
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz         = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.xyz[selected_pts_mask].repeat(N, 1)
        new_scaling     = self.scaling_inverse_activation(self.scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation    = self._rotation  [selected_pts_mask].repeat(N, 1)
        new_features    = self._features  [selected_pts_mask].repeat(N, 1)
        new_opacities   = self._opacity   [selected_pts_mask].repeat(N, 1)
        new_importances = self._importance[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_scaling, new_rotation, new_features, new_opacities, new_importances)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device='cuda', dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads:Tensor, grad_threshold:float, scene_extent:float):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(self.scaling, dim=1).values <= self.percent_dense*scene_extent)

        new_xyz         = self._xyz       [selected_pts_mask]
        new_scaling     = self._scaling   [selected_pts_mask]
        new_rotation    = self._rotation  [selected_pts_mask]
        new_features    = self._features  [selected_pts_mask]
        new_opacities   = self._opacity   [selected_pts_mask]
        new_importances = self._importance[selected_pts_mask]

        self.densification_postfix(new_xyz, new_scaling, new_rotation, new_features, new_opacities, new_importances)

    def densify_and_prune(self, max_grad:float, min_opacity:float, max_importance:float, extent:float, max_screen_size:int):
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

        # 删除不透明度小于设定阈值 and 椭球体半径过大的 3D Gaussian
        # 1、选择不透明度小于一定阈值的 3D Gaussian
        prune_mask = (self.opacity < min_opacity).squeeze()
        # 2、选择椭球体半径过大的 3D Gaussian
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        # # 3、选择重要性大于一定阈值的 Gaussian
        # importance_mask = (self.get_importance > max_importance).squeeze()
        # print(importance_mask.sum())
        # prune_mask = torch.logical_or(importance_mask, prune_mask)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor:Tensor, update_filter):
        self.xyz_grad_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.xyz_grad_count[update_filter] += 1

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.opacity, torch.ones_like(self.opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, 'opacity')
        self._opacity = optimizable_tensors['opacity']
