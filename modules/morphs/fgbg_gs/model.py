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

from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
from plyfile import PlyElement
import numpy as np
from numpy import ndarray
from simple_knn._C import distCUDA2

from modules.data import BasicPointCloud
from modules.layers import ColorMLP, Embedding
from modules.model import GaussianModel_Neural
from modules.utils.modeling_utils import *
from modules.utils.training_utils import *

from .hparam import HyperParams
from .camera import Camera


class GaussianModel(GaussianModel_Neural):

    ''' double buffer fg-bg '''

    def __init__(self, hp:HyperParams):
        super().__init__(hp)

        self.hp: HyperParams

        # props
        self._importance: Parameter = None

        # networks
        self.appear_embedding: Embedding = nn.Identity()
        self.occlus_embedding: Embedding = nn.Identity()

        self.image_color_mlp = ColorMLP(hp.feat_dim + hp.appear_embedding_dim + 4, hp.hidden_dim)
        self.image_alpha_mlp = ColorMLP(hp.feat_dim + hp.appear_embedding_dim + 4, hp.hidden_dim, out_dim=1)
        self.occlu_color_mlp = ColorMLP(hp.feat_dim + hp.occlus_embedding_dim + 4, hp.hidden_dim)
        self.occlu_alpha_mlp = ColorMLP(hp.feat_dim + hp.occlus_embedding_dim + 4, hp.hidden_dim, out_dim=1)

    def feature_encoder(self, vp_cam:Camera, visible_mask:Tensor=slice(None)) -> List[Tuple[Tensor, Tensor]]:
        feats = self.features[visible_mask]

        hp = self.hp
        if 'view feats':
            pts = self.xyz[visible_mask]
            ob_view = pts - vp_cam.camera_center            # (N, 3)
            ob_dist = ob_view.norm(dim=1, keepdim=True)     # (N, 1)
            ob_view = ob_view / ob_dist                     # (N, 3)
            ob_dist = torch.log(ob_dist)
            view_feat = torch.cat([ob_view, ob_dist], -1)

            cam_idx = torch.ones_like(feats[:, 0], dtype=torch.long) * vp_cam.uid
            appear_embed = self.appear_embedding(cam_idx)
            occlus_embed = self.occlus_embedding(cam_idx)

        image_color = self.image_color_mlp(torch.cat([feats, appear_embed, view_feat], -1))
        image_alpha = self.image_alpha_mlp(torch.cat([feats, appear_embed, view_feat], -1))
        occlu_color = self.occlu_color_mlp(torch.cat([feats, occlus_embed, view_feat], -1))
        occlu_alpha = self.occlu_alpha_mlp(torch.cat([feats, occlus_embed, view_feat], -1))
        return [
            (image_color, image_alpha),
            (occlu_color, occlu_alpha),
        ]

    @property
    def importance(self):
        return self._importance

    def init_embeddings(self, num_cameras:int):
        hp = self.hp
        self.appear_embedding = Embedding(num_cameras, hp.appear_embedding_dim)
        self.occlus_embedding = Embedding(num_cameras, hp.occlus_embedding_dim)

    def state_dict(self) -> Dict[str, Any]:
        state_dict = super().state_dict()
        state_dict.update({
            'appear_embedding': self.appear_embedding.state_dict(),
            'occlus_embedding': self.occlus_embedding.state_dict(),
            'image_color_mlp': self.image_color_mlp.state_dict(),
            'occlu_color_mlp': self.occlu_color_mlp.state_dict(),
        })
        return state_dict

    def load_state_dict(self, state_dict:Dict[str, Any]):
        self.appear_embedding.load_state_dict(state_dict['appear_embedding'])
        self.occlus_embedding.load_state_dict(state_dict['occlus_embedding'])
        self.image_color_mlp.load_state_dict(state_dict['image_color_mlp'])
        self.occlu_color_mlp.load_state_dict(state_dict['occlu_color_mlp'])
        super().load_state_dict(state_dict)

    def save_pth(self, fp:Path):
        fp.parent.mkdir(exist_ok=True, parents=True)
        state_dict = {
            'view_embedding': self.appear_embedding.state_dict(),
            'image_color_mlp': self.image_color_mlp.state_dict(),
            'occlu_color_mlp': self.occlu_color_mlp.state_dict(),
        }
        torch.save(state_dict, fp)

    def load_pth(self, fp:Path):
        state_dict = torch.load(fp)
        self.appear_embedding.load_state_dict(state_dict['appear_embedding'])
        self.occlus_embedding.load_state_dict(state_dict['occlus_embedding'])
        self.image_color_mlp.load_state_dict(state_dict['image_color_mlp'])
        self.occlu_color_mlp.load_state_dict(state_dict['occlu_color_mlp'])

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
        importances = torch.ones_like(opacities) * 0.5

        self._xyz        = nn.Parameter(points,      requires_grad=True)
        self._scaling    = nn.Parameter(scales,      requires_grad=True)
        self._rotation   = nn.Parameter(rots,        requires_grad=True)
        self._features   = nn.Parameter(features,    requires_grad=True)
        self._opacity    = nn.Parameter(opacities,   requires_grad=True)
        self._importance = nn.Parameter(importances, requires_grad=True)

    def load_ply(self, elem:PlyElement):
        super().load_ply(elem)

        importances = np.asarray(elem["importance"])[..., np.newaxis]
        self._importance = nn.Parameter(torch.tensor(importances, dtype=torch.float, device="cuda"), requires_grad=True)

    def save_ply(self) -> Tuple[List[ndarray], List[str]]:
        property_data, property_names = super().save_ply()
        property_data.extend([
            self._importance.detach().cpu().numpy(),
        ])
        property_names.extend([
            'importance',
        ])
        return property_data, property_names   

    def make_param_group(self) -> List[Dict[str, Any]]:
        hp = self.hp
        param_group = super().make_param_group()
        param_group.extend([
            {'name': 'importance',       'params': [self._importance],                 'lr': hp.importance_lr},
            {'name': 'bg_mlp',           'params': self.image_color_mlp .parameters(), 'lr': hp.mlp_lr_init},
            {'name': 'fg_mlp',           'params': self.occlu_color_mlp .parameters(), 'lr': hp.mlp_lr_init},
            {'name': 'appear_embedding', 'params': self.appear_embedding.parameters(), 'lr': hp.embedding_lr_init},
            {'name': 'occlus_embedding', 'params': self.occlus_embedding.parameters(), 'lr': hp.embedding_lr_init},
        ])
        return param_group

    def setup_training(self):
        super().setup_training()
        self.mlp_scheduler_args = get_expon_lr_func(**make_expon_lr_func_args(self.hp, 'mlp'))
        self.embedding_scheduler_args = get_expon_lr_func(**make_expon_lr_func_args(self.hp, 'embedding'))

    def update_learning_rate(self, steps:int):
        super().update_learning_rate(steps)
        for param_group in self.optimizer.param_groups:
            if param_group['name'].endswith('_mlp'):
                lr = self.mlp_scheduler_args(steps)
            elif param_group['name'].endswith('_embedding'):
                lr = self.embedding_scheduler_args(steps)
            else: continue  # skip
            param_group['lr'] = lr

    def cat_tensors_to_optimizer(self, tensors_dict:Dict[str, Tensor], excludes:List[str]=[]) -> Dict[str, Tensor]:
        return super().cat_tensors_to_optimizer(tensors_dict, excludes + ['*_mlp', '*_embedding'])

    def prune_optimizer(self, mask:Tensor, excludes:List[str]=[]) -> Dict[str, Tensor]:
        return super().prune_optimizer(mask, excludes + ['*_mlp', '*_embedding'])

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

        #prune_mask = (self.opacity < min_opacity).squeeze()
        #if max_screen_size:
        #    big_points_vs = self.max_radii2D > max_screen_size
        #    big_points_ws = self.scaling.max(dim=1).values > 0.1 * extent
        #    prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        #self.prune_points(prune_mask)

        torch.cuda.empty_cache()
