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
import numpy as np
from typing import List, Dict, Callable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch.optim import Optimizer
from plyfile import PlyData, PlyElement, PlyProperty
from simple_knn._C import distCUDA2

from modules.arguments import ModelParams, OptimizationParams
from modules.scene import Camera
from modules.data import BasicPointCloud
from modules.utils.sh_utils import RGB2SH


def strip_lowerdiag(L:Tensor) -> Tensor:
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device='cuda')
    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym:Tensor) -> Tensor:
    return strip_lowerdiag(sym)


def build_rotation(r:Tensor) -> Tensor:
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


def build_scaling_rotation(s:Tensor, r:Tensor) -> Tensor:
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device='cuda')
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L


def inverse_sigmoid(x:Tensor) -> Tensor:
    return torch.log(x / (1 - x))


def make_expon_lr_func_args(opt:OptimizationParams, prefix:str):
    kwargs = {
        'lr_init':        getattr(opt, f'{prefix}_lr_init'),
        'lr_final':       getattr(opt, f'{prefix}_lr_final'),
        'lr_delay_steps': getattr(opt, f'{prefix}_lr_delay_steps', None),
        'lr_delay_mult':  getattr(opt, f'{prefix}_lr_delay_mult', None),
        'max_steps':      getattr(opt, f'{prefix}_lr_max_steps', None),
    }
    return {k: v for k, v in kwargs.items() if v is not None}


def get_expon_lr_func(lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000):
    '''
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    '''

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


class GaussianModel(nn.Module):

    def __init__(self, mp:ModelParams):
        super().__init__()

        # props
        self._xyz:           Parameter = None
        self._scaling:       Parameter = None
        self._rotation:      Parameter = None
        self._features_dc:   Parameter = None
        self._features_rest: Parameter = None
        self._opacity:       Parameter = None
        self._importance:    Parameter = None
        # optim
        self.optimizer:        Optimizer = None
        self.xyz_scheduler:    Callable  = None
        self.xyz_grad_accum:   Tensor    = None
        self.xyz_grad_count:   Tensor    = None
        self.max_radii2D:      Tensor    = None
        self.max_sh_degree:    int       = mp.sh_degree
        self.cur_sh_degree:    int       = 0
        self.percent_dense:    float     = 0.01
        self.spatial_lr_scale: float     = 1.0

        self.mp = mp
        self.setup_transform_functions()

        # ↓↓↓ new add neural decoder (from bhy)
        self.embedding_appearance: nn.Embedding = nn.Identity()    # dummy
        self.embedding_occlusion: nn.Embedding = nn.Identity()

        if mp.use_view_emb:
            self.mlp_view = nn.Sequential(
                nn.Linear(4, mp.view_emb_dim), 
            )
            view_emb_dim = mp.view_emb_dim
        else:
            self.mlp_view = nn.Identity()
            view_emb_dim = 4
        view_emb_dim = 0        # tmp force ignore

        self.color_mlp_in_dim = mp.sh_feat_dim + view_emb_dim + (mp.appearance_dim if mp.add_view_emb_to_color else 0)
        self.mlp_color = nn.Sequential(
            nn.Linear(self.color_mlp_in_dim, mp.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mp.feat_dim, 3),
            nn.Sigmoid(),
        )

        self.occlu_mlp_in_dim = mp.sh_feat_dim + view_emb_dim + (mp.occlusion_dim if mp.add_view_emb_to_occlu else 0)
        self.mlp_occlu = nn.Sequential(
            nn.Linear(self.occlu_mlp_in_dim, mp.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mp.feat_dim, 1),
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

    @property
    def importance(self):
        return self.importance_activation(self._importance)

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
            '_importance':      self._importance,
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

    def load_state_dict(self, state_dict:Dict[str, Any], opt:OptimizationParams):
        # load data first
        self._xyz           = state_dict['_xyz']
        self._scaling       = state_dict['_scaling']
        self._rotation      = state_dict['_rotation']
        self._features_dc   = state_dict['_features_dc']
        self._features_rest = state_dict['_features_rest']
        self._opacity       = state_dict['_opacity']
        self._importance    = state_dict['_importance']
        # then recover optim state
        self.setup_training(opt)
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
            if not 'show':
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

            if not 'show':
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

    def save_ply(self, path:str):
        xyz          = self._xyz          .detach().cpu().numpy()
        scaling      = self._scaling      .detach().cpu().numpy()
        rotation     = self._rotation     .detach().cpu().numpy()
        feature_dc   = self._features_dc  .detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        feature_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities    = self._opacity      .detach().cpu().numpy()
        importances  = self._importance   .detach().cpu().numpy()

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
        vertexes = np.empty(xyz.shape[0], dtype=[(prop, 'f4') for prop in property_names])
        properties = np.concatenate((xyz, scaling, rotation, feature_dc, feature_rest, opacities, importances), axis=1)
        vertexes[:] = list(map(tuple, properties))
        elem = PlyElement.describe(vertexes, 'vertex')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        PlyData([elem]).write(path)

    def setup_training(self, opt:OptimizationParams):
        param_groups = [
            {'name': 'xyz',        'params': [self._xyz],           'lr': opt.position_lr_init * self.spatial_lr_scale},
            {'name': 'scaling',    'params': [self._scaling],       'lr': opt.scaling_lr},
            {'name': 'rotation',   'params': [self._rotation],      'lr': opt.rotation_lr},
            {'name': 'f_dc',       'params': [self._features_dc],   'lr': opt.feature_lr},
            {'name': 'f_rest',     'params': [self._features_rest], 'lr': opt.feature_lr / 20.0},
            {'name': 'opacity',    'params': [self._opacity],       'lr': opt.opacity_lr},
            {'name': 'importance', 'params': [self._importance],    'lr': opt.importance_lr},
            {'name': 'mlp_color',  'params': self.mlp_color.parameters(), 'lr': opt.m_loss_depth} if self.mp.use_neural_decoder else None,
        ]
        param_groups = [grp for grp in param_groups if grp is not None]
        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
        self.xyz_scheduler = get_expon_lr_func(
            lr_init=opt.position_lr_init * self.spatial_lr_scale,
            lr_final=opt.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=opt.position_lr_delay_mult,
            max_steps=opt.position_lr_max_steps,
        )
        self.xyz_grad_accum = torch.zeros((self.xyz.shape[0], 1), dtype=torch.float, device='cuda')
        self.xyz_grad_count = torch.zeros((self.xyz.shape[0], 1), dtype=torch.int,   device='cuda')
        self.max_radii2D    = torch.zeros((self.xyz.shape[0]),    dtype=torch.int,   device='cuda')
        self.percent_dense = opt.percent_dense

        if self.mp.use_neural_decoder:
            self.color_scheduler_args = get_expon_lr_func(**make_expon_lr_func_args(opt, 'mlp_color'))

    def optimizer_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def update_learning_rate(self, steps:int):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'xyz':
                lr = self.xyz_scheduler(steps)
            elif param_group["name"] == 'mlp_color':
                lr = self.color_scheduler_args(steps)
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
            if group['name'].startswith('mlp'): continue
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
            if group['name'].startswith('mlp'): continue
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

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_grad_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.xyz_grad_count[update_filter] += 1

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.opacity, torch.ones_like(self.opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, 'opacity')
        self._opacity = optimizable_tensors['opacity']

    def oneup_SH_degree(self):
        if self.cur_sh_degree < self.max_sh_degree:
            self.cur_sh_degree += 1
