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
import torch.nn as nn
import numpy as np
from simple_knn._C import distCUDA2

from modules.data import BasicPointCloud
from modules.utils.modeling_utils import inverse_sigmoid
from modules.morphs.mlp_gs.model import GaussianModel as SingleFreqGaussianModel

from .hparam import HyperParams


class MutilFreqGaussianModel:

    ''' based on mlp-gs '''

    def __init__(self, hp:HyperParams):
        self.hp: HyperParams = hp

        self.gaussians = {idx: SingleFreqGaussianModel(hp) for idx in range(hp.n_freqs)}
        self.cur_idx = 0

        self.spatial_lr_scale = 1.0

    @property
    def n_gaussians(self):
        return len(self.gaussians)

    @property
    def cur_gaussian(self):
        return self.gaussians[self.cur_idx]

    def get_gaussian(self, idx:int):
        return self.gaussians[idx]

    def activate_gaussian(self, idx:int=0, swap:bool=True):
        last_idx = self.cur_idx
        if idx == last_idx:
            self.gaussians[self.cur_idx].cuda()
        else:
            self.cur_idx = idx
            if swap:
                self.gaussians[last_idx].cpu()
            self.gaussians[self.cur_idx].cuda()

    def from_pcd(self, pcd:BasicPointCloud):
        hp = self.hp

        init_xyz = torch.tensor(np.asarray(pcd.points), dtype=torch.float, device='cuda')
        n_pts = init_xyz.shape[0]
        init_features = torch.randn(n_pts, hp.feat_dim).float().cuda()
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        init_scaling = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)  # (N, 3)
        init_rotation = torch.zeros((n_pts, 4), dtype=torch.float, device='cuda')  # (N, 4)
        init_rotation[:, 0] = 1
        init_opacity = inverse_sigmoid(0.1 * torch.ones((n_pts, 1), dtype=torch.float, device='cuda'))

        if hp.sub_gauss_init == 'copy':
            for idx in range(hp.n_freqs):
                gaussians = self.get_gaussian(idx)
                gaussians.spatial_lr_scale = self.spatial_lr_scale
                gaussians._xyz      = nn.Parameter(init_xyz,      requires_grad=True)
                gaussians._scaling  = nn.Parameter(init_scaling,  requires_grad=True)
                gaussians._rotation = nn.Parameter(init_rotation, requires_grad=True)
                gaussians._features = nn.Parameter(init_features, requires_grad=True)
                gaussians._opacity  = nn.Parameter(init_opacity,  requires_grad=True)
                print(f'Number of points of freq_{idx} at initialization:', gaussians.n_points)
        elif hp.sub_gauss_init == 'uniform_sparse':
            for idx in range(hp.n_freqs):
                gaussians = self.get_gaussian(idx)
                gaussians.spatial_lr_scale = self.spatial_lr_scale
                gaussians._xyz      = nn.Parameter(init_xyz     [idx::hp.n_freqs], requires_grad=True)
                gaussians._scaling  = nn.Parameter(init_scaling [idx::hp.n_freqs], requires_grad=True)
                gaussians._rotation = nn.Parameter(init_rotation[idx::hp.n_freqs], requires_grad=True)
                gaussians._features = nn.Parameter(init_features[idx::hp.n_freqs], requires_grad=True)
                gaussians._opacity  = nn.Parameter(init_opacity [idx::hp.n_freqs], requires_grad=True)
                print(f'Number of points of freq_{idx} at initialization:', gaussians.n_points)
        else:
            raise ValueError(f'Unknown kind: {hp.sub_gauss_init}')


# unify interface name :)
GaussianModel = MutilFreqGaussianModel
