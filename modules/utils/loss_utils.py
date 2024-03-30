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
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable


def mse(img1:Tensor, img2:Tensor) -> Tensor:
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1:Tensor, img2:Tensor) -> Tensor:
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def l1_loss(render:Tensor, gt:Tensor, reduction='mean') -> Tensor:
    assert reduction in ['none', 'mean']
    loss = torch.abs((render - gt))
    return loss.mean() if reduction == 'mean' else loss

def l2_loss(render:Tensor, gt:Tensor) -> Tensor:
    return ((render - gt) ** 2).mean()

def nerfw_loss(render:Tensor, gt:Tensor, beta:Tensor, beta_min:float=0.03) -> Tensor:
    # shift minimal according to the essay
    beta = beta_min + beta
    loss_pix = torch.square(render - gt) / (2 * beta ** 2)
    # by def. log(β²)/2 = log(β)
    # NOTE: this term can be negative, so +3 to make loss positive, trick impl. borrowed from in nerf_pl and nerfstudio :(
    loss_reg = torch.log(beta) + 3
    loss = loss_pix + loss_reg
    return loss.mean()


# ↓↓↓ borrowed from https://github.com/Po-Hsun-Su/pytorch-ssim/

def _gaussian(window_size, sigma):
    from math import exp
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def _create_window(window_size, channel):
    _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, reduction='mean'):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if reduction == 'none':
        return ssim_map
    elif reduction == 'batchmean':
        return ssim_map.mean()
    else:   # 'mean'
        return ssim_map.mean(1).mean(1).mean(1)

def ssim(img1, img2, window_size=11, reduction='mean'):
    assert reduction in ['none', 'mean', 'batchmean']
    channel = img1.size(-3)
    window = _create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, reduction)
