#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/18

from typing import Tuple, List

import torch
import torch.nn.functional as F
from torch.fft import fft2, fftshift, ifft2, ifftshift
from torch import Tensor
import torchvision.transforms.functional as TF
import numpy as np


def split_freqs_torch(X:Tensor, n_bands:int=4, scale_w:float=0.1, method='cumulative') -> List[Tensor]:
    '''
    ## Parameters
    - X: Tensor in shape [C, H, W]. Input image.
    - n_bands: int, optional. How many freqency bands to split to.
    - scale_w: int, optional. Balancing factor for cut-points, larger value biases to linear while smaller biases to exponential.
    - method: str, optional. Split method, choose from ["cumulative", "addictive"].
    ## Returns
    - X_freqs: List[Tensor]. The splitted images of each freq range.
    '''
    assert len(X.shape) == 3, 'X should be shape of [C, H, W]'
    assert method in ['cumulative', 'addictive'], 'choose method from ["cumulative", "addictive"]'

    def get_cp(scale_w:float) -> Tuple[List[int]]:
        nonlocal H, W
        cp_l_h = np.linspace(0, H // 2, n_bands + 1)
        cp_l_w = np.linspace(0, W // 2, n_bands + 1)
        cp_m_h = np.exp(np.linspace(0, np.log(H) // 2 + 1, n_bands + 1)) - 1
        cp_m_w = np.exp(np.linspace(0, np.log(W) // 2 + 1, n_bands + 1)) - 1
        cp_h = cp_l_h * scale_w + cp_m_h * (1 - scale_w)
        cp_w = cp_l_w * scale_w + cp_m_w * (1 - scale_w)
        cp_h[-1] = cp_l_h[-1]
        cp_w[-1] = cp_l_w[-1]
        cp_h = np.clip(np.round(cp_h).astype(int), 0, H//2).tolist()
        cp_w = np.clip(np.round(cp_w).astype(int), 0, W//2).tolist()
        return cp_h, cp_w

    def get_masked(D:Tensor, h_out:int, w_out:int, h_in:int=None, w_in:int=None) -> Tensor:
        nonlocal H, W
        h, w = H//2, W//2     # central point
        D_hat = torch.zeros_like(D)
        slicer_h = slice(max(0, h-h_out), min(H-1, h+h_out))
        slicer_w = slice(max(0, w-w_out), min(W-1, w+w_out))
        D_hat[..., slicer_h, slicer_w] = D[..., slicer_h, slicer_w]
        if h_in and w_in:
            slicer_h = slice(max(0, h-h_in), min(H-1, h+h_in))
            slicer_w = slice(max(0, w-w_in), min(W-1, w+w_in))
            D_hat[..., slicer_h, slicer_w] = 0.0
        return D_hat

    D = fftshift(fft2(X))
    H, W = X.shape[-2:]
    cp_h, cp_w = get_cp(scale_w)
    if method == 'cumulative':        # [low, low+mid, ..., gt]
        D_freqs = [get_masked(D, cp_h[i+1], cp_w[i+1]) for i in range(n_bands)] + [D]
    elif method == 'addictive':       # [gt, low, mid, ..., high]
        D_freqs = [D] + [get_masked(D, cp_h[i+1], cp_w[i+1], cp_h[i], cp_w[i]) for i in range(n_bands)]
    return [ifft2(ifftshift(D_freq)).real for D_freq in D_freqs]


def combine_freqs_torch(X_freqs:List[Tensor]) -> Tensor:
    '''
    ## Parameters
    - X_freqs: List[Tensor] with Tensor in shape [C, H, W]. Input sub-freq images.
    ## Returns
    - X: Tensor. The combined image.
    '''
    for X_freq in X_freqs: assert len(X_freq.shape) == 3, 'X should be shape of [C, H, W]'

    return ifft2(torch.stack([fft2(X_freq) for X_freq in X_freqs], dim=0).sum(dim=0)).real


if __name__ == '__main__':
    from PIL import Image
    import matplotlib.pyplot as plt
    from modules.utils.general_utils import DATA_PATH
    from torchvision.utils import make_grid

    compose = lambda imgs: make_grid(torch.stack(imgs, dim=0))
    to_np = lambda X: torch.permute(X, (1, 2, 0)).clamp_(0.0, 1.0).cpu().numpy()

    fp = DATA_PATH / 'tandt' / 'train' / 'images' / '00001.jpg'
    img = Image.open(fp).convert('RGB')
    X = TF.to_tensor(img)     # [C, H, W]

    print('[torch.fft] error')
    X_recon = ifft2(ifftshift(fftshift(fft2(X)))).real
    print('  Linf:', torch.abs(X_recon - X).max().item())
    print('  l1_loss:', F.l1_loss(X_recon, X).item())
    print('  mse_loss:', F.mse_loss(X_recon, X).item())

    L_freq = 3
    scale_w = 0.05

    split_add = split_freqs_torch(X, L_freq, scale_w, 'addictive')
    split_cum = split_freqs_torch(X, L_freq, scale_w, 'cumulative')

    plt.clf()
    plt.subplot(211) ; plt.imshow(to_np(compose(split_add))) ; plt.axis('off') ; plt.title('split (addictive)')
    plt.subplot(212) ; plt.imshow(to_np(compose(split_cum))) ; plt.axis('off') ; plt.title('split (cumulative)')
    plt.tight_layout()
    plt.show()
    plt.close()

    combine_add = combine_freqs_torch(split_add[1:])    # exclude gt

    print('[split-combine] error')
    print('  Linf:', torch.abs(combine_add - X).max().item())
    print('  l1_loss:', F.l1_loss(combine_add, X).item())
    print('  mse_loss:', F.mse_loss(combine_add, X).item())

    plt.clf()
    plt.imshow(to_np(combine_add))
    plt.axis('off')
    plt.title('combine (cumulative)')
    plt.tight_layout()
    plt.show()
    plt.close()
