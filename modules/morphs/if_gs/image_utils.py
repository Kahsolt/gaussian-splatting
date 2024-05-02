#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/18

import os
from copy import deepcopy
from typing import Callable, List, Union

import torch
import torch.nn.functional as F
from torch.fft import fft2, fftshift, ifft2, ifftshift
from torch import Tensor
import torchvision.transforms.functional as TF
import numpy as np
from numpy import ndarray

from modules.utils.general_utils import minmax_norm

try:
    # https://pytorch-wavelets.readthedocs.io/en/latest/readme.html
    from pytorch_wavelets import DWT, IDWT
    HAS_PYTORCH_WAVELETS = True
except ImportError:
    print('>> pytorch_wavelets not installed, run "pip install submodules/pytorch_wavelets" for it!')
    HAS_PYTORCH_WAVELETS = False

DEBUG = os.getenv('DEBUG', False)


''' Interface '''

def split_freqs(method:str, X:Tensor, *args, **kwargs) -> List[Tensor]:
    return globals().get(f'split_freqs_{method}')(X, *args, **kwargs)

def combine_freqs(method:str, X_freqs:List[Tensor], *args, **kwargs) -> Tensor:
    return globals().get(f'combine_freqs_{method}')(X_freqs, *args,  **kwargs)

def _mix_cp(vmax:int, n_cp:int, scale_w:float=0.1) -> List[int]:
    cp_l = np.linspace(0, vmax, n_cp)
    cp_m = np.expm1(np.linspace(0, np.log1p(vmax), n_cp))
    cp_mix = cp_l * scale_w + cp_m * (1 - scale_w)
    cp = np.clip(np.round(cp_mix), 0, vmax).astype(np.int32).tolist()
    if DEBUG: print('cp:', cp)
    return cp


''' Implementations '''

def split_freqs_unsharp_mask(X:Tensor, r:int=5, s:float=None, *args, **kwargs) -> List[Tensor]:
    low = TF.gaussian_blur(X, kernel_size=r, sigma=s)
    high = X - low
    return [low, high]

def combine_freqs_unsharp_mask(X_freqs:List[Tensor], *args, **kwargs) -> Tensor:
    return torch.stack(X_freqs, dim=0).sum(dim=0)


def split_freqs_fft(X:Tensor, n_bands:int=3, scale_w:float=0.1, *args, **kwargs) -> List[Tensor]:
    '''
    ## Parameters
    - X: Tensor in shape [C, H, W]. Input image.
    - n_bands: int, optional. How many freqency bands to split to.
    - scale_w: int, optional. Balancing factor for cut-points, larger value biases to linear while smaller biases to exponential.
    ## Returns
    - X_freqs: List[Tensor]. The splitted images of each freq range.
    '''
    assert len(X.shape) == 3, 'X should be shape of [C, H, W]'

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
    cp_h = _mix_cp(np.ceil(H / 2), n_bands + 1, scale_w)
    cp_w = _mix_cp(np.ceil(W / 2), n_bands + 1, scale_w)
    D_freqs = [get_masked(D, cp_h[i+1], cp_w[i+1], cp_h[i], cp_w[i]) for i in range(n_bands)]
    return [ifft2(ifftshift(D_freq)).real for D_freq in D_freqs]

def combine_freqs_fft(X_freqs:List[Tensor], *args, **kwargs) -> Tensor:
    '''
    ## Parameters
    - X_freqs: List[Tensor] with Tensor in shape [C, H, W]. Input sub-freq images.
    ## Returns
    - X: Tensor. The combined image.
    '''
    for X_freq in X_freqs: assert len(X_freq.shape) == 3, 'X should be shape of [C, H, W]'

    return ifft2(torch.stack([fft2(X_freq) for X_freq in X_freqs], dim=0).sum(dim=0)).real


def split_freqs_svd(X:Tensor, n_bands:int=3, scale_w:float=0.1, *args, **kwargs) -> List[Tensor]:
    def get_masked(S:Tensor, right:int, left:int=None) -> Tensor:
        nonlocal K
        S_hat = torch.zeros_like(S)
        slicer = slice(0, min(right, K))
        S_hat[..., slicer] = S[..., slicer]
        if left:
            slicer = slice(0, max(left, 0))
            S_hat[..., slicer] = 0.0
        return S_hat

    U, S, V = torch.svd(X)
    K = S.shape[-1]
    cp = _mix_cp(K, n_bands + 1, scale_w)
    S_freqs = [get_masked(S, cp[i+1], cp[i]) for i in range(n_bands)]
    return [torch.matmul(torch.matmul(U, torch.diag_embed(S_freq)), V.mT) for S_freq in S_freqs]

def combine_freqs_svd(X_freqs:List[Tensor], *args, **kwargs) -> Tensor:
    return torch.stack(X_freqs, dim=0).sum(dim=0)


DWT_CACHE = {}

def inject_dwt(fn:Callable[[Union[Tensor, List[Tensor]], DWT, IDWT], Union[List[Tensor], Tensor]]):
    def wrapper(X_or_X_freqs:Union[Tensor, List[Tensor]], n_freqs:int=3, wavlet:str='db3', padding:str='zero', *args, **kwargs):
        n_layers = n_freqs - 1
        key = f'DWT-L={n_layers}_W={wavlet}_P={padding}'
        if key not in DWT_CACHE:
            DWT_CACHE[key] = DWT(J=n_layers, wave=wavlet, mode=padding)
        xfm = DWT_CACHE[key]
        key = f'iDWT-_W={wavlet}_P={padding}'
        if key not in DWT_CACHE:
            DWT_CACHE[key] = IDWT(wave=wavlet, mode=padding)
        ifm = DWT_CACHE[key]
        return fn(X_or_X_freqs, xfm, ifm)
    return wrapper

@inject_dwt
def split_freqs_dwt(X:Tensor, xfm:DWT, ifm:IDWT) -> List[Tensor]:
    # split to components
    YL, YH = xfm(X.unsqueeze(0))    # LL, H-list
    # make blanks
    YL_blank = torch.zeros_like(YL)
    YH_blank = [torch.zeros_like(it) for it in YH]
    # decode components
    imgs = []
    img = _resize_match(ifm((YL, YH_blank)).squeeze(0), X)
    imgs.append(img)
    for i in range(len(YH)):
        YH_tmp = deepcopy(YH_blank)
        YH_tmp[i] = YH[i]
        img = _resize_match(ifm((YL_blank, YH_tmp)).squeeze(0), X)
        imgs.append(img)
    return imgs

@inject_dwt
def combine_freqs_dwt(X_freqs:List[Tensor], xfm:DWT, ifm:IDWT) -> Tensor:
    # split to components
    YLs, YHs = [], []
    for X_freq in X_freqs:
        YL, YH = xfm(X_freq.unsqueeze(0))
        YLs.append(YL)
        YHs.append(YH)
    # merge components
    YL = sum(YLs)
    YH = [sum(YH_layer) for YH_layer in zip(*YHs)]
    # decode components
    return _resize_match(ifm((YL, YH)).squeeze(0), X_freqs[0])


''' Tests '''

def _resize_match(x:Tensor, y:Tensor) -> Tensor:
    if x.shape[-2:] == y.shape[-2:]: return x
    return F.interpolate(x.unsqueeze(0), size=y.shape[-2:], mode='nearest').squeeze(0)

def _show_error(x:Tensor, y:Tensor, title:str='img'):
    x = _resize_match(x, y)    # occrus in dwt

    print(f'[{title}] error')
    print('  Linf:', torch.abs(x - y).max().item())
    print('  l1_loss:', F.l1_loss(x, y).item())
    print('  mse_loss:', F.mse_loss(x, y).item())

def _make_grid(imgs:List[Tensor]) -> ndarray:
    from torchvision.utils import make_grid
    return torch.permute(make_grid(torch.stack(imgs, dim=0)), (1, 2, 0)).clamp_(0.0, 1.0).cpu().numpy()

def test_unsharp_mask(X:Tensor) -> ndarray:
    # settings
    r = 11
    s = 100

    splitted = split_freqs_unsharp_mask(X, r, s)
    combined = combine_freqs_unsharp_mask(splitted)
    _show_error(combined, X, 'split_freqs_unsharp_mask')

    splitted[-1] = minmax_norm(splitted[-1])       # renorm high for display
    return _make_grid(splitted + [combined, X])

def test_fft(X:Tensor) -> ndarray:
    X_recon = ifft2(ifftshift(fftshift(fft2(X)))).real
    _show_error(X_recon, X, 'torch.fft')

    # settings
    n_freqs = 3
    scale_w = 0.05

    splitted = split_freqs_fft(X, n_freqs, scale_w)
    combined = combine_freqs_fft(splitted)
    _show_error(combined, X, 'split_freqs_fft')

    return _make_grid(splitted + [combined, X])

def test_svd(X:Tensor) -> ndarray:
    U, S, V = torch.svd(X)
    X_recon = torch.matmul(torch.matmul(U, torch.diag_embed(S)), V.mT)
    _show_error(X_recon, X, 'torch.svd')

    # settings
    n_freqs = 3
    scale_w = 0.001

    splitted = split_freqs_svd(X, n_freqs, scale_w)
    combined = combine_freqs_svd(splitted)
    _show_error(combined, X, 'split_freqs_svd')

    return _make_grid(splitted + [combined, X])

def test_dwt(X:Tensor) -> ndarray:
    if not HAS_PYTORCH_WAVELETS: return

    xfm = DWT()
    ifm = IDWT()
    YL, YH = xfm(X.unsqueeze(0))
    X_recon = ifm((YL, YH)).squeeze(0)
    _show_error(X_recon, X, 'pytorch_wavelets')

    # settings
    n_freqs = 3
    wavlet = 'db1'
    padding = 'zero'

    splitted = split_freqs_dwt(X, n_freqs, wavlet, padding)
    combined = combine_freqs_dwt(splitted, n_freqs, wavlet, padding)
    _show_error(combined, X, 'split_freqs_fft')

    return _make_grid(splitted + [combined, X])


if __name__ == '__main__':
    from PIL import Image
    import matplotlib.pyplot as plt
    from modules.utils.general_utils import DATA_PATH

    fp = DATA_PATH / 'tandt' / 'train' / 'images' / '00001.jpg'
    img = Image.open(fp).convert('RGB')
    X = TF.to_tensor(img)     # [C, H, W]

    split_fns = [
        test_unsharp_mask,
        test_fft,
        test_dwt,
        test_svd,
    ]
    grids = [split_fn(X) for split_fn in split_fns]
    grids_names = [(it, split_fns[i].__name__) for i, it in enumerate(grids) if it is not None]

    n_figs = len(grids_names)
    plt.clf()
    for i, (grid, name) in enumerate(grids_names):
        plt.subplot(n_figs, 1, i+1)
        plt.imshow(grid)
        plt.axis('off')
        plt.title(name)
    plt.tight_layout()
    plt.show()
    plt.close()
