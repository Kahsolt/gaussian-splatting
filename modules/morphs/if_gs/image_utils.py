#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/12

from typing import List, Union, Literal

from PIL import Image
from PIL.Image import Image as PILImage
import numpy as np
try:
    from pyfftw.interfaces.numpy_fft import fft2, fftshift, ifft2, ifftshift
except ImportError:
    print('>> [Warn] You need run "pip install pyfftw" to speed up the fucking FFT processes!!')
    from numpy.fft import fft2, fftshift, ifft2, ifftshift
from numpy.typing import NDArray

npimg = NDArray[np.float32]
spec  = NDArray[np.complex128]
SpecKind = Union[
  Literal['addictive'],
  Literal['cumulative'],
]


def np_to_pil(im:npimg) -> PILImage:
    return Image.fromarray((im * 255).clip(0, 255).astype(np.uint8)).convert('RGB')

def pil_to_np(img:PILImage) -> npimg:
    return np.asarray(img, dtype=np.float32)

def imread(fp:str, mode='RGBA') -> npimg:
    return pil_to_np(Image.open(fp).convert(mode))


def split_and_fft(im:npimg) -> List[spec]:
    ''' split image to grey layers, then apply fft2d '''
    if len(im.shape) == 2: im = np.expand_dims(im, axis=-1)  # [H, W, C]
    return [fftshift(fft2(layer)) for layer in [im[:, :, i] for i in range(im.shape[-1])]]   # [H, W]

def ifft_and_merge(freqs_dft:List[List[spec]]) -> List[npimg]:
    ''' apply ifft2d, then merge all grey layers back to image '''
    back = [[np.abs(ifft2(ifftshift(freq))) for freq in layer] for layer in freqs_dft]
    return [np.stack([*layers], axis=-1) for layers in zip(*back)]    # [H, W, C]

def split_freqs(im:npimg, n_bands:int=4, scale_w:float=0.1, kind='cumulative') -> List[npimg]:
    ''' scale_w: smaller value close to log, larger value close to linear '''

    M = split_and_fft(im)
    H, W = im.shape[:-1]

    cp_l_h = np.linspace(0, H // 2, n_bands + 1)
    cp_l_w = np.linspace(0, W // 2, n_bands + 1)
    cp_m_h = np.exp(np.linspace(0, np.log(H) // 2 + 1, n_bands+1)) - 1
    cp_m_w = np.exp(np.linspace(0, np.log(W) // 2 + 1, n_bands+1)) - 1
    cp_h = cp_l_h * scale_w + cp_m_h * (1 - scale_w)
    cp_w = cp_l_w * scale_w + cp_m_w * (1 - scale_w)
    cp_h = np.clip(np.round(cp_h).astype(int), 0, None).tolist()
    cp_w = np.clip(np.round(cp_w).astype(int), 0, None).tolist()

    def get_band(M:spec, h_out, w_out, h_in=None, w_in=None):
        H, W = M.shape
        h, w = H//2, W//2     # central point

        M_hat = np.zeros_like(M)
        slicer = slice(h-h_out, h+h_out), slice(w-w_out, w+w_out)
        M_hat[slicer] = M[slicer]
        if h_in:
            slicer = slice(h-h_in, h+h_in), slice(w-w_in, w+w_in)
            M_hat[slicer] = 0.0
        return M_hat

    if kind == 'addictive':
        M_bands = [[layer] + [get_band(layer, cp_h[i+1], cp_w[i+1], cp_h[i], cp_w[i]) for i in range(len(cp_h)-1)]  for layer in M]
    elif kind == 'cumulative': 
        M_bands = [[get_band(layer, cp_h[i+1], cp_w[i+1]) for i in range(len(cp_h)-1)] + [layer] for layer in M]
    else: 
        raise ValueError(f'unknown kind: {kind}')
    
    return ifft_and_merge(M_bands)
