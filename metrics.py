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

import warnings ; warnings.filterwarnings(action='ignore', category=UserWarning)

import json
from pathlib import Path
from argparse import ArgumentParser
from traceback import print_exc
from typing import List, Tuple

import torch
from torch import Tensor
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from modules.lpipsPyTorch import LPIPS
from modules.utils.loss_utils import psnr, ssim

torch.cuda.set_device(torch.device('cuda:0'))

mean = lambda x: sum(x) / len(x) if len(x) else float('nan')
pil_to_tensor = lambda x: TF.to_tensor(x)[:3, :, :].unsqueeze(0).cuda()


def read_images(renders_dir:Path, gt_dir:Path) -> Tuple[List[Tensor], List[Tensor], List[str]]:
    renders = []
    gts     = []
    names   = []
    for fp in renders_dir.iterdir():
        fn = fp.name
        renders.append(pil_to_tensor(Image.open(renders_dir / fn)))
        gts.append(pil_to_tensor(Image.open(gt_dir / fn)))
        names.append(fn)
    return renders, gts, names


def evaluate(model_paths:List[Path], split:str='test'):
    lpips = LPIPS(net_type='vgg').cuda()

    for scene_dir in model_paths:
        try:
            print(f'Scene: {scene_dir} [{split}]')
            results_agg      = {}
            results_per_view = {}

            test_dir = scene_dir / split
            for method_dir in test_dir.iterdir():
                method = method_dir.name
                print(f'Method: {method}')
                results_agg     [method] = {}
                results_per_view[method] = {}

                renders_dir = method_dir / 'renders'
                gt_dir = method_dir / 'gt'
                renders, gts, image_names = read_images(renders_dir, gt_dir)

                ssims  = []
                psnrs  = []
                lpipss = []
                for idx in tqdm(range(len(renders)), desc='Metric evaluation progress'):
                    ssims. append(ssim (renders[idx], gts[idx]).item())
                    psnrs. append(psnr (renders[idx], gts[idx]).item())
                    lpipss.append(lpips(renders[idx], gts[idx]).item())

                print('  SSIM: {:>12.7f}' .format(mean(ssims),  '.5'))
                print('  PSNR: {:>12.7f}' .format(mean(psnrs),  '.5'))
                print('  LPIPS: {:>12.7f}'.format(mean(lpipss), '.5'))
                print()

                results_agg[method] = {
                    'SSIM':  mean(ssims),
                    'PSNR':  mean(psnrs),
                    'LPIPS': mean(lpipss),
                }
                results_per_view[method] = {
                    'SSIM':  {name: ssim for ssim, name in zip(ssims,  image_names)},
                    'PSNR':  {name: psnr for psnr, name in zip(psnrs,  image_names)},
                    'LPIPS': {name: lp   for lp,   name in zip(lpipss, image_names)},
                }

            with open(scene_dir/  'results.json', 'w', encoding='utf-8') as fp:
                json.dump(results_agg, fp, indent=2, ensure_ascii=False)
            with open(scene_dir / 'results_per_view.json', 'w', encoding='utf-8') as fp:
                json.dump(results_per_view, fp, indent=2, ensure_ascii=False)
        except:
            print_exc()
            print('Unable to compute metrics for model:', scene_dir)


if __name__ == '__main__':
    parser = ArgumentParser(description='Evaluating script parameters')
    parser.add_argument('--model_paths', '-m', required=True, nargs='+', type=Path, default=[])
    args = parser.parse_args()

    evaluate(args.model_paths, 'train')
    evaluate(args.model_paths, 'test' )
