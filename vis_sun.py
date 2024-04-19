#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/15

# 对于引入了视角坐标的模型，能否推算一下太阳在哪里？

from pprint import pprint
from argparse import ArgumentParser
from traceback import print_exc
from typing import Callable

import torch
from torch import Tensor
import numpy as np
from numpy import ndarray
from scipy.optimize import minimize, Bounds

from modules.model import GaussianModel_Neural

from render import *

torch.backends.cudnn.benchmark = True


def get_brightness(img:Tensor, reduction:str='mean') -> Tensor:
  # ref: https://blog.csdn.net/afei__/article/details/81184711
  brightness = 0.299 * img[0] + 0.587 * img[1] * + 0.114 * img[2]
  if reduction is None: return brightness
  if reduction == 'mean': return brightness.mean().item()


@torch.inference_mode()
def run(args, scene:Scene, render_func_tmpl:Callable):
  gaussians = scene.gaussians
  if isinstance(gaussians, GaussianModel_Neural): gaussians.cuda()
  render_func = lambda vp_cam, pc=gaussians: render_func_tmpl(pc, vp_cam, scene.background)

  def loss_fn_min(x:ndarray):
    nonlocal vp_cam
    for i in range(len(x)):
      vp_cam.camera_center[i] = x[i].item()
    rendered: Tensor = render_func(vp_cam)['render'].clamp_(0, 1)
    return get_brightness(rendered) 

  def loss_fn_max(x:ndarray):
    return -loss_fn_min(x)    # maxmize it!

  for idx, vp_cam in enumerate(scene.get_train_cameras()):
    init_loc: ndarray = vp_cam.camera_center.cpu().numpy()
    # find the view_point minimizing the brightness of rendered image
    res = minimize(loss_fn_min, x0=init_loc.copy(), method=args.method, bounds=Bounds(-30, 30), options={'maxiter': args.maxiter, 'disp': False})
    bri_A = res.fun
    loc_A = np.asarray(res.x)
    # find the view_point maximizing the brightness of rendered image
    res = minimize(loss_fn_max, x0=init_loc.copy(), method=args.method, bounds=Bounds(-30, 30), options={'maxiter': args.maxiter, 'disp': False})
    bri_B = -res.fun
    loc_B = np.asarray(res.x)
    # two points decides one line in 3D space
    vec = loc_B - loc_A
    vec /= np.linalg.norm(vec)  # normalize
    if vec[0] < 0: vec = -vec   # adjust direction
    print(f'[Camera {idx}] vec: {[round(e, 5) for e in vec]}, bri: {round(bri_A, 4)} ~ {round(bri_B, 4)}')


if __name__ == '__main__':
  parser = ArgumentParser()
  # ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
  parser.add_argument('--method', default='COBYLA', choices=['Nelder-Mead', 'BFGS', 'COBYLA'], help='optimize method')
  parser.add_argument('--maxiter', default=1000, type=int, help='optimize iter times')

  # Initialize system state (RNG)
  safe_state(silent=False)

  # Recover -M/--morph at training
  morph = get_ckpt_morph()
  print('>> morph:', morph)

  # Resolve real implemetations
  try:
    try:
      mod = import_module(f'modules.morphs.{morph}.hparam')
      HyperParams_cls = getattr(mod, 'HyperParams')
    except AttributeError:
      HyperParams_cls = HyperParams
      print('>> no overrided HyperParams class found, use default')
    try:
      mod = import_module(f'modules.morphs.{morph}.scene')
      Scene_cls = getattr(mod, 'Scene')
    except (ModuleNotFoundError, AttributeError):
      Scene_cls = Scene
      print('>> no overrided Scene class found, use default')
    mod = import_module(f'modules.morphs.{morph}.render')
    render_func = getattr(mod, 'render')
  except: print_exc()

  # Restore run env
  hp = HyperParams_cls()
  hp.send_to(parser)
  cmd_args, _ = parser.parse_known_args()
  cmd_args.eval = None
  args = get_combined_args(cmd_args, hp)
  hp.extract_from(args)
  
  # gogogo!!
  print('Hparams:')
  pprint(vars(hp))
  scene = Scene_cls(hp)
  run(cmd_args, scene, render_func)
