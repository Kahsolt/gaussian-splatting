#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/15

from copy import deepcopy
from pprint import pprint
from time import time
from argparse import ArgumentParser
from traceback import print_exc, format_exc
from typing import *

import tkinter as tk
import tkinter.ttk as ttk

import torch
from torch import Tensor
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends._backend_tk import NavigationToolbar2Tk

from modules.camera import Camera
from modules.model import GaussianModel_Neural
from modules.morphs.cd_gs.model import GaussianModel as GaussianModel_cd_gs
from modules.morphs.gs_w.model import GaussianModel as GaussianModel_gs_w
from modules.lpipsPyTorch import LPIPS
from modules.utils.loss_utils import psnr, ssim
from modules.utils.general_utils import ImageState

from render import *

torch.backends.cudnn.benchmark = True

# camera center shift
CAM_CENTER_LIM = 10

WINDOW_TITLE = '3D-GS Viewer'
WINDOW_SIZE  = (912, 860)
FIG_SIZE = 4
FIG_DPI = 400
HIST_BINS = 48


def GaussianModel_cd_gs_feature_encoder_hijack(self:GaussianModel_cd_gs, camera:Camera, visible_mask=None):
  assert isinstance(camera.uid, list), f'>> failed to hijack camera.uid: {camera.uid}, should be List[int]'

  if visible_mask is None: visible_mask = slice(None)
  feats = self.features[visible_mask]

  hp = self.hp
  if hp.add_view:
    pts = self.xyz[visible_mask]
    ob_view = pts - camera.camera_center            # (N, 3)
    ob_dist = ob_view.norm(dim=1, keepdim=True)     # (N, 1)
    ob_view = ob_view / ob_dist                     # (N, 3)
    ob_dist = torch.log(ob_dist)

  colors: List[Tensor] = []
  for i, mlp in enumerate(self.mlps):
    feat = feats[:, hp.per_feat_dim*i: hp.per_feat_dim*(i+1)]
    if hp.add_view:
      feat = torch.cat([feat, ob_view, ob_dist], -1)
    if self.embeddings[i] is not None:   
      camera_indicies = torch.ones_like(feats[:, 0], dtype=torch.long) * camera.uid[i]
      embedding = self.embeddings[i](camera_indicies)
      feat = torch.cat([feat, embedding], -1)
    colors.append(mlp(feat))

  return colors

GaussianModel_cd_gs.feature_encoder = GaussianModel_cd_gs_feature_encoder_hijack


def GaussianModel_gs_w_feature_encoder_hijack(self:GaussianModel_gs_w, vp_cam:Camera, visible_mask:Tensor=None):
  assert isinstance(vp_cam.uid, list), f'>> failed to hijack vp_cam.uid: {vp_cam.uid}, should be List[int]'

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
    camera_indicies = torch.ones_like(cat_view[:, 0], dtype=torch.long) * vp_cam.uid[0]
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
    camera_indicies = torch.ones_like(cat_view[:, 0], dtype=torch.long) * vp_cam.uid[1]
    occlusion = self.embedding_occlusion(camera_indicies)
    if hp.add_view_emb_to_occlu:
      cat_occlu_feat = torch.cat([cat_occlu_feat, occlusion], -1)    # (N, 80)
    else:
      cat_occlu_feat = torch.cat([importance, occlusion], -1)
  else:
    cat_occlu_feat = importance
  occlus = self.mlp_occlu(cat_occlu_feat)   # (N, 1), vrng [0, inf]

  return colors, occlus

GaussianModel_gs_w.feature_encoder = GaussianModel_gs_w_feature_encoder_hijack


def timer(fn):
  def wrapper(*args, **kwargs):
    start = time()
    r = fn(*args, **kwargs)
    end = time()
    print(f'[Timer]: {fn.__name__} took {end - start:.3f}s')
    return r
  return wrapper


COLOR_PALETTES = {
  1: ['grey'],
  3: ['r', 'g', 'b'],
  4: ['r', 'g', 'b', 'yellow'],
}

@timer
def make_img_hist(X:Tensor) -> Tensor:
  if len(X.shape) == 2: X = X.unsqueeze(dim=0)
  C, H, W = X.shape
  X = X.cpu().flatten(start_dim=1).numpy()
  colors = COLOR_PALETTES[C]
  fig: Figure = plt.figure()
  ax = fig.gca()
  for i, c in enumerate(colors):
    ax.hist(X[i], alpha=0.5, color=c, bins=HIST_BINS, range=(0.0, 1.0))
    ax.axis('off')
  fig.tight_layout()
  cvs: FigureCanvasAgg = fig.canvas
  cvs.draw()
  im = np.frombuffer(cvs.tostring_rgb(), dtype=np.uint8).reshape(cvs.get_width_height()[::-1] + (3,))
  plt.close(fig)
  X_hist = torch.from_numpy(im).permute(2, 0, 1).div(255)
  return TF.resize(X_hist, (H, W), interpolation=TF.InterpolationMode.NEAREST)


class App:

  def __init__(self, args, scene:Scene, render_func:Callable):
    self.args = args
    self.scene = scene
    self.hp = scene.hp
    self.morph = self.hp.morph
    self.vp_cams = scene.get_train_cameras()

    if isinstance(scene.gaussians, GaussianModel_Neural):
      scene.gaussians.cuda()
    self.render_func = lambda vp_cam, pc=scene.gaussians, scale=1.0: render_func(pc, vp_cam, scene.background, scale)
  
    if args.show_metrics:
      self.lpips = LPIPS(net_type='vgg').cuda()

    self.setup_gui()
    self.setup_inits()

    try:
      self.wnd.mainloop()
    except KeyboardInterrupt:
      self.wnd.quit()
    except: print_exc()

  def setup_inits(self):
    self.sc_vp_cam.config(to=len(self.vp_cams)-1)
    for sc in self.sc_embed:
      sc.config(to=len(self.vp_cams)-1)
    self.refresh(vp_cam_chg=True)
    print('>> Ready!')

  def setup_gui(self):
    # window
    wnd = tk.Tk()
    W, H = wnd.winfo_screenwidth(), wnd.winfo_screenheight()
    w, h = WINDOW_SIZE
    wnd.geometry(f'{w}x{h}+{(W-w)//2}+{(H-h)//2}')
    #wnd.resizable(False, False)
    wnd.title(WINDOW_TITLE)
    wnd.protocol('WM_DELETE_WINDOW', wnd.quit)
    self.wnd = wnd

    # top: controls
    frm1 = ttk.Frame(wnd)
    frm1.pack(side=tk.TOP, anchor=tk.N, expand=tk.YES, fill=tk.X)
    if True:
      frm11 = ttk.LabelFrame(frm1, text='Viewpoint Camera')
      frm11.pack(side=tk.TOP, anchor=tk.N, expand=tk.YES, fill=tk.X)
      if True:
        self.var_vp_cam = tk.IntVar(wnd)
        sc = tk.Scale(frm11, command=lambda _: self.refresh(vp_cam_chg=True), variable=self.var_vp_cam, orient=tk.HORIZONTAL, from_=0, to=10, resolution=1, tickinterval=20)
        sc.pack(side=tk.TOP, expand=tk.YES, fill=tk.X)
        self.sc_vp_cam = sc

      frm12 = ttk.LabelFrame(frm1, text='Scaling Modifier')
      frm12.pack(side=tk.TOP, anchor=tk.N, expand=tk.YES, fill=tk.X)
      if True:
        self.var_scale = tk.DoubleVar(wnd, 1.0)
        sc = tk.Scale(frm12, command=self._refresh, variable=self.var_scale, orient=tk.HORIZONTAL, from_=0.01, to=1.25, resolution=0.02, tickinterval=0.1)
        sc.pack(side=tk.TOP, expand=tk.YES, fill=tk.X)

      frm13 = ttk.LabelFrame(frm1, text='Camera-Center Shifter')
      frm13.pack(side=tk.TOP, anchor=tk.N, expand=tk.YES, fill=tk.X) if self.morph in ['mlp_gs', 'cd_gs'] else None
      if True:
        self.var_cam_center = [tk.DoubleVar(wnd) for i in range(3)]
        for var in self.var_cam_center:
          sc = tk.Scale(frm13, command=self._refresh, variable=var, orient=tk.HORIZONTAL, from_=-CAM_CENTER_LIM, to=CAM_CENTER_LIM, resolution=0.1, tickinterval=1)
          sc.pack(side=tk.LEFT, expand=tk.YES, fill=tk.X)

      frm14 = ttk.LabelFrame(frm1, text='Embedding Replacer')
      frm14.pack(side=tk.TOP, anchor=tk.N, expand=tk.YES, fill=tk.X) if self.morph in ['cd_gs', 'gs_w'] else None
      if True:
        n_embed = len(self.scene.gaussians.embeddings) if isinstance(self.scene.gaussians, GaussianModel_Neural) else 0
        self.var_embed = [tk.IntVar(wnd) for i in range(n_embed)]
        self.sc_embed: List[tk.Scale] = []
        for var in self.var_embed:
          sc = tk.Scale(frm14, command=self._refresh, variable=var, orient=tk.HORIZONTAL, from_=0, to=100, resolution=1, tickinterval=50)
          sc.pack(side=tk.LEFT, expand=tk.YES, fill=tk.X)
          self.sc_embed.append(sc)

    # middle: render
    frm2 = ttk.Frame(wnd)
    frm2.pack(expand=tk.YES, fill=tk.BOTH)
    if True:
      fig = plt.figure(figsize=(FIG_SIZE, FIG_SIZE), dpi=FIG_DPI)
      fig.tight_layout()
      ax: Axes = fig.gca()
      cvs = FigureCanvasTkAgg(fig, frm2)
      cvstk = cvs.get_tk_widget()
      cvstk.pack(expand=tk.YES, fill=tk.BOTH)
      toolbar = NavigationToolbar2Tk(cvs, frm2, pack_toolbar=False)
      toolbar.update()
      toolbar.pack(side=tk.BOTTOM, fill=tk.X)
      self.fig, self.ax, self.cvs = fig, ax, cvs

    # bottom: status
    frm3 = ttk.Frame(wnd)
    frm3.pack(side=tk.BOTTOM, anchor=tk.S, expand=tk.YES, fill=tk.X)
    if True:
      var = tk.StringVar(wnd)
      self.var_info = var
      lbl = ttk.Label(frm3, textvariable=var)
      lbl.pack(expand=tk.YES, fill=tk.X)

  def _refresh(self, *args, vp_cam_chg:bool=False):
    return self.refresh(vp_cam_chg)

  @torch.inference_mode()
  @timer
  def refresh(self, vp_cam_chg:bool=False):
    idx = self.var_vp_cam.get()
    vp_cam = vp_cam_original = self.vp_cams[idx]
    scale = self.var_scale.get()

    # hijack vp_cam
    if self.morph in ['mlp_gs', 'cd_gs']:
      vp_cam = deepcopy(vp_cam)
      if vp_cam_chg:
        for i, var in enumerate(self.var_cam_center):
          var.set(vp_cam.camera_center[i].item())
      else:
        for i, var in enumerate(self.var_cam_center):
          vp_cam.camera_center[i] = var.get()
    if self.morph in ['cd_gs', 'gs_w']:
      if vp_cam is vp_cam_original: vp_cam = deepcopy(vp_cam)
      if vp_cam_chg:
        for i, var in enumerate(self.var_embed):
          var.set(vp_cam.uid)
      vp_cam.uid = [var.get() for var in self.var_embed]

    # gt
    if self.morph == 'if_gs':
      from modules.morphs.if_gs.camera import Camera as Camera_if_gs
      assert isinstance(vp_cam, Camera_if_gs)
      gt = vp_cam.gt_image.cuda()
    else:
      gt = vp_cam.image.cuda()   # [C, H, W]

    # render
    if self.morph == 'if_gs':
      from modules.morphs.if_gs.scene import Scene as Scene_if_gs
      assert isinstance(self.scene, Scene_if_gs)
      rendered_set = []
      for freq_idx in range(self.scene.gaussians.n_gaussians):
        self.scene.activate_gaussian(freq_idx)
        gaussian = self.scene.cur_gaussian.cuda()
        render_pkg = self.render_func(vp_cam, pc=gaussian, scale=scale)
        rendered_set.append(render_pkg['render'].clamp_(0, 1))
    else:
      render_pkg = self.render_func(vp_cam, scale=scale)
      if self.morph == 'cd_gs':
        from modules.morphs.cd_gs.render import mix_image
        rendered_set = render_pkg['render']
        rendered: Tensor = mix_image(rendered_set).clamp_(0, 1)
      else:
        rendered: Tensor = render_pkg['render'].clamp_(0, 1)
    imgs = [img.cpu() for img in [*locals().get('rendered_set', []), rendered, gt]]
    img_hists = [make_img_hist(img) for img in imgs] if args.show_histogram else []

    # render (aux.)
    if self.morph == 'gs_w':
      occlusions = rendered['occlusions']
    if self.morph == 'dev':
      img_state = rendered['img_state']
      final_T = img_state.final_T if isinstance(img_state, ImageState) else None
      n_contrib = rendered['n_contrib']
      importance_map = rendered['importance_map']
      depth_map = rendered['depth_map']
      weight_map = rendered['weight_map']
    auxs = [aux.cpu() for aux in [
      locals().get('occlusions'),
      locals().get('final_T'),
      locals().get('n_contrib'),
      locals().get('importance_map'),
      locals().get('depth_map'),
      locals().get('weight_map'),
    ] if aux is not None]

    # metrics
    if self.args.show_metrics:
      rendered_ = rendered.unsqueeze(0)
      gt_ = gt.unsqueeze(0)
      lpips = self.lpips
      metrics = {
        'ssim':  ssim (rendered_, gt_).item(),
        'psnr':  psnr (rendered_, gt_).item(),
        'lpips': lpips(rendered_, gt_).item(),
      }
      self.refresh_status(metrics)

    # draw
    im_grid = make_grid(imgs + img_hists + auxs, nrow=len(imgs)).permute(1, 2, 0).cpu().numpy()
    self.ax.clear()
    self.ax.imshow(im_grid)
    self.ax.axis('off')
    self.cvs.draw()

  def refresh_status(self, metrics:Dict[str, float]):
    metrics_str = ', '.join([f'{k}: {v:.5}' for k, v in metrics.items()])
    self.var_info.set('>> ' + metrics_str)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--show_metrics', action='store_true', help='show sample-wise metrics')
  parser.add_argument('--show_histogram', action='store_true', help='show histograms of rendered images')

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
  App(cmd_args, scene, render_func)
