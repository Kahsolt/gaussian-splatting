#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/15

from typing import List

import torch
from torch import Tensor

from modules.camera import Camera
from modules.morphs.cd_gs.model import GaussianModel as GaussianModel_cd_gs
from modules.morphs.gs_w.model import GaussianModel as GaussianModel_gs_w

originals = {}


def GaussianModel_cd_gs_feature_encoder_hijack(self:GaussianModel_cd_gs, camera:Camera, visible_mask=slice(None)):
  assert isinstance(camera.uid, list), f'>> failed to hijack camera.uid: {camera.uid}, should be List[int]'

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


def GaussianModel_gs_w_feature_encoder_hijack(self:GaussianModel_gs_w, vp_cam:Camera, visible_mask:Tensor=slice(None)):
  assert isinstance(vp_cam.uid, list), f'>> failed to hijack vp_cam.uid: {vp_cam.uid}, should be List[int]'

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


def hijack_feature_encoders():
  originals['GaussianModel_cd_gs_feature_encoder'] = GaussianModel_cd_gs.feature_encoder
  GaussianModel_cd_gs.feature_encoder = GaussianModel_cd_gs_feature_encoder_hijack
  originals['GaussianModel_gs_w_feature_encoder'] = GaussianModel_gs_w.feature_encoder
  GaussianModel_gs_w.feature_encoder = GaussianModel_gs_w_feature_encoder_hijack

def unhijack_feature_encoders():
  GaussianModel_cd_gs.feature_encoder = originals['GaussianModel_cd_gs_feature_encoder']
  del originals['GaussianModel_cd_gs_feature_encoder']
  GaussianModel_gs_w.feature_encoder = originals['GaussianModel_gs_w_feature_encoder']
  del originals['GaussianModel_gs_w_feature_encoder']
