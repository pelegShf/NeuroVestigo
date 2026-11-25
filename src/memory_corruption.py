"""Memory-omission training: delete recent growth from what the memory bank stores.

Motivation (deep failure analysis): the tracker's dominant late-video error is
omission made permanent by memory. A new branch missed once is written to memory
as background, and on the following frames the memory actively suppresses it
(logits on missed branches fall below plain background). Ordinary training never
shows the model a memory that is missing structure, because the frames it
remembers are almost always nearly right.

Here, on a fraction of training clips, the masks written to memory have a set of
growth regions removed: connected components of GT(c) that lie outside
GT(0) dilated, i.e. structure that appeared during the clip. The loss target stays
the full GT, so the model is trained to re-add structure its memory omitted.

Only the memory write is corrupted. The prediction itself (and so the loss)
is untouched, and evaluation never sees any of this.

  mode "window": memory writes for frames 1..c are corrupted (c uniform in
                 [2, T-4]); later frames write clean memory, so the clip also
                 contains the recovery a real missed branch would need.
  mode "all":    every memory write from frame 1 on is corrupted.
"""
from __future__ import annotations

import random
import types
from typing import Optional, Tuple

import numpy as np
import torch
from scipy import ndimage

DEL_LOGIT = -20.0   # what the memory encoder sees in deleted pixels (sigmoid ~ 0)
MIN_AREA = 40       # px at 1024; smaller growth components are ignored
DIL0 = 6            # px at 1024 (~2 native px): GT(0) dilation defining "new"
PAD = 2             # px at 1024: margin around each deleted component


def install(sam2_model) -> None:
    """Wrap _encode_new_memory so a pending deletion mask is applied to its input."""
    if getattr(sam2_model, "_memdel_installed", False):
        return
    orig = sam2_model._encode_new_memory
    sam2_model._memdel = None

    def _encode_new_memory(self, current_vision_feats, feat_sizes, pred_masks_high_res,
                           object_score_logits, is_mask_from_pts):
        d = self._memdel
        if d is not None:
            pred_masks_high_res = pred_masks_high_res.masked_fill(d, DEL_LOGIT)
        return orig(current_vision_feats, feat_sizes, pred_masks_high_res,
                    object_score_logits, is_mask_from_pts)

    sam2_model._encode_new_memory = types.MethodType(_encode_new_memory, sam2_model)
    sam2_model._memdel_installed = True


def plan(gt_masks: torch.Tensor, p: float, mode: str, keep: float
         ) -> Tuple[Optional[torch.Tensor], int, float]:
    """Choose this clip's deletion.

    gt_masks: [B, T, N_obj, H, W] GT at training resolution.
    Returns (D, c, frac): D is a [B*N_obj, 1, H, W] bool deletion mask (None when the
    clip is not corrupted or has no growth), c the last corrupted frame, frac the
    share of GT(c) pixels deleted (for logging).
    """
    if p <= 0.0 or random.random() >= p:
        return None, 0, 0.0
    B, T, N, H, W = gt_masks.shape
    c = T - 1 if mode == "all" else random.randint(2, max(2, T - 4))
    g0 = (gt_masks[:, 0] > 0).reshape(B * N, H, W).cpu().numpy()
    gc = (gt_masks[:, c] > 0).reshape(B * N, H, W).cpu().numpy()
    D = np.zeros_like(gc)
    for i in range(B * N):
        old = ndimage.binary_dilation(g0[i], iterations=DIL0)
        lab, n = ndimage.label(gc[i] & ~old, np.ones((3, 3), int))
        if n == 0:
            continue
        areas = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
        chosen = [k + 1 for k in range(n) if areas[k] >= MIN_AREA and random.random() < keep]
        if chosen:
            sel = np.isin(lab, chosen)
            D[i] = ndimage.binary_dilation(sel, iterations=PAD) & ~ndimage.binary_dilation(g0[i], iterations=PAD)
    if not D.any():
        return None, 0, 0.0
    frac = float((D & gc).sum() / max(gc.sum(), 1))
    return torch.from_numpy(D).to(gt_masks.device)[:, None], c, frac
