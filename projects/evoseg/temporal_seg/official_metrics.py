"""Long-RVOS official J and boundary-F implementation.

Vendored from iSEE-Laboratory/Long_RVOS commit
447682a4fba314e81897645a91ff1a178da493d3, file
``eval/eval_long_rvos/metrics.py``.  Only comments/import layout were shortened;
the metric operations and empty-mask semantics are unchanged.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


def _disk(radius):
    """Exact default ``skimage.morphology.disk`` footprint, without skimage."""
    radius = int(radius)
    axis = np.arange(-radius, radius + 1)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return (xx * xx + yy * yy) <= radius * radius


def db_eval_iou(annotation, segmentation, void_pixels=None):
    assert annotation.shape == segmentation.shape, (
        f"Annotation({annotation.shape}) and segmentation:{segmentation.shape} "
        "dimensions do not match."
    )
    annotation = annotation.astype(bool)
    segmentation = segmentation.astype(bool)
    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape
        void_pixels = void_pixels.astype(bool)
    else:
        void_pixels = np.zeros_like(segmentation)
    inters = np.sum(
        (segmentation & annotation) & np.logical_not(void_pixels), axis=(-2, -1)
    )
    union = np.sum(
        (segmentation | annotation) & np.logical_not(void_pixels), axis=(-2, -1)
    )
    j = inters / union
    if j.ndim == 0:
        j = 1 if np.isclose(union, 0) else j
    else:
        j[np.isclose(union, 0)] = 1
    return j


def db_eval_boundary(annotation, segmentation, void_pixels=None, bound_th=0.008):
    assert annotation.shape == segmentation.shape
    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape
    if annotation.ndim == 3:
        result = np.zeros(annotation.shape[0])
        for frame_id in range(annotation.shape[0]):
            void = None if void_pixels is None else void_pixels[frame_id]
            result[frame_id] = f_measure(
                segmentation[frame_id], annotation[frame_id], void, bound_th=bound_th
            )
        return result
    if annotation.ndim == 2:
        return f_measure(segmentation, annotation, void_pixels, bound_th=bound_th)
    raise ValueError(
        f"db_eval_boundary does not support tensors with {annotation.ndim} dimensions"
    )


def f_measure(foreground_mask, gt_mask, void_pixels=None, bound_th=0.008):
    assert np.atleast_3d(foreground_mask).shape[2] == 1
    if void_pixels is not None:
        void_pixels = void_pixels.astype(bool)
    else:
        void_pixels = np.zeros_like(foreground_mask).astype(bool)
    bound_pix = (
        bound_th
        if bound_th >= 1
        else np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))
    )
    fg_boundary = _seg2bmap(foreground_mask * np.logical_not(void_pixels))
    gt_boundary = _seg2bmap(gt_mask * np.logical_not(void_pixels))
    fg_dil = cv2.dilate(
        fg_boundary.astype(np.uint8), _disk(bound_pix).astype(np.uint8)
    )
    gt_dil = cv2.dilate(
        gt_boundary.astype(np.uint8), _disk(bound_pix).astype(np.uint8)
    )
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)
    if n_fg == 0 and n_gt > 0:
        precision, recall = 1, 0
    elif n_fg > 0 and n_gt == 0:
        precision, recall = 0, 1
    elif n_fg == 0 and n_gt == 0:
        precision, recall = 1, 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)
    return 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _seg2bmap(seg, width=None, height=None):
    seg = seg.astype(bool)
    seg[seg > 0] = 1
    assert np.atleast_3d(seg).shape[2] == 1
    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height
    h, w = seg.shape[:2]
    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)
    assert not (width > w | height > h | abs(ar1 - ar2) > 0.01), (
        "Can't convert %dx%d seg to %dx%d bmap." % (w, h, width, height)
    )
    east = np.zeros_like(seg)
    south = np.zeros_like(seg)
    southeast = np.zeros_like(seg)
    east[:, :-1] = seg[:, 1:]
    south[:-1, :] = seg[1:, :]
    southeast[:-1, :-1] = seg[1:, 1:]
    boundary = seg ^ east | seg ^ south | seg ^ southeast
    boundary[-1, :] = seg[-1, :] ^ east[-1, :]
    boundary[:, -1] = seg[:, -1] ^ south[:, -1]
    boundary[-1, -1] = 0
    if w == width and h == height:
        return boundary
    boundary_map = np.zeros((height, width))
    for x in range(w):
        for y in range(h):
            if boundary[y, x]:
                j = 1 + math.floor((y - 1) + height / h)
                i = 1 + math.floor((x - 1) + width / h)
                boundary_map[j, i] = 1
    return boundary_map
