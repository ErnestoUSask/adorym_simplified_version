"""
Utility helpers for logging, patch extraction, and object/probe manipulation.
"""

from __future__ import annotations

import os
from typing import Iterable, Tuple

import numpy as np

from ..backends import Array


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def slice_from_center(center_y: float, center_x: float, height: int, width: int) -> Tuple[slice, slice]:
    top = int(round(center_y - height / 2))
    left = int(round(center_x - width / 2))
    return slice(top, top + height), slice(left, left + width)


def extract_patch(arr: Array, center_y: float, center_x: float, height: int, width: int) -> Tuple[Array, Tuple[int, int]]:
    """
    Pull a rectangular patch around ``(center_y, center_x)``.

    Returns both the extracted patch (zero-padded when the window extends
    beyond the array bounds) and the top-left location that the patch maps to.
    """
    top = int(round(center_y - height / 2))
    left = int(round(center_x - width / 2))
    patch = np.zeros((height, width), dtype=arr.dtype)
    y_start = max(0, top)
    x_start = max(0, left)
    y_end = min(arr.shape[0], top + height)
    x_end = min(arr.shape[1], left + width)
    patch_y_start = y_start - top
    patch_x_start = x_start - left
    patch_y_end = patch_y_start + (y_end - y_start)
    patch_x_end = patch_x_start + (x_end - x_start)
    if y_end > y_start and x_end > x_start:
        patch[patch_y_start:patch_y_end, patch_x_start:patch_x_end] = arr[y_start:y_end, x_start:x_end]
    return patch, (top, left)


def accumulate_patch(target: Array, patch: Array, top_left: Tuple[int, int]) -> None:
    """
    Add ``patch`` into ``target`` starting at ``top_left`` in-place, clipping to bounds.
    """
    top, left = top_left
    h, w = patch.shape
    y_start = max(0, top)
    x_start = max(0, left)
    y_end = min(target.shape[0], top + h)
    x_end = min(target.shape[1], left + w)
    patch_y_start = y_start - top
    patch_x_start = x_start - left
    patch_y_end = patch_y_start + (y_end - y_start)
    patch_x_end = patch_x_start + (x_end - x_start)
    if y_end > y_start and x_end > x_start:
        target[y_start:y_end, x_start:x_end] += patch[patch_y_start:patch_y_end, patch_x_start:patch_x_end]


def l2_relative_error(a: Array, b: Array) -> float:
    num = np.linalg.norm(a - b)
    denom = max(np.linalg.norm(b), 1e-12)
    return float(num / denom)


def max_abs_diff(a: Array, b: Array) -> float:
    return float(np.max(np.abs(a - b)))


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / max(len(vals), 1)
