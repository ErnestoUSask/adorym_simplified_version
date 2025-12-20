"""
Minimal numerical helpers built only on the Python standard library.

The goal is to provide a handful of array operations that are sufficient for a
tiny 2D ptychography reconstruction loop without relying on NumPy, Torch, or
other heavy dependencies that are unavailable in the execution environment.

Arrays are represented as nested Python lists. Complex numbers use Python's
native ``complex`` type. All helpers return new lists and never mutate their
inputs unless explicitly stated.
"""

from __future__ import annotations

import cmath
import math
import random
from typing import Iterable, List, Sequence, Tuple


Array2D = List[List[complex]]


def zeros(shape: Tuple[int, int], value: complex = 0j) -> Array2D:
    """Allocate a 2D array filled with a constant."""
    y, x = shape
    return [[value for _ in range(x)] for _ in range(y)]


def clone(arr: Array2D) -> Array2D:
    """Deep copy a 2D array."""
    return [row[:] for row in arr]


def random_complex(shape: Tuple[int, int], mean: float = 0.0, std: float = 1.0) -> Array2D:
    """Draw complex random numbers from independent Gaussians on real/imag axes."""
    y, x = shape
    return [
        [complex(random.gauss(mean, std), random.gauss(mean, std)) for _ in range(x)] for _ in range(y)
    ]


def shape(arr: Array2D) -> Tuple[int, int]:
    return (len(arr), len(arr[0]) if arr else 0)


def pad(arr: Array2D, padding: Tuple[int, int, int, int], value: complex = 0j) -> Array2D:
    """Pad a 2D array: (top, bottom, left, right)."""
    top, bottom, left, right = padding
    h, w = shape(arr)
    new_h = h + top + bottom
    new_w = w + left + right
    padded = zeros((new_h, new_w), value=value)
    for iy in range(h):
        for ix in range(w):
            padded[iy + top][ix + left] = arr[iy][ix]
    return padded


def crop(arr: Array2D, top: int, left: int, height: int, width: int) -> Array2D:
    """Extract a sub-array, clipping to bounds if necessary."""
    h, w = shape(arr)
    result = zeros((height, width))
    for iy in range(height):
        for ix in range(width):
            sy = iy + top
            sx = ix + left
            if 0 <= sy < h and 0 <= sx < w:
                result[iy][ix] = arr[sy][sx]
    return result


def elementwise_apply(a: Array2D, b: Array2D, fn) -> Array2D:
    """Apply a binary function to two equally-shaped arrays."""
    h, w = shape(a)
    out = zeros((h, w))
    for iy in range(h):
        row_a = a[iy]
        row_b = b[iy]
        row_o = out[iy]
        for ix in range(w):
            row_o[ix] = fn(row_a[ix], row_b[ix])
    return out


def add(a: Array2D, b: Array2D) -> Array2D:
    return elementwise_apply(a, b, lambda x, y: x + y)


def sub(a: Array2D, b: Array2D) -> Array2D:
    return elementwise_apply(a, b, lambda x, y: x - y)


def mul(a: Array2D, b: Array2D) -> Array2D:
    return elementwise_apply(a, b, lambda x, y: x * y)


def scalar_mul(a: Array2D, scalar: complex) -> Array2D:
    h, w = shape(a)
    out = zeros((h, w))
    for iy in range(h):
        row_a = a[iy]
        row_o = out[iy]
        for ix in range(w):
            row_o[ix] = row_a[ix] * scalar
    return out


def conj(a: Array2D) -> Array2D:
    h, w = shape(a)
    out = zeros((h, w))
    for iy in range(h):
        row_a = a[iy]
        row_o = out[iy]
        for ix in range(w):
            row_o[ix] = row_a[ix].conjugate()
    return out


def abs_array(a: Array2D) -> List[List[float]]:
    h, w = shape(a)
    out: List[List[float]] = [[0.0 for _ in range(w)] for _ in range(h)]
    for iy in range(h):
        row_a = a[iy]
        row_o = out[iy]
        for ix in range(w):
            row_o[ix] = abs(row_a[ix])
    return out


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / max(len(vals), 1)


def mean_square_diff(a: Array2D, b: Array2D) -> float:
    h, w = shape(a)
    total = 0.0
    count = h * w
    for iy in range(h):
        row_a = a[iy]
        row_b = b[iy]
        for ix in range(w):
            diff = row_a[ix] - row_b[ix]
            total += (diff.real ** 2 + diff.imag ** 2)
    return total / count


def fft1d(vec: Sequence[complex], inverse: bool = False) -> List[complex]:
    n = len(vec)
    out = [0j] * n
    sign = 1 if inverse else -1
    for k in range(n):
        accum = 0j
        for t, val in enumerate(vec):
            angle = sign * 2 * math.pi * (k * t / n)
            accum += val * cmath.exp(1j * angle)
        out[k] = accum / (n if inverse else 1)
    return out


def fft2d(arr: Array2D, inverse: bool = False) -> Array2D:
    h, w = shape(arr)
    # Transform rows.
    row_transformed = [fft1d(row, inverse=inverse) for row in arr]
    # Transform columns.
    out: Array2D = zeros((h, w))
    for x in range(w):
        col = [row_transformed[y][x] for y in range(h)]
        col_fft = fft1d(col, inverse=inverse)
        for y in range(h):
            out[y][x] = col_fft[y]
    return out


def fftshift(arr: Array2D) -> Array2D:
    h, w = shape(arr)
    out = zeros((h, w))
    mid_y = h // 2
    mid_x = w // 2
    for iy in range(h):
        for ix in range(w):
            new_y = (iy + mid_y) % h
            new_x = (ix + mid_x) % w
            out[new_y][new_x] = arr[iy][ix]
    return out


def ifftshift(arr: Array2D) -> Array2D:
    # For integer-sized grids, fftshift is its own inverse.
    return fftshift(arr)


def fft2_and_shift(arr: Array2D) -> Array2D:
    return fftshift(fft2d(arr, inverse=False))


def ifft2_and_shift(arr: Array2D) -> Array2D:
    return ifftshift(fft2d(arr, inverse=True))


def l2_relative_error(a: Array2D, b: Array2D) -> float:
    """Compute ||a-b||_2 / ||a||_2 using flattened complex vectors."""
    num = 0.0
    denom = 0.0
    h, w = shape(a)
    for iy in range(h):
        for ix in range(w):
            diff = a[iy][ix] - b[iy][ix]
            aval = a[iy][ix]
            num += diff.real ** 2 + diff.imag ** 2
            denom += aval.real ** 2 + aval.imag ** 2
    if denom == 0:
        return 0.0
    return math.sqrt(num / denom)


def max_abs_diff(a: Array2D, b: Array2D) -> float:
    h, w = shape(a)
    best = 0.0
    for iy in range(h):
        for ix in range(w):
            best = max(best, abs(a[iy][ix] - b[iy][ix]))
    return best
