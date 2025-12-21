"""
Minimal NumPy-based array helpers used by the simplified ptychography workflow.

The original Adorym code supports multiple array backends (Autograd, PyTorch,
Cupy). This reduction keeps a single NumPy implementation because the demo does
not exercise GPU or multi-backend execution paths.
"""

from __future__ import annotations

import numpy as np


Array = np.ndarray


def ensure_complex(arr: Array) -> Array:
    """Cast real arrays to complex128 to match the original algorithm."""
    if np.iscomplexobj(arr):
        return arr
    return arr.astype(np.complex128)


def zeros(shape, dtype=np.complex128) -> Array:
    return np.zeros(shape, dtype=dtype)


def random_complex(shape, mean=0.0, std=1e-3, seed=None) -> Array:
    rng = np.random.default_rng(seed)
    real = rng.normal(mean, std, size=shape)
    imag = rng.normal(mean, std, size=shape)
    return real + 1j * imag


def fft2c(arr: Array, normalize: bool = False) -> Array:
    shifted = np.fft.fftshift(np.fft.fft2(arr))
    if normalize:
        shifted /= arr.size ** 0.5
    return shifted


def ifft2c(arr: Array, normalize: bool = False) -> Array:
    unshift = np.fft.ifftshift(arr)
    inv = np.fft.ifft2(unshift)
    if normalize:
        inv /= arr.size ** 0.5
    return inv


def to_magnitude(intensity: Array) -> Array:
    return np.sqrt(np.maximum(intensity, 0.0))


def safe_divide(num: Array, denom: Array, eps: float = 1e-8) -> Array:
    return num / (denom + eps)
