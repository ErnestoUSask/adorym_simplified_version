"""
Forward and backward passes for the simplified 2D ptychography problem.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Tuple

import numpy as np

from ..backends import Array, fft2c, ifft2c, random_complex, safe_divide, to_magnitude, zeros
from ..utils import accumulate_patch, extract_patch


def initialize_object(obj_size: Tuple[int, int, int], initial_guess=None, random_guess_means_sigmas=(1.0, 0.0, 0.001, 0.002), seed: int | None = None) -> Array:
    """
    Create a complex object volume. Only the first slice is used in two_d_mode.
    """
    height, width, depth = obj_size
    rng = np.random.default_rng(seed)
    mean_r, mean_i, sigma_r, sigma_i = random_guess_means_sigmas
    if initial_guess is not None:
        delta, beta = initial_guess
        complex_guess = (delta + 1j * beta).astype(np.complex128)
        if complex_guess.ndim == 3 and complex_guess.shape[-1] == 1:
            complex_guess = complex_guess[..., 0]
        return complex_guess
    real = rng.normal(mean_r, sigma_r, size=(height, width))
    imag = rng.normal(mean_i, sigma_i, size=(height, width))
    return (real + 1j * imag).astype(np.complex128)


def aperture_probe(radius: int, det_shape: Tuple[int, int], n_modes: int, beamstop_radius: int = 0, defocus_cm: float | None = None, wavelength_cm: float | None = None, psize_cm: float | None = None, seed: int | None = None) -> List[Array]:
    """
    Generate circular probes with an optional quadratic phase term to emulate defocus.
    """
    h, w = det_shape
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    base_amp = np.where(rr <= radius, 1.0, 0.0)
    if beamstop_radius:
        base_amp = np.where(rr < beamstop_radius, 0.0, base_amp)

    phase = np.zeros_like(base_amp)
    if defocus_cm is not None and wavelength_cm is not None and psize_cm is not None and defocus_cm != "inf":
        k = 2 * math.pi / wavelength_cm
        quad = ((yy - cy) ** 2 + (xx - cx) ** 2) * (psize_cm ** 2)
        phase = -1j * k * quad / (2 * defocus_cm)
    elif defocus_cm == "inf":
        phase = np.zeros_like(base_amp)

    rng = np.random.default_rng(seed)
    probes: List[Array] = []
    for mode in range(n_modes):
        noise = rng.normal(0.0, 0.01, size=base_amp.shape)
        probes.append(base_amp * np.exp(phase + 1j * noise))
    return probes


def forward_intensity(object_slice: Array, probe: Array, position: Tuple[float, float], normalize_fft: bool = False) -> Tuple[Array, Array, Tuple[int, int]]:
    """
    Compute exit wave and detector wavefront for a single probe position.
    """
    patch, top_left = extract_patch(object_slice, position[0], position[1], probe.shape[0], probe.shape[1])
    exit_wave = patch * probe
    wave_det = fft2c(exit_wave, normalize=normalize_fft)
    intensity = np.abs(wave_det) ** 2
    return wave_det, intensity, top_left


def loss_and_gradients(object_slice: Array, probes: List[Array], measured_intensity: Array, position: Tuple[float, float], normalize_fft: bool = False) -> Tuple[float, Array, List[Array], Tuple[int, int]]:
    """
    Compute LSQ loss and gradients for a single diffraction pattern.
    """
    det_shape = measured_intensity.shape
    combined_intensity = np.zeros(det_shape, dtype=np.float64)
    exit_waves: List[Array] = []
    wave_dets: List[Array] = []

    top_left: Tuple[int, int] | None = None
    for probe in probes:
        wave_det, intensity, tl = forward_intensity(object_slice, probe, position, normalize_fft=normalize_fft)
        combined_intensity += intensity
        exit_waves.append(wave_det)
        wave_dets.append(wave_det)
        top_left = tl

    assert top_left is not None
    predicted_mag = to_magnitude(combined_intensity)
    target_mag = to_magnitude(measured_intensity)

    diff = predicted_mag - target_mag
    loss = float(np.mean(diff ** 2))

    grad_mag = safe_divide(diff, predicted_mag + 1e-8)

    obj_grad_patch = np.zeros_like(object_slice, dtype=np.complex128)
    probe_grads: List[Array] = []

    for wave_det, probe in zip(wave_dets, probes):
        mag = np.abs(wave_det)
        grad_wave = grad_mag * safe_divide(wave_det, mag)
        grad_exit = ifft2c(grad_wave, normalize=normalize_fft)
        patch, tl = extract_patch(object_slice, position[0], position[1], probe.shape[0], probe.shape[1])
        accumulate_patch(obj_grad_patch, grad_exit * np.conj(probe), tl)
        probe_grads.append(grad_exit * np.conj(patch))

    return loss, obj_grad_patch, probe_grads, top_left
