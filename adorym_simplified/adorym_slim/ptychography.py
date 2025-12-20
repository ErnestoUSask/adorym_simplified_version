"""
Simplified 2D ptychography reconstruction loop.

This module intentionally mirrors only the execution path used by
``demos/2d_ptychography_experimental_data.py`` while remaining dependency-free.
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Tuple

from . import math_utils as mu
from .optim import AdamOptimizer


def load_dataset(path: str) -> Dict:
    """
    Load a tiny synthetic dataset stored as JSON.

    The structure mirrors the original HDF5 layout::
        {
            "exchange_data": [[[... intensity ...]]],
            "probe_pos_px": [[y, x], ...],
            "energy_ev": float,
            "psize_cm": float
        }
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_tiff_like(path: str, arr: mu.Array2D) -> None:
    """Persist a complex array as a JSON list-of-lists for inspection."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    serializable = [[(v.real, v.imag) for v in row] for row in arr]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f)


def initialize_probe(aperture_radius: int, size: Tuple[int, int], n_modes: int) -> List[mu.Array2D]:
    """
    Create a stack of circular-aperture probes with small random phase noise.
    """
    h, w = size
    cy = (h - 1) / 2
    cx = (w - 1) / 2
    base = mu.zeros(size)
    for iy in range(h):
        for ix in range(w):
            dist = ((iy - cy) ** 2 + (ix - cx) ** 2) ** 0.5
            if dist <= aperture_radius:
                base[iy][ix] = 1.0 + 0j
    probes = []
    for _ in range(n_modes):
        phase_perturbed = mu.zeros(size)
        for iy in range(h):
            for ix in range(w):
                noise = random.uniform(-0.05, 0.05)
                val = base[iy][ix] * complex(math_utils_cos(noise), math_utils_sin(noise))
                phase_perturbed[iy][ix] = val
        probes.append(phase_perturbed)
    return probes


def forward_patch(obj: mu.Array2D, probe: mu.Array2D) -> Tuple[mu.Array2D, mu.Array2D]:
    """Forward propagate a single probe/object patch to detector magnitude."""
    exit_wave = mu.mul(obj, probe)
    wave_det = mu.fft2_and_shift(exit_wave)
    magnitude = mu.abs_array(wave_det)
    return exit_wave, [[m ** 0.5 for m in row] for row in magnitude]


def gradient_step(
    obj: mu.Array2D,
    probes: List[mu.Array2D],
    target: List[List[float]],
    obj_optimizer: AdamOptimizer,
    probe_optimizer: AdamOptimizer,
) -> Tuple[mu.Array2D, List[mu.Array2D], float]:
    """
    Compute gradients for a single diffraction pattern and apply Adam updates.
    """
    obj_h, obj_w = mu.shape(obj)
    # Forward per-mode and accumulate intensity.
    forward_waves = []
    intensity = [[0.0 for _ in range(obj_w)] for _ in range(obj_h)]
    for probe in probes:
        exit_wave, mag = forward_patch(obj, probe)
        forward_waves.append((exit_wave, probe, mag))
        for iy in range(obj_h):
            for ix in range(obj_w):
                intensity[iy][ix] += mag[iy][ix] ** 2
    predicted = [[i ** 0.5 for i in row] for row in intensity]
    # Loss and gradient w.r.t. magnitude.
    loss = 0.0
    grad_mag = [[0.0 for _ in range(obj_w)] for _ in range(obj_h)]
    for iy in range(obj_h):
        for ix in range(obj_w):
            diff = predicted[iy][ix] - target[iy][ix]
            loss += diff * diff
            denom = predicted[iy][ix] if predicted[iy][ix] != 0 else 1e-8
            grad_mag[iy][ix] = diff / denom
    loss /= obj_h * obj_w
    # Backpropagate to exit waves (chain rule through sqrt and sum-of-modes).
    grad_obj = mu.zeros((obj_h, obj_w))
    grad_probes = [mu.zeros((obj_h, obj_w)) for _ in probes]
    for mode_idx, (exit_wave, probe, mag) in enumerate(forward_waves):
        # dL/d|wave_det| -> dL/dwave_det via complex chain rule.
        wave_det = mu.fft2_and_shift(exit_wave)
        grad_wave_det = mu.zeros((obj_h, obj_w))
        for iy in range(obj_h):
            for ix in range(obj_w):
                m = abs(wave_det[iy][ix]) + 1e-8
                grad_wave_det[iy][ix] = grad_mag[iy][ix] * (wave_det[iy][ix] / m)
        grad_exit = mu.ifft2_and_shift(grad_wave_det)
        conj_probe = mu.conj(probe)
        conj_obj = mu.conj(obj)
        grad_obj = mu.add(grad_obj, mu.mul(grad_exit, conj_probe))
        grad_probes[mode_idx] = mu.mul(grad_exit, conj_obj)
    # Apply Adam updates.
    updated_obj = obj_optimizer.step("obj", obj, grad_obj)
    updated_probes: List[mu.Array2D] = []
    for i, (probe, grad_probe) in enumerate(zip(probes, grad_probes)):
        updated_probes.append(probe_optimizer.step(f"probe_{i}", probe, grad_probe))
    return updated_obj, updated_probes, loss


def reconstruct_ptychography(
    data_path: str,
    output_folder: str,
    n_epochs: int = 3,
    minibatch_size: int = 1,
    probe_radius: int = 6,
    n_probe_modes: int = 1,
    obj_size: Tuple[int, int] = (32, 32),
    seed: int = 0,
) -> Dict[str, mu.Array2D]:
    """
    Run a tiny ptychography optimization on the bundled synthetic dataset.
    """
    random.seed(seed)
    dataset = load_dataset(data_path)
    raw = dataset["exchange_data"]
    # Dataset is [n_theta, n_spots, y, x]; we only support n_theta == 1 here.
    frames = raw[0]
    probe_pos = dataset["probe_pos_px"]
    det_h = len(frames[0])
    det_w = len(frames[0][0])

    obj = mu.random_complex(obj_size, mean=0.0, std=1e-3)
    probes = initialize_probe(probe_radius, (det_h, det_w), n_probe_modes)
    obj_opt = AdamOptimizer(step_size=1e-3)
    probe_opt = AdamOptimizer(step_size=1e-3)

    history = []
    for epoch in range(n_epochs):
        order = list(range(len(frames)))
        random.shuffle(order)
        for start in range(0, len(order), minibatch_size):
            batch_ids = order[start : start + minibatch_size]
            for idx in batch_ids:
                measured_intensity = frames[idx]
                measured_mag = [[val ** 0.5 for val in row] for row in measured_intensity]
                obj, probes, loss = gradient_step(obj, probes, measured_mag, obj_opt, probe_opt)
                history.append(loss)
        save_epoch_outputs(output_folder, epoch, obj, probes)
    return {"object": obj, "probes": probes, "loss_history": history}


def save_epoch_outputs(folder: str, epoch: int, obj: mu.Array2D, probes: List[mu.Array2D]) -> None:
    os.makedirs(folder, exist_ok=True)
    save_tiff_like(os.path.join(folder, f"epoch_{epoch}_object.json"), obj)
    for i, probe in enumerate(probes):
        save_tiff_like(os.path.join(folder, f"epoch_{epoch}_probe_mode_{i}.json"), probe)


# --- small trig wrappers to avoid importing math everywhere --- #
def math_utils_cos(x: float) -> float:
    import math

    return math.cos(x)


def math_utils_sin(x: float) -> float:
    import math

    return math.sin(x)
