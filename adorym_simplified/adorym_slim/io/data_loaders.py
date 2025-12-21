"""
Data loading utilities for diffraction patterns and checkpoints.
"""

from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import h5py
import numpy as np

_dxchange_spec = importlib.util.find_spec("dxchange")
if _dxchange_spec:
    import dxchange  # type: ignore
else:  # pragma: no cover - optional dependency for TIFF I/O
    dxchange = None  # type: ignore


@dataclass
class DiffractionDataset:
    intensities: List[np.ndarray]
    probe_positions: List[Tuple[float, float]]

    @property
    def detector_shape(self) -> Tuple[int, int]:
        return self.intensities[0].shape


def _load_probe_positions(f: h5py.File) -> List[Tuple[float, float]]:
    for key in ["exchange/probe_pos", "probe/positions", "exchange/probe_pos_pixels"]:
        if key in f:
            pos_arr = np.array(f[key][...])
            return [(float(y), float(x)) for y, x in pos_arr.reshape(-1, 2)]
    # Default to center scanning if metadata is absent.
    n_spots = f["exchange/data"].shape[1]
    det_h, det_w = f["exchange/data"].shape[2:4]
    center = (det_h / 2.0, det_w / 2.0)
    return [center for _ in range(n_spots)]


def load_dataset(path: str, probe_pos_override: Sequence[Sequence[float]] | None = None) -> DiffractionDataset:
    """
    Load diffraction intensities and probe positions.

    If ``path`` is missing, fall back to a tiny synthetic dataset stored next to
    this module.
    """
    if not os.path.exists(path):
        synthetic_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "data", "ptycho_synthetic.json")
        )
        path = synthetic_path
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        raw = np.array(payload["exchange_data"], dtype=np.float32)
        intensities = [np.array(frame, dtype=np.float32) for frame in raw.reshape(-1, raw.shape[-2], raw.shape[-1])]
        positions = [(float(y), float(x)) for y, x in payload["probe_pos_px"]]
        return DiffractionDataset(intensities=intensities, probe_positions=positions)

    with h5py.File(path, "r") as f:
        data = f["exchange/data"]
        intensities = [np.array(dp, dtype=np.float32) for dp in data.reshape(-1, data.shape[-2], data.shape[-1])]
        positions = probe_pos_override or _load_probe_positions(f)
    return DiffractionDataset(intensities=intensities, probe_positions=list(positions))


def load_initial_guess(save_path: str, output_folder: str, epoch: int) -> List[np.ndarray]:
    """
    Load delta/beta checkpoint pairs from the output directory.
    """
    delta_path = os.path.join(save_path, output_folder, f"epoch_{epoch}/delta_ds_1.tiff")
    beta_path = os.path.join(save_path, output_folder, f"epoch_{epoch}/beta_ds_1.tiff")
    if dxchange is None:
        raise RuntimeError("dxchange is required to load TIFF checkpoints.")
    delta = dxchange.read_tiff(delta_path)
    beta = dxchange.read_tiff(beta_path)
    return [np.array(delta, dtype=np.complex128), np.array(beta, dtype=np.complex128)]
