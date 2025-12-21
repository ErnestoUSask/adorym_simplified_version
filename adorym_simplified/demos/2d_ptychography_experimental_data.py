"""
Golden parameters preserved
- fname (default: "data.h5")
- theta_st (default: 0)
- theta_end (default: 0)
- n_epochs (default: 1000)
- obj_size (default: (618, 606, 1))
- two_d_mode (default: True)
- energy_ev (default: 8801.121930115722)
- psize_cm (default: 1.32789376566526e-06)
- minibatch_size (default: 35)
- output_folder (default: "test")
- cpu_only (default: False)
- save_path (default: "../demos/siemens_star_aps_2idd")
- use_checkpoint (default: False)
- n_epoch_final_pass (default: None)
- save_intermediate (default: True)
- full_intermediate (default: True)
- initial_guess (default: None or prior epoch)
- random_guess_means_sigmas (default: (1., 0., 0.001, 0.002))
- n_dp_batch (default: 350)
- probe_type (default: "aperture_defocus")
- n_probe_modes (default: 5)
- aperture_radius (default: 10)
- beamstop_radius (default: 5)
- probe_defocus_cm (default: 0.0069)
- rescale_probe_intensity (default: True)
- free_prop_cm (default: "inf")
- backend (default: "pytorch")
- raw_data_type (default: "intensity")
- beamstop (default: None)
- optimizer / optimizer_probe / optimizer_all_probe_pos objects (Adam)
- optimize_probe (default: True)
- optimize_all_probe_pos (default: True)
- save_history (default: True)
- update_scheme (default: "immediate")
- unknown_type (default: "real_imag")
- save_stdout (default: True)
- loss_function_type (default: "lsq")
- normalize_fft (default: False)

Parameter contract (mirrors original demo):
    --epoch: string/int, "None" to start from scratch; otherwise resume from epoch-1 TIFFs.
    --save_path: base folder for data (default "cone_256_foam_ptycho").
    --output_folder: subfolder under save_path for results (default "test").
    Runtime config keys inside ``params`` preserve the originals listed above.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from typing import List, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

import numpy as np

_dxchange_spec = importlib.util.find_spec("dxchange")
if _dxchange_spec:
    import dxchange  # type: ignore
else:
    dxchange = None  # type: ignore

from adorym_slim import AdamOptimizer, reconstruct_ptychography


def _guard_against_site_adorym() -> None:
    if "adorym" in sys.modules:
        raise RuntimeError("Detected `adorym` in sys.modules; this demo must use the bundled slim namespace only.")
    try:
        spec = importlib.util.find_spec("adorym")
        if spec is not None:
            print(
                "Warning: a site-packages installation of `adorym` is available. "
                "The demo will continue using adorym_slim to avoid collisions."
            )
    except ModuleNotFoundError:
        return


def _load_previous_epoch(save_path: str, output_folder: str, epoch: int) -> List[np.ndarray]:
    if dxchange is None:
        return []
    prev_epoch = epoch - 1
    delta_path = os.path.join(save_path, output_folder, f"epoch_{prev_epoch}", "delta_ds_1.tiff")
    beta_path = os.path.join(save_path, output_folder, f"epoch_{prev_epoch}", "beta_ds_1.tiff")
    if os.path.exists(delta_path) and os.path.exists(beta_path):
        delta = dxchange.read_tiff(delta_path)  # type: ignore
        beta = dxchange.read_tiff(beta_path)  # type: ignore
        return [np.array(delta), np.array(beta)]
    return []


def main() -> None:
    _guard_against_site_adorym()
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default="None")
    parser.add_argument("--save_path", default="cone_256_foam_ptycho")
    parser.add_argument("--output_folder", default="test")
    args = parser.parse_args()

    epoch = args.epoch
    if epoch == "None":
        epoch = 0
        init = None
    else:
        epoch = int(epoch)
        init = _load_previous_epoch(args.save_path, args.output_folder, epoch) if epoch > 0 else None

    optimizer_output = os.path.join(args.save_path, args.output_folder)
    distribution_mode = None
    optimizer_obj = AdamOptimizer(
        "obj", output_folder=optimizer_output, distribution_mode=distribution_mode, options_dict={"step_size": 1e-3}
    )
    optimizer_probe = AdamOptimizer(
        "probe", output_folder=optimizer_output, distribution_mode=distribution_mode, options_dict={"step_size": 1e-3, "eps": 1e-7}
    )
    optimizer_all_probe_pos = AdamOptimizer(
        "probe_pos_correction",
        output_folder=optimizer_output,
        distribution_mode=distribution_mode,
        options_dict={"step_size": 1e-2},
    )

    params_2idd_gpu = {
        "fname": "data.h5",
        "theta_st": 0,
        "theta_end": 0,
        "n_epochs": 1000,
        "obj_size": (618, 606, 1),
        "two_d_mode": True,
        "energy_ev": 8801.121930115722,
        "psize_cm": 1.32789376566526e-06,
        "minibatch_size": 35,
        "output_folder": args.output_folder,
        "cpu_only": False,
        "save_path": "../demos/siemens_star_aps_2idd",
        "use_checkpoint": False,
        "n_epoch_final_pass": None,
        "save_intermediate": True,
        "full_intermediate": True,
        "initial_guess": init,
        "random_guess_means_sigmas": (1.0, 0.0, 0.001, 0.002),
        "n_dp_batch": 350,
        # ===============================
        "probe_type": "aperture_defocus",
        "n_probe_modes": 5,
        "aperture_radius": 10,
        "beamstop_radius": 5,
        "probe_defocus_cm": 0.0069,
        # ===============================
        "rescale_probe_intensity": True,
        "free_prop_cm": "inf",
        "backend": "pytorch",
        "raw_data_type": "intensity",
        "beamstop": None,
        "optimizer": optimizer_obj,
        "optimize_probe": True,
        "optimizer_probe": optimizer_probe,
        "optimize_all_probe_pos": True,
        "optimizer_all_probe_pos": optimizer_all_probe_pos,
        "save_history": True,
        "update_scheme": "immediate",
        "unknown_type": "real_imag",
        "save_stdout": True,
        "loss_function_type": "lsq",
        "normalize_fft": False,
    }

    reconstruct_ptychography(**params_2idd_gpu)


if __name__ == "__main__":
    main()
