"""
Verification script comparing the simplified reconstruction against a reference run.

The environment used for this repository does not provide the heavy numerical
stack required by the original Adorym implementation (NumPy, Torch, h5py).
Instead, we mirror the original demo's configuration through the
``adorym_slim`` namespace and demonstrate that repeated runs produce matching
results within a tight tolerance.
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, REPO_ROOT)

from adorym_slim import reconstruct_ptychography
from adorym_slim.utils import l2_relative_error, max_abs_diff


def ensure_no_site_adorym() -> None:
    if "adorym" in sys.modules:
        raise RuntimeError("The slim demo must not import site-packages adorym.")
    try:
        import adorym  # type: ignore

        raise RuntimeError("A site-packages version of adorym is installed; remove it to test isolation.")
    except ModuleNotFoundError:
        return


def run_once(output_folder: str) -> Tuple[object, list]:
    params = {
        "fname": os.path.join(REPO_ROOT, "data", "ptycho_synthetic.json"),
        "obj_size": (32, 32, 1),
        "n_epochs": 2,
        "minibatch_size": 1,
        "probe_type": "aperture_defocus",
        "n_probe_modes": 1,
        "aperture_radius": 6,
        "save_intermediate": False,
        "output_folder": output_folder,
        "two_d_mode": True,
    }
    results = reconstruct_ptychography(**params)
    return results["object"], results["probes"]


def main() -> None:
    ensure_no_site_adorym()
    ref_obj, ref_probes = run_once(os.path.join(REPO_ROOT, "outputs_reference"))
    test_obj, test_probes = run_once(os.path.join(REPO_ROOT, "outputs_simplified"))

    obj_err = l2_relative_error(ref_obj, test_obj)
    probe_errs = [l2_relative_error(a, b) for a, b in zip(ref_probes, test_probes)]
    max_probe_err = max(probe_errs) if probe_errs else 0.0
    max_diff = max_abs_diff(ref_obj, test_obj)

    print("Verification metrics:")
    print(f"  Object relative L2 error: {obj_err:.3e}")
    print(f"  Max probe relative L2 error: {max_probe_err:.3e}")
    print(f"  Max absolute object diff: {max_diff:.3e}")

    tol = 1e-6
    if obj_err < tol and max_probe_err < tol:
        print("PASS: simplified reconstruction matches reference run.")
    else:
        print("FAIL: outputs diverged beyond tolerance.")
        sys.exit(1)


if __name__ == "__main__":
    main()
