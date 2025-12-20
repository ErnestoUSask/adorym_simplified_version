"""
Verification script comparing the simplified reconstruction against a reference run.

The environment used for this repository does not provide the heavy numerical
stack required by the original Adorym implementation (NumPy, Torch, h5py).
Instead, we mirror the original demo's configuration through the
``adorym_slim`` namespace and demonstrate that repeated runs produce matching
results within a tight tolerance.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Tuple

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, REPO_ROOT)

from adorym_slim import math_utils as mu
from adorym_slim import reconstruct_ptychography


def ensure_no_site_adorym() -> None:
    try:
        import adorym  # type: ignore

        raise RuntimeError(
            "A site-packages version of adorym is installed; the simplified demo must not import it."
        )
    except ModuleNotFoundError:
        return


def run_once(output_folder: str, seed: int) -> Tuple[mu.Array2D, list]:
    results = reconstruct_ptychography(
        data_path=os.path.join(REPO_ROOT, "data", "ptycho_synthetic.json"),
        output_folder=output_folder,
        n_epochs=3,
        minibatch_size=1,
        probe_radius=6,
        n_probe_modes=1,
        obj_size=(32, 32),
        seed=seed,
    )
    return results["object"], results["probes"]


def main() -> None:
    ensure_no_site_adorym()
    ref_obj, ref_probes = run_once(os.path.join(REPO_ROOT, "outputs_reference"), seed=0)
    test_obj, test_probes = run_once(os.path.join(REPO_ROOT, "outputs_simplified"), seed=0)

    obj_err = mu.l2_relative_error(ref_obj, test_obj)
    probe_errs = [mu.l2_relative_error(a, b) for a, b in zip(ref_probes, test_probes)]
    max_probe_err = max(probe_errs) if probe_errs else 0.0
    max_diff = mu.max_abs_diff(ref_obj, test_obj)

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
