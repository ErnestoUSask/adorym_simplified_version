"""
Run the simplified 2D ptychography reconstruction using only standard library code.

The script mirrors the original Adorym demo but points all imports to the
``adorym_slim`` namespace and uses a tiny synthetic dataset bundled with this
repository so that it can execute in an offline environment.
"""

from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from adorym_slim import reconstruct_ptychography


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(REPO_ROOT, "data", "ptycho_synthetic.json"))
    parser.add_argument("--output", default=os.path.join(REPO_ROOT, "outputs"))
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    results = reconstruct_ptychography(
        data_path=args.data,
        output_folder=args.output,
        n_epochs=args.epochs,
        minibatch_size=1,
        probe_radius=6,
        n_probe_modes=1,
        obj_size=(32, 32),
        seed=0,
    )
    print(f"Reconstruction complete. Final loss {results['loss_history'][-1]:.6f}")
    print(f"Outputs saved to: {args.output}")


if __name__ == "__main__":
    main()
