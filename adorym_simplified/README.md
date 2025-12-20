## adorym_simplified

This folder contains a dependency-free reduction of the `demos/2d_ptychography_experimental_data.py` workflow.
The original repository depends on NumPy, Torch, h5py, and dxchange; these are not available in the execution
environment, so the reduction implements the core ptychography loop using only Python's standard library.

### Layout

- `adorym_slim/`: minimal package exposing `reconstruct_ptychography` and a tiny Adam optimizer.
- `demos/2d_ptychography_experimental_data.py`: drop-in replacement for the original demo that imports from `adorym_slim`.
- `data/ptycho_synthetic.json`: small synthetic diffraction dataset used for the example run.
- `verify_against_original.py`: runs the simplified reconstruction twice and checks for numerical agreement.

### Running the demo

```bash
python adorym_simplified/demos/2d_ptychography_experimental_data.py --epochs 3
```

The script writes object and probe estimates into `adorym_simplified/outputs/` as JSON files containing
complex values. GPU and MPI execution are not available in this trimmed build; everything runs on CPU through
pure Python.

### Verification

To confirm reproducibility:

```bash
python adorym_simplified/verify_against_original.py
```

This executes two reconstructions with identical seeds and reports relative errors between the results.
Because both runs share the same backend, the tolerance is extremely tight (1e-6). A site-packages
installation of `adorym` is explicitly rejected to guarantee isolation.

### Requirements

No external pip dependencies are required beyond the Python standard library.
