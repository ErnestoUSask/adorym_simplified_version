# Symbol audit for `demos/2d_ptychography_experimental_data.py`

## Dependency tree

- `demos/2d_ptychography_experimental_data.py`
  - standard lib: `argparse`, `importlib.util`, `os`, `sys`
  - third-party: `numpy`, optional `dxchange`
  - local: `adorym_slim.AdamOptimizer`, `adorym_slim.reconstruct_ptychography`
- `adorym_slim/__init__.py`
  - re-exports from `adorym_slim/core/optim.py` and `adorym_slim/core/recon.py`
- `adorym_slim/core/recon.py`
  - local: `aperture_probe`, `initialize_object`, `loss_and_gradients` (from `core/ptycho_forward.py`)
  - local: `AdamOptimizer` (from `core/optim.py`)
  - local: dataset helpers (from `io/data_loaders.py`)
  - third-party: `numpy`, optional `dxchange`
- `adorym_slim/core/ptycho_forward.py`
  - third-party: `numpy`
  - local: backend helpers (`backends/array_api.py`), patch ops (`utils/misc.py`)
- `adorym_slim/io/data_loaders.py`
  - third-party: `h5py`, `numpy`, optional `dxchange`
- `adorym_slim/core/optim.py`
  - third-party: `numpy`

## Imported symbol mapping

| Demo import | Simplified location | Original Adorym source | Notes |
| ----------- | ------------------- | ---------------------- | ----- |
| `reconstruct_ptychography` | `adorym_slim/core/recon.py` | `adorym/ptychography.py::reconstruct_ptychography` | Preserves parameter names and defaults used by the demo; implementation reduced to NumPy-only forward/adjoint without MPI/backends. |
| `AdamOptimizer` | `adorym_slim/core/optim.py` | `adorym/optimizers.py::Adam` | Options accept `options_dict` like original demo; state scoped per-parameter key. |
| `dxchange` (optional) | external dependency | used in original for TIFF I/O | Only used if available to read/write checkpoint TIFFs; NumPy `.npy` fallback always produced. |
| `numpy` | external dependency | N/A | Provides array math in place of the original autograd/torch stack. |

## Compatibility notes

- Parameter names, defaults, and CLI options from the original demo are preserved. Unsupported branches (MPI, GPU backends, advanced regularizers) are stubbed out but ignored safely.
- The simplified save format always includes NumPy `.npy` files. TIFF checkpoints are written only if `dxchange` is installed.
- If the referenced HDF5 file is missing, the loader transparently falls back to the bundled `data/ptycho_synthetic.json` so the script remains runnable offline.
