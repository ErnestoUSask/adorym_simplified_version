## adorym_simplified

A slim, self-contained extraction of the original Adorym
`demos/2d_ptychography_experimental_data.py` workflow. The script keeps the same
user-facing parameters but routes all functionality through the `adorym_slim`
namespace to avoid importing the full `adorym` package.

### Layout

- `adorym_slim/`: minimal package with NumPy-only forward/adjoint operators and Adam optimizers.
- `demos/2d_ptychography_experimental_data.py`: entrypoint mirroring the original demo with a no-`adorym` guard.
- `data/ptycho_synthetic.json`: bundled fallback dataset used when the referenced HDF5 is absent.
- `checkpoints/import_audit.py`: static import graph checker that fails if external `adorym` is referenced.
- `checkpoints/symbol_audit.md`: mapping between imported symbols and their simplified implementations.
- `verify_against_original.py`: deterministic replay to ensure repeatability within the slim stack.

### Running the demo

From the repository root:

```bash
python adorym_simplified/demos/2d_ptychography_experimental_data.py --epoch None --save_path cone_256_foam_ptycho --output_folder test
```

Notes:
- The demo inserts `adorym_simplified/` onto `sys.path`; it will warn if a site-packages `adorym` is present.
- If the target HDF5 (`save_path`/`fname`) is missing, the bundled JSON dataset is used instead.
- Intermediate reconstructions are written under `save_path/output_folder/`, with NumPy and (optionally) TIFF outputs.

### Requirements

Install the minimal dependencies:

```bash
pip install -r adorym_simplified/requirements_min.txt
```

Required packages: `numpy`, `h5py`, `dxchange` (for checkpoint I/O). GPU and MPI code paths are intentionally removed.
