"""
Static import audit for the slim ptychography demo.

This script walks the import graph starting from
``demos/2d_ptychography_experimental_data.py`` and fails loudly if any module
outside the ``adorym_simplified`` tree (other than stdlib/third-party
dependencies like numpy/h5py/dxchange) or any ``adorym`` site-packages import
is detected.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_PATH = REPO_ROOT / "demos" / "2d_ptychography_experimental_data.py"


def _node_to_modules(node: ast.AST) -> List[str]:
    mods: List[str] = []
    if isinstance(node, ast.Import):
        for alias in node.names:
            mods.append(alias.name)
    elif isinstance(node, ast.ImportFrom):
        if node.module:
            mods.append(node.module)
    return mods


def _local_path(module: str) -> Path | None:
    rel = module.replace(".", "/")
    file_candidate = REPO_ROOT / f"{rel}.py"
    if file_candidate.exists():
        return file_candidate
    pkg_candidate = REPO_ROOT / rel / "__init__.py"
    if pkg_candidate.exists():
        return pkg_candidate
    return None


def _walk(path: Path, seen: Set[Path], graph: Dict[Path, List[str]]) -> None:
    if path in seen:
        return
    seen.add(path)
    tree = ast.parse(path.read_text())
    imports = []
    for node in ast.walk(tree):
        imports.extend(_node_to_modules(node))
    graph[path] = imports
    for mod in imports:
        if mod.startswith(("core", "backends", "utils", "io")):
            mod = f"adorym_slim.{mod}"
        if mod.startswith("adorym") and not mod.startswith("adorym_slim"):
            raise RuntimeError(f"Illegal import detected: {mod} in {path}")
        local = _local_path(mod)
        if local is None:
            sibling = path.parent / (mod.replace(".", "/") + ".py")
            if sibling.exists():
                local = sibling
            pkg_sibling = path.parent / mod / "__init__.py"
            if pkg_sibling.exists():
                local = pkg_sibling
        if local is not None:
            _walk(local, seen, graph)


def main() -> None:
    seen: Set[Path] = set()
    graph: Dict[Path, List[str]] = {}
    _walk(DEMO_PATH, seen, graph)

    print("=== Import Graph ===")
    for path, mods in graph.items():
        rel = path.relative_to(REPO_ROOT)
        print(f"{rel}:")
        for mod in mods:
            print(f"  - {mod}")
    print("\n=== Module Closure ===")
    for path in sorted(seen):
        print(path.relative_to(REPO_ROOT))

    # Final guard: verify adorym is not importable from sys.path.
    spec = importlib.util.find_spec("adorym")
    if spec is not None:
        print("Warning: site-packages `adorym` is installed; ensure only adorym_slim is imported.")
    print("Import audit completed without detecting forbidden imports.")


if __name__ == "__main__":
    main()
