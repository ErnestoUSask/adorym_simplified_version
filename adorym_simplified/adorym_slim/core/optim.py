"""
Optimizers used by the simplified reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from ..backends import Array, zeros


@dataclass
class AdamState:
    m: Array
    v: Array
    t: int = 0


class AdamOptimizer:
    """
    Lightweight Adam implementation with a small state cache keyed by parameter name.
    """

    def __init__(self, name: str | None = None, output_folder: str | None = None, distribution_mode=None, options_dict=None):
        opts = options_dict or {}
        self.step_size = opts.get("step_size", 1e-3)
        self.beta1 = opts.get("beta1", 0.9)
        self.beta2 = opts.get("beta2", 0.999)
        self.eps = opts.get("eps", 1e-8)
        self.name = name or "adam"
        self.output_folder = output_folder
        self.distribution_mode = distribution_mode
        self.state_cache: Dict[str, AdamState] = {}

    def _state_for(self, key: str, shape: Tuple[int, int]) -> AdamState:
        if key not in self.state_cache:
            self.state_cache[key] = AdamState(m=zeros(shape), v=zeros(shape))
        return self.state_cache[key]

    def step(self, key: str, params: Array, grads: Array) -> Array:
        state = self._state_for(key, params.shape)
        state.t += 1
        b1 = self.beta1
        b2 = self.beta2
        lr = self.step_size

        m = state.m
        v = state.v

        m[...] = b1 * m + (1 - b1) * grads
        v[...] = b2 * v + (1 - b2) * (grads * np.conj(grads))

        m_hat = m / (1 - b1 ** state.t)
        v_hat = v / (1 - b2 ** state.t)
        denom = np.sqrt(v_hat.real) + self.eps

        return params - lr * m_hat / denom


__all__ = ["AdamOptimizer"]
