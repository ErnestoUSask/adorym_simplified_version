"""
Lightweight optimizers used by the simplified ptychography loop.

Only Adam is implemented because the reference demo relies on it.
"""

from __future__ import annotations

from typing import List, Tuple

from . import math_utils as mu


class AdamState:
    def __init__(self, shape: Tuple[int, int]):
        self.m = mu.zeros(shape)
        self.v = mu.zeros(shape)
        self.t = 0


class AdamOptimizer:
    def __init__(self, step_size: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.step_size = step_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.state_cache = {}

    def _get_state(self, key: str, shape: Tuple[int, int]) -> AdamState:
        if key not in self.state_cache:
            self.state_cache[key] = AdamState(shape)
        return self.state_cache[key]

    def step(self, key: str, params: mu.Array2D, grads: mu.Array2D) -> mu.Array2D:
        state = self._get_state(key, mu.shape(params))
        state.t += 1
        m = state.m
        v = state.v
        b1 = self.beta1
        b2 = self.beta2
        h, w = mu.shape(params)

        updated = mu.zeros((h, w))
        for iy in range(h):
            for ix in range(w):
                g = grads[iy][ix]
                m[iy][ix] = b1 * m[iy][ix] + (1 - b1) * g
                v[iy][ix] = b2 * v[iy][ix] + (1 - b2) * (g * g.conjugate())
                m_hat = m[iy][ix] / (1 - b1 ** state.t)
                v_hat = v[iy][ix] / (1 - b2 ** state.t)
                denom = (v_hat.real ** 0.5) + self.eps
                updated[iy][ix] = params[iy][ix] - self.step_size * m_hat / denom
        return updated
