"""Array backend configuration for the slim ptychography stack."""

from .array_api import (
    Array,
    ensure_complex,
    fft2c,
    ifft2c,
    random_complex,
    safe_divide,
    to_magnitude,
    zeros,
)

__all__ = [
    "Array",
    "ensure_complex",
    "fft2c",
    "ifft2c",
    "random_complex",
    "safe_divide",
    "to_magnitude",
    "zeros",
]
