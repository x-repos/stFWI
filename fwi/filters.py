"""
fwi/filters.py — JAX-differentiable signal processing for FWI.

Provides SOS (biquad) filtering and cosine tapering, fully differentiable
through JAX autodiff so they can be used inside the loss function.

Author: Minh Nhat Tran
Date: 2026
"""
import jax
import jax.numpy as jnp


def jax_biquad(x, b0, b1, b2, a1, a2):
    """Apply one biquad (second-order IIR) section along axis=0. Differentiable in JAX."""
    def step(carry, xi):
        x1, x2, y1, y2 = carry
        yi = b0*xi + b1*x1 + b2*x2 - a1*y1 - a2*y2
        return (xi, x1, yi, y1), yi

    init = (jnp.zeros_like(x[0]),) * 4
    _, y = jax.lax.scan(step, init, x)
    return y


def cosine_taper_end(x, n_taper):
    """Taper the end of axis=0 with a cosine window (1→0). x: (nt, n_rec)."""
    n_taper = min(n_taper, x.shape[0])
    if n_taper <= 0:
        return x
    taper = (jnp.cos(jnp.arange(1, n_taper + 1) / n_taper * jnp.pi) + 1) / 2
    return x.at[-n_taper:].multiply(taper[:, None])


def jax_sosfilt(x, sos):
    """Apply SOS filter (chain of biquad sections). sos: (n_sections, 6)."""
    for i in range(sos.shape[0]):
        s = sos[i]
        b0, b1, b2 = s[0]/s[3], s[1]/s[3], s[2]/s[3]
        a1, a2 = s[4]/s[3], s[5]/s[3]
        x = jax_biquad(x, b0, b1, b2, a1, a2)
    return x
