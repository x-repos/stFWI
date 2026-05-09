"""
fwi/loss.py — Loss function builder for FWI.

Builds a JAX-differentiable loss function (value_and_grad) that computes
the L2 misfit between observed and estimated seismograms.

Author: Minh Nhat Tran
Date: 2026
"""
import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve2d

from .filters import cosine_taper_end, jax_sosfilt


def _gaussian_kernel_2d(sigma, truncate=3.0):
    """Build a normalized 2-D Gaussian kernel (like scipy gaussian_filter)."""
    radius = int(truncate * sigma + 0.5)
    size = 2 * radius + 1
    x = jnp.arange(size) - radius
    g1d = jnp.exp(-0.5 * (x / sigma) ** 2)
    g2d = jnp.outer(g1d, g1d)
    return g2d / g2d.sum()


def gaussian_smooth_2d(field, sigma):
    """Smooth a 2-D field with a Gaussian kernel using JAX convolution."""
    kernel = _gaussian_kernel_2d(sigma)
    return convolve2d(field, kernel, mode='same')


def build_loss_fn(run_shot, observed, src_x_list, src_z, src_wavelet,
                  component=1, scale=1e20,
                  invert='vs', sos=None, n_taper=0,
                  source_taper=None,
                  grad_smooth_sigma=None,
                  tikhonov_alpha=None,
                  illumination=None):
    """
    Build a differentiable loss function for FWI.

    Parameters
    ----------
    run_shot : callable
        JIT-compiled forward function from build_forward_fn.
    observed : array
        Observed data (n_shots, nt, n_rec).
    src_x_list : array
        Source x positions.
    src_z : array
        Source z positions.
    src_wavelet : jax array
        Source wavelet (nt,) — passed through to run_shot.
    component : int
        Which component to compare (0=vx, 1=vz, 2=exx, 3=ezz).
    scale : float
        Loss scaling factor.
    invert : str
        'vs' = invert Vs only, 'vs_vp' = invert Vs & Vp, 'all' = invert all.
    sos : array-like, optional
        SOS filter coefficients from scipy.signal.butter(..., output='sos').
    n_taper : int, optional
        Number of samples for cosine taper at the end of the trace.
    source_taper : jax array, optional
        2-D source taper (nz, nx) to damp gradient near sources (DENISE style).
    grad_smooth_sigma : float, optional
        Gaussian smoothing sigma (in grid points) applied to gradients.
    tikhonov_alpha : float, optional
        Tikhonov regularization weight as fraction of misfit (e.g. 0.01 = 1%).
    illumination : jax array, optional
        2-D illumination scaling (nz, nx) from compute_illumination.

    Returns
    -------
    loss_fn : callable
        Function that takes params=(Vs, Vp, rho) and returns (loss, grad).
    """
    n_shots = len(src_x_list)

    # Pre-process observed data (once): taper → filter
    obs_proc = jnp.array(observed)
    if n_taper > 0:
        obs_proc = jax.vmap(lambda obs: cosine_taper_end(obs, n_taper))(obs_proc)
    if sos is not None:
        sos_jax = jnp.array(sos)
        obs_proc = jax.vmap(lambda obs: jax_sosfilt(obs, sos_jax))(obs_proc)

    src_x_jax = jnp.array(src_x_list, dtype=jnp.int32)
    src_z_val = jnp.int32(src_z[0])

    def _single_shot_misfit(Vs, Vp, rho, sx, obs_i):
        """Compute L2 misfit for one shot."""
        pred = run_shot(Vs, Vp, rho, sx, src_z_val, src_wavelet)
        est = pred[component]
        if n_taper > 0:
            est = cosine_taper_end(est, n_taper)
        if sos is not None:
            est = jax_sosfilt(est, sos_jax)
        return jnp.mean((est - obs_i) ** 2)

    def loss_fn(params):
        Vs, Vp, rho = params

        # Apply inversion constraints: fix parameters via stop_gradient
        if invert == 'vs':
            Vp  = jax.lax.stop_gradient(Vp)
            rho = jax.lax.stop_gradient(rho)
        elif invert == 'vs_vp':
            rho = jax.lax.stop_gradient(rho)

        # lax.map: sequential like for-loop but JAX compiles only ONCE
        def _body(carry):
            sx, obs_i = carry
            return _single_shot_misfit(Vs, Vp, rho, sx, obs_i)

        misfits = jax.lax.map(_body, (src_x_jax, obs_proc))
        total = scale * jnp.mean(misfits)

        # Tikhonov regularization: alpha is fraction of misfit
        # e.g. alpha=0.001 → reg contributes ~0.1% of misfit magnitude
        # stop_gradient on scaling so gradient flows only through reg, not through misfit/reg ratio
        if tikhonov_alpha is not None:
            reg = jnp.sum(jnp.diff(Vs, axis=0) ** 2) + jnp.sum(jnp.diff(Vs, axis=1) ** 2)
            if invert in ('vs_vp', 'all'):
                reg = reg + jnp.sum(jnp.diff(Vp, axis=0) ** 2) + jnp.sum(jnp.diff(Vp, axis=1) ** 2)
            if invert == 'all':
                reg = reg + jnp.sum(jnp.diff(rho, axis=0) ** 2) + jnp.sum(jnp.diff(rho, axis=1) ** 2)
            reg_scale = jax.lax.stop_gradient(total / (reg + 1e-30))
            total = total + tikhonov_alpha * reg_scale * reg

        return total

    vg = jax.value_and_grad(loss_fn)

    apply_taper  = source_taper is not None
    apply_smooth = grad_smooth_sigma is not None
    apply_illum  = illumination is not None

    if apply_taper or apply_smooth or apply_illum:
        def postproc_loss_fn(params):
            loss, grads = vg(params)
            if apply_smooth:
                grads = tuple(gaussian_smooth_2d(g, grad_smooth_sigma) for g in grads)
            if apply_taper:
                grads = tuple(g * source_taper for g in grads)
            if apply_illum:
                grads = tuple(g * illumination for g in grads)
            return loss, grads
        return postproc_loss_fn

    return vg
