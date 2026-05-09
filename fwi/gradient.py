"""
fwi/gradient.py — unified FWI gradient: AD vs hand-coded adjoint + imaging.

Both backends compute (∂L/∂Vs, ∂L/∂Vp, ∂L/∂ρ) for the SAME loss

    L = scale · (1/n_shots) · mean_k{ mean_{t,r} (F·T·(est_k - obs_k))² }

where F is an optional Butterworth lowpass (`sos`) and T is an optional
cosine end-taper (`n_taper`). They use the same `invert` flag, the same
filter / taper / scale, and the same post-AD pipeline (Gaussian
smoothing, illumination, source taper). The ONLY difference is the
operator that walks the per-shot receiver-level cotangent back through
the wave equation:

    backend='ad'        →  reverse-mode AD (jax.value_and_grad on the
                            full multi-shot loss).
    backend='adjoint'   →  per-shot {forward(record fields) → adjoint_jax
                            with the same cotangent → imaging_condition}
                            then accumulated.

The cotangent itself (= AD's would-be receiver injection) is computed
identically by `jax.grad` on the small filter+taper+L2 sub-graph — so
the two paths consume the same residual and only differ in the wave-
equation transpose.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp

from .forward import build_forward_fn
from .adjoint_jax_kernel import adjoint_jax
from .imaging import imaging_condition
from .loss import build_loss_fn, gaussian_smooth_2d
from .filters import cosine_taper_end, jax_sosfilt


_COMP_TO_RECKEY = {0: "rec_vx_a", 1: "rec_vz_a", 2: "rec_ex_a", 3: "rec_ez_a"}


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def compute_gradient(
    backend: str,
    *,
    Vs, Vp, rho,
    src_x, src_z, src_wavelet,
    observed,                              # (n_shots, nt, nrec)
    rec_x, rec_z,
    nx_dom, nz_dom, dx, dz, dt, nt, fc, pad,
    component: int = 2,
    invert: str = "vs",
    scale: float = 1.0,
    sos=None,
    n_taper: int = 0,
    fd_order: int = 2,
    free_surface: bool = False,
    src_type: str = "force",
    block_size: Optional[int] = None,
    # post-raw-gradient preconditioning (applied identically to BOTH backends)
    grad_smooth_sigma: Optional[float] = None,
    source_taper=None,
    illumination=None,
):
    """
    Compute (g_Vs, g_Vp, g_rho) at the current model.

    Returns
    -------
    grads : (g_Vs, g_Vp, g_rho)   — physical-domain (nz_dom, nx_dom) arrays.
    loss  : scalar (numpy float)
    """
    block_size = block_size or nt

    common = dict(
        Vs=Vs, Vp=Vp, rho=rho,
        src_x=src_x, src_z=src_z, src_wavelet=src_wavelet,
        observed=observed, rec_x=rec_x, rec_z=rec_z,
        nx_dom=nx_dom, nz_dom=nz_dom, dx=dx, dz=dz, dt=dt, nt=nt, fc=fc, pad=pad,
        component=component, invert=invert, scale=scale,
        sos=sos, n_taper=n_taper,
        fd_order=fd_order, free_surface=free_surface, src_type=src_type,
        block_size=block_size,
    )

    if backend == "ad":
        grads, loss = _backend_ad(**common)
    elif backend == "adjoint":
        grads, loss = _backend_adjoint(**common)
    else:
        raise ValueError(f"unknown backend {backend!r}; use 'ad' or 'adjoint'")

    grads = _postprocess(grads, grad_smooth_sigma, source_taper, illumination)
    return grads, loss


# ─────────────────────────────────────────────────────────────────────────────
# AD backend (reuses build_loss_fn machinery)
# ─────────────────────────────────────────────────────────────────────────────

def _backend_ad(*, Vs, Vp, rho, src_x, src_z, src_wavelet,
                observed, rec_x, rec_z,
                nx_dom, nz_dom, dx, dz, dt, nt, fc, pad,
                component, invert, scale, sos, n_taper,
                fd_order, free_surface, src_type, block_size):

    run_shot = build_forward_fn(
        nz_dom, nx_dom, dx, dz, dt, nt, fc, pad, block_size,
        rec_x, rec_z, fd_order=fd_order, free_surface=free_surface,
    )

    # build_loss_fn already handles  sos + n_taper + scale + mean-over-shots +
    # invert. Pass None for the post-AD knobs — those are applied externally
    # by `_postprocess` (so AD and adjoint backends share the same wrapping).
    loss_fn = build_loss_fn(
        run_shot, observed, np.asarray(src_x), np.asarray(src_z), src_wavelet,
        component=component, scale=scale, invert=invert,
        sos=sos, n_taper=n_taper,
        source_taper=None, grad_smooth_sigma=None,
        tikhonov_alpha=None, illumination=None,
    )

    loss, grads = loss_fn((jnp.asarray(Vs), jnp.asarray(Vp), jnp.asarray(rho)))
    g_Vs = np.asarray(grads[0])
    g_Vp = np.asarray(grads[1])
    g_rho = np.asarray(grads[2])
    return (g_Vs, g_Vp, g_rho), float(loss)


# ─────────────────────────────────────────────────────────────────────────────
# Adjoint backend (per-shot loop: forward → cotangent → adjoint → imaging)
# ─────────────────────────────────────────────────────────────────────────────

def _shot_residual_prep(pred, obs, sos, n_taper, scale, n_shots):
    """
    Per-shot scalar misfit term  (scale / n_shots) · mean( (F·T·(pred - obs))² ).

    `jax.grad` of this w.r.t `pred` produces exactly the receiver-level
    cotangent that AD would back-propagate through the wave equation
    (i.e.  (2·scale/(n_shots·N)) · T^T·F^T·F·T·(pred-obs)  with all
    transposes routed by AD).
    """
    est = pred
    obs_p = obs
    if n_taper > 0:
        est = cosine_taper_end(est, n_taper)
        obs_p = cosine_taper_end(obs_p, n_taper)
    if sos is not None:
        est = jax_sosfilt(est, sos)
        obs_p = jax_sosfilt(obs_p, sos)
    return (scale / n_shots) * jnp.mean((est - obs_p) ** 2)


def _backend_adjoint(*, Vs, Vp, rho, src_x, src_z, src_wavelet,
                     observed, rec_x, rec_z,
                     nx_dom, nz_dom, dx, dz, dt, nt, fc, pad,
                     component, invert, scale, sos, n_taper,
                     fd_order, free_surface, src_type, block_size):

    src_x = np.asarray(src_x, dtype=np.int32)
    src_z = np.asarray(src_z, dtype=np.int32)
    rec_x = np.asarray(rec_x, dtype=np.int32)
    rec_z = np.asarray(rec_z, dtype=np.int32)

    n_shots = len(src_x)
    nrec = len(rec_x)
    rx_pad = jnp.asarray(rec_x + pad)
    rz_pad = jnp.asarray(rec_z + (0 if free_surface else pad))

    return_vars = ["vx_full", "vz_full", "ex_full", "ez_full", "es_full"]
    run_shot_full = build_forward_fn(
        nz_dom, nx_dom, dx, dz, dt, nt, fc, pad, block_size,
        rec_x, rec_z, fd_order=fd_order, free_surface=free_surface,
        return_vars=return_vars,
    )

    sos_jax = jnp.asarray(sos) if sos is not None else None

    Vs_j = jnp.asarray(Vs);  Vp_j = jnp.asarray(Vp);  rho_j = jnp.asarray(rho)
    g_Vs = np.zeros_like(np.asarray(Vs), dtype=np.float64)
    g_Vp = np.zeros_like(g_Vs);  g_rho = np.zeros_like(g_Vs)
    loss_total = 0.0

    rec_extract_axis = {0: "vx", 1: "vz", 2: "ex", 3: "ez"}[component]
    rec_key = _COMP_TO_RECKEY[component]
    zero_rec = jnp.zeros((nt, nrec))

    for k in range(n_shots):
        sx_k = int(src_x[k]);  sz_k = int(src_z[k])
        obs_k = jnp.asarray(observed[k])

        # 1) Forward at current model with full wavefield recording.
        vx_h, vz_h, ex_h, ez_h, es_h = run_shot_full(
            Vs_j, Vp_j, rho_j, jnp.int32(sx_k), jnp.int32(sz_k), src_wavelet,
        )

        # 2) Pred at receivers on the chosen component.
        comp_full_map = {"vx": vx_h, "vz": vz_h, "ex": ex_h, "ez": ez_h}
        full_field = comp_full_map[rec_extract_axis]
        pred_k = full_field[:, rz_pad, rx_pad]            # (nt, nrec)

        # 3) Per-shot loss + receiver cotangent via small-AD on residual prep.
        def _shot_loss(pred_):
            return _shot_residual_prep(pred_, obs_k, sos_jax, n_taper, scale, n_shots)
        loss_k, cotan_rec = jax.value_and_grad(_shot_loss)(pred_k)
        loss_total += float(loss_k)

        # 4) Pack the cotangent into the right adjoint-receiver channel.
        adj_kwargs = dict(rec_vx_a=zero_rec, rec_vz_a=zero_rec,
                          rec_ex_a=zero_rec, rec_ez_a=zero_rec)
        adj_kwargs[rec_key] = cotan_rec

        # 5) Hand-coded adjoint with adjoint-VELOCITY recording only
        #    (skip ex_a, ez_a, es_a — imaging condition doesn't use them
        #    and they cost ~3× extra cube memory).
        _, U_a, V_a = adjoint_jax(
            Vs_j, Vp_j, rho_j,
            src_x=jnp.int32(sx_k), src_z=jnp.int32(sz_k),
            rec_x=jnp.asarray(rec_x), rec_z=jnp.asarray(rec_z),
            nx_dom=nx_dom, nz_dom=nz_dom,
            dx=dx, dz=dz, dt=float(dt), nt=nt, fc=fc, pad=pad,
            **adj_kwargs,
            fd_order=fd_order, free_surface=free_surface, src_type=src_type,
            return_wavefields="velocity",
        )

        # 6) Imaging condition for this shot.
        gVs_k, gVp_k, grho_k, _, _, _ = imaging_condition(
            Vs=Vs_j, Vp=Vp_j, rho=rho_j,
            ex_hist=ex_h, ez_hist=ez_h, es_hist=es_h,
            U_a_hist=U_a, V_a_hist=V_a,
            dx=dx, dz=dz, dt=float(dt),
            pad=pad, free_surface=free_surface,
            nz_dom=nz_dom, nx_dom=nx_dom,
            vx_hist=vx_h, vz_hist=vz_h,
        )
        g_Vs += np.asarray(gVs_k)
        g_Vp += np.asarray(gVp_k)
        g_rho += np.asarray(grho_k)

        # Drop the big cubes before the next shot (memory hygiene)
        del vx_h, vz_h, ex_h, ez_h, es_h, U_a, V_a, full_field

    # Apply `invert` (zero out frozen components — AD does this via stop_gradient)
    if invert == "vs":
        g_Vp = np.zeros_like(g_Vp);  g_rho = np.zeros_like(g_rho)
    elif invert == "vs_vp":
        g_rho = np.zeros_like(g_rho)
    elif invert == "all":
        pass
    else:
        raise ValueError(f"invert={invert!r}; use 'vs', 'vs_vp', or 'all'")

    return (g_Vs, g_Vp, g_rho), float(loss_total)


# ─────────────────────────────────────────────────────────────────────────────
# Shared post-processing (applied identically after either backend)
# ─────────────────────────────────────────────────────────────────────────────

def _postprocess(grads, grad_smooth_sigma, source_taper, illumination):
    out = list(grads)
    if grad_smooth_sigma is not None:
        out = [np.asarray(gaussian_smooth_2d(jnp.asarray(g), grad_smooth_sigma))
               for g in out]
    if source_taper is not None:
        st = np.asarray(source_taper)
        out = [g * st for g in out]
    if illumination is not None:
        il = np.asarray(illumination)
        out = [g * il for g in out]
    return tuple(out)
