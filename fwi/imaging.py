"""
fwi/imaging.py — strain-FWI imaging conditions.

Continuous-formula (Zhou-style) imaging conditions that turn a pair of
forward + adjoint wavefield histories into a model gradient. Coupled with
`adjoint_jax_kernel.adjoint_jax(... return_wavefields=True)` this is the
hand-coded counterpart to AD's `jax.value_and_grad`.

NOTE on exactness
-----------------
This is a *continuous-PDE* imaging condition discretised with centred
finite differences via `jnp.roll`. It will NOT match `jax.grad` of the
discrete forward kernel bit-for-bit because:

  • The forward uses staggered + high-order FD; we use centred 2-point.
  • The forward has C-PML auxiliary states that AD walks through exactly;
    the continuous formula has no PML term.
  • Leap-frog time staggering between v and ε is not modelled here.

What you DO get is a gradient with the same sign and qualitatively the
same shape as AD in the interior of the physical domain — sufficient for
sanity checking and for the strain-FWI imaging derivation we'll later
plug into a hand-rolled L-BFGS loop.

Math
----
Forward strains  : ex(t), ez(t), es(t)   (es = 2 εxz)
Adjoint vels     : v̄_x(t) = U_a(t),  v̄_z(t) = V_a(t)

Lamé-space imaging:
    g_λ = Σ_t  (ex + ez) · (∂_x v̄_x + ∂_z v̄_z)  · dt
    g_μ = Σ_t  [2 ex ∂_x v̄_x + 2 ez ∂_z v̄_z
                + es (∂_z v̄_x + ∂_x v̄_z)] · dt
    g_ρ_lame = − Σ_t (v_x · ∂_t v̄_x + v_z · ∂_t v̄_z) · dt
    (sign on g_ρ_lame chosen to match `ρ ∂_t v = ∇·σ + f` convention.)

Chain to velocity (Vp, Vs, ρ):
    g_Vs   = 2 ρ Vs (g_μ − 2 g_λ)
    g_Vp   = 2 ρ Vp · g_λ
    g_ρ_v  = (Vp² − 2 Vs²) g_λ + Vs² g_μ + g_ρ_lame
"""
from __future__ import annotations

import jax.numpy as jnp


def _Dx(f, dx, ax):
    """Centred difference along spatial-x axis `ax` (works on (T, Z, X) cubes)."""
    return (jnp.roll(f, -1, axis=ax) - jnp.roll(f, 1, axis=ax)) / (2.0 * dx)


def _Dz(f, dz, ax):
    return (jnp.roll(f, -1, axis=ax) - jnp.roll(f, 1, axis=ax)) / (2.0 * dz)


def _extend_to_full_grid(field_phys, pad, free_surface):
    """Mirror the nearest-neighbour PML extension done by `forward.forward_jax`.
    Input shape (nz_dom, nx_dom); output shape (nz_t, nx_t)."""
    nz_dom, nx_dom = field_phys.shape
    if free_surface:
        nz_t = nz_dom + pad
        z_start = 0
    else:
        nz_t = nz_dom + 2 * pad
        z_start = pad
    nx_t = nx_dom + 2 * pad

    f = jnp.zeros((nz_t, nx_t), dtype=field_phys.dtype)
    f = f.at[z_start:z_start + nz_dom, pad:pad + nx_dom].set(field_phys)
    # left / right
    f = f.at[z_start:z_start + nz_dom, :pad].set(
        f[z_start:z_start + nz_dom, pad:pad + 1])
    f = f.at[z_start:z_start + nz_dom, nx_t - pad:].set(
        f[z_start:z_start + nz_dom, pad + nx_dom - 1:pad + nx_dom])
    # bottom
    f = f.at[z_start + nz_dom:, :].set(f[z_start + nz_dom - 1:z_start + nz_dom, :])
    # top (only when no free-surface)
    if not free_surface:
        f = f.at[:pad, :].set(f[pad:pad + 1, :])
    return f


def imaging_condition(
    Vs, Vp, rho,
    ex_hist, ez_hist, es_hist,
    U_a_hist, V_a_hist,
    dx, dz, dt,
    pad, free_surface, nz_dom, nx_dom,
    vx_hist=None, vz_hist=None,
    time_chunk: int = 64,
):
    """
    Compute model gradients via Zhou-style strain-FWI imaging.

    Inputs (all on the FULL padded grid (nz_t, nx_t) except the model):
      Vs, Vp, rho      : (nz_dom, nx_dom)
      ex/ez/es_hist    : (nt, nz_t, nx_t)   forward strains
      U_a/V_a_hist     : (nt, nz_t, nx_t)   adjoint velocities
      vx_hist, vz_hist : (nt, nz_t, nx_t)   forward velocities (for ρ; optional)
      dx, dz, dt       : grid / time step
      pad, free_surface, nz_dom, nx_dom : geometry for cropping

    Returns
    -------
    g_Vs, g_Vp, g_rho : (nz_dom, nx_dom)   velocity-space gradients
    g_lam, g_mu, g_rho_lame : (nz_dom, nx_dom)   intermediate Lamé gradients
    """
    # The discrete forward chain has the form  ΔU = (dt·B) · D(stuff),
    # so the discrete adjoint keeps B INSIDE the spatial-derivative
    # transpose: i.e. derivatives act on (B · v̄), not on bare v̄.
    # Extend ρ to the full padded grid (mirroring forward.forward_jax),
    # then build B = 1/ρ.
    rho_full = _extend_to_full_grid(rho, pad=pad, free_surface=free_surface)
    B_full = 1.0 / rho_full

    nt_total = U_a_hist.shape[0]
    spatial_shape = U_a_hist.shape[1:]
    g_lam_full = jnp.zeros(spatial_shape)
    g_mu_full  = jnp.zeros(spatial_shape)

    # Time integrals processed in chunks to bound peak memory.
    # Per-chunk cubes are (chunk, Z, X); intermediates (BU, BV, derivs)
    # are freed between chunks rather than living for the whole nt.
    # Sign chosen so that g_λ, g_μ have the same sign as AD's ∂L/∂λ, ∂L/∂μ.
    for k in range(0, nt_total, time_chunk):
        e = min(k + time_chunk, nt_total)
        BU = B_full * U_a_hist[k:e]            # (chunk, Z, X)
        BV = B_full * V_a_hist[k:e]
        dx_BU = _Dx(BU, dx, 2)
        dz_BV = _Dz(BV, dz, 1)
        dz_BU = _Dz(BU, dz, 1)
        dx_BV = _Dx(BV, dx, 2)
        div_Bva = dx_BU + dz_BV
        g_lam_full = g_lam_full - dt * jnp.sum(
            (ex_hist[k:e] + ez_hist[k:e]) * div_Bva, axis=0)
        g_mu_full  = g_mu_full - dt * jnp.sum(
            2.0 * ex_hist[k:e] * dx_BU
            + 2.0 * ez_hist[k:e] * dz_BV
            + es_hist[k:e] * (dz_BU + dx_BV),
            axis=0,
        )

    # ρ contribution (optional: needs forward velocities). Centred ∂_t v̄
    # at index n needs v̄[n−1], v̄[n+1]; loop over the interior of [1, T−1).
    if vx_hist is not None and vz_hist is not None:
        g_rho_lame_full = jnp.zeros(spatial_shape)
        for k in range(1, nt_total - 1, time_chunk):
            e = min(k + time_chunk, nt_total - 1)
            dt_Ua = (U_a_hist[k + 1:e + 1] - U_a_hist[k - 1:e - 1]) / (2.0 * dt)
            dt_Va = (V_a_hist[k + 1:e + 1] - V_a_hist[k - 1:e - 1]) / (2.0 * dt)
            g_rho_lame_full = g_rho_lame_full - dt * jnp.sum(
                vx_hist[k:e] * dt_Ua + vz_hist[k:e] * dt_Va, axis=0)
    else:
        g_rho_lame_full = jnp.zeros(spatial_shape)

    # Crop to physical domain
    z_start = 0 if free_surface else pad
    sl_z = slice(z_start, z_start + nz_dom)
    sl_x = slice(pad, pad + nx_dom)
    g_lam      = g_lam_full[sl_z, sl_x]
    g_mu       = g_mu_full[sl_z, sl_x]
    g_rho_lame = g_rho_lame_full[sl_z, sl_x]

    # Chain rule to velocity (Vp, Vs, ρ)
    g_Vs   = 2.0 * rho * Vs * (g_mu - 2.0 * g_lam)
    g_Vp   = 2.0 * rho * Vp * g_lam
    g_rho  = (Vp ** 2 - 2.0 * Vs ** 2) * g_lam + Vs ** 2 * g_mu + g_rho_lame

    return g_Vs, g_Vp, g_rho, g_lam, g_mu, g_rho_lame
