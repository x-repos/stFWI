"""
fwi/adjoint_jax_kernel.py — hand-coded discrete adjoint of fwi/forward.py.

Implements the exact discrete transpose of `forward_jax` viewed as a linear
operator from the source wavelet to the receiver data, at fixed model
parameters. NO autodiff is used; every transpose is derived by hand from
the forward kernel.

Phases implemented (this file):
    fd_order      ∈ {2, 4, 8}        (staggered FD stencil transposes)
    free_surface  ∈ {False, True}    (PML on 4 or 3 sides)
    src_type      ∈ {'force', 'moment'}

The forward at fixed model is linear in src_wavelet, so the discrete
adjoint produced here passes the dot-product test
    < L s, y >  ==  < s, L^T y >
to machine precision (~1e-14 in float64) regardless of PML decay.

Public API:
    pml_apply_T(D_new_a, psi_new_a, k, b, a) -> (D_a, psi_a)
    adjoint_jax(...) -> src_wavelet_a, of shape (nt,)
"""
from __future__ import annotations

from functools import partial
import jax
import jax.numpy as jnp
from jax import lax

from .forward import _pml_coefficients


# ─────────────────────────────────────────────────────────────────────────────
# Transpose of the C-PML recursion (per-point linear map)
#
# Forward (per-point):
#     psi_new = b * psi + a * D
#     D_used  = D / k + psi_new = (1/k + a) * D + b * psi
#
# Matrix form  [D_used; psi_new]  =  [[1/k+a, b]; [a, b]] [D; psi]
#
# Transpose:
#     D_a    = (1/k + a) * D_used_a + a * psi_new_a
#     psi_a  = b * D_used_a + b * psi_new_a   =   b * (D_used_a + psi_new_a)
#
# In the backward time sweep, psi_new_a comes from the carry (it is the
# adjoint that flowed in from the next-later timestep). psi_a goes out as
# the new carry, to be consumed by the next-earlier timestep.
# ─────────────────────────────────────────────────────────────────────────────

def pml_apply_T(D_new_a, psi_new_a, k, b, a):
    D_a   = (1.0 / k + a) * D_new_a + a * psi_new_a
    psi_a = b * (D_new_a + psi_new_a)
    return D_a, psi_a


# ─────────────────────────────────────────────────────────────────────────────
# Hand-coded adjoint kernel
# ─────────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnames=[
    'nx_dom', 'nz_dom', 'nt', 'pad', 'fd_order', 'free_surface', 'src_type',
    'return_wavefields',
])
def adjoint_jax(Vs, Vp, rho, src_x, src_z, rec_x, rec_z,
                nx_dom, nz_dom, dx, dz, dt, nt, fc, pad,
                rec_vx_a, rec_vz_a, rec_ex_a, rec_ez_a,
                fd_order=2, free_surface=False, src_type='force',
                return_wavefields=False):
    """
    Discrete transpose of `forward_jax` mapping receiver-data adjoint cubes
    back to a source-wavelet adjoint trace.

    Parameters
    ----------
    Vs, Vp, rho : (nz_dom, nx_dom)
        Velocity / density model at which to linearise.
    src_x, src_z : int
        Source location indices (physical-domain coordinates).
    rec_x, rec_z : (nrec,) int
        Receiver location indices.
    nx_dom, nz_dom, dx, dz, dt, nt, fc, pad : grid / time / PML params.
    rec_vx_a, rec_vz_a, rec_ex_a, rec_ez_a : (nt, nrec)
        Adjoint of the four recorded fields at receivers.
    fd_order, free_surface, src_type : configuration flags.
        Only fd_order=2, free_surface=False, src_type='force' is implemented
        in this file (Phase 1).

    Returns
    -------
    If return_wavefields=False (default):
        src_wavelet_a : (nt,) array
    If return_wavefields=True:
        (src_wavelet_a, U_a_hist, V_a_hist, ex_a_hist, ez_a_hist, es_a_hist)
        where each *_a_hist has shape (nt, nz_t, nx_t) on the FULL padded grid,
        stacked in forward-time order. Frame `it` is the adjoint state at the
        carry-OUT of adjoint step `it` (≈ adjoint at forward time `it-1`).
        For a backward-propagation animation, play the frames in reverse:
            U_a_hist[::-1]
    """
    if fd_order not in (2, 4, 8):
        raise NotImplementedError(f"fd_order={fd_order} not supported (use 2, 4, or 8).")
    if src_type not in ('force', 'moment'):
        raise NotImplementedError(f"src_type={src_type!r} not supported (use 'force' or 'moment').")

    # ── Setup must mirror forward_jax exactly (model extension + PML) ─────
    Vs = jnp.array(Vs)
    Vp = jnp.array(Vp)
    rho = jnp.array(rho)
    rec_x = jnp.array(rec_x)
    rec_z = jnp.array(rec_z)
    rec_vx_a = jnp.asarray(rec_vx_a)
    rec_vz_a = jnp.asarray(rec_vz_a)
    rec_ex_a = jnp.asarray(rec_ex_a)
    rec_ez_a = jnp.asarray(rec_ez_a)

    # FD stencil coefficients (must match fwi/forward.py exactly)
    if fd_order == 8:
        c1, c2, c3, c4 = 1225.0/1024.0, -245.0/3072.0, 49.0/5120.0, -5.0/7168.0
    elif fd_order >= 4:
        c1, c2, c3, c4 = 9.0/8.0, -1.0/24.0, 0.0, 0.0
    else:
        c1, c2, c3, c4 = 1.0, 0.0, 0.0, 0.0

    if free_surface:
        nx_t, nz_t = nx_dom + 2 * pad, nz_dom + pad
        z_start = 0
    else:
        nx_t, nz_t = nx_dom + 2 * pad, nz_dom + 2 * pad
        z_start = pad
    nz, nx = nz_t, nx_t

    Vs_full  = jnp.zeros((nz, nx)).at[z_start:z_start+nz_dom, pad:pad+nx_dom].set(Vs)
    Vp_full  = jnp.zeros((nz, nx)).at[z_start:z_start+nz_dom, pad:pad+nx_dom].set(Vp)
    rho_full = jnp.zeros((nz, nx)).at[z_start:z_start+nz_dom, pad:pad+nx_dom].set(rho)

    Vs_full  = Vs_full.at[z_start:z_start+nz_dom, :pad].set(
        Vs_full[z_start:z_start+nz_dom, pad:pad+1])
    Vp_full  = Vp_full.at[z_start:z_start+nz_dom, :pad].set(
        Vp_full[z_start:z_start+nz_dom, pad:pad+1])
    rho_full = rho_full.at[z_start:z_start+nz_dom, :pad].set(
        rho_full[z_start:z_start+nz_dom, pad:pad+1])
    Vs_full  = Vs_full.at[z_start:z_start+nz_dom, nx_t-pad:].set(
        Vs_full[z_start:z_start+nz_dom, pad+nx_dom-1:pad+nx_dom])
    Vp_full  = Vp_full.at[z_start:z_start+nz_dom, nx_t-pad:].set(
        Vp_full[z_start:z_start+nz_dom, pad+nx_dom-1:pad+nx_dom])
    rho_full = rho_full.at[z_start:z_start+nz_dom, nx_t-pad:].set(
        rho_full[z_start:z_start+nz_dom, pad+nx_dom-1:pad+nx_dom])
    Vs_full  = Vs_full.at[z_start+nz_dom:, :].set(
        Vs_full[z_start+nz_dom-1:z_start+nz_dom, :])
    Vp_full  = Vp_full.at[z_start+nz_dom:, :].set(
        Vp_full[z_start+nz_dom-1:z_start+nz_dom, :])
    rho_full = rho_full.at[z_start+nz_dom:, :].set(
        rho_full[z_start+nz_dom-1:z_start+nz_dom, :])
    if not free_surface:
        Vs_full  = Vs_full.at[:pad, :].set(Vs_full[pad:pad+1, :])
        Vp_full  = Vp_full.at[:pad, :].set(Vp_full[pad:pad+1, :])
        rho_full = rho_full.at[:pad, :].set(rho_full[pad:pad+1, :])

    M    = rho_full * Vs_full**2          # μ
    L2M  = rho_full * Vp_full**2          # λ + 2μ
    L    = L2M - 2 * M                    # λ
    B    = 1.0 / rho_full

    # Free-surface ratio used in BC at iz=0:  ∂Vz/∂z = -λ/(λ+2μ) ∂Vx/∂x
    L_ratio = -L / L2M if free_surface else None

    Vp_max = jnp.max(Vp)
    kx, kz, bx, bz, ax, az = _pml_coefficients(
        pad, dx, dt, Vp_max, fc, nz, nx, nz_dom, free_surface=free_surface)

    sx, sz = src_x + pad, src_z + z_start
    rx, rz = rec_x + pad, rec_z + z_start

    # Initial adjoint carry (reverse-time): all zeros
    z_field = jnp.zeros((nz, nx))
    initial_carry = (z_field,) * 15

    # ── Adjoint timestep ──────────────────────────────────────────────────
    # Inputs (carry_a, ys):
    #   carry_a — adjoint of forward carry-out at this timestep, layout
    #             same as forward.time_step's `carry`.
    #   ys      — receiver-data adjoint at this timestep:
    #             (rec_vx_a_t, rec_vz_a_t, rec_ex_a_t, rec_ez_a_t),
    #             each of shape (nrec,)
    # Outputs:
    #   new_carry_a — adjoint of forward carry-in at this timestep
    #   src_wav_a_t — scalar adjoint of src_wavelet[it]
    def adjoint_step(carry_a, ys):
        (U_a, V_a, ex_a, ez_a, es_a,
         p_Dx_L2M_ex_a, p_Dx_L_ez_a, p_Dz_M_es_a,
         p_Dz_L2M_ez_a, p_Dz_L_ex_a, p_Dx_M_es_a,
         p_Dx_U_a, p_Dz_V_a, p_Dz_U_a, p_Dx_V_a) = carry_a

        rec_vx_a_t, rec_vz_a_t, rec_ex_a_t, rec_ez_a_t = ys

        # Source-wavelet adjoint contribution at this step. Filled in by
        # exactly one of the two branches below (force vs moment).
        src_wav_a_t = jnp.array(0.0, dtype=B.dtype)

        # ── 0-T) Inject adjoint of receiver sampling ─────────────────────
        # forward: rec_data = (U[rz, rx], V[rz, rx], ex[rz, rx], ez[rz, rx])
        U_a  = U_a.at[rz, rx].add(rec_vx_a_t)
        V_a  = V_a.at[rz, rx].add(rec_vz_a_t)
        ex_a = ex_a.at[rz, rx].add(rec_ex_a_t)
        ez_a = ez_a.at[rz, rx].add(rec_ez_a_t)

        # ── 5-T) Adjoint of εxz update ────────────────────────────────────
        # forward: es[:nz-1, :nx-1] += 0.5 * (dt/dz Dz_U[:nz-1, :nx-1]
        #                                     + dt/dx Dx_V[:nz-1, :nx-1])
        # es_a passes through unchanged (identity in the +=).
        Dz_U_a = jnp.zeros((nz, nx)).at[:nz-1, :nx-1].set(
            0.5 * dt/dz * es_a[:nz-1, :nx-1])
        Dx_V_a = jnp.zeros((nz, nx)).at[:nz-1, :nx-1].set(
            0.5 * dt/dx * es_a[:nz-1, :nx-1])

        Dz_U_a, p_Dz_U_a = pml_apply_T(Dz_U_a, p_Dz_U_a, kz, bz, az)
        Dx_V_a, p_Dx_V_a = pml_apply_T(Dx_V_a, p_Dx_V_a, kx, bx, ax)

        # forward Dz_U (D⁺_z applied to U)
        # c1: Dz_U[:nz-1, 1:nx] = c1*(U[1:nz, 1:nx] - U[:nz-1, 1:nx])
        U_a = U_a.at[1:nz,  1:nx].add( c1 * Dz_U_a[:nz-1, 1:nx])
        U_a = U_a.at[:nz-1, 1:nx].add(-c1 * Dz_U_a[:nz-1, 1:nx])
        if fd_order >= 4:
            # c2: Dz_U[1:nz-2, 1:nx] += c2*(U[3:nz, 1:nx] - U[:nz-3, 1:nx])
            U_a = U_a.at[3:nz,  1:nx].add( c2 * Dz_U_a[1:nz-2, 1:nx])
            U_a = U_a.at[:nz-3, 1:nx].add(-c2 * Dz_U_a[1:nz-2, 1:nx])
        if fd_order >= 8:
            # c3: Dz_U[2:nz-3, 1:nx] += c3*(U[5:nz, 1:nx] - U[:nz-5, 1:nx])
            U_a = U_a.at[5:nz,  1:nx].add( c3 * Dz_U_a[2:nz-3, 1:nx])
            U_a = U_a.at[:nz-5, 1:nx].add(-c3 * Dz_U_a[2:nz-3, 1:nx])
            # c4: Dz_U[3:nz-4, 1:nx] += c4*(U[7:nz, 1:nx] - U[:nz-7, 1:nx])
            U_a = U_a.at[7:nz,  1:nx].add( c4 * Dz_U_a[3:nz-4, 1:nx])
            U_a = U_a.at[:nz-7, 1:nx].add(-c4 * Dz_U_a[3:nz-4, 1:nx])

        # forward Dx_V (D⁺_x applied to V)
        # c1: Dx_V[:nz-1, :nx-1] = c1*(V[:nz-1, 1:nx] - V[:nz-1, :nx-1])
        V_a = V_a.at[:nz-1, 1:nx ].add( c1 * Dx_V_a[:nz-1, :nx-1])
        V_a = V_a.at[:nz-1, :nx-1].add(-c1 * Dx_V_a[:nz-1, :nx-1])
        if fd_order >= 4:
            # c2: Dx_V[:nz-1, 1:nx-2] += c2*(V[:nz-1, 3:nx] - V[:nz-1, :nx-3])
            V_a = V_a.at[:nz-1, 3:nx ].add( c2 * Dx_V_a[:nz-1, 1:nx-2])
            V_a = V_a.at[:nz-1, :nx-3].add(-c2 * Dx_V_a[:nz-1, 1:nx-2])
        if fd_order >= 8:
            # c3: Dx_V[:nz-1, 2:nx-3] += c3*(V[:nz-1, 5:nx] - V[:nz-1, :nx-5])
            V_a = V_a.at[:nz-1, 5:nx ].add( c3 * Dx_V_a[:nz-1, 2:nx-3])
            V_a = V_a.at[:nz-1, :nx-5].add(-c3 * Dx_V_a[:nz-1, 2:nx-3])
            # c4: Dx_V[:nz-1, 3:nx-4] += c4*(V[:nz-1, 7:nx] - V[:nz-1, :nx-7])
            V_a = V_a.at[:nz-1, 7:nx ].add( c4 * Dx_V_a[:nz-1, 3:nx-4])
            V_a = V_a.at[:nz-1, :nx-7].add(-c4 * Dx_V_a[:nz-1, 3:nx-4])

        # ── 4.5-T) Adjoint of moment-tensor source injection ─────────────
        # forward (between εzz update and εxz update):
        #   lam_mu_s    = L[sz, sx] + M[sz, sx]
        #   src_term_mt = src_wavelet[it] * dt / (2 * lam_mu_s * dx * dz)
        #   ex[sz, sx] += src_term_mt
        #   ez[sz, sx] += src_term_mt
        # ex_a, ez_a are unchanged by the +=; src_wav_a_t collects from both.
        if src_type == 'moment':
            lam_mu_s = L[sz, sx] + M[sz, sx]
            src_wav_a_t = (dt / (2.0 * lam_mu_s * dx * dz)) * (
                ex_a[sz, sx] + ez_a[sz, sx])

        # ── 4-T) Adjoint of εzz update ───────────────────────────────────
        # forward: ez[0:nz, :] += dt/dz * Dz_V[0:nz, :]   (whole grid)
        Dz_V_a = (dt/dz) * ez_a  # shape (nz, nx); this is Dz_V_used_a

        Dz_V_a, p_Dz_V_a = pml_apply_T(Dz_V_a, p_Dz_V_a, kz, bz, az)
        # Dz_V_a is now the pre-PML adjoint (i.e. raw FD/BC adjoint).

        # Free-surface BC at iz=0 in forward:
        #   Dz_V[0, :nx-1] = L_ratio[0, :nx-1] * Dx_U[0, :nx-1] * dz/dx
        # Transpose: contribute into Dx_U_a[0, :nx-1] BEFORE Dx_U's PML T.
        # Allocate Dx_U_a here so the BC contribution can land first.
        Dx_U_a = jnp.zeros((nz, nx))
        if free_surface:
            Dx_U_a = Dx_U_a.at[0, :nx-1].add(
                L_ratio[0, :nx-1] * (dz/dx) * Dz_V_a[0, :nx-1])

        # forward Dz_V (D⁻_z applied to V)
        # c1: Dz_V[1:nz, :nx-1] = c1*(V[1:nz, :nx-1] - V[:nz-1, :nx-1])
        V_a = V_a.at[1:nz,  :nx-1].add( c1 * Dz_V_a[1:nz, :nx-1])
        V_a = V_a.at[:nz-1, :nx-1].add(-c1 * Dz_V_a[1:nz, :nx-1])
        if fd_order >= 4:
            # c2: Dz_V[2:nz-1, :nx-1] += c2*(V[3:nz, :nx-1] - V[:nz-3, :nx-1])
            V_a = V_a.at[3:nz,  :nx-1].add( c2 * Dz_V_a[2:nz-1, :nx-1])
            V_a = V_a.at[:nz-3, :nx-1].add(-c2 * Dz_V_a[2:nz-1, :nx-1])
        if fd_order >= 8:
            # c3: Dz_V[3:nz-2, :nx-1] += c3*(V[5:nz, :nx-1] - V[:nz-5, :nx-1])
            V_a = V_a.at[5:nz,  :nx-1].add( c3 * Dz_V_a[3:nz-2, :nx-1])
            V_a = V_a.at[:nz-5, :nx-1].add(-c3 * Dz_V_a[3:nz-2, :nx-1])
            # c4: Dz_V[4:nz-3, :nx-1] += c4*(V[7:nz, :nx-1] - V[:nz-7, :nx-1])
            V_a = V_a.at[7:nz,  :nx-1].add( c4 * Dz_V_a[4:nz-3, :nx-1])
            V_a = V_a.at[:nz-7, :nx-1].add(-c4 * Dz_V_a[4:nz-3, :nx-1])

        # ── 3-T) Adjoint of εxx update ───────────────────────────────────
        # forward: ex[:, 1:nx] += dt/dx * Dx_U[:, 1:nx]
        # Add ex contribution to Dx_U_a (which already holds the FS-BC term).
        Dx_U_a = Dx_U_a.at[:, 1:nx].add((dt/dx) * ex_a[:, 1:nx])

        Dx_U_a, p_Dx_U_a = pml_apply_T(Dx_U_a, p_Dx_U_a, kx, bx, ax)

        # forward Dx_U (D⁻_x applied to U)
        # c1: Dx_U[:nz, 1:nx] = c1*(U[:nz, 1:nx] - U[:nz, :nx-1])
        U_a = U_a.at[:nz, 1:nx ].add( c1 * Dx_U_a[:nz, 1:nx])
        U_a = U_a.at[:nz, :nx-1].add(-c1 * Dx_U_a[:nz, 1:nx])
        if fd_order >= 4:
            # c2: Dx_U[:nz, 2:nx-1] += c2*(U[:nz, 3:nx] - U[:nz, :nx-3])
            U_a = U_a.at[:nz, 3:nx ].add( c2 * Dx_U_a[:nz, 2:nx-1])
            U_a = U_a.at[:nz, :nx-3].add(-c2 * Dx_U_a[:nz, 2:nx-1])
        if fd_order >= 8:
            # c3: Dx_U[:nz, 3:nx-2] += c3*(U[:nz, 5:nx] - U[:nz, :nx-5])
            U_a = U_a.at[:nz, 5:nx ].add( c3 * Dx_U_a[:nz, 3:nx-2])
            U_a = U_a.at[:nz, :nx-5].add(-c3 * Dx_U_a[:nz, 3:nx-2])
            # c4: Dx_U[:nz, 4:nx-3] += c4*(U[:nz, 7:nx] - U[:nz, :nx-7])
            U_a = U_a.at[:nz, 7:nx ].add( c4 * Dx_U_a[:nz, 4:nx-3])
            U_a = U_a.at[:nz, :nx-7].add(-c4 * Dx_U_a[:nz, 4:nx-3])

        # ── 2-T) Adjoint of source injection (force) ─────────────────────
        # forward: V[sz, sx] += dt/(dx*dz) * B[sz, sx] * src_wavelet[it]
        # V_a is unchanged; src_wavelet_a[it] is extracted.
        if src_type == 'force':
            src_wav_a_t = (dt / (dx * dz)) * B[sz, sx] * V_a[sz, sx]

        # ── 2-T) Adjoint of V update ─────────────────────────────────────
        # forward V[0:nz-1, 1:nx-1] += dt/dx*B*Dx_M_es + dt/dz*B*Dz_L2M_ez
        #                              + dt/dz*B*Dz_L_ex
        Dx_M_es_a   = jnp.zeros((nz, nx)).at[:nz-1, 1:nx-1].set(
            (dt/dx) * B[:nz-1, 1:nx-1] * V_a[:nz-1, 1:nx-1])
        Dz_L2M_ez_a = jnp.zeros((nz, nx)).at[:nz-1, 1:nx-1].set(
            (dt/dz) * B[:nz-1, 1:nx-1] * V_a[:nz-1, 1:nx-1])
        Dz_L_ex_a   = jnp.zeros((nz, nx)).at[:nz-1, 1:nx-1].set(
            (dt/dz) * B[:nz-1, 1:nx-1] * V_a[:nz-1, 1:nx-1])

        Dx_M_es_a,   p_Dx_M_es_a   = pml_apply_T(Dx_M_es_a,   p_Dx_M_es_a,   kx, bx, ax)
        Dz_L2M_ez_a, p_Dz_L2M_ez_a = pml_apply_T(Dz_L2M_ez_a, p_Dz_L2M_ez_a, kz, bz, az)
        Dz_L_ex_a,   p_Dz_L_ex_a   = pml_apply_T(Dz_L_ex_a,   p_Dz_L_ex_a,   kz, bz, az)

        # forward Dx_M_es (D⁻_x applied to M_es)
        M_es_a = jnp.zeros((nz, nx))
        # c1: Dx_M_es[:nz-1, 1:nx-1] = c1*(M_es[:nz-1, 1:nx-1] - M_es[:nz-1, :nx-2])
        M_es_a = M_es_a.at[:nz-1, 1:nx-1].add( c1 * Dx_M_es_a[:nz-1, 1:nx-1])
        M_es_a = M_es_a.at[:nz-1, :nx-2 ].add(-c1 * Dx_M_es_a[:nz-1, 1:nx-1])
        if fd_order >= 4:
            # c2: Dx_M_es[:nz-1, 2:nx-1] += c2*(M_es[:nz-1, 3:nx] - M_es[:nz-1, :nx-3])
            M_es_a = M_es_a.at[:nz-1, 3:nx ].add( c2 * Dx_M_es_a[:nz-1, 2:nx-1])
            M_es_a = M_es_a.at[:nz-1, :nx-3].add(-c2 * Dx_M_es_a[:nz-1, 2:nx-1])
        if fd_order >= 8:
            # c3: Dx_M_es[:nz-1, 3:nx-2] += c3*(M_es[:nz-1, 5:nx] - M_es[:nz-1, :nx-5])
            M_es_a = M_es_a.at[:nz-1, 5:nx ].add( c3 * Dx_M_es_a[:nz-1, 3:nx-2])
            M_es_a = M_es_a.at[:nz-1, :nx-5].add(-c3 * Dx_M_es_a[:nz-1, 3:nx-2])
            # c4: Dx_M_es[:nz-1, 4:nx-3] += c4*(M_es[:nz-1, 7:nx] - M_es[:nz-1, :nx-7])
            M_es_a = M_es_a.at[:nz-1, 7:nx ].add( c4 * Dx_M_es_a[:nz-1, 4:nx-3])
            M_es_a = M_es_a.at[:nz-1, :nx-7].add(-c4 * Dx_M_es_a[:nz-1, 4:nx-3])

        # forward Dz_L2M_ez (D⁺_z applied to L2M_ez)
        L2M_ez_a = jnp.zeros((nz, nx))
        # c1: Dz_L2M_ez[:nz-1, 1:nx-1] = c1*(L2M_ez[1:nz, 1:nx-1] - L2M_ez[:nz-1, 1:nx-1])
        L2M_ez_a = L2M_ez_a.at[1:nz,  1:nx-1].add( c1 * Dz_L2M_ez_a[:nz-1, 1:nx-1])
        L2M_ez_a = L2M_ez_a.at[:nz-1, 1:nx-1].add(-c1 * Dz_L2M_ez_a[:nz-1, 1:nx-1])
        if fd_order >= 4:
            # c2: Dz_L2M_ez[1:nz-2, 1:nx-1] += c2*(L2M_ez[3:nz, 1:nx-1] - L2M_ez[:nz-3, 1:nx-1])
            L2M_ez_a = L2M_ez_a.at[3:nz,  1:nx-1].add( c2 * Dz_L2M_ez_a[1:nz-2, 1:nx-1])
            L2M_ez_a = L2M_ez_a.at[:nz-3, 1:nx-1].add(-c2 * Dz_L2M_ez_a[1:nz-2, 1:nx-1])
        if fd_order >= 8:
            # c3: Dz_L2M_ez[2:nz-3, 1:nx-1] += c3*(L2M_ez[5:nz, 1:nx-1] - L2M_ez[:nz-5, 1:nx-1])
            L2M_ez_a = L2M_ez_a.at[5:nz,  1:nx-1].add( c3 * Dz_L2M_ez_a[2:nz-3, 1:nx-1])
            L2M_ez_a = L2M_ez_a.at[:nz-5, 1:nx-1].add(-c3 * Dz_L2M_ez_a[2:nz-3, 1:nx-1])
            # c4: Dz_L2M_ez[3:nz-4, 1:nx-1] += c4*(L2M_ez[7:nz, 1:nx-1] - L2M_ez[:nz-7, 1:nx-1])
            L2M_ez_a = L2M_ez_a.at[7:nz,  1:nx-1].add( c4 * Dz_L2M_ez_a[3:nz-4, 1:nx-1])
            L2M_ez_a = L2M_ez_a.at[:nz-7, 1:nx-1].add(-c4 * Dz_L2M_ez_a[3:nz-4, 1:nx-1])

        # forward Dz_L_ex (D⁺_z applied to L_ex; same slice pattern as Dz_L2M_ez)
        L_ex_a = jnp.zeros((nz, nx))
        # c1
        L_ex_a = L_ex_a.at[1:nz,  1:nx-1].add( c1 * Dz_L_ex_a[:nz-1, 1:nx-1])
        L_ex_a = L_ex_a.at[:nz-1, 1:nx-1].add(-c1 * Dz_L_ex_a[:nz-1, 1:nx-1])
        if fd_order >= 4:
            L_ex_a = L_ex_a.at[3:nz,  1:nx-1].add( c2 * Dz_L_ex_a[1:nz-2, 1:nx-1])
            L_ex_a = L_ex_a.at[:nz-3, 1:nx-1].add(-c2 * Dz_L_ex_a[1:nz-2, 1:nx-1])
        if fd_order >= 8:
            L_ex_a = L_ex_a.at[5:nz,  1:nx-1].add( c3 * Dz_L_ex_a[2:nz-3, 1:nx-1])
            L_ex_a = L_ex_a.at[:nz-5, 1:nx-1].add(-c3 * Dz_L_ex_a[2:nz-3, 1:nx-1])
            L_ex_a = L_ex_a.at[7:nz,  1:nx-1].add( c4 * Dz_L_ex_a[3:nz-4, 1:nx-1])
            L_ex_a = L_ex_a.at[:nz-7, 1:nx-1].add(-c4 * Dz_L_ex_a[3:nz-4, 1:nx-1])

        # ── 1-T) Adjoint of U update ─────────────────────────────────────
        # forward U[0:nz-1, 1:nx-1] += dt/dx*B*Dx_L2M_ex + dt/dx*B*Dx_L_ez
        #                              + dt/dz*B*Dz_M_es
        Dx_L2M_ex_a = jnp.zeros((nz, nx)).at[:nz-1, 1:nx-1].set(
            (dt/dx) * B[:nz-1, 1:nx-1] * U_a[:nz-1, 1:nx-1])
        Dx_L_ez_a   = jnp.zeros((nz, nx)).at[:nz-1, 1:nx-1].set(
            (dt/dx) * B[:nz-1, 1:nx-1] * U_a[:nz-1, 1:nx-1])
        Dz_M_es_a   = jnp.zeros((nz, nx)).at[:nz-1, 1:nx-1].set(
            (dt/dz) * B[:nz-1, 1:nx-1] * U_a[:nz-1, 1:nx-1])

        Dx_L2M_ex_a, p_Dx_L2M_ex_a = pml_apply_T(Dx_L2M_ex_a, p_Dx_L2M_ex_a, kx, bx, ax)
        Dx_L_ez_a,   p_Dx_L_ez_a   = pml_apply_T(Dx_L_ez_a,   p_Dx_L_ez_a,   kx, bx, ax)
        Dz_M_es_a,   p_Dz_M_es_a   = pml_apply_T(Dz_M_es_a,   p_Dz_M_es_a,   kz, bz, az)

        # forward Dx_L2M_ex (D⁺_x applied to L2M_ex)
        L2M_ex_a = jnp.zeros((nz, nx))
        # c1
        L2M_ex_a = L2M_ex_a.at[:nz-1, 1:nx ].add( c1 * Dx_L2M_ex_a[:nz-1, :nx-1])
        L2M_ex_a = L2M_ex_a.at[:nz-1, :nx-1].add(-c1 * Dx_L2M_ex_a[:nz-1, :nx-1])
        if fd_order >= 4:
            # c2: target [:nz-1, 1:nx-2], reads (3:nx) - (:nx-3)
            L2M_ex_a = L2M_ex_a.at[:nz-1, 3:nx ].add( c2 * Dx_L2M_ex_a[:nz-1, 1:nx-2])
            L2M_ex_a = L2M_ex_a.at[:nz-1, :nx-3].add(-c2 * Dx_L2M_ex_a[:nz-1, 1:nx-2])
        if fd_order >= 8:
            # c3: target [:nz-1, 2:nx-3], reads (5:nx) - (:nx-5)
            L2M_ex_a = L2M_ex_a.at[:nz-1, 5:nx ].add( c3 * Dx_L2M_ex_a[:nz-1, 2:nx-3])
            L2M_ex_a = L2M_ex_a.at[:nz-1, :nx-5].add(-c3 * Dx_L2M_ex_a[:nz-1, 2:nx-3])
            # c4: target [:nz-1, 3:nx-4], reads (7:nx) - (:nx-7)
            L2M_ex_a = L2M_ex_a.at[:nz-1, 7:nx ].add( c4 * Dx_L2M_ex_a[:nz-1, 3:nx-4])
            L2M_ex_a = L2M_ex_a.at[:nz-1, :nx-7].add(-c4 * Dx_L2M_ex_a[:nz-1, 3:nx-4])

        # forward Dx_L_ez (D⁺_x applied to L_ez; same slice pattern as Dx_L2M_ex)
        L_ez_a = jnp.zeros((nz, nx))
        # c1
        L_ez_a = L_ez_a.at[:nz-1, 1:nx ].add( c1 * Dx_L_ez_a[:nz-1, :nx-1])
        L_ez_a = L_ez_a.at[:nz-1, :nx-1].add(-c1 * Dx_L_ez_a[:nz-1, :nx-1])
        if fd_order >= 4:
            L_ez_a = L_ez_a.at[:nz-1, 3:nx ].add( c2 * Dx_L_ez_a[:nz-1, 1:nx-2])
            L_ez_a = L_ez_a.at[:nz-1, :nx-3].add(-c2 * Dx_L_ez_a[:nz-1, 1:nx-2])
        if fd_order >= 8:
            L_ez_a = L_ez_a.at[:nz-1, 5:nx ].add( c3 * Dx_L_ez_a[:nz-1, 2:nx-3])
            L_ez_a = L_ez_a.at[:nz-1, :nx-5].add(-c3 * Dx_L_ez_a[:nz-1, 2:nx-3])
            L_ez_a = L_ez_a.at[:nz-1, 7:nx ].add( c4 * Dx_L_ez_a[:nz-1, 3:nx-4])
            L_ez_a = L_ez_a.at[:nz-1, :nx-7].add(-c4 * Dx_L_ez_a[:nz-1, 3:nx-4])

        # forward Dz_M_es (D⁻_z applied to M_es)
        # ACCUMULATE into the SAME M_es_a from the V-update (M_es is one variable
        # used in both U and V updates in the forward).
        # c1: Dz_M_es[1:nz-1, 1:nx-1] = c1*(M_es[1:nz-1, 1:nx-1] - M_es[:nz-2, 1:nx-1])
        M_es_a = M_es_a.at[1:nz-1, 1:nx-1].add( c1 * Dz_M_es_a[1:nz-1, 1:nx-1])
        M_es_a = M_es_a.at[:nz-2,  1:nx-1].add(-c1 * Dz_M_es_a[1:nz-1, 1:nx-1])
        if fd_order >= 4:
            # c2: Dz_M_es[2:nz-1, 1:nx-1] += c2*(M_es[3:nz, 1:nx-1] - M_es[:nz-3, 1:nx-1])
            M_es_a = M_es_a.at[3:nz,  1:nx-1].add( c2 * Dz_M_es_a[2:nz-1, 1:nx-1])
            M_es_a = M_es_a.at[:nz-3, 1:nx-1].add(-c2 * Dz_M_es_a[2:nz-1, 1:nx-1])
        if fd_order >= 8:
            # c3: Dz_M_es[3:nz-2, 1:nx-1] += c3*(M_es[5:nz, 1:nx-1] - M_es[:nz-5, 1:nx-1])
            M_es_a = M_es_a.at[5:nz,  1:nx-1].add( c3 * Dz_M_es_a[3:nz-2, 1:nx-1])
            M_es_a = M_es_a.at[:nz-5, 1:nx-1].add(-c3 * Dz_M_es_a[3:nz-2, 1:nx-1])
            # c4: Dz_M_es[4:nz-3, 1:nx-1] += c4*(M_es[7:nz, 1:nx-1] - M_es[:nz-7, 1:nx-1])
            M_es_a = M_es_a.at[7:nz,  1:nx-1].add( c4 * Dz_M_es_a[4:nz-3, 1:nx-1])
            M_es_a = M_es_a.at[:nz-7, 1:nx-1].add(-c4 * Dz_M_es_a[4:nz-3, 1:nx-1])

        if free_surface:
            # forward (σxz antisymmetric image at iz=0, M_es ghost = -M_es[0]):
            #   Dz_M_es[0, 1:nx-1] = 2 * c1 * M_es[0, 1:nx-1]
            # (Higher-order terms at iz=0..3 are dropped in the forward — their
            # default-slice patterns already exclude rows that would need ghosts.)
            M_es_a = M_es_a.at[0, 1:nx-1].add(2.0 * c1 * Dz_M_es_a[0, 1:nx-1])

        # ── Adjoint of constitutive products (carry-in ε's) ──────────────
        # forward (start of each step):
        #   L2M_ex = L2M * ex; L_ex = L * ex
        #   L2M_ez = L2M * ez; L_ez = L * ez
        #   M_es   = 2 * M * es
        ex_a = ex_a + L2M * L2M_ex_a + L   * L_ex_a
        ez_a = ez_a + L   * L_ez_a   + L2M * L2M_ez_a
        es_a = es_a + 2.0 * M * M_es_a

        new_carry_a = (U_a, V_a, ex_a, ez_a, es_a,
                       p_Dx_L2M_ex_a, p_Dx_L_ez_a, p_Dz_M_es_a,
                       p_Dz_L2M_ez_a, p_Dz_L_ex_a, p_Dx_M_es_a,
                       p_Dx_U_a, p_Dz_V_a, p_Dz_U_a, p_Dx_V_a)
        if return_wavefields == "velocity":
            # Memory-light path: only the adjoint velocities (used by the
            # imaging condition); skips the 3 adjoint strain cubes entirely.
            return new_carry_a, (src_wav_a_t, U_a, V_a)
        if return_wavefields:
            return new_carry_a, (src_wav_a_t, U_a, V_a, ex_a, ez_a, es_a)
        return new_carry_a, src_wav_a_t

    ys = (rec_vx_a, rec_vz_a, rec_ex_a, rec_ez_a)
    _, scan_out = lax.scan(adjoint_step, initial_carry, ys, reverse=True)
    if return_wavefields:
        return scan_out  # 'velocity' → (src_wav_a, U_a, V_a)
                         # True       → (src_wav_a, U_a, V_a, ex_a, ez_a, es_a)
    return scan_out      # False      → src_wavelet_a (nt,)
