"""
fwi/forward.py — 2D P-SV Elastic Wave Forward Modeling (JAX)

Velocity-STRAIN Finite Difference with C-PML (2nd/4th/8th Order)
Combines: wave propagation, Ricker wavelet, and PML absorbing boundaries.

Public API:
    ricker_jax(t, fc, t0)          — generate source wavelet
    build_forward_fn(...)          — returns JIT-compiled forward function
    forward_jax(...)               — raw forward modeling (used internally)

Authors: Minh Nhat Tran
Date: February 2026
"""
import jax
import jax.numpy as jnp
from jax import lax
from functools import partial


# ─────────────────────────────────────────────────────────────────────────────
# Ricker wavelet
# ─────────────────────────────────────────────────────────────────────────────

def ricker_jax(t, fc, t0, scaling_factor=1):
    """Generate Ricker wavelet using JAX."""
    arg = (jnp.pi * fc * (t - t0)) ** 2
    return scaling_factor * (1.0 - 2.0 * arg) * jnp.exp(-arg)


# ─────────────────────────────────────────────────────────────────────────────
# C-PML (Convolutional Perfectly Matched Layer)
# ─────────────────────────────────────────────────────────────────────────────

def _pml_coefficients(pad, dx, dt, Vp_max, fc, nz_t, nx_t, nz_dom,
                      free_surface=True):
    """
    Setup C-PML coefficients using JAX.
    Returns JAX arrays for use in JIT-compiled functions.

    When free_surface=True:  PML on left, right, bottom only (3 sides).
    When free_surface=False: PML on all 4 sides including top.
    """
    N = 2              # Polynomial order
    Rc = 1e-3          # Reflection coefficient
    L_pml = pad * dx   # PML thickness

    d0 = -(N + 1) * Vp_max * jnp.log(Rc) / (2 * L_pml)

    # Create profile arrays
    i_arr = jnp.arange(pad)
    dist = (i_arr + 1) * dx / L_pml
    d_profile = d0 * (dist ** N)
    k_profile = 1.0 + (2.0 - 1.0) * (dist ** N)
    alpha_profile = jnp.pi * fc * (1.0 - dist)

    # Initialize coefficient arrays
    dx_arr = jnp.zeros((nz_t, nx_t))
    dz_arr = jnp.zeros((nz_t, nx_t))
    kx = jnp.ones((nz_t, nx_t))
    kz = jnp.ones((nz_t, nx_t))
    alpha_x = jnp.zeros((nz_t, nx_t))
    alpha_z = jnp.zeros((nz_t, nx_t))

    # Left PML
    for i in range(pad):
        col = pad - 1 - i
        dx_arr = dx_arr.at[:, col].set(d_profile[i])
        kx = kx.at[:, col].set(k_profile[i])
        alpha_x = alpha_x.at[:, col].set(alpha_profile[i])

    # Right PML
    for i in range(pad):
        col = nx_t - pad + i
        dx_arr = dx_arr.at[:, col].set(d_profile[i])
        kx = kx.at[:, col].set(k_profile[i])
        alpha_x = alpha_x.at[:, col].set(alpha_profile[i])

    # Bottom PML
    bottom_start = nz_dom if free_surface else pad + nz_dom
    for i in range(pad):
        row = bottom_start + i
        dz_arr = dz_arr.at[row, :].set(d_profile[i])
        kz = kz.at[row, :].set(k_profile[i])
        alpha_z = alpha_z.at[row, :].set(alpha_profile[i])

    # Top PML (only when no free surface)
    if not free_surface:
        for i in range(pad):
            row = pad - 1 - i
            dz_arr = dz_arr.at[row, :].set(d_profile[i])
            kz = kz.at[row, :].set(k_profile[i])
            alpha_z = alpha_z.at[row, :].set(alpha_profile[i])

    bx = jnp.exp(-(dx_arr / kx + alpha_x) * dt)
    bz = jnp.exp(-(dz_arr / kz + alpha_z) * dt)

    denom_x = kx * (dx_arr + kx * alpha_x)
    denom_z = kz * (dz_arr + kz * alpha_z)

    ax = jnp.where(denom_x > 1e-10, dx_arr * (bx - 1.0) / denom_x, 0.0)
    az = jnp.where(denom_z > 1e-10, dz_arr * (bz - 1.0) / denom_z, 0.0)

    return kx, kz, bx, bz, ax, az


@jax.jit
def _pml_apply(D, k, psi, b, a):
    """Apply C-PML to derivative (JIT-compiled)"""
    psi_new = b * psi + a * D
    D_new = D / k + psi_new
    return D_new, psi_new


# ─────────────────────────────────────────────────────────────────────────────
# Forward modeling (core)
# ─────────────────────────────────────────────────────────────────────────────

def forward_jax(Vs, Vp, rho, src_x, src_z, rec_x, rec_z,
                nx_dom, nz_dom, dx, dz, dt, nt, fc, pad,
                src_wavelet,
                return_wavefields=False, block_size=100, return_vars=None,
                fd_order=2, free_surface=True, src_type='force'):
    """
    2D P-SV Elastic Wave Forward Modeling - JAX Version (Velocity-Strain Formulation)
    2nd/4th/8th Order Spatial Accuracy with Block Checkpointing

    Parameters:
    -----------
    Vs, Vp, rho : numpy arrays
        Velocity and density models (nz_dom, nx_dom)
    src_x, src_z : int
        Source position indices
    rec_x, rec_z : numpy arrays
        Receiver position indices
    nx_dom, nz_dom : int
        Domain size
    dx, dz, dt : float
        Grid spacing and time step
    nt : int
        Number of time steps
    fc : float
        Source frequency (used for PML tuning)
    pad : int
        PML padding size
    src_wavelet : jax array
        Source wavelet (nt,) — must be created externally (e.g. via ricker_jax)
    return_wavefields : bool
        Whether to return the full wavefields or just receiver data.
    block_size : int
        Number of time steps per checkpoint block.
    return_vars : list of str, optional
        Specific variables to return. If provided, overrides return_wavefields.
        Options: 'vx', 'vz', 'ex', 'ez' (receiver data);
                 'vx_full', 'vz_full', 'ex_full', 'ez_full', 'es_full' (full wavefields).
    fd_order : int
        Finite difference spatial order (2, 4, or 8). Default: 2.
    free_surface : bool
        If True, apply free surface BC at top (z=0). PML on 3 sides.
        If False, apply PML on all 4 sides including top. Default: True.

    Returns:
    --------
    results : tuple
        Requested variables as JAX arrays. Default: (rec_vx, rec_vz, rec_exx, rec_ezz)
    """
    # Convert inputs to JAX arrays
    Vs = jnp.array(Vs)
    Vp = jnp.array(Vp)
    rho = jnp.array(rho)
    rec_x = jnp.array(rec_x)
    rec_z = jnp.array(rec_z)

    # FD stencil coefficients (staggered grid)
    #   D⁺f[i] = c1*(f[i+1]-f[i]) + c2*(f[i+2]-f[i-1]) + c3*(f[i+3]-f[i-2]) + c4*(f[i+4]-f[i-3])
    #   D⁻f[i] = c1*(f[i]-f[i-1]) + c2*(f[i+1]-f[i-2]) + c3*(f[i+2]-f[i-3]) + c4*(f[i+3]-f[i-4])
    if fd_order == 8:
        c1, c2, c3, c4 = 1225.0/1024.0, -245.0/3072.0, 49.0/5120.0, -5.0/7168.0
    elif fd_order >= 4:
        c1, c2, c3, c4 = 9.0/8.0, -1.0/24.0, 0.0, 0.0
    else:
        c1, c2, c3, c4 = 1.0, 0.0, 0.0, 0.0

    # Grid size
    if free_surface:
        nx_t, nz_t = nx_dom + 2 * pad, nz_dom + pad
        z_start = 0       # domain starts at row 0
    else:
        nx_t, nz_t = nx_dom + 2 * pad, nz_dom + 2 * pad
        z_start = pad      # domain starts at row pad
    nz, nx = nz_t, nx_t

    # Setup material parameters — place domain into full grid, extend into PML.
    # Step 1: embed into physical domain
    Vs_full  = jnp.zeros((nz, nx)).at[z_start:z_start+nz_dom, pad:pad+nx_dom].set(Vs)
    Vp_full  = jnp.zeros((nz, nx)).at[z_start:z_start+nz_dom, pad:pad+nx_dom].set(Vp)
    rho_full = jnp.zeros((nz, nx)).at[z_start:z_start+nz_dom, pad:pad+nx_dom].set(rho)

    # Step 2: extend edge values into PML (nearest-neighbour)
    Vs_full  = Vs_full.at[z_start:z_start+nz_dom, :pad].set(Vs_full[z_start:z_start+nz_dom, pad:pad+1])
    Vp_full  = Vp_full.at[z_start:z_start+nz_dom, :pad].set(Vp_full[z_start:z_start+nz_dom, pad:pad+1])
    rho_full = rho_full.at[z_start:z_start+nz_dom, :pad].set(rho_full[z_start:z_start+nz_dom, pad:pad+1])
    Vs_full  = Vs_full.at[z_start:z_start+nz_dom, nx_t-pad:].set(Vs_full[z_start:z_start+nz_dom, pad+nx_dom-1:pad+nx_dom])
    Vp_full  = Vp_full.at[z_start:z_start+nz_dom, nx_t-pad:].set(Vp_full[z_start:z_start+nz_dom, pad+nx_dom-1:pad+nx_dom])
    rho_full = rho_full.at[z_start:z_start+nz_dom, nx_t-pad:].set(rho_full[z_start:z_start+nz_dom, pad+nx_dom-1:pad+nx_dom])
    Vs_full  = Vs_full.at[z_start+nz_dom:, :].set(Vs_full[z_start+nz_dom-1:z_start+nz_dom, :])
    Vp_full  = Vp_full.at[z_start+nz_dom:, :].set(Vp_full[z_start+nz_dom-1:z_start+nz_dom, :])
    rho_full = rho_full.at[z_start+nz_dom:, :].set(rho_full[z_start+nz_dom-1:z_start+nz_dom, :])
    if not free_surface:
        Vs_full  = Vs_full.at[:pad, :].set(Vs_full[pad:pad+1, :])
        Vp_full  = Vp_full.at[:pad, :].set(Vp_full[pad:pad+1, :])
        rho_full = rho_full.at[:pad, :].set(rho_full[pad:pad+1, :])

    # Step 3: stop gradient on PML — freeze the full grid then re-inject gradient
    # only for the physical domain. Prevents edge columns/rows from accumulating
    # ~(pad+1)x gradient from all PML cells that copied from them.
    Vs_full  = lax.stop_gradient(Vs_full).at[z_start:z_start+nz_dom, pad:pad+nx_dom].set(Vs)
    Vp_full  = lax.stop_gradient(Vp_full).at[z_start:z_start+nz_dom, pad:pad+nx_dom].set(Vp)
    rho_full = lax.stop_gradient(rho_full).at[z_start:z_start+nz_dom, pad:pad+nx_dom].set(rho)

    # Material parameters
    M = rho_full * Vs_full**2        # μ
    L2M = rho_full * Vp_full**2      # λ + 2μ
    L = L2M - 2 * M                  # λ

    # Buoyancy (inverse density)
    B = 1.0 / rho_full

    # Ratio for free surface: -λ/(λ+2μ)
    if free_surface:
        L_ratio = -L / L2M

    # PML coefficients
    Vp_max = jnp.max(Vp)
    # Stop gradient on Vp_max to prevent PML parameters from updating the model
    Vp_max = lax.stop_gradient(Vp_max)
    kx, kz, bx, bz, ax, az = _pml_coefficients(
        pad, dx, dt, Vp_max, fc, nz, nx, nz_dom, free_surface=free_surface)

    # Source/receiver positions (shift z by pad when top has PML)
    sx, sz = src_x + pad, src_z + z_start
    rx, rz = rec_x + pad, rec_z + z_start

    # Initialize state
    initial_carry = (jnp.zeros((nz, nx)), jnp.zeros((nz, nx)), jnp.zeros((nz, nx)), jnp.zeros((nz, nx)), jnp.zeros((nz, nx)),
                     jnp.zeros((nz, nx)), jnp.zeros((nz, nx)), jnp.zeros((nz, nx)), jnp.zeros((nz, nx)), jnp.zeros((nz, nx)),
                     jnp.zeros((nz, nx)), jnp.zeros((nz, nx)), jnp.zeros((nz, nx)), jnp.zeros((nz, nx)), jnp.zeros((nz, nx)))

    # Define time_step
    def time_step(carry, it):
        U, V, ex, ez, es, p_Dx_L2M_ex, p_Dx_L_ez, p_Dz_M_es, p_Dz_L2M_ez, p_Dz_L_ex, p_Dx_M_es, p_Dx_U, p_Dz_V, p_Dz_U, p_Dx_V = carry

        # ── 1) Update U (vx) ──────────────────────────────────────────────
        L2M_ex, L_ez, M_es = L2M * ex, L * ez, 2 * M * es

        # Dx_L2M_ex (D⁺_x): c_n*(f[ix+1]-f[ix]), result at ix+½
        Dx_L2M_ex = jnp.zeros((nz, nx))
        Dx_L2M_ex = Dx_L2M_ex.at[:nz-1, :nx-1].set(
            c1 * (L2M_ex[:nz-1, 1:nx] - L2M_ex[:nz-1, :nx-1]))
        if fd_order >= 4:
            Dx_L2M_ex = Dx_L2M_ex.at[:nz-1, 1:nx-2].add(
                c2 * (L2M_ex[:nz-1, 3:nx] - L2M_ex[:nz-1, :nx-3]))
        if fd_order >= 8:
            Dx_L2M_ex = Dx_L2M_ex.at[:nz-1, 2:nx-3].add(
                c3 * (L2M_ex[:nz-1, 5:nx] - L2M_ex[:nz-1, :nx-5]))
            Dx_L2M_ex = Dx_L2M_ex.at[:nz-1, 3:nx-4].add(
                c4 * (L2M_ex[:nz-1, 7:nx] - L2M_ex[:nz-1, :nx-7]))
        Dx_L2M_ex, p_Dx_L2M_ex = _pml_apply(Dx_L2M_ex, kx, p_Dx_L2M_ex, bx, ax)

        # Dx_L_ez (D⁺_x): c_n*(f[ix+1]-f[ix]), result at ix+½
        Dx_L_ez = jnp.zeros((nz, nx))
        Dx_L_ez = Dx_L_ez.at[:nz-1, :nx-1].set(
            c1 * (L_ez[:nz-1, 1:nx] - L_ez[:nz-1, :nx-1]))
        if fd_order >= 4:
            Dx_L_ez = Dx_L_ez.at[:nz-1, 1:nx-2].add(
                c2 * (L_ez[:nz-1, 3:nx] - L_ez[:nz-1, :nx-3]))
        if fd_order >= 8:
            Dx_L_ez = Dx_L_ez.at[:nz-1, 2:nx-3].add(
                c3 * (L_ez[:nz-1, 5:nx] - L_ez[:nz-1, :nx-5]))
            Dx_L_ez = Dx_L_ez.at[:nz-1, 3:nx-4].add(
                c4 * (L_ez[:nz-1, 7:nx] - L_ez[:nz-1, :nx-7]))
        Dx_L_ez, p_Dx_L_ez = _pml_apply(Dx_L_ez, kx, p_Dx_L_ez, bx, ax)

        # Dz_M_es (D⁻_z): c_n*(f[iz]-f[iz-1]), result at iz-½
        Dz_M_es = jnp.zeros((nz, nx))
        Dz_M_es = Dz_M_es.at[1:nz-1, 1:nx-1].set(
            c1 * (M_es[1:nz-1, 1:nx-1] - M_es[:nz-2, 1:nx-1]))
        if fd_order >= 4:
            Dz_M_es = Dz_M_es.at[2:nz-1, 1:nx-1].add(
                c2 * (M_es[3:nz, 1:nx-1] - M_es[:nz-3, 1:nx-1]))
        if fd_order >= 8:
            Dz_M_es = Dz_M_es.at[3:nz-2, 1:nx-1].add(
                c3 * (M_es[5:nz, 1:nx-1] - M_es[:nz-5, 1:nx-1]))
            Dz_M_es = Dz_M_es.at[4:nz-3, 1:nx-1].add(
                c4 * (M_es[7:nz, 1:nx-1] - M_es[:nz-7, 1:nx-1]))
        if free_surface:
            # Hybrid order-2 at surface: only c1 image at iz=0 (σxz antisymm, M_es[-1]=-M_es[0]).
            # High-order terms at iz=0..3 are dropped — their stencil would reach ghost rows.
            Dz_M_es = Dz_M_es.at[0, 1:nx-1].set(2*c1 * M_es[0, 1:nx-1])
        Dz_M_es, p_Dz_M_es = _pml_apply(Dz_M_es, kz, p_Dz_M_es, bz, az)

        # Update U (vx)
        U = U.at[0:nz-1, 1:nx-1].set(
            U[0:nz-1, 1:nx-1]
            + dt/dx * B[0:nz-1, 1:nx-1] * Dx_L2M_ex[0:nz-1, 1:nx-1]
            + dt/dx * B[0:nz-1, 1:nx-1] * Dx_L_ez[0:nz-1, 1:nx-1]
            + dt/dz * B[0:nz-1, 1:nx-1] * Dz_M_es[0:nz-1, 1:nx-1]
        )

        # ── 2) Update V (vz) ──────────────────────────────────────────────
        L2M_ez, L_ex = L2M * ez, L * ex

        # Dx_M_es (D⁻_x): c_n*(f[ix]-f[ix-1]), result at ix-½
        Dx_M_es = jnp.zeros((nz, nx))
        Dx_M_es = Dx_M_es.at[:nz-1, 1:nx-1].set(
            c1 * (M_es[:nz-1, 1:nx-1] - M_es[:nz-1, :nx-2]))
        if fd_order >= 4:
            Dx_M_es = Dx_M_es.at[:nz-1, 2:nx-1].add(
                c2 * (M_es[:nz-1, 3:nx] - M_es[:nz-1, :nx-3]))
        if fd_order >= 8:
            Dx_M_es = Dx_M_es.at[:nz-1, 3:nx-2].add(
                c3 * (M_es[:nz-1, 5:nx] - M_es[:nz-1, :nx-5]))
            Dx_M_es = Dx_M_es.at[:nz-1, 4:nx-3].add(
                c4 * (M_es[:nz-1, 7:nx] - M_es[:nz-1, :nx-7]))
        Dx_M_es, p_Dx_M_es = _pml_apply(Dx_M_es, kx, p_Dx_M_es, bx, ax)

        # Dz_L2M_ez (D⁺_z): c_n*(f[iz+1]-f[iz]), result at iz+½
        Dz_L2M_ez = jnp.zeros((nz, nx))
        Dz_L2M_ez = Dz_L2M_ez.at[:nz-1, 1:nx-1].set(
            c1 * (L2M_ez[1:nz, 1:nx-1] - L2M_ez[:nz-1, 1:nx-1]))
        if fd_order >= 4:
            Dz_L2M_ez = Dz_L2M_ez.at[1:nz-2, 1:nx-1].add(
                c2 * (L2M_ez[3:nz, 1:nx-1] - L2M_ez[:nz-3, 1:nx-1]))
        if fd_order >= 8:
            Dz_L2M_ez = Dz_L2M_ez.at[2:nz-3, 1:nx-1].add(
                c3 * (L2M_ez[5:nz, 1:nx-1] - L2M_ez[:nz-5, 1:nx-1]))
            Dz_L2M_ez = Dz_L2M_ez.at[3:nz-4, 1:nx-1].add(
                c4 * (L2M_ez[7:nz, 1:nx-1] - L2M_ez[:nz-7, 1:nx-1]))
        # Hybrid: drop high-order imaging for σzz at surface rows.
        # Default c1 at iz=0 (f[1]-f[0]) is all physical; c2/c3/c4 default slices
        # already exclude surface rows needing ghosts, so those rows get order-2 only.
        Dz_L2M_ez, p_Dz_L2M_ez = _pml_apply(Dz_L2M_ez, kz, p_Dz_L2M_ez, bz, az)

        # Dz_L_ex (D⁺_z): c_n*(f[iz+1]-f[iz]), result at iz+½
        Dz_L_ex = jnp.zeros((nz, nx))
        Dz_L_ex = Dz_L_ex.at[:nz-1, 1:nx-1].set(
            c1 * (L_ex[1:nz, 1:nx-1] - L_ex[:nz-1, 1:nx-1]))
        if fd_order >= 4:
            Dz_L_ex = Dz_L_ex.at[1:nz-2, 1:nx-1].add(
                c2 * (L_ex[3:nz, 1:nx-1] - L_ex[:nz-3, 1:nx-1]))
        if fd_order >= 8:
            Dz_L_ex = Dz_L_ex.at[2:nz-3, 1:nx-1].add(
                c3 * (L_ex[5:nz, 1:nx-1] - L_ex[:nz-5, 1:nx-1]))
            Dz_L_ex = Dz_L_ex.at[3:nz-4, 1:nx-1].add(
                c4 * (L_ex[7:nz, 1:nx-1] - L_ex[:nz-7, 1:nx-1]))
        # Hybrid: drop high-order imaging for εxx at surface rows.
        Dz_L_ex, p_Dz_L_ex = _pml_apply(Dz_L_ex, kz, p_Dz_L_ex, bz, az)

        V = V.at[0:nz-1, 1:nx-1].set(
            V[0:nz-1, 1:nx-1]
            + dt/dx * B[0:nz-1, 1:nx-1] * Dx_M_es[0:nz-1, 1:nx-1]
            + dt/dz * B[0:nz-1, 1:nx-1] * Dz_L2M_ez[0:nz-1, 1:nx-1]
            + dt/dz * B[0:nz-1, 1:nx-1] * Dz_L_ex[0:nz-1, 1:nx-1]
        )

        if src_type == 'force':
            V = V.at[sz, sx].add(dt/(dx * dz) * B[sz, sx] * src_wavelet[it])

        # ── 3) Update εxx ─────────────────────────────────────────────────
        # Dx_U (D⁻_x): c_n*(f[ix]-f[ix-1]), result at ix-½ = (ix, iz)
        Dx_U = jnp.zeros((nz, nx))
        Dx_U = Dx_U.at[:nz, 1:nx].set(
            c1 * (U[:nz, 1:nx] - U[:nz, :nx-1]))
        if fd_order >= 4:
            Dx_U = Dx_U.at[:nz, 2:nx-1].add(
                c2 * (U[:nz, 3:nx] - U[:nz, :nx-3]))
        if fd_order >= 8:
            Dx_U = Dx_U.at[:nz, 3:nx-2].add(
                c3 * (U[:nz, 5:nx] - U[:nz, :nx-5]))
            Dx_U = Dx_U.at[:nz, 4:nx-3].add(
                c4 * (U[:nz, 7:nx] - U[:nz, :nx-7]))
        Dx_U, p_Dx_U = _pml_apply(Dx_U, kx, p_Dx_U, bx, ax)
        ex = ex.at[:, 1:nx].add(dt/dx * Dx_U[:, 1:nx])

        # ── 4) Update εzz ─────────────────────────────────────────────────
        # Dz_V (D⁻_z): c_n*(f[iz]-f[iz-1]), result at iz-½
        Dz_V = jnp.zeros((nz, nx))
        if free_surface:
            # Free surface BC at z=0: σzz=0 → ∂Vz/∂z = -λ/(λ+2μ) * ∂Vx/∂x = L_ratio*∂Vx/∂x
            Dz_V = Dz_V.at[0, :nx-1].set(L_ratio[0, :nx-1] * Dx_U[0, :nx-1] * dz / dx)
        Dz_V = Dz_V.at[1:nz, :nx-1].set(
            c1 * (V[1:nz, :nx-1] - V[0:nz-1, :nx-1]))
        if fd_order >= 4:
            Dz_V = Dz_V.at[2:nz-1, :nx-1].add(
                c2 * (V[3:nz, :nx-1] - V[:nz-3, :nx-1]))
        if fd_order >= 8:
            Dz_V = Dz_V.at[3:nz-2, :nx-1].add(
                c3 * (V[5:nz, :nx-1] - V[:nz-5, :nx-1]))
            Dz_V = Dz_V.at[4:nz-3, :nx-1].add(
                c4 * (V[7:nz, :nx-1] - V[:nz-7, :nx-1]))
        # Hybrid: drop high-order imaging for Vz at surface rows.
        # Explicit BC at iz=0 (σzz=0 above) already handles the surface row.
        Dz_V, p_Dz_V = _pml_apply(Dz_V, kz, p_Dz_V, bz, az)
        ez = ez.at[0:nz, :].add(dt/dz * Dz_V[0:nz, :])

        # Moment tensor source (explosion: Mxx = Mzz = M0)
        if src_type == 'moment':
            lam_mu_s = L[sz, sx] + M[sz, sx]  # λ + μ
            src_term_mt = src_wavelet[it] * dt / (2 * lam_mu_s * dx * dz)
            ex = ex.at[sz, sx].add(src_term_mt)
            ez = ez.at[sz, sx].add(src_term_mt)

        # ── 5) Update εxz ─────────────────────────────────────────────────
        # Dz_U (D⁺_z): c_n*(f[iz+1]-f[iz]), result at iz+½
        Dz_U = jnp.zeros((nz, nx))
        Dz_U = Dz_U.at[:nz-1, 1:nx].set(
            c1 * (U[1:nz, 1:nx] - U[:nz-1, 1:nx]))
        if fd_order >= 4:
            Dz_U = Dz_U.at[1:nz-2, 1:nx].add(
                c2 * (U[3:nz, 1:nx] - U[:nz-3, 1:nx]))
        if fd_order >= 8:
            Dz_U = Dz_U.at[2:nz-3, 1:nx].add(
                c3 * (U[5:nz, 1:nx] - U[:nz-5, 1:nx]))
            Dz_U = Dz_U.at[3:nz-4, 1:nx].add(
                c4 * (U[7:nz, 1:nx] - U[:nz-7, 1:nx]))
        # Hybrid: drop high-order imaging for Vx at surface rows.
        Dz_U, p_Dz_U = _pml_apply(Dz_U, kz, p_Dz_U, bz, az)

        # Dx_V (D⁺_x): c_n*(f[ix+1]-f[ix]), result at ix+½
        Dx_V = jnp.zeros((nz, nx))
        Dx_V = Dx_V.at[:nz-1, :nx-1].set(
            c1 * (V[:nz-1, 1:nx] - V[:nz-1, :nx-1]))
        if fd_order >= 4:
            Dx_V = Dx_V.at[:nz-1, 1:nx-2].add(
                c2 * (V[:nz-1, 3:nx] - V[:nz-1, :nx-3]))
        if fd_order >= 8:
            Dx_V = Dx_V.at[:nz-1, 2:nx-3].add(
                c3 * (V[:nz-1, 5:nx] - V[:nz-1, :nx-5]))
            Dx_V = Dx_V.at[:nz-1, 3:nx-4].add(
                c4 * (V[:nz-1, 7:nx] - V[:nz-1, :nx-7]))
        Dx_V, p_Dx_V = _pml_apply(Dx_V, kx, p_Dx_V, bx, ax)

        # Update strain xz (seismology convention: εxz = ½(∂zVx + ∂xVz))
        es = es.at[:nz-1, :nx-1].add(0.5 * (dt/dz * Dz_U[:nz-1, :nx-1] + dt/dx * Dx_V[:nz-1, :nx-1]))

        new_carry = (U, V, ex, ez, es, p_Dx_L2M_ex, p_Dx_L_ez, p_Dz_M_es,
                     p_Dz_L2M_ez, p_Dz_L_ex, p_Dx_M_es, p_Dx_U, p_Dz_V, p_Dz_U, p_Dx_V)

        # Variables mapping
        vars_map = {
            'vx': U[rz, rx],
            'vz': V[rz, rx],
            'ex': ex[rz, rx],
            'ez': ez[rz, rx],
            'vx_full': U,
            'vz_full': V,
            'ex_full': ex,
            'ez_full': ez,
            'es_full': es
        }

        if return_vars is not None:
            return new_carry, tuple(vars_map[k] for k in return_vars)

        rec_data = (U[rz, rx], V[rz, rx], ex[rz, rx], ez[rz, rx])
        if return_wavefields:
            return new_carry, rec_data + (U, V, ex, ez)
        else:
            return new_carry, rec_data

    # Block Checkpointing logic
    num_blocks = nt // block_size
    remainder = nt % block_size

    @jax.checkpoint
    def run_block(carry, block_its):
        return lax.scan(time_step, carry, block_its)

    # Process blocks
    if num_blocks > 0:
        main_its = jnp.arange(num_blocks * block_size).reshape(num_blocks, block_size)
        carry, main_results = lax.scan(run_block, initial_carry, main_its)

        # Flatten the results from (num_blocks, block_size, ...) to (num_blocks * block_size, ...)
        main_results = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), main_results)
    else:
        carry = initial_carry
        # We need to determine the correct results structure
        _, sample_results = time_step(initial_carry, 0)
        main_results = jax.tree_util.tree_map(lambda x: jnp.zeros((0,) + x.shape), sample_results)

    # Process remainder
    if remainder > 0:
        remainder_its = jnp.arange(num_blocks * block_size, nt)
        carry, remainder_results = lax.scan(time_step, carry, remainder_its)
        # Combine with main results
        results = jax.tree_util.tree_map(lambda r1, r2: jnp.concatenate([r1, r2], axis=0), main_results, remainder_results)
    else:
        results = main_results

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Build JIT-compiled forward function
# ─────────────────────────────────────────────────────────────────────────────

def build_forward_fn(nz, nx, dx, dz, dt, nt, freq, pad, block_size, rec_x, rec_z,
                     fd_order=2, return_vars=None, free_surface=True):
    """
    Build a JIT-compiled single-shot forward function.

    Fixes grid/wavelet params into forward_jax, leaving only
    (Vs, Vp, rho, src_x, src_z, src_wavelet) as free args.

    Parameters
    ----------
    return_vars : list of str, optional
        e.g. ['ex_full'] to return full wavefield for illumination.
    free_surface : bool
        If True, free surface at top (z=0). If False, PML on all 4 sides.

    Returns:
        run_shot(Vs, Vp, rho, src_x, src_z, src_wavelet) → tuple
    """
    _fwd = partial(
        forward_jax,
        nx_dom=nx, nz_dom=nz,
        dx=dx,     dz=dz,
        dt=dt,     nt=nt,
        fc=freq,
        pad=pad,   block_size=block_size,
        fd_order=fd_order,
        return_vars=return_vars,
        free_surface=free_surface,
    )
    rec_x_jax = jnp.array(rec_x, dtype=jnp.int32)
    rec_z_jax = jnp.array(rec_z, dtype=jnp.int32)

    def run_shot(Vs, Vp, rho, src_x, src_z, src_wavelet):
        return _fwd(Vs, Vp, rho, src_x, src_z, rec_x_jax, rec_z_jax,
                    src_wavelet=src_wavelet)

    return jax.jit(run_shot)
