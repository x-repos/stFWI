"""
Hand-coded (non-AD) JAX adjoint for the velocity-strain elastic wave equation.

Discretization mirrors `fwi-numpy/solver.py` (Zhou 2024, eqs. 3-5, dropping the
adjoint-injection terms Delta e):

    rho d_t v_z = d_z[(lam+2mu) e_z + lam e_x] + d_x[mu e_s] + f_z
    rho d_t v_x = d_x[lam e_z + (lam+2mu) e_x] + d_z[mu e_s] + f_x
    d_t e_z     = d_z v_z
    d_t e_x     = d_x v_x
    d_t e_s     = d_z v_x + d_x v_z          (e_s = 2 eps_xz)

with a collocated periodic grid, centered 2-point differences (antisymmetric:
D^T = -D at the discrete level) and leap-frog time stepping. This combination
makes the discrete adjoint the exact transpose of the implemented forward
linear operator, so the dot-product adjoint test returns a relative mismatch
at O(eps_machine).

This is intentionally *not* the same kernel as `fwi/forward.py`. The production
kernel uses a staggered grid, high-order FD, C-PML and an optional free
surface, which break the clean antisymmetry that makes the hand-coded adjoint
exact. This module is the JAX counterpart of the NumPy reference, scaffolding
for a future hand-coded adjoint of the production kernel.

Exposed:
    Dz, Dx, Dz_T, Dx_T   -- centered periodic differences (and transposes)
    forward_simple       -- L : (fz_src, fx_src) history -> (vz, vx) history
    adjoint_simple       -- L^T : (vz*, vx*) adjoint   -> (fz*, fx*) adjoint-source
    inner_product        -- L2 inner product over space-time grid
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax


# ---------------------------------------------------------------------------
# Centered periodic differences. Both are antisymmetric, so D^T = -D exactly.
# ---------------------------------------------------------------------------
def Dz(u, dz):
    return (jnp.roll(u, -1, axis=0) - jnp.roll(u, 1, axis=0)) / (2.0 * dz)


def Dx(u, dx):
    return (jnp.roll(u, -1, axis=1) - jnp.roll(u, 1, axis=1)) / (2.0 * dx)


def Dz_T(u, dz):
    return -Dz(u, dz)


def Dx_T(u, dx):
    return -Dx(u, dx)


# ---------------------------------------------------------------------------
# Forward solver : source history  ->  velocity history (the "data")
# ---------------------------------------------------------------------------
def forward_simple(rho, lam, mu, fz_src, fx_src, dz, dx, dt):
    """
    Linear map L from body-force source history to velocity history.

    Parameters
    ----------
    rho, lam, mu : (nz, nx) arrays
    fz_src, fx_src : (nt, nz, nx) arrays  -- volumetric body-force per step
    dz, dx, dt : floats

    Returns
    -------
    vz_hist, vx_hist : (nt, nz, nx) arrays  -- velocities recorded AFTER each step
    """
    rho = jnp.asarray(rho)
    lam = jnp.asarray(lam)
    mu = jnp.asarray(mu)
    fz_src = jnp.asarray(fz_src)
    fx_src = jnp.asarray(fx_src)

    nz, nx = rho.shape
    lp2m = lam + 2.0 * mu
    inv_rho = 1.0 / rho

    def step(carry, srcs):
        v_z, v_x, e_z, e_x, e_s = carry
        fz_n, fx_n = srcs

        # strain update (kinematic):  e^{n+1/2} = e^{n-1/2} + dt * G v^n
        e_z = e_z + dt * Dz(v_z, dz)
        e_x = e_x + dt * Dx(v_x, dx)
        e_s = e_s + dt * (Dz(v_x, dz) + Dx(v_z, dx))

        # constitutive (pointwise symmetric stiffness)
        sig_zz = lp2m * e_z + lam * e_x
        sig_xx = lam * e_z + lp2m * e_x
        sig_xz = mu * e_s

        # velocity update:  v^{n+1} = v^n + (dt/rho)(div sigma + f)
        v_z = v_z + (dt * inv_rho) * (Dz(sig_zz, dz) + Dx(sig_xz, dx) + fz_n)
        v_x = v_x + (dt * inv_rho) * (Dx(sig_xx, dx) + Dz(sig_xz, dz) + fx_n)

        return (v_z, v_x, e_z, e_x, e_s), (v_z, v_x)

    z = jnp.zeros((nz, nx), dtype=rho.dtype)
    init = (z, z, z, z, z)
    _, (vz_hist, vx_hist) = lax.scan(step, init, (fz_src, fx_src))
    return vz_hist, vx_hist


# ---------------------------------------------------------------------------
# Adjoint solver : adjoint "data"  ->  adjoint source  (transpose of forward)
# ---------------------------------------------------------------------------
def adjoint_simple(rho, lam, mu, vz_adj_data, vx_adj_data, dz, dx, dt):
    """
    Discrete transpose of `forward_simple`. Same parameter conventions.

    Parameters
    ----------
    vz_adj_data, vx_adj_data : (nt, nz, nx)  -- adjoint "data" cubes

    Returns
    -------
    fz_adj, fx_adj : (nt, nz, nx)  -- adjoint body-force cubes
    """
    rho = jnp.asarray(rho)
    lam = jnp.asarray(lam)
    mu = jnp.asarray(mu)
    vz_adj_data = jnp.asarray(vz_adj_data)
    vx_adj_data = jnp.asarray(vx_adj_data)

    nz, nx = rho.shape
    lp2m = lam + 2.0 * mu
    inv_rho = 1.0 / rho

    def step(carry, adjs):
        v_z_a, v_x_a, e_z_a, e_x_a, e_s_a = carry
        vz_a_in, vx_a_in = adjs

        # Adjoint of "history store": v_hist[n] = v_{post-update}
        #  => inject adjoint-data into v_{post}_adj before undoing the step.
        v_z_a = v_z_a + vz_a_in
        v_x_a = v_x_a + vx_a_in

        # ---- Adjoint of velocity update ----
        # forward: v_z += (dt/rho)(Dz sig_zz + Dx sig_xz + f_z)
        sig_zz_a = Dz_T((dt * inv_rho) * v_z_a, dz)
        sig_xz_a = Dx_T((dt * inv_rho) * v_z_a, dx) + Dz_T((dt * inv_rho) * v_x_a, dz)
        sig_xx_a = Dx_T((dt * inv_rho) * v_x_a, dx)
        fz_out = (dt * inv_rho) * v_z_a
        fx_out = (dt * inv_rho) * v_x_a
        # v_*_a passes through identity (the += is identity in the v_a slot)

        # ---- Adjoint of constitutive (local symmetric stiffness) ----
        e_z_a = e_z_a + lp2m * sig_zz_a + lam * sig_xx_a
        e_x_a = e_x_a + lam * sig_zz_a + lp2m * sig_xx_a
        e_s_a = e_s_a + mu * sig_xz_a

        # ---- Adjoint of strain update ----
        # forward: e_z += dt Dz v_z, e_x += dt Dx v_x, e_s += dt (Dz v_x + Dx v_z)
        v_z_a = v_z_a + Dz_T(dt * e_z_a, dz) + Dx_T(dt * e_s_a, dx)
        v_x_a = v_x_a + Dx_T(dt * e_x_a, dx) + Dz_T(dt * e_s_a, dz)

        return (v_z_a, v_x_a, e_z_a, e_x_a, e_s_a), (fz_out, fx_out)

    z = jnp.zeros((nz, nx), dtype=rho.dtype)
    init = (z, z, z, z, z)
    # Walk time backwards. `reverse=True` feeds xs[nt-1] first and stacks ys
    # in original (forward) order, so fz_adj[n] corresponds to the forward
    # source slot at time n.
    _, (fz_adj, fx_adj) = lax.scan(step, init, (vz_adj_data, vx_adj_data), reverse=True)
    return fz_adj, fx_adj


# ---------------------------------------------------------------------------
# L2 inner product over space-time grid (matches fwi-numpy.solver.ip)
# ---------------------------------------------------------------------------
def inner_product(a, b):
    return jnp.sum(a * b)
