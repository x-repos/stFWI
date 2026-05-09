"""
Dot-product test for the hand-coded adjoint of the production forward
kernel (fwi/forward.py).

Configurations covered (Phases 1-4):
    fd_order      ∈ {2, 4, 8}
    free_surface  ∈ {False, True}
    src_type      ∈ {'force', 'moment'}

Test:
    pick random src_wavelet  s   (shape (nt,))
    pick random adjoint receiver data  y = (y_vx, y_vz, y_ex, y_ez)
                                       (each shape (nt, nrec))
    forward:  d = L s            (4-tuple of (nt, nrec) cubes)
    adjoint:  r = L^T y          (shape (nt,))
    check:   < d, y >  ==  < s, r >    to machine precision (~1e-14 in f64)

The forward IS exactly linear in src_wavelet at fixed model, so the discrete
adjoint produced by adjoint_jax_kernel.py should pass this to ~eps_machine.
"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from fwi.forward import forward_jax
from fwi.adjoint_jax_kernel import adjoint_jax


rng = np.random.default_rng(2026)

# ── Small heterogeneous model so all PDE terms are exercised ──────────────
nz_dom, nx_dom = 24, 32
dz, dx = 10.0, 10.0
nt = 60
dt = 1.0e-3
fc = 15.0
pad = 8

zz, xx = np.meshgrid(np.arange(nz_dom), np.arange(nx_dom), indexing="ij")
Vp  = 2500.0 + 150.0 * np.sin(2*np.pi*zz/nz_dom) * np.cos(2*np.pi*xx/nx_dom)
Vs  = 1400.0 +  80.0 * np.cos(2*np.pi*zz/nz_dom) * np.sin(2*np.pi*xx/nx_dom)
rho = 2000.0 + 100.0 * np.sin(np.pi*zz/nz_dom)   * np.cos(np.pi*xx/nx_dom)

# ── Acquisition: source in interior, several receivers spread across surface
src_x_int = nx_dom // 3
rec_x_int = np.arange(2, nx_dom - 2, 4, dtype=np.int32)
nrec = len(rec_x_int)

# ── Random source wavelet and random adjoint receiver data ────────────────
s_in = rng.standard_normal(nt)
y_vx = rng.standard_normal((nt, nrec))
y_vz = rng.standard_normal((nt, nrec))
y_ex = rng.standard_normal((nt, nrec))
y_ez = rng.standard_normal((nt, nrec))


def run_test(fd_order, free_surface, src_type):
    # Moment source must NOT sit at z=0 with free surface (the L+M pre-factor
    # uses interior values). Place it in the interior in all cases for moment.
    if src_type == 'moment':
        src_z_int = nz_dom // 4
    elif free_surface:
        src_z_int = 0
    else:
        src_z_int = nz_dom // 4

    if free_surface:
        rec_z_int = np.zeros_like(rec_x_int)
    else:
        rec_z_int = np.full_like(rec_x_int, nz_dom // 5)

    rec_vx, rec_vz, rec_ex, rec_ez = forward_jax(
        Vs, Vp, rho,
        src_x=int(src_x_int), src_z=int(src_z_int),
        rec_x=jnp.array(rec_x_int), rec_z=jnp.array(rec_z_int),
        nx_dom=nx_dom, nz_dom=nz_dom,
        dx=dx, dz=dz, dt=dt, nt=nt, fc=fc, pad=pad,
        src_wavelet=jnp.asarray(s_in),
        return_wavefields=False, block_size=nt,
        fd_order=fd_order, free_surface=free_surface, src_type=src_type,
    )

    src_wavelet_a = adjoint_jax(
        Vs, Vp, rho,
        src_x=int(src_x_int), src_z=int(src_z_int),
        rec_x=jnp.array(rec_x_int), rec_z=jnp.array(rec_z_int),
        nx_dom=nx_dom, nz_dom=nz_dom,
        dx=dx, dz=dz, dt=dt, nt=nt, fc=fc, pad=pad,
        rec_vx_a=jnp.asarray(y_vx),
        rec_vz_a=jnp.asarray(y_vz),
        rec_ex_a=jnp.asarray(y_ex),
        rec_ez_a=jnp.asarray(y_ez),
        fd_order=fd_order, free_surface=free_surface, src_type=src_type,
    )

    lhs = float(jnp.sum(rec_vx * jnp.asarray(y_vx))
                + jnp.sum(rec_vz * jnp.asarray(y_vz))
                + jnp.sum(rec_ex * jnp.asarray(y_ex))
                + jnp.sum(rec_ez * jnp.asarray(y_ez)))
    rhs = float(jnp.sum(jnp.asarray(s_in) * src_wavelet_a))
    abs_diff = abs(lhs - rhs)
    rel = abs_diff / max(abs(lhs), abs(rhs), 1e-300)

    tol = 1e-10
    ok = rel < tol
    status = "PASS" if ok else f"FAIL"
    print(f"  fd_order={fd_order}  FS={str(free_surface):5s}  src={src_type:6s}  "
          f"<Ls,y>={lhs: .3e}   <s,L^Ty>={rhs: .3e}   "
          f"rel={rel:.2e}   {status}")
    return ok


print("=" * 100)
print("Adjoint dot-product test  (production kernel)")
print("=" * 100)
all_ok = True
for fd_order in (2, 4, 8):
    for fs in (False, True):
        for src in ('force', 'moment'):
            all_ok &= run_test(fd_order, fs, src)
print("-" * 100)
print("OVERALL:", "PASS" if all_ok else "FAIL")
