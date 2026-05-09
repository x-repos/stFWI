"""
Smoke tests for Phase 1 FWI infrastructure modules:
    fwi/preconditioning.py
    fwi/projections.py

Velocity parameterisation only (no Lamé / impedance conversions).
"""
from __future__ import annotations

import numpy as np

from fwi import (
    eprecond1, eprecond3, taper_grad,
    k_safe_ratio, cfl_vp_cap, project_velocity, R_CONTINUUM,
)


rng = np.random.default_rng(7)
nz, nx, nrec = 32, 40, 10
dz = dx = 10.0
pad = 8


def banner(s):
    print()
    print("=" * 78)
    print(s)
    print("=" * 78)


# ─── 1) Velocity-space feasibility projection ──────────────────────────────
banner("Velocity feasibility projection")

# Cook a model that violates BOTH K-margin AND CFL.
vp_bad = np.where(rng.random((nz, nx)) > 0.5, 4500.0, 2500.0)  # some > vp_cap
vs_bad = vp_bad * 0.85   # vp/vs = 1.176 < 1.20  → violates K-margin
rho_bad = 2000.0 * np.ones((nz, nx))

vp_cap_test = 4000.0
# project_velocity uses (Vs, Vp, rho) order, matching forward_jax.
vs_p, vp_p, rho_p = project_velocity(vs_bad, vp_bad, rho_bad,
                                     K_margin=0.04, vp_cap=vp_cap_test)
r = k_safe_ratio(0.04)
ratio = vp_p / vs_p
ok_K   = bool(np.all(ratio >= r - 1e-12))
ok_cfl = bool(np.all(vp_p <= vp_cap_test + 1e-12))
print(f"  Vp/Vs ≥ {r:.4f} ?    {ok_K}    "
      f"(min ratio after = {ratio.min():.4f})")
print(f"  Vp ≤ {vp_cap_test} ?    {ok_cfl}    "
      f"(max Vp after = {vp_p.max():.4f})")

# helper consistency
print(f"  k_safe_ratio(0)  = 2/√3 ?      {abs(k_safe_ratio(0)-R_CONTINUUM)<1e-15}")
print(f"  cfl_vp_cap(...)  positive ?    "
      f"{cfl_vp_cap(dx=10, dz=10, dt=1e-3, cfl_coeff=0.6) > 0}")


# ─── 2) Preconditioner shapes / positivity ─────────────────────────────────
banner("Preconditioners")

# Wavefield energies — positive by construction; mock them as random**2.
nz_t, nx_t = nz + 2*pad, nx + 2*pad
Ws = rng.random((nz_t, nx_t)) ** 2 + 1e-6
Wr = rng.random((nz_t, nx_t)) ** 2 + 1e-6

We1 = eprecond1(Ws, Wr, eps=5e-7)
ok1 = (We1.shape == Ws.shape) and bool(np.all(We1 > 0))
print(f"  eprecond1   shape={We1.shape}   all>0 = {ok1}")

# Receivers along the surface (free_surface=True so z_start=0 in coord grid)
rec_x = np.arange(2, nx-2, 4)
rec_z = np.zeros_like(rec_x)
We3 = eprecond3(Ws, rec_x=rec_x, rec_z=rec_z, dx=dx, dz=dz,
                free_surface=True, pad=pad, eps=5e-7)
ok3 = (We3.shape == Ws.shape) and bool(np.all(We3 > 0))
print(f"  eprecond3   shape={We3.shape}   all>0 = {ok3}")

# Taper: zeros above gradt1, depth-boost below gradt2
T = taper_grad((nz, nx), dz=dz, gradt1=4, gradt2=8, exp_taper=2.0)
ok_t1 = bool(np.all(T[:4, :] == 0.0))
ok_t2 = bool(np.all(T[8, :] > T[7, :]))   # depth-boost rising past gradt2
print(f"  taper_grad  zeros above gradt1 = {ok_t1}    rising past gradt2 = {ok_t2}")


print()
print("done.")
