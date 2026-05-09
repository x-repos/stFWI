"""
Main_strain_fwi.py — Marmousi Vs-FWI using the unified compute_gradient
backend + SciPy L-BFGS-B multistage loop.

Mirrors `Main_ex_Vs.py`'s recipe (component=2 = εxx, multiscale low→high
frequency, illumination + smoothing) but goes through:

    fwi.run_multiscale_fwi(
        ...,
        backend='ad',       ← swap to 'adjoint' to use the hand-coded
                              adjoint+imaging path under the hood
        stages=[...],
    )

Phase 1 helpers (`project_velocity`, `eprecond3`) are available as
optional knobs but the default stage settings here mirror Main_ex_Vs.py
(illumination_eps=0.0001, grad_smooth_sigma=1).

Knobs at the top: `backend`, `n_shots_used`, `nt`. The tunable dials at
the top control runtime/memory:

    backend='ad'        →  uses block-checkpointing (memory-light)
    backend='adjoint'   →  stores full wavefield history per shot;
                            keep `nt` modest (≤ ~600 on 16-32 GB CPU)
"""
from __future__ import annotations

import os
import time

import numpy as np
import jax
import jax.numpy as jnp
from scipy.ndimage import gaussian_filter

from fwi import (
    build_forward_fn, ricker_jax,
    run_multiscale_fwi,
)


# ── Setup ────────────────────────────────────────────────────────────────
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
# Adjoint backend stores 5 forward + 5 adjoint wavefield cubes per shot
# (≈ 24 GB at nt=2890 on Marmousi). Bump fraction to fit comfortably
# on a 32 GB card.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
jax.config.update("jax_enable_x64", True)

# ── Knobs ────────────────────────────────────────────────────────────────
backend       = "adjoint"     # or "ad"
n_shots_used  = 34            # full Marmousi acquisition
fd_order      = 8
pad           = 30
block_size    = 200
free_surface  = True
src_type      = "force"
component     = 2             # εxx (DAS-style)
tmax          = 6.0           # seconds — production setup (matches Main_ex_Vs.py)

save_dir      = f"Results_strain_fwi/{backend}"


# ── 1) Models ────────────────────────────────────────────────────────────
model_dir = "marmousi_models"
Vp_true  = np.load(os.path.join(model_dir, "vp_true.npy"))
Vs_true  = np.load(os.path.join(model_dir, "vs_true.npy"))
rho_true = np.load(os.path.join(model_dir, "rho_true.npy"))

sigma = 20
Vp_init  = 1.0 / gaussian_filter(1.0 / Vp_true,  sigma)
Vs_init  = 1.0 / gaussian_filter(1.0 / Vs_true,  sigma)
rho_init = gaussian_filter(rho_true, sigma)

nz_dom, nx_dom = Vs_true.shape
dx = dz = 20.0


# ── 2) Acquisition ───────────────────────────────────────────────────────
src_x_full = np.arange(0, nx_dom - 2, 15) + 2
src_z_full = np.zeros(len(src_x_full), dtype=int)
idx = np.linspace(0, len(src_x_full) - 1, n_shots_used).astype(int)
src_x = src_x_full[idx]
src_z = src_z_full[idx]

rec_x = np.arange(0, nx_dom, 2, dtype=np.int32)
rec_z = np.zeros(len(rec_x), dtype=np.int32)
print(f"Acquisition: {len(src_x)} shots, {len(rec_x)} receivers")


# ── 3) Time / wavelet ────────────────────────────────────────────────────
if fd_order == 8:
    S = 1225/1024 + 245/3072 + 49/5120 + 5/7168
elif fd_order >= 4:
    S = 9/8 + 1/24
else:
    S = 1.0
dt = 0.9 / (S * Vp_true.max() * np.sqrt(1/dx**2 + 1/dz**2))
nt = int(tmax / dt)
print(f"dt = {dt:.6e}    nt = {nt}    tmax = {nt*dt:.3f} s")


# ── 4) Observed data on TRUE model (single forward, multistage filter
#       inside the loss handles bandlimitation per stage) ────────────────
print("Generating observed data on TRUE model...")
t0 = time.time()
src_wavelet_obs = ricker_jax(jnp.arange(nt) * dt, 6.0, 1.5 / 6.0)   # broad
run_shot_obs = build_forward_fn(
    nz_dom, nx_dom, dx, dz, dt, nt, 6.0, pad, block_size,
    rec_x, rec_z, fd_order=fd_order, free_surface=free_surface,
)
src_x_arr = jnp.array(src_x, dtype=jnp.int32)
src_z_arr = jnp.array(src_z, dtype=jnp.int32)
all_obs = jax.jit(jax.vmap(run_shot_obs, in_axes=(None, None, None, 0, 0, None)))(
    jnp.array(Vs_true), jnp.array(Vp_true), jnp.array(rho_true),
    src_x_arr, src_z_arr, src_wavelet_obs,
)
observed = np.asarray(all_obs[component])      # (n_shots, nt, nrec)
print(f"  obs.shape = {observed.shape}    [{time.time()-t0:.1f}s]")


# ── 5) Stages (matches Main_ex_Vs.py recipe, fewer iters for demo) ──────
#  AD  : keep production behaviour — H_inv illumination ('illum').
#  adj : Plessix-Mulder eprecond3 (also accounts for receiver-line geometry,
#        more robust to imaging-condition discretisation noise).
precon_for_backend = 'eprecond3' if backend == 'adjoint' else 'illum'
print(f"Preconditioner = {precon_for_backend}")

stages = [
    dict(f_c=6.0, cutoff=3.0,  n_iters=200, invert='vs',
         n_taper_pct=0.1,
         precon=precon_for_backend, illumination_eps=0.0001,
         grad_smooth_sigma=1.0),
    dict(f_c=6.0, cutoff=15.0, n_iters=200, invert='vs',
         n_taper_pct=0.1,
         precon=precon_for_backend, illumination_eps=0.0001,
         grad_smooth_sigma=1.0),
]


# ── 6) Run FWI ───────────────────────────────────────────────────────────
print(f"\nRunning multiscale FWI  (backend={backend})...")
t0 = time.time()
(Vs_inv, Vp_inv, rho_inv), history = run_multiscale_fwi(
    Vs_init=Vs_init, Vp_init=Vp_init, rho_init=rho_init,
    src_x=src_x, src_z=src_z, rec_x=rec_x, rec_z=rec_z,
    nx_dom=nx_dom, nz_dom=nz_dom,
    dx=dx, dz=dz, dt=dt, nt=nt, pad=pad,
    fd_order=fd_order, free_surface=free_surface,
    src_type=src_type, block_size=block_size,
    component=component, backend=backend,
    observed=observed,
    stages=stages,
    K_margin=0.04,
    cfl_coeff=S,
    vs_bounds=(500.0, 4500.0),
    plateau_window=5, plateau_rel_tol=1e-3,
    save_dir=save_dir,
    save_every=10,                  # checkpoint every 10 L-BFGS iterations
    verbose=True,
)
print(f"\nTotal FWI time: {time.time()-t0:.1f}s")


# ── 7) Save final result + summary plots ────────────────────────────────
os.makedirs(save_dir, exist_ok=True)
out_npz = os.path.join(save_dir, "final.npz")
np.savez_compressed(
    out_npz,
    Vs_inv=Vs_inv, Vp_inv=Vp_inv, rho_inv=rho_inv,
    Vs_init=Vs_init, Vp_init=Vp_init, rho_init=rho_init,
    Vs_true=Vs_true, Vp_true=Vp_true, rho_true=rho_true,
    dx=dx, dz=dz, nz=nz_dom, nx=nx_dom,
    src_x=src_x, src_z=src_z, rec_x=rec_x, rec_z=rec_z,
    backend=backend,
)
print(f"Saved {out_npz}")
print("\nStage summary:")
for h in history:
    nl = h["loss_history"]
    print(f"  stage {h['stage']+1}: f_c={h['f_c']}  cutoff={h['cutoff']}  "
          f"loss {nl[0]:.4e} → {nl[-1]:.4e}  ({len(nl)} evals, {h['time_s']:.1f}s)")
