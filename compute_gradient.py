"""
compute_gradient.py — raw AD FWI gradient at the initial Marmousi model.

This is the "benchmark" gradient for comparing with the hand-coded
adjoint kernel: NO post-AD preconditioning is applied (no illumination,
no source taper, no smoothing, no Tikhonov).

Pipeline (one stage, low-frequency band):
    1. Load Marmousi  (Vp, Vs, ρ — true and initial)
    2. Generate observed εxx data on the TRUE model
    3. At the INITIAL model: (loss, gradient) via AD  (component=2 = εxx)
    4. Save  Results_gradient/gradient_data.npz

Plotting lives in `plot_gradient.py` so you can iterate on the figure
without re-running the (~90 s) computation.
"""
from __future__ import annotations

import os
import time

import numpy as np
import jax
import jax.numpy as jnp
from scipy.ndimage import gaussian_filter
from scipy.signal import butter

from fwi import (
    build_forward_fn, build_loss_fn, ricker_jax,
)


# ── GPU / output ────────────────────────────────────────────────────────
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

out_dir = "Results_gradient"
os.makedirs(out_dir, exist_ok=True)

# ── Knobs ────────────────────────────────────────────────────────────────
n_shots_used = 5      # subset of Marmousi sources for speed
component    = 2      # 0=vx, 1=vz, 2=εxx, 3=εzz
fd_order     = 8
pad          = 30
block_size   = 200
free_surface = True
src_type     = "force"

# Single-stage parameters (matches Stage 1 of Main_ex_Vs.py)
f_c    = 6.0          # source dominant freq [Hz]
cutoff = 3.0          # Butterworth lowpass cutoff [Hz]
tmax   = 6.0          # seconds


# ── 1) Models ────────────────────────────────────────────────────────────
model_dir = "marmousi_models"
Vp_true  = np.load(os.path.join(model_dir, "vp_true.npy"))
Vs_true  = np.load(os.path.join(model_dir, "vs_true.npy"))
rho_true = np.load(os.path.join(model_dir, "rho_true.npy"))

# Smoothed initial models (same recipe as Main_ex_Vs.py)
sigma = 20
Vp_init  = 1.0 / gaussian_filter(1.0 / Vp_true,  sigma)
Vs_init  = 1.0 / gaussian_filter(1.0 / Vs_true,  sigma)
rho_init = gaussian_filter(rho_true, sigma)

nz, nx = Vs_true.shape
dx = dz = 20.0

# ── 2) Acquisition ───────────────────────────────────────────────────────
src_x_full = np.arange(0, nx - 2, 15) + 2
src_z_full = np.zeros(len(src_x_full), dtype=int)

# Pick ~n_shots_used shots evenly spread
idx = np.linspace(0, len(src_x_full) - 1, n_shots_used).astype(int)
src_x = src_x_full[idx]
src_z = src_z_full[idx]

rec_x = np.arange(0, nx, 2)
rec_z = np.zeros(len(rec_x), dtype=int)
print(f"Acquisition: {len(src_x)} shots, {len(rec_x)} receivers")

# ── 3) Time stepping ─────────────────────────────────────────────────────
if fd_order == 8:
    S = 1225/1024 + 245/3072 + 49/5120 + 5/7168
elif fd_order >= 4:
    S = 9/8 + 1/24
else:
    S = 1.0
dt = 0.9 / (S * Vp_true.max() * np.sqrt(1/dx**2 + 1/dz**2))
nt = int(tmax / dt)
print(f"dt = {dt:.6e}    nt = {nt}")

# Source wavelet
t0_src = 1.5 / f_c
src_wavelet = ricker_jax(jnp.arange(nt) * dt, f_c, t0_src)

src_x_arr = jnp.array(src_x, dtype=jnp.int32)
src_z_arr = jnp.array(src_z, dtype=jnp.int32)


# ── 4) Observed data on TRUE model ───────────────────────────────────────
print("Generating observed data on TRUE model...")
t0 = time.time()
run_shot = build_forward_fn(nz, nx, dx, dz, dt, nt, f_c, pad, block_size,
                            rec_x, rec_z, fd_order=fd_order,
                            free_surface=free_surface)

all_obs = jax.jit(jax.vmap(run_shot, in_axes=(None, None, None, 0, 0, None)))(
    jnp.array(Vs_true), jnp.array(Vp_true), jnp.array(rho_true),
    src_x_arr, src_z_arr, src_wavelet,
)
observed = all_obs[component]    # (n_shots, nt, n_rec)
print(f"  observed.shape = {observed.shape}    [{time.time()-t0:.1f}s]")


# ── 5) Loss + gradient at INITIAL model ──────────────────────────────────
print("Computing AD gradient at INITIAL model (component=εxx)...")
t0 = time.time()
sos        = butter(6, cutoff, fs=1/dt, output='sos')
n_taper    = int(0.1 * nt)
auto_scale = 1.0 / (jnp.mean(observed ** 2) + 1e-45)

loss_fn = build_loss_fn(
    run_shot, observed, src_x, src_z, src_wavelet, component,
    scale=float(auto_scale),
    invert='vs',                  # only Vs gradient is non-stop_gradient
    sos=sos, n_taper=n_taper,
    source_taper=None,
    grad_smooth_sigma=None,
    tikhonov_alpha=None,
    illumination=None,
)

params0 = (jnp.array(Vs_init), jnp.array(Vp_init), jnp.array(rho_init))
loss0, grads0 = loss_fn(params0)
g_vs, g_vp, g_rho = (np.asarray(g) for g in grads0)
print(f"  loss = {float(loss0):.4e}")
print(f"  |g_vs|_∞ = {np.max(np.abs(g_vs)):.4e}    [{time.time()-t0:.1f}s]")


# ── 6) Save everything to disk for plot_gradient.py ──────────────────────
out_npz = os.path.join(out_dir, "gradient_data.npz")
np.savez_compressed(
    out_npz,
    # raw AD gradients (physical-domain, shape (nz, nx))
    g_vs=g_vs,
    g_vp=g_vp,
    g_rho=g_rho,
    # models
    Vs_true=Vs_true, Vs_init=Vs_init,
    Vp_true=Vp_true, Vp_init=Vp_init,
    rho_true=rho_true, rho_init=rho_init,
    # geometry & params
    src_x=np.asarray(src_x), src_z=np.asarray(src_z),
    rec_x=np.asarray(rec_x), rec_z=np.asarray(rec_z),
    dx=np.float64(dx), dz=np.float64(dz),
    pad=np.int32(pad),
    free_surface=np.bool_(free_surface),
    nz=np.int32(nz), nx=np.int32(nx),
    nt=np.int32(nt), dt=np.float64(dt),
    f_c=np.float64(f_c), cutoff=np.float64(cutoff),
    component=np.int32(component),
    fd_order=np.int32(fd_order),
    n_shots=np.int32(len(src_x)),
    # scalar diagnostics
    loss=np.float64(loss0),
    auto_scale=np.float64(auto_scale),
)
print(f"\nSaved {out_npz}")
print("Run  `python plot_gradient.py`  to render the figure.")
