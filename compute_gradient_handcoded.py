"""
compute_gradient_handcoded.py — AD vs hand-coded-adjoint FWI gradient
benchmark using the unified `fwi.compute_gradient(backend=...)` API.

Same observed data, same residual prep (filter / taper / scale), same
post-processing (smooth / illum / source taper). Only the wave-equation
operator differs:

    backend='ad'        → reverse-mode AD on the multi-shot loss
    backend='adjoint'   → forward(record) + hand-coded adjoint_jax +
                           imaging_condition

Saves both gradients to  Results_gradient/gradient_compare.npz.
Render with `python plot_compare_gradient.py`.

Memory: per-shot peak ~5-7 GB on CPU at nt=400 (forward + adjoint cubes
for the adjoint backend; AD uses block-checkpointing so memory is much
smaller for that path).
"""
from __future__ import annotations

import os
import time

import numpy as np
import jax
import jax.numpy as jnp
from scipy.ndimage import gaussian_filter

from fwi import (
    build_forward_fn, ricker_jax, compute_gradient,
)


# ── Setup ────────────────────────────────────────────────────────────────
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

jax.config.update("jax_enable_x64", True)

out_dir = "Results_gradient"
os.makedirs(out_dir, exist_ok=True)

# ── Knobs ────────────────────────────────────────────────────────────────
n_shots_used = 5
fd_order     = 8
pad          = 30
block_size   = 200
free_surface = True
src_type     = "force"
component    = 2          # εxx
f_c          = 6.0
nt           = 400        # short window for memory budget on adjoint backend

# Loss preprocessing — leave OFF for the bare-operator comparison.
# Flip these on (e.g. cutoff=3.0, n_taper=0.1*nt) if you want to verify
# AD ≡ adjoint with the production filter+taper machinery in place too.
sos_cutoff   = None       # e.g. 3.0  (Hz); None = no Butterworth
n_taper_pct  = 0.0        # fraction of nt for cosine end-taper

# Post-AD preconditioning — same on both backends, OFF for bare comparison.
grad_smooth_sigma = None
source_taper      = None
illumination      = None


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

rec_x_int = np.arange(0, nx_dom, 2, dtype=np.int32)
rec_z_int = np.zeros(len(rec_x_int), dtype=np.int32)
print(f"Acquisition: {len(src_x)} shots ({src_x.tolist()}), {len(rec_x_int)} receivers")


# ── 3) Time / wavelet ────────────────────────────────────────────────────
if fd_order == 8:
    S = 1225/1024 + 245/3072 + 49/5120 + 5/7168
elif fd_order >= 4:
    S = 9/8 + 1/24
else:
    S = 1.0
dt = 0.9 / (S * Vp_true.max() * np.sqrt(1/dx**2 + 1/dz**2))
print(f"dt = {dt:.6e}    nt = {nt}    tmax = {nt*dt:.3f} s")

t0_src      = 1.5 / f_c
src_wavelet = ricker_jax(jnp.arange(nt) * dt, f_c, t0_src)


# ── 4) Observed data on TRUE model ───────────────────────────────────────
print("Generating observed data on TRUE model...")
t0 = time.time()
run_shot = build_forward_fn(nz_dom, nx_dom, dx, dz, dt, nt, f_c, pad, block_size,
                            rec_x_int, rec_z_int, fd_order=fd_order,
                            free_surface=free_surface)

src_x_arr = jnp.array(src_x, dtype=jnp.int32)
src_z_arr = jnp.array(src_z, dtype=jnp.int32)
all_obs = jax.jit(jax.vmap(run_shot, in_axes=(None, None, None, 0, 0, None)))(
    jnp.array(Vs_true), jnp.array(Vp_true), jnp.array(rho_true),
    src_x_arr, src_z_arr, src_wavelet,
)
observed = np.asarray(all_obs[component])      # (n_shots, nt, nrec)
print(f"  obs.shape = {observed.shape}    [{time.time()-t0:.1f}s]")


# ── 5) Build optional residual-prep filters ──────────────────────────────
sos = None
if sos_cutoff is not None:
    from scipy.signal import butter
    sos = butter(6, sos_cutoff, fs=1.0/dt, output='sos')
n_taper = int(n_taper_pct * nt)

auto_scale = float(1.0 / (np.mean(observed ** 2) + 1e-45))
print(f"auto_scale = {auto_scale:.4e}")


# ── 6) Common kwargs for both backends ───────────────────────────────────
common = dict(
    Vs=Vs_init, Vp=Vp_init, rho=rho_init,
    src_x=src_x, src_z=src_z, src_wavelet=src_wavelet,
    observed=observed, rec_x=rec_x_int, rec_z=rec_z_int,
    nx_dom=nx_dom, nz_dom=nz_dom,
    dx=dx, dz=dz, dt=dt, nt=nt, fc=f_c, pad=pad,
    component=component, invert='vs', scale=auto_scale,
    sos=sos, n_taper=n_taper,
    fd_order=fd_order, free_surface=free_surface, src_type=src_type,
    block_size=block_size,
    grad_smooth_sigma=grad_smooth_sigma,
    source_taper=source_taper,
    illumination=illumination,
)


# ── 7) AD backend ────────────────────────────────────────────────────────
print("\n── compute_gradient(backend='ad') ──")
t0 = time.time()
(g_Vs_AD, g_Vp_AD, g_rho_AD), loss_AD = compute_gradient(backend='ad', **common)
print(f"  loss = {loss_AD:.4e}    |g_Vs|_∞ = {np.max(np.abs(g_Vs_AD)):.3e}    [{time.time()-t0:.1f}s]")


# ── 8) Adjoint backend ───────────────────────────────────────────────────
print("\n── compute_gradient(backend='adjoint') ──")
t0 = time.time()
(g_Vs_HC, g_Vp_HC, g_rho_HC), loss_HC = compute_gradient(backend='adjoint', **common)
print(f"  loss = {loss_HC:.4e}    |g_Vs|_∞ = {np.max(np.abs(g_Vs_HC)):.3e}    [{time.time()-t0:.1f}s]")


# ── 9) Save ──────────────────────────────────────────────────────────────
print(f"\nFinal  |g_Vs_AD|_∞ = {np.max(np.abs(g_Vs_AD)):.3e}")
print(f"Final  |g_Vs_HC|_∞ = {np.max(np.abs(g_Vs_HC)):.3e}")
print(f"Loss(AD)  = {loss_AD:.4e}")
print(f"Loss(HC)  = {loss_HC:.4e}")

out_npz = os.path.join(out_dir, "gradient_compare.npz")
np.savez_compressed(
    out_npz,
    g_Vs_AD=g_Vs_AD, g_Vp_AD=g_Vp_AD, g_rho_AD=g_rho_AD,
    g_Vs_HC=g_Vs_HC, g_Vp_HC=g_Vp_HC, g_rho_HC=g_rho_HC,
    Vs_init=Vs_init, Vs_true=Vs_true,
    Vp_init=Vp_init, Vp_true=Vp_true,
    rho_init=rho_init, rho_true=rho_true,
    src_x=np.asarray(src_x), src_z=np.asarray(src_z),
    rec_x=rec_x_int, rec_z=rec_z_int,
    dx=dx, dz=dz, dt=float(dt), nt=nt, pad=pad,
    free_surface=np.bool_(free_surface),
    nz=nz_dom, nx=nx_dom,
    f_c=f_c, fd_order=fd_order, component=component,
    n_shots=n_shots_used,
    auto_scale=auto_scale,
    loss_AD=loss_AD, loss_HC=loss_HC,
)
print(f"\nSaved {out_npz}")
print("Run  `python plot_compare_gradient.py`  to render the AD-vs-adjoint figure.")
