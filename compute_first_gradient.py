"""
compute_first_gradient.py — produce the FIRST FWI gradient at the initial
model on the Marmousi production setup, using BOTH backends (AD and
adjoint) with the same stage-1 preconditioning (illumination + Gaussian
smoothing). One forward+gradient eval per backend, no inversion loop.

Saves to  Results_first_gradient/first_gradient.npz  with both gradients,
illumination, and the model. Plot with plot_first_gradient.py.
"""
from __future__ import annotations

import os
import time
import argparse

# Parse CLI BEFORE importing JAX so we can set env vars accordingly.
_p = argparse.ArgumentParser()
_p.add_argument("--backend", choices=["ad", "adjoint", "both"], default="both",
                help="Which backend to run. 'both' runs AD then adjoint in the "
                     "same process (may OOM); 'ad' or 'adjoint' alone runs "
                     "only that backend in this process — pair with two calls.")
_args = _p.parse_args()
_RUN_AD  = _args.backend in ("ad", "both")
_RUN_ADJ = _args.backend in ("adjoint", "both")

# CRITICAL: these env vars must be set BEFORE jax is imported. They tell
# JAX/XLA to skip the BFC pool and call cudaMalloc directly — slower per
# allocation but avoids the "BFC ran out trying to alloc N GB" failures
# that hit when we jump between forward (single-output) and adjoint kernels.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import numpy as np
import jax
import jax.numpy as jnp
from scipy.ndimage import gaussian_filter
from scipy.signal import butter

from fwi import (
    build_forward_fn, ricker_jax,
    compute_gradient, compute_illumination_for_stage,
    eprecond3,
)


# ── Setup ────────────────────────────────────────────────────────────────
jax.config.update("jax_enable_x64", True)

out_dir = "Results_first_gradient"
os.makedirs(out_dir, exist_ok=True)

n_shots_used     = 34
fd_order         = 8
pad              = 30
block_size       = 200
free_surface     = True
src_type         = "force"
component        = 2          # εxx
tmax             = 6.0

# Stage 1 settings (low frequency, with production preconditioning)
f_c              = 6.0
cutoff           = 3.0
n_taper_pct      = 0.1
illumination_eps = 0.0001
grad_smooth_sigma = 1.0


# ── 1) Models ────────────────────────────────────────────────────────────
model_dir = "marmousi_models"
Vp_true  = np.load(os.path.join(model_dir, "vp_true.npy"))
Vs_true  = np.load(os.path.join(model_dir, "vs_true.npy"))
rho_true = np.load(os.path.join(model_dir, "rho_true.npy"))

sigma_init = 20
Vp_init  = 1.0 / gaussian_filter(1.0 / Vp_true,  sigma_init)
Vs_init  = 1.0 / gaussian_filter(1.0 / Vs_true,  sigma_init)
rho_init = gaussian_filter(rho_true, sigma_init)

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


# ── 3) Time / wavelet ────────────────────────────────────────────────────
S = 1225/1024 + 245/3072 + 49/5120 + 5/7168     # fd_order=8
dt = 0.9 / (S * Vp_true.max() * np.sqrt(1/dx**2 + 1/dz**2))
nt = int(tmax / dt)
print(f"dt = {dt:.6e}    nt = {nt}    tmax = {nt*dt:.3f} s")

src_wavelet = ricker_jax(jnp.arange(nt) * dt, f_c, 1.5 / f_c)
sos = butter(6, cutoff, fs=1.0/dt, output='sos')
n_taper = int(n_taper_pct * nt)


# ── 4) Observed data on TRUE model ──────────────────────────────────────
print("Generating observed data on TRUE model (34 shots)...")
t0 = time.time()
run_shot_obs = build_forward_fn(
    nz_dom, nx_dom, dx, dz, dt, nt, f_c, pad, block_size,
    rec_x, rec_z, fd_order=fd_order, free_surface=free_surface,
)
src_x_arr = jnp.array(src_x, dtype=jnp.int32)
src_z_arr = jnp.array(src_z, dtype=jnp.int32)
all_obs = jax.jit(jax.vmap(run_shot_obs, in_axes=(None, None, None, 0, 0, None)))(
    jnp.array(Vs_true), jnp.array(Vp_true), jnp.array(rho_true),
    src_x_arr, src_z_arr, src_wavelet,
)
observed = np.asarray(all_obs[component])
# CRITICAL: free the 4-tuple immediately. With nt=2890 it's ~27 GB on GPU
# (4 components × 34 shots × nt × nrec × 8 B). Without del, this stays
# alive throughout the script and breaks the adjoint pass at production
# scale.
del all_obs, run_shot_obs
import gc as _gc; _gc.collect()
auto_scale = float(1.0 / (np.mean(observed ** 2) + 1e-45))
print(f"  obs.shape = {observed.shape}    auto_scale = {auto_scale:.4e}    [{time.time()-t0:.1f}s]")


# ── 5a) Illumination H_inv  (used by AD, matches production behaviour) ─
illumination = None
if _RUN_AD:
    print("Computing illumination on INITIAL model (34 shots)...")
    t0 = time.time()
    illumination = compute_illumination_for_stage(
        nz_dom, nx_dom, dx, dz, dt, nt, f_c, pad, block_size,
        rec_x, rec_z, fd_order, component,
        jnp.asarray(Vs_init), jnp.asarray(Vp_init), jnp.asarray(rho_init),
        src_x, src_z, src_wavelet, eps=illumination_eps,
    )
    illumination = np.asarray(illumination)
    print(f"  illumination shape = {illumination.shape}    [{time.time()-t0:.1f}s]")


# ── 5b) eprecond3  (Plessix-Mulder; used ONLY by adjoint) ──────────────
# Recompute Ws on the FULL padded grid (the H_inv routine cropped to the
# physical domain). Then apply eprecond3 = sqrt(Ws) · arcsinh(receivers).
illumination_eprecond3 = None
if _RUN_ADJ:
    print("Computing eprecond3 on INITIAL model (34 shots)...")
    t0 = time.time()
    comp_to_var = {0: "vx_full", 1: "vz_full", 2: "ex_full", 3: "ez_full"}
    run_shot_full = build_forward_fn(
        nz_dom, nx_dom, dx, dz, dt, nt, f_c, pad, block_size,
        rec_x, rec_z, fd_order=fd_order, free_surface=free_surface,
        return_vars=[comp_to_var[component]],
    )
    nz_t = nz_dom + pad if free_surface else nz_dom + 2 * pad
    nx_t = nx_dom + 2 * pad
    Ws_full = np.zeros((nz_t, nx_t))
    for i in range(len(src_x)):
        (field,) = run_shot_full(
            jnp.asarray(Vs_init), jnp.asarray(Vp_init), jnp.asarray(rho_init),
            jnp.int32(int(src_x[i])), jnp.int32(int(src_z[i])), src_wavelet,
        )
        Ws_full += np.asarray(jnp.sum(field ** 2, axis=0))
        if (i + 1) % 10 == 0 or i == len(src_x) - 1:
            print(f"    eprecond3 Ws: shot {i+1}/{len(src_x)}")
    We_full = eprecond3(
        Ws_full, rec_x=rec_x, rec_z=rec_z, dx=dx, dz=dz,
        free_surface=free_surface, pad=pad, eps=illumination_eps,
    )
    z_start = 0 if free_surface else pad
    We_phys = We_full[z_start:z_start + nz_dom, pad:pad + nx_dom]
    illumination_eprecond3 = 1.0 / We_phys
    illumination_eprecond3 = illumination_eprecond3 / np.max(illumination_eprecond3)  # → [0, 1]
    print(f"  eprecond3 shape = {illumination_eprecond3.shape}    "
          f"min={illumination_eprecond3.min():.4e}   "
          f"max={illumination_eprecond3.max():.4f}    "
          f"[{time.time()-t0:.1f}s]")


# ── 6) AD gradient ──────────────────────────────────────────────────────
common = dict(
    Vs=Vs_init, Vp=Vp_init, rho=rho_init,
    src_x=src_x, src_z=src_z, src_wavelet=src_wavelet,
    observed=observed, rec_x=rec_x, rec_z=rec_z,
    nx_dom=nx_dom, nz_dom=nz_dom,
    dx=dx, dz=dz, dt=dt, nt=nt, fc=f_c, pad=pad,
    component=component, invert='vs', scale=auto_scale,
    sos=sos, n_taper=n_taper,
    fd_order=fd_order, free_surface=free_surface, src_type=src_type,
    block_size=block_size,
    grad_smooth_sigma=grad_smooth_sigma,
    source_taper=None,
    # `illumination` is passed PER BACKEND below (AD: H_inv ; adj: eprecond3)
)

out_npz = os.path.join(out_dir, "first_gradient.npz")

# Metadata always saved alongside whichever gradient(s) we computed.
# When a preconditioner wasn't computed in this run (e.g. AD-only doesn't
# need eprecond3), we omit it here and rely on the prior-merge step below
# to preserve whatever was previously on disk.
common_meta = dict(
    Vs_init=Vs_init, Vs_true=Vs_true,
    Vp_init=Vp_init, Vp_true=Vp_true,
    rho_init=rho_init, rho_true=rho_true,
    src_x=np.asarray(src_x), src_z=np.asarray(src_z),
    rec_x=rec_x, rec_z=rec_z,
    dx=dx, dz=dz, dt=float(dt), nt=nt, pad=pad,
    free_surface=np.bool_(free_surface),
    nz=nz_dom, nx=nx_dom,
    f_c=f_c, cutoff=cutoff, fd_order=fd_order, component=component,
    n_shots=n_shots_used,
    auto_scale=auto_scale,
    grad_smooth_sigma=grad_smooth_sigma,
    illumination_eps=illumination_eps,
)
if illumination is not None:
    common_meta["illumination"] = illumination
if illumination_eprecond3 is not None:
    common_meta["illumination_eprecond3"] = illumination_eprecond3


# Load EVERYTHING from prior npz so we can MERGE (not overwrite). When
# this is an adjoint-only run we want to keep the H_inv 'illumination'
# array AD wrote earlier; vice-versa for an AD-only run.
prior = {}
if os.path.exists(out_npz):
    try:
        with np.load(out_npz, allow_pickle=True) as d:
            for k in d.files:
                prior[k] = np.asarray(d[k])
        print(f"Loaded prior fields from {out_npz}: {sorted(prior.keys())}")
    except Exception as e:
        print(f"(could not read prior {out_npz}: {e})")


# ── 6) AD gradient ──────────────────────────────────────────────────────
if _RUN_AD:
    print("\n── compute_gradient(backend='ad')   [illumination = H_inv] ──")
    t0 = time.time()
    (g_Vs_AD, g_Vp_AD, g_rho_AD), loss_AD = compute_gradient(
        backend='ad', illumination=illumination, **common,
    )
    print(f"  loss = {loss_AD:.4e}    |g_Vs|_∞ = {np.max(np.abs(g_Vs_AD)):.3e}    "
          f"[{time.time()-t0:.1f}s]")
    prior.update(g_Vs_AD=g_Vs_AD, g_Vp_AD=g_Vp_AD, g_rho_AD=g_rho_AD,
                 loss_AD=loss_AD)
    np.savez_compressed(out_npz, **prior, **common_meta)
    print(f"  -> Saved (AD merged) to {out_npz}")


# ── 7) Adjoint + eprecond3 ──────────────────────────────────────────────
if _RUN_ADJ:
    # Aggressive cleanup before the adjoint pass.
    import gc
    for name in ("all_obs", "Ws_full", "We_full", "We_phys", "run_shot_full"):
        if name in globals():
            del globals()[name]
    jax.clear_caches()
    for _ in range(3):
        gc.collect()
    try:
        jax.lib.xla_bridge.get_backend().clear_compile_cache()
    except Exception:
        pass

    print("\n── compute_gradient(backend='adjoint')   [illumination = eprecond3] ──")
    print("    (per-shot: forward record + adjoint_jax + imaging condition)")
    t0 = time.time()
    try:
        (g_Vs_HC, g_Vp_HC, g_rho_HC), loss_HC = compute_gradient(
            backend='adjoint', illumination=illumination_eprecond3, **common,
        )
        print(f"  loss = {loss_HC:.4e}    |g_Vs|_∞ = {np.max(np.abs(g_Vs_HC)):.3e}    "
              f"[{time.time()-t0:.1f}s]")
        prior.update(g_Vs_HC=g_Vs_HC, g_Vp_HC=g_Vp_HC, g_rho_HC=g_rho_HC,
                     loss_HC=loss_HC)
        np.savez_compressed(out_npz, **prior, **common_meta)
        print(f"\nSaved (adjoint merged) to {out_npz}")
    except Exception as e:
        print(f"\n!! Adjoint pass failed: {type(e).__name__}: {e}")
        print(f"   AD-only / prior result(s) preserved at {out_npz}")

print("\nPlot with `python plot_first_gradient.py`.")
