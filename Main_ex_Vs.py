# %%
# Multiscale Strain-Velocity code for Full Waveform Inversion (FWI) — Invert Vs for Marmousi model
# Author: Minh Nhat Tran
# Date: 2026

# ── Imports ──────────────────────────────────────────────────────
# Your environment may require all the packages to be imported before setting CUDA_VISIBLE_DEVICES.
import os
import time

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import butter

from fwi import (
    build_forward_fn, build_loss_fn, run_lbfgs, ricker_jax,
    build_source_taper, save_source_taper_plot,
    save_shot_qc,
    compute_illumination_for_stage, save_illumination_plot,
    save_acquisition_plot, save_wavelet_plot,
    save_multistage_loss_curve,
)

# ── GPU / Environment ───────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

# ── Output directories ──────────────────────────────────────────
result_dir = 'Results_ex_Vs/elastic_FWI'
setup_dir  = os.path.join(result_dir, 'setup')
os.makedirs(setup_dir, exist_ok=True)

# %%
# ═════════════════════════════════════════════════════════════════
# 1. MODELS
# ═════════════════════════════════════════════════════════════════
# Model Discription:
# - Marmousi model (2D elastic) removed water layer and top 200m
# - Size: 3040m (depth) x 10000m (width) - 152 x 500 grid points
# - Grid spacing: 20m

model_dir = 'marmousi_models'

Vp_true  = np.load(os.path.join(model_dir, 'vp_true.npy'))
Vp_init  = np.load(os.path.join(model_dir, 'vp_init.npy'))
Vs_true  = np.load(os.path.join(model_dir, 'vs_true.npy'))
Vs_init  = np.load(os.path.join(model_dir, 'vs_init.npy'))
rho_true = np.load(os.path.join(model_dir, 'rho_true.npy'))
rho_init = np.load(os.path.join(model_dir, 'rho_init.npy'))

nz, nx = Vs_true.shape
dx = 20.0
dz = 20.0

# Smooth initial models (applied later based on each stage's invert mode)
smooth_sigma = 20
Vs_init  = 1 / gaussian_filter(1 / Vs_true, smooth_sigma)
Vp_init  = 1 / gaussian_filter(1 / Vp_true, smooth_sigma)
rho_init = gaussian_filter(rho_true, smooth_sigma)

# %%
# ── Plot true models ────────────────────────────────────────────
true_models = (Vp_true, Vs_true, rho_true)

plt.figure(figsize=(26, 6))
for i, (m, title) in enumerate([
    (Vp_true, "True Vp"), (Vs_true, "True Vs"), (rho_true, "True Rho")
]):
    plt.subplot(1, 3, i + 1)
    plt.imshow(m, cmap='jet')
    plt.title(title, fontsize=8)
    plt.colorbar()
plt.tight_layout()
plt.show()

# ── Plot initial models ────────────────────────────────────────
plt.figure(figsize=(26, 6))
for i, (m, title) in enumerate([
    (Vp_init, "Initial Vp"), (Vs_init, "Initial Vs"), (rho_init, "Initial Rho")
]):
    plt.subplot(1, 3, i + 1)
    plt.imshow(m, cmap='jet')
    plt.title(title, fontsize=8)
    plt.colorbar()
plt.tight_layout()
plt.show()

# %%
# ═════════════════════════════════════════════════════════════════
# 2. ACQUISITION
# ═════════════════════════════════════════════════════════════════
# Component to invert: 0=vx, 1=vz, 2=exx, 3=ezz
component = 2  

# Source Locations: 1 every 15 grid points, starting from x=2 
src_x = np.arange(0, nx - 2, 15) + 2
src_z = np.zeros(len(src_x))

# Receiver Locations: 1 every 2 grid points, starting from x=0
rec_x = np.arange(0, nx, 2)
rec_z = np.zeros(len(rec_x), dtype=int)

# Plot acquisition geometry and save acquisition plot (See in results folder)
save_acquisition_plot(true_models, src_x, src_z, rec_x, rec_z, setup_dir, dx=dx, dz=dz)
print(f"  Acquisition plot saved to {setup_dir}/")

# %%
# ═════════════════════════════════════════════════════════════════
# 3. GRID PARAMETERS
# ═════════════════════════════════════════════════════════════════
# PML grid padding  
pad        = 30
# For the forward modeling, we choose block_size=200 by checking point method to reduce the GPU memory usage.
block_size = 200

# %%
# ═════════════════════════════════════════════════════════════════
# 4. MULTISCALE STAGES
# ═════════════════════════════════════════════════════════════════
# Time for forward modeling
tmax = 6  # seconds

# CFL stability limit for staggered-grid FD
fd_order = 8

if fd_order == 8:
    S = 1225/1024 + 245/3072 + 49/5120 + 5/7168  # ~1.286
elif fd_order >= 4:
    S = 9/8 + 1/24                                 # ~1.167
else:
    S = 1.0

dt = 0.9 / (S * Vp_true.max() * np.sqrt(1/dx**2 + 1/dz**2))
nt = int(tmax / dt)

# Multiscale stages: simultaneous Vs+Vp inversion, increasing frequency
stages = [
    dict(f_c=6, cutoff=3,  nt=nt, n_epochs=200, invert='vs', save_dir=os.path.join(result_dir, 'stage1_cut3')),
    dict(f_c=6, cutoff=15,  nt=nt, n_epochs=200, invert='vs', save_dir=os.path.join(result_dir, 'stage2_cut15')),
]

# %%
# ═════════════════════════════════════════════════════════════════
# 5. GENERATE OBSERVED DATA
# ═════════════════════════════════════════════════════════════════
obs_f_c = 6
run_shot_obs = build_forward_fn(nz, nx, dx, dz, dt, nt, obs_f_c, pad, block_size,
                                rec_x, rec_z, fd_order=fd_order)

t0_obs      = 1.5 / obs_f_c
src_wavelet = ricker_jax(jnp.arange(nt) * dt, obs_f_c, t0_obs)

Vp_j  = jnp.array(Vp_true)
Vs_j  = jnp.array(Vs_true)
rho_j = jnp.array(rho_true)

print(f"Generating observed data (fc={obs_f_c} Hz) ...")
src_x_arr = jnp.array(src_x, dtype=jnp.int32)
src_z_arr = jnp.array(src_z, dtype=jnp.int32)

all_obs = jax.jit(jax.vmap(run_shot_obs, in_axes=(None, None, None, 0, 0, None)))(
    Vs_j, Vp_j, rho_j, src_x_arr, src_z_arr, src_wavelet
)
observed = all_obs[component]  # (n_shots, nt, n_rec)
print(f"  All {len(src_x)} shots done")

# ── Save wavelet plot ───────────────────────────────────────────
save_wavelet_plot(np.array(src_wavelet), dt, setup_dir, fc=obs_f_c)
print(f"  Wavelet plot saved to {setup_dir}/")

# ── QC: shot gathers before/after filter ────────────────────────
save_shot_qc(observed, stages, rec_x, dt, nt, obs_f_c, setup_dir)

# %%
# ═════════════════════════════════════════════════════════════════
# 6. SOURCE TAPER
# ═════════════════════════════════════════════════════════════════
srt_radius    = 0.5 * Vp_true.max() // 10
src_positions = list(zip(src_x.astype(int), src_z.astype(int)))
source_taper  = build_source_taper(nz, nx, dx, src_positions, srt_radius, filt_size=0)
source_taper_jax = jnp.array(source_taper, dtype=jnp.float32)

save_source_taper_plot(source_taper, dx, dz, src_positions,
                       os.path.join(setup_dir, "source_taper.png"))

# %%
# ═════════════════════════════════════════════════════════════════
# 7. MAIN INVERSION LOOP
# ═════════════════════════════════════════════════════════════════
early_stop_ratio = 0.001   # stop when loss/loss_init < this (None to disable)
tikhonov_alpha   = 0.0001  # Tikhonov weight (None to disable)
illumination_eps = 0.001   # illumination stabilization (None to disable)
smooth_sigma = 15      # Gaussian smoothing sigma for illumination of fixed parameters
# Initial parameters: always start with init for params that WILL be inverted in any stage
any_invert_vp  = any(s['invert'] in ('vs_vp', 'all') for s in stages)
any_invert_rho = any(s['invert'] == 'all' for s in stages)

params = (jnp.array(Vs_init),
          jnp.array(Vp_init if any_invert_vp else Vp_true),
          jnp.array(rho_init if any_invert_rho else rho_true))

# Counter for total time
t_start_all      = time.time()
all_stage_losses = []

for stage in stages:
    f_c          = stage['f_c']
    cutoff       = stage['cutoff']
    nt_stage     = stage['nt']
    stage_invert = stage['invert']

    print(f"\n{'='*60}")
    print(f"  Stage: f_c={f_c} Hz  cutoff={cutoff} Hz  invert={stage_invert}  nt={nt_stage}  dt={dt:.6f}s")
    print(f"{'='*60}")

    # Butterworth lowpass filter
    sos = butter(6, cutoff, fs=1/dt, output='sos')

    # Forward operator for this stage
    run_shot = build_forward_fn(nz, nx, dx, dz, dt, nt_stage, f_c, pad, block_size,
                                rec_x, rec_z, fd_order=fd_order)

    # Source wavelet
    t0_stage         = 1.5 / f_c
    src_wavelet_stage = ricker_jax(jnp.arange(nt_stage) * dt, f_c, t0_stage)

    n_taper    = int(0.1 * nt_stage)
    auto_scale = 1.0 / (jnp.mean(observed ** 2) + 1e-45)
    print(f"  Auto scale: {auto_scale:.4e}")

    # Illumination compensation - for scaling the gradient each stages (See in results folder)
    # Inverted params → use current estimate directly; fixed params → smooth from true
    illum = None
    if illumination_eps is not None:
        Vs_illum  = params[0]
        Vp_illum  = params[1] if stage_invert in ('vs_vp', 'all') else jnp.array(1 / gaussian_filter(1 / Vp_true, smooth_sigma))
        rho_illum = params[2] if stage_invert == 'all' else jnp.array(gaussian_filter(rho_true, smooth_sigma))
        illum = compute_illumination_for_stage(
            nz, nx, dx, dz, dt, nt_stage, f_c, pad, block_size,
            rec_x, rec_z, fd_order, component,
            Vs_illum, Vp_illum, rho_illum,
            src_x, src_z, src_wavelet_stage,
            eps=illumination_eps,
        )
        save_illumination_plot(illum, dx, dz,
                               os.path.join(setup_dir, f'illumination_fc{f_c}Hz_cut{cutoff}Hz.png'),
                               f_c=f_c, eps=illumination_eps)


    # ==================== Loss function ============================
    loss_fn = build_loss_fn(
        run_shot, observed, src_x, src_z, src_wavelet_stage, component,
        scale=auto_scale, invert=stage_invert,
        sos=sos, n_taper=n_taper,
        source_taper=None , grad_smooth_sigma=1,
        tikhonov_alpha=None, illumination=illum,
    )

    # L-BFGS optimization
    t_stage_start = time.time()
    params, loss_hist = run_lbfgs(
        loss_fn, params, stage['n_epochs'],
        save_every   = 10,
        save_dir     = stage['save_dir'],
        true_models  = true_models,
        dx           = dx,
        dz           = dz,
        run_shot     = run_shot,
        observed     = observed,
        src_x_list   = src_x,
        src_z        = src_z,
        src_wavelet  = src_wavelet_stage,
        dt           = dt,
        component    = component,
        invert       = stage_invert,
        early_stop_ratio = early_stop_ratio,
    )

    # ==================== End of stage ============================
    t_stage = time.time() - t_stage_start

    label = f"fc={f_c}Hz cut={cutoff}Hz"
    all_stage_losses.append((label, loss_hist))
    print(f"  Stage done in {t_stage/60:.1f} min ({t_stage:.0f} s)")

t_total = time.time() - t_start_all
print(f"\nMultiscale FWI complete. Total time: {t_total/60:.1f} min ({t_total:.0f} s)")

save_multistage_loss_curve(all_stage_losses, os.path.join(setup_dir, 'loss_convergence.png'))
print(f"  Multistage loss curve saved to {setup_dir}/loss_convergence.png")

# %%
# ═════════════════════════════════════════════════════════════════
# 8. SAVE INVERSION RESULTS
# ═════════════════════════════════════════════════════════════════
inv_dir = os.path.join(result_dir, 'inverted_results')
os.makedirs(inv_dir, exist_ok=True)

Vs_inv, Vp_inv, rho_inv = params
np.save(os.path.join(inv_dir, 'Vs_final.npy'),  np.array(Vs_inv))
np.save(os.path.join(inv_dir, 'Vp_final.npy'),  np.array(Vp_inv))
np.save(os.path.join(inv_dir, 'rho_final.npy'), np.array(rho_inv))

info = f"""FWI Inversion Info
==================
Grid: nz={nz}, nx={nx}, dx={dx}, dz={dz}
tmax={tmax}s, dt={dt:.6f}s, nt={nt}
fd_order={fd_order}
pad={pad}, block_size={block_size}
component={component}  (0=vx, 1=vz, 2=exx, 3=ezz)

Sources: {len(src_x)} shots, x={src_x[0]}..{src_x[-1]} (step {src_x[1]-src_x[0]}), z=0
Receivers: {len(rec_x)} recs, x={rec_x[0]}..{rec_x[-1]} (step {rec_x[1]-rec_x[0]}), z=0

Stages:
"""
for i, stage in enumerate(stages):
    info += f"  {i+1}. f_c={stage['f_c']}Hz, cutoff={stage['cutoff']}Hz, invert={stage['invert']}, n_epochs={stage['n_epochs']}\n"
info += f"\nTotal time: {t_total/60:.1f} min\n"

with open(os.path.join(inv_dir, 'inversion_info.txt'), 'w') as f:
    f.write(info)
print(f"  Final model + info saved to {inv_dir}/")
