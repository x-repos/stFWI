"""
tmp.py — quick visualization of backward (adjoint) wave propagation
on a HOMOGENEOUS half-space, using fwi/adjoint_jax_kernel.adjoint_jax.

Pipeline:
  1. forward Ricker shot at one source, record (vx, vz, εxx, εzz) at receivers
  2. inject the recorded vz back as the adjoint receiver data (rec_vz_a)
     and run the hand-coded adjoint with return_wavefields=True
  3. animate the backward V_a wavefield with src/rec markers,
     save as backward_wavefield.gif
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU is fine for this size

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.axes_grid1 import make_axes_locatable

from fwi.forward import forward_jax, ricker_jax
from fwi.adjoint_jax_kernel import adjoint_jax


# ── Setup ────────────────────────────────────────────────────────────────
nz_dom, nx_dom = 80, 120
dz, dx = 10.0, 10.0
pad = 20
nt = 400
dt = 1.0e-3
fc = 15.0

fd_order = 4
free_surface = False
src_type = 'force'

# Homogeneous half-space (sediment-like)
Vs  = np.full((nz_dom, nx_dom), 1500.0)
Vp  = np.full((nz_dom, nx_dom), 2500.0)
rho = np.full((nz_dom, nx_dom), 2000.0)

# Acquisition: source slightly buried, receivers along top surface
src_x_int = nx_dom // 2
src_z_int = 5
rec_x_int = np.arange(8, nx_dom - 8, 6, dtype=np.int32)
rec_z_int = np.full_like(rec_x_int, src_z_int)   # receivers at source depth
nrec = len(rec_x_int)

# Ricker source wavelet
t = jnp.arange(nt) * dt
src_wavelet = ricker_jax(t, fc, 1.5 / fc)

# ── 1) Forward shot ──────────────────────────────────────────────────────
print(f"Forward  ({nt} steps)...")
rec_vx, rec_vz, rec_ex, rec_ez = forward_jax(
    Vs, Vp, rho,
    src_x=int(src_x_int), src_z=int(src_z_int),
    rec_x=jnp.array(rec_x_int), rec_z=jnp.array(rec_z_int),
    nx_dom=nx_dom, nz_dom=nz_dom,
    dx=dx, dz=dz, dt=dt, nt=nt, fc=fc, pad=pad,
    src_wavelet=src_wavelet, block_size=nt,
    fd_order=fd_order, free_surface=free_surface, src_type=src_type,
)

# ── 2) Backward (adjoint) propagation ────────────────────────────────────
# Use the recorded vz as the adjoint receiver data; zero on the others.
print(f"Adjoint  ({nt} steps)...")
zero_rec = jnp.zeros((nt, nrec))
src_wav_a, U_a_hist, V_a_hist, ex_a_hist, ez_a_hist, es_a_hist = adjoint_jax(
    Vs, Vp, rho,
    src_x=int(src_x_int), src_z=int(src_z_int),
    rec_x=jnp.array(rec_x_int), rec_z=jnp.array(rec_z_int),
    nx_dom=nx_dom, nz_dom=nz_dom,
    dx=dx, dz=dz, dt=dt, nt=nt, fc=fc, pad=pad,
    rec_vx_a=zero_rec,
    rec_vz_a=rec_vz,            # adjoint "data" = forward recorded vz
    rec_ex_a=zero_rec,
    rec_ez_a=zero_rec,
    fd_order=fd_order, free_surface=free_surface, src_type=src_type,
    return_wavefields=True,
)

# ── 3) Slice physical domain, reverse to backward-time, animate ──────────
z_start = 0 if free_surface else pad


def to_phys(hist):
    a = np.asarray(hist)[:, z_start:z_start + nz_dom, pad:pad + nx_dom]
    return a[::-1]   # reverse to backward-time order


U_a  = to_phys(U_a_hist)    # vx adjoint
V_a  = to_phys(V_a_hist)    # vz adjoint
EX_a = to_phys(ex_a_hist)   # εxx adjoint
EZ_a = to_phys(ez_a_hist)   # εzz adjoint

stride = 4
fields = [U_a[::stride], V_a[::stride], EX_a[::stride], EZ_a[::stride]]
labels = [r"Backward $V_x$", r"Backward $V_z$",
          r"Backward $\varepsilon_{xx}$", r"Backward $\varepsilon_{zz}$"]
n_frames = fields[0].shape[0]

# Per-subplot symmetric colour limit (each component has its own scale)
vmaxes = [0.15 * np.max(np.abs(f)) for f in fields]

extent = [0, nx_dom * dx, nz_dom * dz, 0]
fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
axes = axes.ravel()
ims, sub_titles = [], []
for ax, f, lab, vmax in zip(axes, fields, labels, vmaxes):
    im = ax.imshow(f[0], cmap="seismic", vmin=-vmax, vmax=vmax,
                   aspect="equal", extent=extent, interpolation="bilinear")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.08)
    fig.colorbar(im, cax=cax)
    ax.scatter([src_x_int * dx], [src_z_int * dz], marker="*", s=180,
               c="yellow", edgecolors="black", linewidths=0.8, zorder=5)
    ax.scatter(rec_x_int * dx, rec_z_int * dz, marker="v", s=25,
               c="lime", edgecolors="black", linewidths=0.5, zorder=5)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    sub_titles.append(ax.set_title(lab))
    ims.append(im)

suptitle = fig.suptitle("")


def update(i):
    backward_it = nt - 1 - i * stride
    for im, f in zip(ims, fields):
        im.set_data(f[i])
    suptitle.set_text(f"reverse time step {backward_it}/{nt-1}")
    return ims + [suptitle]


print(f"Animating  ({n_frames} frames)...")
anim = FuncAnimation(fig, update, frames=n_frames, interval=40, blit=False)
out = "backward_wavefield.gif"
anim.save(out, writer=PillowWriter(fps=25))
plt.close(fig)
print(f"Saved {out}")
