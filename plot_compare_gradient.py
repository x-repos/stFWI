"""
plot_compare_gradient.py — render AD vs hand-coded gradient comparison
from `Results_gradient/gradient_compare.npz`.

Three stacked imshow panels (matches the style of plot_gradient.py):
    1) AD ∂L/∂Vs
    2) hand-coded ∂L/∂Vs
    3) Vs_init − Vs_true   (target update direction)
"""
from __future__ import annotations

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt


# ── Plotting knobs (tweak freely) ────────────────────────────────────────
GAIN_AD     = 0.1    # raw AD gradient (top)
GAIN_HC     = 0.1    # hand-coded gradient (middle)
GAIN_MODEL  = 0.15    # Vs_init − Vs_true (bottom)
Q_GRAD  = 0.99
Q_MODEL = 0.995
CMAP    = "seismic"

OUT_PNG = "gradient_compare.png"
DPI     = 140
FIGSIZE = (12, 10)

SRC_KW = dict(marker='*', s=80, c='yellow', edgecolors='black', zorder=5)
REC_KW = dict(marker='v', s=8,  c='lime',   edgecolors='black', linewidths=0.4, zorder=5)


def sym_clip(arr, q=0.99, gain=1.0):
    return float(np.quantile(np.abs(arr), q) * gain)


def render(npz_path, out_path):
    d = np.load(npz_path)
    g_AD = d["g_Vs_AD"]
    g_HC = d["g_Vs_HC"]
    Vs_init = d["Vs_init"]
    Vs_true = d["Vs_true"]
    src_x = np.atleast_1d(d["src_x"]).astype(float)
    src_z = np.atleast_1d(d["src_z"]).astype(float)
    rec_x = d["rec_x"];  rec_z = d["rec_z"]
    dx, dz = float(d["dx"]), float(d["dz"])
    nz, nx = int(d["nz"]), int(d["nx"])
    nt = int(d["nt"]); fc = float(d["f_c"])
    n_shots = int(d.get("n_shots", len(src_x)))

    extent = [0, nx*dx/1000, nz*dz/1000, 0]

    fig, axes = plt.subplots(3, 1, figsize=FIGSIZE, constrained_layout=True)

    vmax = sym_clip(g_AD, q=Q_GRAD, gain=GAIN_AD)
    im0 = axes[0].imshow(g_AD, cmap=CMAP, vmin=-vmax, vmax=vmax,
                         extent=extent, aspect="auto")
    axes[0].set_title(f"∂L/∂Vs   (AD, raw, no preconditioning)   "
                      f"n_shots={n_shots}, nt={nt}, fc={fc:g} Hz")
    fig.colorbar(im0, ax=axes[0])

    vmax = sym_clip(g_HC, q=Q_GRAD, gain=GAIN_HC)
    im1 = axes[1].imshow(g_HC, cmap=CMAP, vmin=-vmax, vmax=vmax,
                         extent=extent, aspect="auto")
    axes[1].set_title("∂L/∂Vs   (adjoint + imaging condition)")
    fig.colorbar(im1, ax=axes[1])

    dvs = Vs_init - Vs_true
    vmax = sym_clip(dvs, q=Q_MODEL, gain=GAIN_MODEL)
    im2 = axes[2].imshow(dvs, cmap=CMAP, vmin=-vmax, vmax=vmax,
                         extent=extent, aspect="auto")
    axes[2].set_title("Vs_init − Vs_true   (target update direction)")
    fig.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.set_xlabel("x [km]")
        ax.set_ylabel("z [km]")
        ax.scatter(src_x * dx / 1000, src_z * dz / 1000, **SRC_KW)
        ax.scatter(np.asarray(rec_x) * dx / 1000,
                   np.asarray(rec_z) * dz / 1000, **REC_KW)

    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default="Results_gradient/gradient_compare.npz")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out = args.out or os.path.join(os.path.dirname(args.npz) or ".", OUT_PNG)
    render(args.npz, out)


if __name__ == "__main__":
    main()
