"""
plot_first_gradient_no_illum.py — render the first-FWI gradients with the
illumination preconditioner H_inv UNDONE (i.e., divided back out).

Reads `Results_first_gradient/first_gradient.npz` and produces a plot
showing the gradients as if `_postprocess` had been run with
`illumination=None`. The Gaussian smoothing remains applied.

4 stacked panels:
    1) ∂L/∂Vs  (AD,        smoothed, NO illumination)
    2) ∂L/∂Vs  (adjoint+imaging, smoothed, NO illumination)
    3) H_inv  (the multiplier we just divided out — for reference)
    4) Vs_init − Vs_true
"""
from __future__ import annotations

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt


# ── Plotting knobs ───────────────────────────────────────────────────────
GAIN_GRAD  = 0.05
GAIN_MODEL = 0.15
Q_GRAD  = 0.99
Q_MODEL = 0.995
CMAP_DIVERGING = "seismic"
CMAP_ILLUM     = "jet"

OUT_PNG = "first_gradient_no_illum.png"
DPI     = 140
FIGSIZE = (12, 14)

SRC_KW = dict(marker='*', s=80, c='yellow', edgecolors='black', zorder=5)
REC_KW = dict(marker='v', s=8,  c='lime',   edgecolors='black', linewidths=0.4, zorder=5)


def sym_clip(arr, q=0.99, gain=1.0):
    return float(np.quantile(np.abs(arr), q) * gain)


def render(npz_path, out_path):
    d = np.load(npz_path)
    g_AD_pre = d["g_Vs_AD"]                                 # smoothed × H_inv
    g_HC_pre = d["g_Vs_HC"] if "g_Vs_HC" in d.files else None
    illum    = d["illumination"]
    Vs_init  = d["Vs_init"]
    Vs_true  = d["Vs_true"]
    src_x    = np.atleast_1d(d["src_x"]).astype(float)
    src_z    = np.atleast_1d(d["src_z"]).astype(float)
    rec_x    = d["rec_x"]
    rec_z    = d["rec_z"]
    dx, dz   = float(d["dx"]), float(d["dz"])
    nz, nx   = int(d["nz"]), int(d["nx"])
    n_shots  = int(d["n_shots"])
    f_c      = float(d["f_c"])
    cutoff   = float(d["cutoff"])

    # Crop illumination if it's still on the padded grid (defensive)
    if illum.shape != (nz, nx):
        pad = (illum.shape[1] - nx) // 2
        illum_phys = illum[:nz, pad:pad + nx]
    else:
        illum_phys = illum

    # Undo the H_inv multiplication.  H_inv is always > 0 thanks to the
    # ε·max(W_s) regulariser, so dividing is numerically safe.
    g_AD = g_AD_pre / illum_phys
    g_HC = g_HC_pre / illum_phys if g_HC_pre is not None else None

    extent = [0, nx * dx / 1000, nz * dz / 1000, 0]

    has_hc = g_HC is not None
    n_panels = 4 if has_hc else 3
    fig, axes = plt.subplots(n_panels, 1, figsize=FIGSIZE, constrained_layout=True)

    # Shared gradient colour-bar across the AD and adjoint panels
    if has_hc:
        vmax_grad = max(sym_clip(g_AD, Q_GRAD, GAIN_GRAD),
                        sym_clip(g_HC, Q_GRAD, GAIN_GRAD))
    else:
        vmax_grad = sym_clip(g_AD, Q_GRAD, GAIN_GRAD)

    im0 = axes[0].imshow(g_AD, cmap=CMAP_DIVERGING, vmin=-vmax_grad, vmax=vmax_grad,
                         extent=extent, aspect="auto")
    axes[0].set_title(f"∂L/∂Vs   AD   (smoothed, NO H_inv applied)   "
                      f"n_shots={n_shots}, fc={f_c:g} Hz, cutoff={cutoff:g} Hz")
    fig.colorbar(im0, ax=axes[0])
    nxt = 1

    if has_hc:
        im1 = axes[nxt].imshow(g_HC, cmap=CMAP_DIVERGING,
                               vmin=-vmax_grad, vmax=vmax_grad,
                               extent=extent, aspect="auto")
        axes[nxt].set_title("∂L/∂Vs   adjoint + imaging   (smoothed, NO H_inv applied)")
        fig.colorbar(im1, ax=axes[nxt])
        nxt += 1

    im2 = axes[nxt].imshow(illum_phys, cmap=CMAP_ILLUM, vmin=0, vmax=1,
                           extent=extent, aspect="auto")
    axes[nxt].set_title("H_inv  (just for reference; divided OUT of panels 1 & 2)")
    fig.colorbar(im2, ax=axes[nxt])
    nxt += 1

    dvs = Vs_init - Vs_true
    vmax = sym_clip(dvs, q=Q_MODEL, gain=GAIN_MODEL)
    im3 = axes[nxt].imshow(dvs, cmap=CMAP_DIVERGING, vmin=-vmax, vmax=vmax,
                           extent=extent, aspect="auto")
    axes[nxt].set_title("Vs_init − Vs_true   (target update direction)")
    fig.colorbar(im3, ax=axes[nxt])

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
    p.add_argument("--npz", default="Results_first_gradient/first_gradient.npz")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out = args.out or os.path.join(os.path.dirname(args.npz) or ".", OUT_PNG)
    render(args.npz, out)


if __name__ == "__main__":
    main()
