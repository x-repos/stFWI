"""
plot_gradient.py — render the gradient figure from
`Results_gradient/gradient_data.npz` produced by compute_gradient.py.

Iterate on colour scaling / panel selection here; you do NOT need to
re-run the (slow) gradient computation.
"""
from __future__ import annotations

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt


# ── Plotting knobs (tweak freely) ────────────────────────────────────────
# Per-panel saturation gains (smaller = more punchy; 1.0 = q-th percentile).
GAIN_AD     = 0.005   # raw AD gradient (top panel)
GAIN_MODEL  = 0.15    # Vs_init − Vs_true reference (bottom panel)
Q_GRAD  = 0.99  # percentile for gradient panel
Q_MODEL = 0.995 # percentile for the Vs_init − Vs_true reference panel
CMAP    = "seismic"

OUT_PNG = "gradient_panels.png"
DPI     = 140
FIGSIZE = (12, 7)

# Source / receiver markers
SRC_KW = dict(marker='*', s=80, c='yellow', edgecolors='black', zorder=5)
REC_KW = dict(marker='v', s=8,  c='lime',   edgecolors='black', linewidths=0.4, zorder=5)


# ── Helpers ───────────────────────────────────────────────────────────────
def sym_clip(arr, q=0.99, gain=1.0):
    """symmetric percentile clip for diverging colormap."""
    return float(np.quantile(np.abs(arr), q) * gain)


def render(npz_path, out_path):
    d = np.load(npz_path)
    g_vs     = d["g_vs"]
    Vs_init  = d["Vs_init"]
    Vs_true  = d["Vs_true"]
    src_x    = d["src_x"]
    src_z    = d["src_z"]
    rec_x    = d["rec_x"]
    rec_z    = d["rec_z"]
    dx       = float(d["dx"])
    dz       = float(d["dz"])
    nz       = int(d["nz"])
    nx       = int(d["nx"])
    n_shots  = int(d["n_shots"])
    f_c      = float(d["f_c"])
    cutoff   = float(d["cutoff"])

    extent = [0, nx * dx / 1000, nz * dz / 1000, 0]

    fig, axes = plt.subplots(2, 1, figsize=FIGSIZE, constrained_layout=True)

    vmax = sym_clip(g_vs, q=Q_GRAD, gain=GAIN_AD)
    im0 = axes[0].imshow(g_vs, cmap=CMAP, vmin=-vmax, vmax=vmax,
                         extent=extent, aspect="auto")
    axes[0].set_title(f"∂L/∂Vs (raw AD, no preconditioning)  —  εxx misfit  "
                      f"(n_shots={n_shots}, fc={f_c:g} Hz, cutoff={cutoff:g} Hz)")
    fig.colorbar(im0, ax=axes[0])

    dvs = Vs_init - Vs_true
    vmax = sym_clip(dvs, q=Q_MODEL, gain=GAIN_MODEL)
    im1 = axes[1].imshow(dvs, cmap=CMAP, vmin=-vmax, vmax=vmax,
                         extent=extent, aspect="auto")
    axes[1].set_title("Vs_init − Vs_true   (target update direction)")
    fig.colorbar(im1, ax=axes[1])

    for ax in axes:
        ax.set_xlabel("x [km]")
        ax.set_ylabel("z [km]")
        ax.scatter(np.asarray(src_x) * dx / 1000,
                   np.asarray(src_z) * dz / 1000, **SRC_KW)
        ax.scatter(np.asarray(rec_x) * dx / 1000,
                   np.asarray(rec_z) * dz / 1000, **REC_KW)

    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default="Results_gradient/gradient_data.npz",
                   help="Input .npz from compute_gradient.py")
    p.add_argument("--out", default=None,
                   help="Output PNG path (default: alongside the .npz)")
    args = p.parse_args()

    out_path = args.out or os.path.join(os.path.dirname(args.npz) or ".", OUT_PNG)
    render(args.npz, out_path)


if __name__ == "__main__":
    main()
