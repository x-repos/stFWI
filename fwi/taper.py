"""
fwi/taper.py — Gradient tapers for FWI.

- Source taper: damp gradient artifacts near source positions (DENISE style).

Author: Minh Nhat Tran
Date: 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def build_source_taper(nz, nx, dh, src_positions, srt_radius, filt_size=2):
    """
    Build a 2-D source taper to damp gradient artifacts near source positions.

    Parameters
    ----------
    nz, nx : int
        Grid dimensions (depth, lateral).
    dh : float
        Grid spacing (m), assumed equal in x and z.
    src_positions : list of (src_x_grid, src_z_grid)
        Source positions in **grid indices** (0-based).
    srt_radius : float
        Taper radius around each source (m).
    filt_size : int
        Half-width (grid points) of the hard-zero box stamped on each source.

    Returns
    -------
    taper : ndarray, shape (nz, nx), values in [0, 1]
    """
    from scipy.special import erf

    # 1. Build local erf-based radial patch
    patch_size = max(int(np.ceil(2.0 * srt_radius / dh)), 5)
    center = patch_size / 2.0
    maxrad = np.sqrt(2.0) * srt_radius

    jj, ii = np.mgrid[0:patch_size, 0:patch_size]
    x = (ii - center + 0.5) * dh
    y = (jj - center + 0.5) * dh
    rad = np.sqrt(x**2 + y**2)
    patch = erf(rad / maxrad)

    # Normalize: shift so min=0, then scale by edge-center values
    patch -= patch.min()
    mid = patch_size // 2
    edge_vals = [patch[0, mid], patch[mid, 0], patch[mid, -1], patch[-1, mid]]
    patch /= max(edge_vals)
    patch = np.clip(patch, 0.0, 1.0)

    ijc = patch_size // 2

    # 2. Stamp patch onto global grid (take minimum across all sources)
    taper = np.ones((nz, nx), dtype=np.float64)

    for src_x, src_z in src_positions:
        for pj in range(patch_size):
            for pi in range(patch_size):
                gx = src_x + pi - ijc
                gz = src_z + pj - ijc
                if 0 <= gx < nx and 0 <= gz < nz:
                    taper[gz, gx] = min(taper[gz, gx], patch[pj, pi])

    # 3. Normalize to [0, 1]
    taper -= taper.min()
    if taper.max() > 0:
        taper /= taper.max()

    # 4. Hard-zero a small box around each source
    for src_x, src_z in src_positions:
        z0 = max(src_z - filt_size, 0)
        z1 = min(src_z + filt_size + 1, nz)
        x0 = max(src_x - filt_size, 0)
        x1 = min(src_x + filt_size + 1, nx)
        taper[z0:z1, x0:x1] = 0.0

    return taper



def save_source_taper_plot(taper, dx, dz, src_positions, save_path):
    """Save a 2-D image of the source taper with source positions marked."""
    nz, nx = taper.shape
    extent = [0, nx * dx, nz * dz, 0]

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(taper, cmap='RdYlGn', vmin=0, vmax=1, extent=extent, aspect='auto')
    plt.colorbar(im, ax=ax, label='Taper weight')

    # Mark source positions
    sx = [s[0] * dx for s in src_positions]
    sz = [s[1] * dz for s in src_positions]
    ax.scatter(sx, sz, c='red', marker='*', s=80, edgecolors='k', linewidths=0.5, label='Sources')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title('Source Taper (gradient damping near sources)')
    ax.legend(loc='lower right')

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Source taper plot saved to {save_path}")
