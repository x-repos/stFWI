"""
fwi/illumination.py — Source illumination scaling for gradient preconditioning.

Author: Minh Nhat Tran
Date: 2026
"""
import os
import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .forward import build_forward_fn



def compute_illumination_for_stage(nz, nx, dx, dz, dt, nt, f_c, pad, block_size,
                                    rec_x, rec_z, fd_order, component,
                                    Vs, Vp, rho, src_x, src_z, src_wavelet, eps=0.001):
    """
    Compute source illumination for one stage (all shots).

    Returns
    -------
    H_inv : jax array (nz, nx), normalized to [0, 1].
    """
    comp_to_var = {0: 'vx_full', 1: 'vz_full', 2: 'ex_full', 3: 'ez_full'}
    run_shot = build_forward_fn(nz, nx, dx, dz, dt, nt, f_c, pad, block_size,
                                 rec_x, rec_z, fd_order=fd_order,
                                 return_vars=[comp_to_var[component]])

    src_x_arr = jnp.array(src_x, dtype=jnp.int32)
    src_z_arr = jnp.array(src_z, dtype=jnp.int32)

    nz_pad, nx_pad = nz + pad, nx + 2 * pad
    Ws = jnp.zeros((nz_pad, nx_pad))
    for i in range(len(src_x)):
        (field,) = run_shot(Vs, Vp, rho, src_x_arr[i], src_z_arr[i], src_wavelet)
        Ws = Ws + jnp.sum(field ** 2, axis=0)
        if (i + 1) % 10 == 0 or i == len(src_x) - 1:
            print(f"    Illumination: shot {i+1}/{len(src_x)}")

    Ws = Ws[:nz, pad:pad + nx]
    lamda = eps * jnp.max(Ws)
    H_inv = 1.0 / (Ws + lamda)
    H_inv = H_inv / jnp.max(H_inv)
    return H_inv


def save_illumination_plot(H_inv, dx, dz, save_path, f_c=None, eps=None):
    """Save illumination scaling H_inv as a 2D image + 1D depth profile."""
    H = np.array(H_inv)
    nz, nx = H.shape
    extent = [0, nx * dx / 1000, nz * dz / 1000, 0]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 4),
                                    gridspec_kw={'width_ratios': [3, 1]})

    im = ax0.imshow(H, aspect='auto', cmap='jet', extent=extent, vmin=0, vmax=1)
    title = 'Source Illumination Scaling'
    if f_c is not None:
        title += f' (fc={f_c}Hz)'
    if eps is not None:
        title += f' eps={eps}'
    ax0.set_title(title)
    ax0.set_xlabel('X (km)')
    ax0.set_ylabel('Z (km)')
    plt.colorbar(im, ax=ax0, label='H_inv (0–1)')

    # 1D depth profile (mean over x)
    depth_km = np.arange(nz) * dz / 1000
    ax1.plot(H.mean(axis=1), depth_km, 'k', lw=1.2)
    ax1.set_xlabel('Mean H_inv')
    ax1.set_ylabel('Z (km)')
    ax1.invert_yaxis()
    ax1.set_title('Depth profile')
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
