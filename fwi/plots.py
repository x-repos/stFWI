"""
fwi/plots.py — Visualization helpers for FWI.

All functions save figures to disk and close them immediately to avoid
memory leaks during long inversions.

Author: Minh Nhat Tran
Date: 2026
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker


# ─────────────────────────────────────────────────────────────────────────────
# Velocity / density comparison
# ─────────────────────────────────────────────────────────────────────────────

def _plot_one_param(true, init, inverted, name, extent_km, save_path):
    """Plot true / initial / inverted for one parameter (3 rows)."""
    true_km = true / 1000.0
    init_km = init / 1000.0
    inv_km  = inverted / 1000.0
    vmin, vmax = true_km.min(), true_km.max()
    unit = f'{name} (km/s)' if name != 'Rho' else f'{name} (g/cm³)'
    if name == 'Rho':
        true_km, init_km, inv_km = true, init, inverted
        vmin, vmax = true.min(), true.max()
        unit = f'{name} (kg/m³)'

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(8, 9))

    ax0.imshow(true_km, aspect='auto', cmap='jet',
               vmin=vmin, vmax=vmax, extent=extent_km)
    ax0.set_title(f'True {name}', fontsize=12)
    ax0.set_ylabel('Z (km)')
    ax0.tick_params(labelbottom=False)

    ax1.imshow(init_km, aspect='auto', cmap='jet',
               vmin=vmin, vmax=vmax, extent=extent_km)
    ax1.set_title(f'Initial {name}', fontsize=12)
    ax1.set_ylabel('Z (km)')
    ax1.tick_params(labelbottom=False)

    im = ax2.imshow(inv_km, aspect='auto', cmap='jet',
                    vmin=vmin, vmax=vmax, extent=extent_km)
    ax2.set_title(f'Inverted {name}', fontsize=12)
    ax2.set_xlabel('X (km)')
    ax2.set_ylabel('Z (km)')

    fig.colorbar(im, ax=[ax0, ax1, ax2], orientation='horizontal',
                 label=unit, fraction=0.04, pad=0.1)

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_velocity_plot(epoch, params, true_models, save_dir, dx=1.0, dz=1.0,
                       invert='all', init_params=None):
    """Save velocity plots: side-by-side subplots for multi-parameter inversion."""
    vs_est,  vp_est,  rho_est  = [np.array(p) for p in params]
    vp_true, vs_true, rho_true = true_models

    nz, nx = vs_true.shape
    nx_km = nx * dx / 1000.0
    nz_km = nz * dz / 1000.0
    extent_km = [0, nx_km, nz_km, 0]

    prefix = 'epoch_0000_initial' if epoch == 0 else f'epoch_{epoch:04d}'

    vs_init  = np.array(init_params[0]) if init_params is not None else vs_est
    vp_init  = np.array(init_params[1]) if init_params is not None else vp_est
    rho_init = np.array(init_params[2]) if init_params is not None else rho_est

    # Build list of (true, init, inverted, name) for each inverted param
    panels = [(vs_true, vs_init, vs_est, 'Vs')]
    if invert in ('vs_vp', 'all'):
        panels.append((vp_true, vp_init, vp_est, 'Vp'))
    if invert == 'all':
        panels.append((rho_true, rho_init, rho_est, 'Rho'))

    ncols = len(panels)
    fig, axes = plt.subplots(3, ncols, figsize=(8 * ncols, 9), squeeze=False)

    for col, (true, init, inv, name) in enumerate(panels):
        # Convert to display units
        if name == 'Rho':
            d_true, d_init, d_inv = true, init, inv
            vmin, vmax = true.min(), true.max()
            unit = f'{name} (kg/m³)'
        else:
            d_true, d_init, d_inv = true / 1000, init / 1000, inv / 1000
            vmin, vmax = true.min() / 1000, true.max() / 1000
            unit = f'{name} (km/s)'

        axes[0][col].imshow(d_true, aspect='auto', cmap='jet',
                            vmin=vmin, vmax=vmax, extent=extent_km)
        axes[0][col].set_title(f'True {name}', fontsize=12)

        axes[1][col].imshow(d_init, aspect='auto', cmap='jet',
                            vmin=vmin, vmax=vmax, extent=extent_km)
        axes[1][col].set_title(f'Initial {name}', fontsize=12)

        im = axes[2][col].imshow(d_inv, aspect='auto', cmap='jet',
                                 vmin=vmin, vmax=vmax, extent=extent_km)
        axes[2][col].set_title(f'Inverted {name}', fontsize=12)
        axes[2][col].set_xlabel('X (km)')

        # Y labels only on leftmost column
        if col == 0:
            for row in range(3):
                axes[row][col].set_ylabel('Z (km)')
        else:
            for row in range(3):
                axes[row][col].tick_params(labelleft=False)

        # Hide x labels on top two rows
        for row in range(2):
            axes[row][col].tick_params(labelbottom=False)

        # Colorbar per column
        fig.colorbar(im, ax=[axes[r][col] for r in range(3)],
                     orientation='horizontal', label=unit,
                     fraction=0.04, pad=0.1)

    fig.savefig(os.path.join(save_dir, f'{prefix}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Gradient
# ─────────────────────────────────────────────────────────────────────────────

def save_gradient_plot(epoch, grads, save_dir, dx=1.0, dz=1.0, invert='all'):
    """
    Show gradients only for inverted parameters.
    grads must be in the same order as params: (Vs_grad, Vp_grad, rho_grad).
    """
    vs_grad, vp_grad, rho_grad = [np.array(g) for g in grads]

    if invert == 'vs':
        panels = [(vs_grad, '∇Vs')]
    elif invert == 'vs_vp':
        panels = [(vs_grad, '∇Vs'), (vp_grad, '∇Vp')]
    else:
        panels = [(vp_grad, '∇Vp'), (vs_grad, '∇Vs'), (rho_grad, '∇ρ')]

    n = len(panels)
    nz, nx = vs_grad.shape
    nx_m   = nx * dx
    nz_m   = nz * dz
    extent = [0, nx_m, nz_m, 0]

    col_w  = 5.0
    img_h  = col_w * (nz_m / nx_m)
    row_h  = img_h + 1.2
    fig, axes = plt.subplots(1, n, figsize=(n * col_w, row_h), squeeze=False)
    for ax, (g, name) in zip(axes[0], panels):
        vmax = np.abs(g).max()
        vmax = vmax if vmax > 0 else 1.0
        im = ax.imshow(g, aspect='equal', cmap='jet',
                       vmin=-vmax, vmax=vmax, extent=extent)
        ax.set_title(f'Epoch {epoch:04d} — {name}  (max={vmax:.2e})')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f'gradient_{epoch:04d}.png'), dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Loss curve
# ─────────────────────────────────────────────────────────────────────────────

def save_loss_curve(loss_history, save_dir):
    """Normalized loss curve (starts at 100%, decreases)."""
    losses = np.array(loss_history)
    loss0  = losses[0]
    norm   = (losses / abs(loss0) * 100.0) if loss0 != 0 else np.ones_like(losses) * 100.0
    epochs = np.arange(1, len(losses) + 1)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(epochs, norm, linewidth=1.5, color='steelblue')
    ax.scatter(epochs[-1], norm[-1], color='crimson', s=50, zorder=5)
    ax.annotate(f'{norm[-1]:.2f}%',
                xy=(epochs[-1], norm[-1]),
                xytext=(-40, 10), textcoords='offset points',
                fontsize=10, color='crimson', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='crimson'))

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Normalized Loss (%)', fontsize=12)
    ax.set_title('FWI Loss Convergence', fontsize=14, fontweight='bold')
    ax.set_xlim(1, len(epochs))
    ax.set_ylim(bottom=0, top=105)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.0f%%'))
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'loss_curve.png'), dpi=150)
    plt.close(fig)


def save_multistage_loss_curve(stage_losses, save_path):
    """Plot all stages on one continuous loss curve, normalized to the very first loss value.

    Parameters
    ----------
    stage_losses : list of (label, loss_history)
        Each entry is (str, list[float]).  Stages are plotted sequentially.
    save_path : str
        Full file path for the output PNG.
    """
    # Global normalization: first loss value of the first stage
    loss0 = abs(stage_losses[0][1][0]) if stage_losses[0][1] else 1.0
    if loss0 == 0:
        loss0 = 1.0

    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(10, 4))
    epoch_offset = 0

    for idx, (label, hist) in enumerate(stage_losses):
        losses = np.array(hist)
        norm = losses / loss0 * 100.0
        epochs = np.arange(1, len(losses) + 1) + epoch_offset
        color = colors[idx % len(colors)]

        ax.plot(epochs, norm, linewidth=1.5, color=color, label=label)
        # Mark final value
        ax.scatter(epochs[-1], norm[-1], color=color, s=40, zorder=5)
        ax.annotate(f'{norm[-1]:.2f}%',
                    xy=(epochs[-1], norm[-1]),
                    xytext=(-40, 12), textcoords='offset points',
                    fontsize=9, color=color, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=color))

        # Draw stage boundary
        if idx > 0:
            ax.axvline(epoch_offset + 0.5, color='grey', ls=':', lw=0.8, alpha=0.6)

        epoch_offset += len(losses)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Normalized Loss (%)', fontsize=12)
    ax.set_title('FWI Multistage Loss Convergence', fontsize=14, fontweight='bold')
    ax.set_xlim(1, epoch_offset)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.0f%%'))
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Shot QC: raw vs filtered wiggle + spectrum
# ─────────────────────────────────────────────────────────────────────────────

def save_shot_qc(observed, stages, rec_x, dt, nt, obs_f_c, save_dir):
    """Plot shot 1 wiggles and spectra before/after Butterworth filter per stage."""
    from scipy.signal import butter, sosfilt

    shot1_raw = np.array(observed[0])
    t_axis = np.arange(nt) * dt
    freqs = np.fft.rfftfreq(nt, d=dt)

    for stage in stages:
        cutoff = stage['cutoff']
        sos_qc = butter(6, cutoff, fs=1/dt, output='sos')
        shot1_filt = sosfilt(sos_qc, shot1_raw, axis=0)

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        skip = max(1, len(rec_x) // 40)

        # Raw wiggle
        ax = axes[0, 0]
        for i in range(0, shot1_raw.shape[1], skip):
            tr = shot1_raw[:, i]
            peak = np.abs(tr).max() + 1e-30
            ax.plot(tr / peak * 0.8 * skip + rec_x[i], t_axis, 'k', lw=0.4)
        ax.set_title(f'Shot 1 — Raw (fc={obs_f_c}Hz)')
        ax.set_xlabel('Receiver x (grid)')
        ax.set_ylabel('Time (s)')
        ax.invert_yaxis()

        # Filtered wiggle
        ax = axes[0, 1]
        for i in range(0, shot1_filt.shape[1], skip):
            tr = shot1_filt[:, i]
            peak = np.abs(tr).max() + 1e-30
            ax.plot(tr / peak * 0.8 * skip + rec_x[i], t_axis, 'b', lw=0.4)
        ax.set_title(f'Shot 1 — Filtered (cutoff={cutoff}Hz)')
        ax.set_xlabel('Receiver x (grid)')
        ax.set_ylabel('Time (s)')
        ax.invert_yaxis()

        # Raw spectrum
        ax = axes[1, 0]
        spec_raw = np.abs(np.fft.rfft(shot1_raw, axis=0)).mean(axis=1)
        ax.plot(freqs, spec_raw / spec_raw.max(), 'k', lw=1.2)
        ax.set_title('Spectrum — Raw')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Normalised amp')
        ax.set_xlim(0, min(60, freqs[-1]))
        ax.axvline(obs_f_c, color='r', ls='--', label=f'fc={obs_f_c}Hz')
        ax.axvline(cutoff, color='b', ls='--', label=f'cutoff={cutoff}Hz')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Filtered spectrum
        ax = axes[1, 1]
        spec_filt = np.abs(np.fft.rfft(shot1_filt, axis=0)).mean(axis=1)
        ax.plot(freqs, spec_filt / (spec_raw.max() + 1e-30), 'b', lw=1.2)
        ax.set_title(f'Spectrum — After cutoff={cutoff}Hz')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Normalised amp (same scale)')
        ax.set_xlim(0, min(60, freqs[-1]))
        ax.axvline(cutoff, color='b', ls='--', label=f'cutoff={cutoff}Hz')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, f'shot1_qc_cut{cutoff}Hz.png'), dpi=150)
        plt.close(fig)
        print(f"  Shot QC plot saved: shot1_qc_cut{cutoff}Hz.png")


# ─────────────────────────────────────────────────────────────────────────────
# Acquisition geometry
# ─────────────────────────────────────────────────────────────────────────────

def save_acquisition_plot(true_models, src_x, src_z, rec_x, rec_z,
                          save_dir, dx=1.0, dz=1.0):
    """Plot source-receiver geometry overlaid on the true Vs model."""
    vp_true, vs_true, rho_true = true_models
    nz, nx = vs_true.shape
    nx_km = nx * dx / 1000.0
    nz_km = nz * dz / 1000.0
    extent_km = [0, nx_km, nz_km, 0]

    vs_km = vs_true / 1000.0

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(vs_km, aspect='auto', cmap='jet',
                   vmin=vs_km.min(), vmax=vs_km.max(), extent=extent_km)

    # Convert grid indices to km
    src_x_km = np.asarray(src_x) * dx / 1000.0
    src_z_km = np.asarray(src_z) * dz / 1000.0
    rec_x_km = np.asarray(rec_x) * dx / 1000.0
    rec_z_km = np.asarray(rec_z) * dz / 1000.0

    ax.scatter(rec_x_km, rec_z_km, c='black', s=10, marker='v', label=f'Receivers ({len(rec_x)})', zorder=3)
    ax.scatter(src_x_km, src_z_km, c='red', s=80, marker='*', label=f'Sources ({len(src_x)})', zorder=4)

    ax.set_xlabel('X (km)')
    ax.set_ylabel('Z (km)')
    ax.set_title('Acquisition Geometry on True Vs Model')
    ax.legend(loc='lower right', fontsize=9)
    fig.colorbar(im, ax=ax, label='Vs (km/s)')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, 'acquisition.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Wiggle comparison (observed vs estimated)
# ─────────────────────────────────────────────────────────────────────────────

def save_wiggle_comparison(epoch, obs, est, dt, save_dir, skip=10, shot_idx=0):
    """Wiggle overlay of obs (black) vs est (red), global normalization.

    Parameters
    ----------
    obs, est : ndarray, shape (nt, n_rec)
    dt       : float, time step in seconds
    skip     : int, plot every skip-th receiver
    shot_idx : int, shot index (for title only)
    """
    obs = np.array(obs)
    est = np.array(est)
    nt  = obs.shape[0]
    t_axis = np.arange(nt) * dt
    n_rec  = obs.shape[1]
    rec_idx = np.arange(0, n_rec, skip)
    clip = 0.8 * skip

    fig, ax0 = plt.subplots(1, 1, figsize=(10, 8))

    global_peak = max(np.abs(obs).max(), np.abs(est).max(), 1e-30)

    for i in range(len(rec_idx)):
        ri = rec_idx[i]
        x_off = ri

        tr_obs = obs[:, ri] / global_peak * clip
        label_obs = 'Observed' if i == 0 else None
        ax0.plot(tr_obs + x_off, t_axis, 'k', lw=0.5, label=label_obs)

        tr_est = est[:, ri] / global_peak * clip
        label_est = 'Estimated' if i == 0 else None
        ax0.plot(tr_est + x_off, t_axis, 'r', lw=0.5, label=label_est)

    ax0.set_xlabel('Receiver index')
    ax0.set_ylabel('Time (s)')
    ax0.set_title(f'Epoch {epoch:04d} — Shot {shot_idx}')
    ax0.set_xlim(rec_idx[0] - skip, rec_idx[-1] + skip)
    ax0.invert_yaxis()
    ax0.legend(loc='upper right')

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f'wiggle_{epoch:04d}.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Source wavelet
# ─────────────────────────────────────────────────────────────────────────────

def save_wavelet_plot(wavelet, dt, save_dir, fc=None):
    """Plot source wavelet in time and save to disk.

    Parameters
    ----------
    wavelet : array-like, shape (nt,)
    dt      : float, time step in seconds
    fc      : float, optional, central frequency for title
    """
    wavelet = np.array(wavelet)
    nt = len(wavelet)
    t_axis = np.arange(nt) * dt

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 3))

    # Time domain
    ax0.plot(t_axis, wavelet, 'k', lw=1.2)
    ax0.axhline(0, color='grey', lw=0.5, ls='--')
    ax0.set_xlabel('Time (s)')
    ax0.set_ylabel('Amplitude')
    title = 'Source Wavelet'
    if fc is not None:
        title += f' (fc = {fc} Hz)'
    ax0.set_title(title)
    ax0.grid(True, alpha=0.3)

    # Frequency domain
    freqs = np.fft.rfftfreq(nt, d=dt)
    spec = np.abs(np.fft.rfft(wavelet))
    spec_norm = spec / (spec.max() + 1e-30)
    ax1.plot(freqs, spec_norm, 'b', lw=1.2)
    if fc is not None:
        ax1.axvline(fc, color='r', ls='--', lw=1, label=f'fc = {fc} Hz')
        ax1.legend()
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Normalised amplitude')
    ax1.set_title('Amplitude Spectrum')
    ax1.set_xlim(0, min(freqs[-1], 10 * fc if fc else freqs[-1]))
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, 'source_wavelet.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
