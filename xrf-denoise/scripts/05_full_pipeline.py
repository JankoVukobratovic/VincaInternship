"""
Phase 6: Full end-to-end pipeline.

    Denoise -> Element Maps -> NMF -> CVI -> SAM Segmentation -> Risk Report

Loads the trained UNet1D model, denoises the raw datacube, extracts element
maps from the denoised data, runs NMF blind decomposition, computes the CVI
(Chemical Vulnerability Index), runs SAM segmentation, and generates a
conservator report with a risk table.

Companion code to Pešić et al., "Automated Chemical Vulnerability Assessment
of Canvas Paintings from XRF Spectral Imaging Using Deep Learning and
Foundation Models", ICETRAN 2026.

Usage:
    python scripts/05_full_pipeline.py [--no-sam] [--dataset prova1]
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# Parent repo root (only used to locate the shared SAM checkpoint)
VINCA_ROOT = PROJECT_ROOT.parent

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from sklearn.decomposition import NMF
import json
import time

from src.config import Config
from src.data.loader import load_datacube
from src.models.unet1d import UNet1D
from src.analysis.cross_validation import datacube_to_element_map

# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═════════════════════════════════════════════════════════════════════════════

cfg = Config()

ELEMENTS  = list(cfg.elements.keys())  # ['Ca', 'Ti', 'Fe', 'Cu', 'Pb_La']
EL_LABEL  = {el: el.replace('_La', ' Lα') for el in ELEMENTS}   # display names

RISK_CMAP = LinearSegmentedColormap.from_list('risk', [
    (0.0, '#1a9641'), (0.3, '#a6d96a'), (0.5, '#ffffbf'),
    (0.7, '#fdae61'), (1.0, '#d7191c'),
])
ZONE_COLORS = ['#1a9641', '#ffffbf', '#fdae61', '#d7191c']
ZONE_LABELS = ['Low', 'Moderate', 'Elevated', 'Critical']

KNOWN_PEAKS = {
    'K':     3.31, 'Ca':    3.69, 'Ti':    4.51,
    'Fe Ka': 6.40, 'Fe Kb': 7.06, 'Cu Ka': 8.05,
    'Zn Ka': 8.64, 'Pb La': 10.55, 'Pb Lb': 12.61,
    'Sr Ka': 14.16, 'Sn Ka': 25.27,
}

# Degradation rules - paper Table II, chemistry-first severity weights
RISK_RULES = [
    {'id': 'R1', 'el_a': 'Ti',    'el_b': 'Ca',    'w': 0.40,
     'name': 'TiO2/CaCO3 thermal mismatch',
     'desc': 'TiO2 layer over a CaCO3-bearing ground',
     'mechanism': 'Mismatched thermal-expansion coefficients (TiO2 vs CaCO3)'},
    {'id': 'R2', 'el_a': 'Cu',    'el_b': 'Cu',    'w': 1.00,
     'name': 'Cu-based green pigment degradation',
     'desc': 'Basic copper carbonates transform in humid air',
     'mechanism': 'Cu carbonate conversion in the presence of moisture/CO2'},
    {'id': 'R3', 'el_a': 'Pb_La', 'el_b': 'Pb_La', 'w': 1.00,
     'name': 'Lead white darkening (PbS)',
     'desc': 'PbCO3 -> PbS in the presence of sulfur compounds',
     'mechanism': '2PbCO3*Pb(OH)2 -> PbS'},
    {'id': 'R4', 'el_a': 'Ti',    'el_b': 'Cu',    'w': 0.60,
     'name': 'Trapped moisture under TiO2',
     'desc': 'TiO2 blocks moisture diffusion out of the Cu pigment',
     'mechanism': 'Impermeable TiO2 accelerates Cu degradation beneath'},
    {'id': 'R5', 'el_a': 'Fe',    'el_b': 'Pb_La', 'w': 0.80,
     'name': 'Fe-catalyzed Pb oxidation',
     'desc': 'Fe oxides catalyse oxidative degradation of lead white',
     'mechanism': 'Fe2O3 -> PbO2 formation at the layer boundary'},
]

MATERIAL_DESC = {
    'Ca':    'Lime/chalk layer (CaCO3)',
    'Ti':    'Titanium white ground (TiO2)',
    'Fe':    'Ochre (Fe-based pigment)',
    'Cu':    'Cu-based green pigment',
    'Pb_La': 'Lead white (Pb-based pigment)',
}

_MAP_CB = dict(fraction=0.046 * cfg.rows / cfg.cols, pad=0.02)


# ═════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def norm_percentile(m, q_lo=8, q_hi=99):
    bg   = np.percentile(m, q_lo)
    peak = np.percentile(m, q_hi)
    return np.clip((m - bg) / (peak - bg + 1e-10), 0, 1)


def _style_cb(cb, label=None):
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_edgecolor('#CCCCCC')
    if label:
        cb.set_label(label, fontsize=9)


def _blank(ax):
    ax.set_xticks([]); ax.set_yticks([])


def denoise_datacube(model, datacube, global_scale, device, batch_size=256):
    H, W, C = datacube.shape
    flat = datacube.reshape(-1, C)
    N = flat.shape[0]
    denoised = np.zeros_like(flat)

    model.eval()
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = flat[i:i+batch_size]
            x = torch.from_numpy(batch / global_scale).unsqueeze(1).to(device)
            y = model(x).squeeze(1).cpu().numpy() * global_scale
            denoised[i:i+batch_size] = np.maximum(y, 0)

    return denoised.reshape(H, W, C)


def extract_element_maps(datacube):
    """Extract element maps from a datacube using configured elements."""
    maps = {}
    for el, info in cfg.elements.items():
        maps[el] = datacube_to_element_map(
            datacube, info['kev'], cfg.cal_slope, cfg.cal_intercept
        )
    return maps


def run_nmf(spectra_flat, K_range=range(3, 9)):
    """NMF blind decomposition with elbow method for optimal K."""
    n_ch = spectra_flat.shape[1]
    energy = np.arange(n_ch) * cfg.cal_slope + cfg.cal_intercept

    ch_lo = max(0, int((1.0 - cfg.cal_intercept) / cfg.cal_slope))
    ch_hi = min(n_ch, int((14.0 - cfg.cal_intercept) / cfg.cal_slope))
    D_trim = np.maximum(spectra_flat[:, ch_lo:ch_hi], 0)
    energy_trim = energy[ch_lo:ch_hi]

    # --- Remove Hg artifact channels ---
    # Mercury (Hg La ~9.99 keV, Hg Lb ~11.82 keV) appears as a rectangular
    # scan artifact in the inner region (rows 16-45). Zero out these channels
    # so NMF doesn't waste a component on the acquisition artifact.
    hg_ranges_kev = [(9.75, 10.20), (11.60, 12.10)]  # Hg La, Hg Lb
    for kev_lo, kev_hi in hg_ranges_kev:
        mask_lo = max(0, int((kev_lo - cfg.cal_intercept) / cfg.cal_slope) - ch_lo)
        mask_hi = min(D_trim.shape[1], int((kev_hi - cfg.cal_intercept) / cfg.cal_slope) - ch_lo)
        D_trim[:, mask_lo:mask_hi] = 0

    # Normalize each spectrum by total counts to remove rectangular
    # acquisition-intensity variation across the scan grid
    row_sums = D_trim.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    D_trim = D_trim / row_sums * row_sums.mean()

    print("  Determining optimal K...")
    errors = []
    for k in K_range:
        model = NMF(n_components=k, init='nndsvda', max_iter=500, random_state=42)
        model.fit_transform(D_trim)
        errors.append(model.reconstruction_err_)
        print(f"    K={k}: error={model.reconstruction_err_:.0f}")

    K_list = list(K_range)
    K_opt = K_list[np.argmax(np.diff(np.diff(errors))) + 2]
    print(f"  Optimal K = {K_opt}")

    model = NMF(n_components=K_opt, init='nndsvda', max_iter=1000, random_state=42)
    W = model.fit_transform(D_trim)
    H = model.components_

    # Identify emission lines in each component
    names = []
    for k in range(K_opt):
        peaks_idx, _ = find_peaks(H[k], height=np.max(H[k]) * 0.1,
                                  distance=5, prominence=np.max(H[k]) * 0.05)
        found = []
        for kev in energy_trim[peaks_idx]:
            best_el, best_d = None, 999
            for el, el_kev in KNOWN_PEAKS.items():
                d = abs(kev - el_kev)
                if d < best_d:
                    best_d = d
                    best_el = el
            if best_d < 0.4:
                found.append(best_el)
        names.append(" + ".join(dict.fromkeys(found)) if found else f"Component {k+1}")

    nmf_maps = W.reshape(cfg.rows, cfg.cols, K_opt)

    return {
        'W': W, 'H': H, 'K': K_opt,
        'energy': energy_trim, 'names': names,
        'maps': nmf_maps, 'errors': errors, 'K_range': K_list,
    }


def compute_cvi(norm_maps):
    """Compute CVI per pixel from normalized element maps."""
    cvi = np.zeros((cfg.rows, cfg.cols))
    dominant_risk = np.zeros((cfg.rows, cfg.cols), dtype=int)
    risk_maps = {}

    for ri, rule in enumerate(RISK_RULES):
        a = norm_maps[rule['el_a']]
        b = norm_maps[rule['el_b']]
        w = rule['w']

        risk = w * a if rule['el_a'] == rule['el_b'] else w * np.sqrt(a * b)
        risk = np.clip(gaussian_filter(risk, sigma=1.0), 0, 1)
        risk_maps[rule['id']] = risk

        mask = risk > cvi
        cvi[mask] = risk[mask]
        dominant_risk[mask] = ri

        print(f"  {rule['id']} ({rule['name'][:42]:42s}) "
              f"max={risk.max():.3f}, mean={risk.mean():.3f}, "
              f">0.5: {np.mean(risk > 0.5) * 100:.1f}%")

    stats = {
        'low':      np.mean(cvi < 0.25) * 100,
        'moderate': np.mean((cvi >= 0.25) & (cvi < 0.50)) * 100,
        'elevated': np.mean((cvi >= 0.50) & (cvi < 0.75)) * 100,
        'critical': np.mean(cvi >= 0.75) * 100,
    }

    print(f"\n  Composite CVI: max={cvi.max():.3f}, mean={cvi.mean():.3f}")
    print("  Zones: " + ", ".join(f"{k}={v:.1f}%" for k, v in stats.items()))

    return {
        'cvi': cvi,
        'dominant_risk': dominant_risk,
        'risk_maps': risk_maps,
        'stats': stats,
    }


def run_sam_segmentation(norm_maps):
    """Run SAM automatic segmentation on the false-color XRF image."""
    from scipy.ndimage import zoom as ndi_zoom, distance_transform_edt

    # RGB: R=Fe, G=Cu, B=Pb
    rgb_input = np.stack([norm_maps['Fe'], norm_maps['Cu'], norm_maps['Pb_La']], axis=2)
    rgb_input = np.clip(rgb_input, 0, 1)

    SCALE = 8
    rgb_upscaled = np.stack([
        ndi_zoom(rgb_input[:, :, c], SCALE, order=3)
        for c in range(3)
    ], axis=2)
    rgb_upscaled = np.clip(rgb_upscaled, 0, 1)
    sam_input = (rgb_upscaled * 255).astype(np.uint8)
    print(f"  SAM input: {sam_input.shape} (scaled {SCALE}x)")

    # Load SAM
    sam_checkpoint = VINCA_ROOT / 'models' / 'sam_vit_b_01ec64.pth'
    if not sam_checkpoint.exists():
        print(f"  WARNING: SAM checkpoint not found at {sam_checkpoint}")
        print(f"  Skipping SAM segmentation.")
        return None, rgb_input

    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    sam = sam_model_registry['vit_b'](checkpoint=str(sam_checkpoint))
    sam.eval()

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=100,
    )

    print("  Running SAM segmentation...")
    masks = mask_generator.generate(sam_input)
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    print(f"  SAM found {len(masks)} segments")

    # Downscale masks to the native resolution (majority vote)
    segments = np.zeros((cfg.rows, cfg.cols), dtype=int)
    segment_info = []
    valid_id = 0

    for mask_data in masks:
        mask_hr = mask_data['segmentation']
        mask_lr = np.zeros((cfg.rows, cfg.cols), dtype=bool)
        for r in range(cfg.rows):
            for c in range(cfg.cols):
                patch = mask_hr[r*SCALE:(r+1)*SCALE, c*SCALE:(c+1)*SCALE]
                mask_lr[r, c] = patch.mean() > 0.5

        n_pixels = mask_lr.sum()
        if n_pixels < 5:
            continue

        valid_id += 1
        unassigned = (segments == 0) & mask_lr
        segments[unassigned] = valid_id

        segment_info.append({
            'id': valid_id,
            'area_px': int(n_pixels),
            'area_pct': n_pixels / cfg.n_pixels * 100,
            'stability': mask_data['stability_score'],
            'iou': mask_data['predicted_iou'],
            'mask': mask_lr,
        })

    # Assign unassigned pixels to the nearest segment
    unassigned_mask = segments == 0
    if unassigned_mask.any() and segment_info:
        min_dist = np.full((cfg.rows, cfg.cols), np.inf)
        for seg in segment_info:
            dist = distance_transform_edt(~seg['mask'])
            closer = dist < min_dist
            segments[unassigned_mask & closer] = seg['id']
            min_dist[unassigned_mask & closer] = dist[unassigned_mask & closer]

    print(f"  Valid segments: {len(segment_info)}")
    return {'segments': segments, 'segment_info': segment_info}, rgb_input


def compute_region_reports(segments_data, norm_maps, cvi_data):
    """Compute per-region CVI statistics from SAM segments."""
    segments = segments_data['segments']
    segment_info = segments_data['segment_info']
    cvi = cvi_data['cvi']
    dominant_risk = cvi_data['dominant_risk']

    region_reports = []
    for seg in segment_info:
        mask = segments == seg['id']
        n_px = mask.sum()
        if n_px == 0:
            continue

        el_means = {el: float(norm_maps[el][mask].mean()) for el in ELEMENTS}
        cvi_vals = cvi[mask]

        dom_vals = dominant_risk[mask]
        dom_counts = np.bincount(dom_vals, minlength=len(RISK_RULES))
        dom_idx = dom_counts.argmax()

        dominant_el = max(el_means, key=el_means.get)

        region_reports.append({
            'id': seg['id'],
            'area_px': n_px,
            'area_pct': n_px / cfg.n_pixels * 100,
            'el_means': el_means,
            'dominant_el': dominant_el,
            'material': MATERIAL_DESC.get(dominant_el, '?'),
            'cvi_mean': float(cvi_vals.mean()),
            'cvi_max': float(cvi_vals.max()),
            'dominant_risk': RISK_RULES[dom_idx],
            'pct_elevated': float(np.mean(cvi_vals >= 0.5) * 100),
            'pct_critical': float(np.mean(cvi_vals >= 0.75) * 100),
        })

    region_reports.sort(key=lambda r: r['cvi_mean'], reverse=True)
    return region_reports


# ═════════════════════════════════════════════════════════════════════════════
#  VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════

def plot_element_maps(maps_raw, maps_denoised, norm_den, fig_dir):
    """Raw vs denoised element maps, with a difference row."""
    n_el = len(ELEMENTS)
    fig, axes = plt.subplots(3, n_el, figsize=(3.3 * n_el, 6.4),
                             layout='constrained')

    for j, el in enumerate(ELEMENTS):
        vmax = np.percentile(np.concatenate([
            maps_raw[el].ravel(), maps_denoised[el].ravel()
        ]), 99)

        axes[0, j].imshow(maps_raw[el], origin='upper', cmap='inferno',
                          vmin=0, vmax=vmax, interpolation='bilinear',
                          aspect='equal')
        axes[0, j].set_title(EL_LABEL[el], fontsize=11, fontweight='bold')

        axes[1, j].imshow(maps_denoised[el], origin='upper', cmap='inferno',
                          vmin=0, vmax=vmax, interpolation='bilinear',
                          aspect='equal')

        # Difference: Raw - Denoised (what the model removed = noise estimate)
        diff = maps_raw[el].astype(np.float64) - maps_denoised[el].astype(np.float64)
        diff_vmax = np.percentile(np.abs(diff), 99)
        im = axes[2, j].imshow(diff, origin='upper', cmap='RdBu_r',
                               vmin=-diff_vmax, vmax=diff_vmax,
                               interpolation='bilinear', aspect='equal')
        _style_cb(plt.colorbar(im, ax=axes[2, j], **_MAP_CB))

    for ax in axes.flatten():
        _blank(ax)
    for i, row_name in enumerate(['Raw', 'Denoised', 'Raw − Denoised']):
        axes[i, 0].set_ylabel(row_name, fontsize=10, fontweight='bold')

    fig.suptitle('Element maps - raw vs denoised (difference = removed noise)',
                 fontsize=13, fontweight='bold')
    plt.savefig(fig_dir / '01_element_maps_raw_vs_denoised.png',
                dpi=180, bbox_inches='tight')
    plt.close()
    print("  Saved: 01_element_maps_raw_vs_denoised.png")


def plot_nmf(nmf_res, fig_dir):
    """NMF spectral endmembers + spatial abundance maps."""
    K, H = nmf_res['K'], nmf_res['H']
    energy, nmf_maps, names = nmf_res['energy'], nmf_res['maps'], nmf_res['names']
    colors = plt.cm.Dark2(np.linspace(0, 0.85, K))

    fig = plt.figure(figsize=(15, 2.9 * K), layout='constrained')
    gs = GridSpec(K, 2, figure=fig, width_ratios=[1.6, 1])

    for k in range(K):
        ax_sp = fig.add_subplot(gs[k, 0])
        ax_sp.fill_between(energy, H[k], alpha=0.25, color=colors[k])
        ax_sp.plot(energy, H[k], color=colors[k], linewidth=1.4)
        ax_sp.set_title(f'Component {k+1}: {names[k]}',
                        fontsize=11, fontweight='bold', color=colors[k])
        ax_sp.set_xlim(0, 14)
        ax_sp.grid(True, alpha=0.15)
        ax_sp.set_ylabel('Intensity (a.u.)', fontsize=9)
        ax_sp.tick_params(labelsize=8)
        if k == K - 1:
            ax_sp.set_xlabel('Energy (keV)', fontsize=10)

        peaks_idx, _ = find_peaks(H[k], height=np.max(H[k]) * 0.1,
                                  distance=5, prominence=np.max(H[k]) * 0.05)
        for pi in peaks_idx:
            kev = energy[pi]
            best_el, best_d = '', 999
            for el, el_kev in KNOWN_PEAKS.items():
                if abs(kev - el_kev) < best_d:
                    best_d = abs(kev - el_kev)
                    best_el = el
            if best_d < 0.4:
                ax_sp.annotate(best_el, xy=(kev, H[k][pi]), fontsize=8,
                               fontweight='bold', ha='center', va='bottom',
                               xytext=(0, 5), textcoords='offset points')

        ax_map = fig.add_subplot(gs[k, 1])
        m = nmf_maps[:, :, k]
        p99 = np.percentile(m, 99)
        im = ax_map.imshow(m / max(p99, 1e-10), origin='upper', aspect='equal',
                           cmap='inferno', interpolation='bilinear',
                           vmin=0, vmax=1)
        ax_map.set_title('Spatial abundance', fontsize=9)
        _blank(ax_map)
        _style_cb(plt.colorbar(im, ax=ax_map, **_MAP_CB))

    fig.suptitle('NMF blind decomposition of the denoised spectra',
                 fontsize=13, fontweight='bold')
    plt.savefig(fig_dir / '02_nmf_components.png', dpi=180, bbox_inches='tight')
    plt.close()
    print("  Saved: 02_nmf_components.png")


def plot_cvi(cvi_data, fig_dir):
    """CVI composite map with zone classification."""
    cvi = cvi_data['cvi']
    stats = cvi_data['stats']

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.4), layout='constrained')

    im = axes[0].imshow(cvi, origin='upper', aspect='equal', cmap=RISK_CMAP,
                        interpolation='bilinear', vmin=0, vmax=1)
    axes[0].set_title('Chemical Vulnerability Index (continuous)',
                      fontsize=12, fontweight='bold')
    _style_cb(plt.colorbar(im, ax=axes[0], **_MAP_CB), 'CVI score')
    axes[0].text(0.02, 0.04,
                 f"Low (<0.25): {stats['low']:.1f}%\n"
                 f"Moderate (0.25–0.50): {stats['moderate']:.1f}%\n"
                 f"Elevated (0.50–0.75): {stats['elevated']:.1f}%\n"
                 f"Critical (≥0.75): {stats['critical']:.1f}%",
                 transform=axes[0].transAxes, fontsize=8.5, color='white',
                 va='bottom', family='monospace',
                 bbox=dict(boxstyle='round,pad=0.35', facecolor='black',
                           alpha=0.7, edgecolor='none'))

    zones = np.digitize(cvi, [0.25, 0.50, 0.75])
    zone_cmap = LinearSegmentedColormap.from_list('zone', ZONE_COLORS, N=4)
    axes[1].imshow(zones, origin='upper', aspect='equal', cmap=zone_cmap,
                   interpolation='nearest', vmin=0, vmax=3)
    axes[1].set_title('Risk-zone classification', fontsize=12, fontweight='bold')
    axes[1].legend(handles=[mpatches.Patch(color=c, label=f'{l} risk')
                            for c, l in zip(ZONE_COLORS, ZONE_LABELS)],
                   loc='lower left', fontsize=8.5, frameon=True,
                   facecolor='white', framealpha=0.9, edgecolor='#CCCCCC')

    for ax in axes:
        _blank(ax)

    fig.suptitle('Chemical vulnerability map (from denoised XRF data)',
                 fontsize=13, fontweight='bold')
    plt.savefig(fig_dir / '03_cvi_map.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 03_cvi_map.png")


def plot_risk_rules(cvi_data, fig_dir):
    """Individual risk-rule maps."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 6.2), layout='constrained')
    axes_flat = axes.flatten()

    for i, rule in enumerate(RISK_RULES):
        ax = axes_flat[i]
        r = cvi_data['risk_maps'][rule['id']]
        im = ax.imshow(r, origin='upper', aspect='equal', cmap=RISK_CMAP,
                       interpolation='bilinear', vmin=0, vmax=1)
        pair = (rule['el_a'] if rule['el_a'] == rule['el_b']
                else f"{rule['el_a']}/{rule['el_b']}").replace('_La', '')
        ax.set_title(f"{rule['id']}: {rule['name']}\n({pair}, w={rule['w']:.2f})",
                     fontsize=9.5, fontweight='bold')
        _blank(ax)
        _style_cb(plt.colorbar(im, ax=ax, **_MAP_CB))

    axes_flat[len(RISK_RULES)].axis('off')

    fig.suptitle('Individual risk-rule maps (from denoised data)',
                 fontsize=13, fontweight='bold')
    plt.savefig(fig_dir / '04_risk_rules.png', dpi=180, bbox_inches='tight')
    plt.close()
    print("  Saved: 04_risk_rules.png")


def plot_publication_figure(maps_raw, maps_denoised, nmf_res, cvi_data,
                            norm_maps, fig_dir):
    """Publication-quality composite figure (paper Fig. 1 layout).
    3 rows: (a) raw element maps, (b) denoised, (c) CVI + zones + overlay.
    NMF excluded - one component captures background (support), not pigment.
    """
    n_el = len(ELEMENTS)
    fig = plt.figure(figsize=(16, 8.2), layout='constrained')
    gs = GridSpec(3, n_el, figure=fig, height_ratios=[1, 1, 1.25])

    cvi = cvi_data['cvi']

    # --- Rows 1-2: raw and denoised element maps ---
    for j, el in enumerate(ELEMENTS):
        vmax = np.percentile(maps_raw[el], 99)

        ax = fig.add_subplot(gs[0, j])
        ax.imshow(maps_raw[el], origin='upper', cmap='inferno', vmin=0,
                  vmax=vmax, interpolation='bilinear', aspect='equal')
        ax.set_title(EL_LABEL[el], fontsize=11, fontweight='bold')
        if j == 0:
            ax.set_ylabel('(a) Raw', fontsize=10, fontweight='bold')
        _blank(ax)

        ax = fig.add_subplot(gs[1, j])
        ax.imshow(maps_denoised[el], origin='upper', cmap='inferno', vmin=0,
                  vmax=vmax, interpolation='bilinear', aspect='equal')
        if j == 0:
            ax.set_ylabel('(b) Denoised', fontsize=10, fontweight='bold')
        _blank(ax)

    # --- Row 3: CVI map + zone classification + overlay ---
    ax_cvi = fig.add_subplot(gs[2, 0:2])
    im = ax_cvi.imshow(cvi, origin='upper', cmap=RISK_CMAP, vmin=0, vmax=1,
                       interpolation='bilinear', aspect='equal')
    ax_cvi.set_title('Chemical Vulnerability Index', fontsize=10, fontweight='bold')
    ax_cvi.set_ylabel('(c) Risk', fontsize=10, fontweight='bold')
    _blank(ax_cvi)
    _style_cb(plt.colorbar(im, ax=ax_cvi, **_MAP_CB), 'CVI')

    ax_zone = fig.add_subplot(gs[2, 2:4])
    zones = np.digitize(cvi, [0.25, 0.50, 0.75])
    zone_cmap = LinearSegmentedColormap.from_list('zone', ZONE_COLORS, N=4)
    ax_zone.imshow(zones, origin='upper', cmap=zone_cmap,
                   interpolation='nearest', vmin=0, vmax=3, aspect='equal')
    ax_zone.set_title('Risk-zone classification', fontsize=10, fontweight='bold')
    _blank(ax_zone)
    ax_zone.legend(handles=[mpatches.Patch(color=c, label=l)
                            for c, l in zip(ZONE_COLORS, ZONE_LABELS)],
                   loc='lower left', fontsize=7, frameon=True,
                   facecolor='white', framealpha=0.9, edgecolor='#CCCCCC')

    ax_over = fig.add_subplot(gs[2, 4])
    rgb_fc = np.stack([norm_maps['Fe'], norm_maps['Cu'], norm_maps['Pb_La']], axis=2)
    rgb_fc = np.clip(rgb_fc, 0, 1)
    ax_over.imshow(rgb_fc, origin='upper', interpolation='bilinear', aspect='equal')
    cs = ax_over.contour(cvi, levels=[0.5, 0.7], colors=['#ff9900', '#ff1a1a'],
                         linewidths=[0.9, 1.4], origin='upper')
    ax_over.clabel(cs, inline=True, fontsize=6, fmt='%.1f')
    ax_over.set_title('CVI overlay', fontsize=10, fontweight='bold')
    _blank(ax_over)

    fig.suptitle('XRF spectral denoising and chemical vulnerability mapping',
                 fontsize=14, fontweight='bold')

    plt.savefig(fig_dir / '00_publication_figure.png', dpi=300, bbox_inches='tight')
    plt.savefig(fig_dir / '00_publication_figure.pdf', bbox_inches='tight')
    plt.close()
    print("  Saved: 00_publication_figure.png / .pdf")


def plot_sam_results(segments_data, region_reports, cvi_data, norm_maps,
                     rgb_input, fig_dir):
    """SAM segmentation + per-region CVI visualization."""
    segments = segments_data['segments']
    segment_info = segments_data['segment_info']
    cvi = cvi_data['cvi']
    n_segments = len(segment_info)

    # --- Figure 5: SAM segmentation overview ---
    rng = np.random.default_rng(42)
    seg_rgb = np.zeros((cfg.rows, cfg.cols, 3))
    for seg in segment_info:
        seg_rgb[segments == seg['id']] = rng.random(3) * 0.7 + 0.3

    region_cvi_map = np.zeros((cfg.rows, cfg.cols))
    for r in region_reports:
        region_cvi_map[segments == r['id']] = r['cvi_mean']

    fig, axes = plt.subplots(1, 3, figsize=(18, 3.9), layout='constrained')
    axes[0].imshow(rgb_input, origin='upper', aspect='equal', interpolation='bilinear')
    axes[0].set_title('SAM input - false-color composite\n(R=Fe, G=Cu, B=Pb)',
                      fontsize=11, fontweight='bold')

    axes[1].imshow(seg_rgb, origin='upper', aspect='equal', interpolation='nearest')
    axes[1].set_title(f'SAM segmentation\n({n_segments} regions)',
                      fontsize=11, fontweight='bold')

    im = axes[2].imshow(region_cvi_map, origin='upper', aspect='equal',
                        cmap=RISK_CMAP, interpolation='nearest', vmin=0, vmax=1)
    axes[2].set_title('Mean CVI per SAM region', fontsize=11, fontweight='bold')
    _style_cb(plt.colorbar(im, ax=axes[2], **_MAP_CB), 'mean CVI')

    for ax in axes:
        _blank(ax)

    fig.suptitle('SAM segmentation + chemical vulnerability analysis',
                 fontsize=13, fontweight='bold')
    plt.savefig(fig_dir / '05_sam_segmentation.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 05_sam_segmentation.png")

    # --- Figure 6: summary panel ---
    material_colors = {
        'Ca': [0.93, 0.87, 0.72], 'Ti': [0.96, 0.96, 0.94],
        'Fe': [0.68, 0.20, 0.03], 'Cu': [0.09, 0.27, 0.70],
        'Pb_La': [0.80, 0.75, 0.95],
    }
    mat_rgb = np.zeros((cfg.rows, cfg.cols, 3))
    for r in region_reports:
        mat_rgb[segments == r['id']] = material_colors[r['dominant_el']]

    fig, axes = plt.subplots(2, 3, figsize=(18, 7.4), layout='constrained')
    axes[0, 0].imshow(rgb_input, origin='upper', aspect='equal', interpolation='bilinear')
    axes[0, 0].set_title('XRF false-color (R=Fe, G=Cu, B=Pb)',
                         fontsize=10.5, fontweight='bold')

    axes[0, 1].imshow(seg_rgb, origin='upper', aspect='equal', interpolation='nearest')
    axes[0, 1].set_title(f'SAM segmentation ({n_segments} regions)',
                         fontsize=10.5, fontweight='bold')

    axes[0, 2].imshow(mat_rgb, origin='upper', aspect='equal', interpolation='nearest')
    axes[0, 2].set_title('Dominant material', fontsize=10.5, fontweight='bold')

    im = axes[1, 0].imshow(cvi, origin='upper', aspect='equal', cmap=RISK_CMAP,
                           interpolation='bilinear', vmin=0, vmax=1)
    axes[1, 0].set_title('CVI - pixel level', fontsize=10.5, fontweight='bold')
    _style_cb(plt.colorbar(im, ax=axes[1, 0], **_MAP_CB))

    im = axes[1, 1].imshow(region_cvi_map, origin='upper', aspect='equal',
                           cmap=RISK_CMAP, interpolation='nearest', vmin=0, vmax=1)
    axes[1, 1].set_title('Mean CVI per SAM region', fontsize=10.5, fontweight='bold')
    _style_cb(plt.colorbar(im, ax=axes[1, 1], **_MAP_CB))

    # CVI overlay on the XRF composite
    axes[1, 2].imshow(rgb_input, origin='upper', aspect='equal', interpolation='bilinear')
    cs = axes[1, 2].contour(cvi, levels=[0.5, 0.7], colors=['#ff9900', '#ff1a1a'],
                            linewidths=[1.0, 1.5], origin='upper')
    axes[1, 2].clabel(cs, inline=True, fontsize=7, fmt='%.1f')
    axes[1, 2].set_title('CVI overlay on XRF', fontsize=10.5, fontweight='bold')

    for ax in axes.flatten():
        _blank(ax)

    fig.suptitle('Full pipeline summary: Denoise + NMF + CVI + SAM',
                 fontsize=13, fontweight='bold')
    plt.savefig(fig_dir / '06_full_summary.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 06_full_summary.png")


def generate_risk_table(region_reports, cvi_data, nmf_res, elapsed, fig_dir):
    """Generate the risk table and conservator report as text + JSON."""
    cvi = cvi_data['cvi']

    # Text report
    lines = []
    lines.append("=" * 75)
    lines.append("  CANVAS PAINTING CHEMICAL VULNERABILITY REPORT")
    lines.append("  Auto-generated: Denoise + NMF + CVI + SAM Pipeline")
    lines.append("=" * 75)
    lines.append("")
    lines.append("METHODOLOGY:")
    lines.append("  1. UNet1D denoising (self-supervised Poisson splitting)")
    lines.append("  2. NMF blind spectral decomposition")
    lines.append("  3. Chemical Vulnerability Index (CVI) - 5 degradation rules")
    if region_reports:
        lines.append("  4. SAM (Segment Anything Model) automatic segmentation")
    lines.append("")
    lines.append("STATISTICS:")
    lines.append(f"  Grid: {cfg.rows}x{cfg.cols} = {cfg.n_pixels} pixels")
    if nmf_res:
        lines.append(f"  NMF components: {nmf_res['K']}")
    lines.append(f"  Mean CVI: {cvi.mean():.3f}")
    lines.append(f"  Elevated risk (CVI >= 0.5): {np.mean(cvi >= 0.5)*100:.1f}%")
    lines.append(f"  Critical risk (CVI >= 0.75): {np.mean(cvi >= 0.75)*100:.1f}%")
    lines.append(f"  Pipeline time: {elapsed:.0f}s")
    lines.append("")

    if region_reports:
        lines.append("=" * 75)
        lines.append("  RISK TABLE - TOP REGIONS (sorted by CVI)")
        lines.append("=" * 75)
        lines.append("")
        lines.append(f"  {'#':>3} {'Region':>7} {'Area%':>6} {'CVI':>6} "
                     f"{'Max':>6} {'Level':>10} {'Material':>28} {'Risk':>34}")
        lines.append(f"  {'-'*104}")

        for i, r in enumerate(region_reports[:20]):
            level = ('CRITICAL' if r['cvi_mean'] >= 0.75 else
                     'ELEVATED' if r['cvi_mean'] >= 0.5 else
                     'MODERATE' if r['cvi_mean'] >= 0.25 else 'LOW')
            lines.append(
                f"  {i+1:3d} R{r['id']:>5d} {r['area_pct']:5.1f}% "
                f"{r['cvi_mean']:6.3f} {r['cvi_max']:6.3f} {level:>10} "
                f"{r['material']:>28} {r['dominant_risk']['id']}: "
                f"{r['dominant_risk']['name'][:30]}"
            )

        lines.append("")
        lines.append("=" * 75)
        lines.append("  RECOMMENDATIONS")
        lines.append("=" * 75)
        lines.append("")

        priority = [r for r in region_reports if r['cvi_mean'] >= 0.5]
        moderate = [r for r in region_reports if 0.25 <= r['cvi_mean'] < 0.5]

        if priority:
            lines.append(f"  URGENT: {len(priority)} region(s) require priority assessment:")
            for r in priority[:5]:
                lines.append(f"    - Region {r['id']}: {r['material']} "
                             f"(CVI={r['cvi_mean']:.3f}, "
                             f"{r['dominant_risk']['id']}: {r['dominant_risk']['name']})")
            lines.append("")

        if moderate:
            lines.append(f"  MONITOR: {len(moderate)} region(s) with moderate risk:")
            for r in moderate[:5]:
                lines.append(f"    - Region {r['id']}: {r['material']} "
                             f"(CVI={r['cvi_mean']:.3f})")
            lines.append("")

    lines.append("=" * 75)
    lines.append("  RISK RULES LEGEND")
    lines.append("=" * 75)
    for p in RISK_RULES:
        lines.append(f"  {p['id']}: {p['name']} (w={p['w']})")
        lines.append(f"      {p['desc']}")
        lines.append(f"      Mechanism: {p['mechanism']}")
        lines.append("")

    report_text = "\n".join(lines)

    report_path = fig_dir / 'risk_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"  Saved: risk_report.txt")

    # JSON summary
    json_data = {
        'pipeline': 'denoise_nmf_cvi_sam',
        'grid': f'{cfg.rows}x{cfg.cols}',
        'cvi_mean': float(cvi.mean()),
        'cvi_max': float(cvi.max()),
        'pct_elevated': float(np.mean(cvi >= 0.5) * 100),
        'pct_critical': float(np.mean(cvi >= 0.75) * 100),
        'cvi_stats': cvi_data['stats'],
        'time_seconds': round(elapsed, 1),
    }
    if nmf_res:
        json_data['nmf_K'] = nmf_res['K']
    if region_reports:
        json_data['n_sam_regions'] = len(region_reports)
        json_data['top_regions'] = [
            {
                'id': r['id'],
                'area_pct': round(r['area_pct'], 1),
                'cvi_mean': round(r['cvi_mean'], 3),
                'cvi_max': round(r['cvi_max'], 3),
                'material': r['material'],
                'dominant_risk': r['dominant_risk']['id'],
            }
            for r in region_reports[:10]
        ]

    json_path = fig_dir / 'pipeline_summary.json'
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"  Saved: pipeline_summary.json")

    return report_text


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Full XRF denoising + analysis pipeline')
    parser.add_argument('--no-sam', action='store_true',
                        help='Skip SAM segmentation (if checkpoint missing or slow)')
    parser.add_argument('--dataset', default='prova1',
                        help='Dataset name (default: prova1)')
    args = parser.parse_args()

    t0 = time.time()

    # Output directory
    out_dir = cfg.abs_path('experiments') / 'full_pipeline'
    fig_dir = out_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  FULL PIPELINE: Denoise -> Maps -> NMF -> CVI -> SAM -> Report")
    print("=" * 70)

    # ─── Step 1: Load raw datacube ─────────────────────────────────────────
    print("\n[1/7] Loading raw datacube...")
    dataset_name = f"aurora-antico1-{args.dataset}"
    dataset_path = Path(cfg.raw_data_dir) / dataset_name
    cache_dir = cfg.abs_path(cfg.processed_dir)

    cube_raw, _ = load_datacube(dataset_path, cfg.detector_a, cfg.rows, cfg.cols,
                                cache_path=cache_dir / f"{cfg.detector_a}_raw.npy")
    print(f"  Datacube shape: {cube_raw.shape}")

    # ─── Step 2: Denoise ───────────────────────────────────────────────────
    print("\n[2/7] Denoising datacube with the trained UNet1D...")
    model_path = cfg.abs_path(cfg.exp_a_dir) / "checkpoints" / "best_model.pt"
    if not model_path.exists():
        print(f"  ERROR: trained model not found at {model_path}")
        print(f"  Run 03a_train_scratch.py first!")
        sys.exit(1)

    # Load the global scale used during training
    train_summary = cfg.abs_path(cfg.exp_a_dir) / "results" / "phase4a_summary.json"
    with open(train_summary) as f:
        global_scale = json.load(f)['global_scale']

    model = UNet1D(base_filters=cfg.base_filters, n_blocks=cfg.n_encoder_blocks,
                   dropout=0).to(cfg.device)
    model.load_state_dict(torch.load(model_path, map_location=cfg.device,
                                     weights_only=True))

    t_denoise = time.time()
    cube_denoised = denoise_datacube(model, cube_raw, global_scale, cfg.device)
    t_denoise = time.time() - t_denoise
    print(f"  Denoised in {t_denoise:.1f}s "
          f"({t_denoise/cfg.n_pixels*1000:.2f} ms/spectrum)")

    # ─── Step 3: Extract element maps ──────────────────────────────────────
    print("\n[3/7] Extracting element maps...")
    maps_raw = extract_element_maps(cube_raw)
    maps_denoised = extract_element_maps(cube_denoised)
    norm_maps = {el: norm_percentile(maps_denoised[el]) for el in ELEMENTS}
    print(f"  Elements: {', '.join(ELEMENTS)}")

    # ─── Step 4: NMF ──────────────────────────────────────────────────────
    print("\n[4/7] NMF blind decomposition on denoised spectra...")
    spectra_flat = cube_denoised.reshape(-1, cube_denoised.shape[-1])
    nmf_res = run_nmf(spectra_flat)

    # ─── Step 5: CVI ──────────────────────────────────────────────────────
    print("\n[5/7] Computing the Chemical Vulnerability Index...")
    cvi_data = compute_cvi(norm_maps)

    # ─── Step 6: SAM (optional) ───────────────────────────────────────────
    segments_data = None
    region_reports = []
    rgb_input = np.stack([norm_maps['Fe'], norm_maps['Cu'], norm_maps['Pb_La']], axis=2)
    rgb_input = np.clip(rgb_input, 0, 1)

    if not args.no_sam:
        print("\n[6/7] SAM segmentation...")
        segments_data, rgb_input = run_sam_segmentation(norm_maps)
        if segments_data:
            region_reports = compute_region_reports(segments_data, norm_maps, cvi_data)
            print(f"\n  Top 5 regions by risk:")
            for r in region_reports[:5]:
                level = ('CRITICAL' if r['cvi_mean'] >= 0.75 else
                         'ELEVATED' if r['cvi_mean'] >= 0.5 else
                         'MODERATE' if r['cvi_mean'] >= 0.25 else 'LOW')
                print(f"    R{r['id']:2d}: CVI={r['cvi_mean']:.3f} [{level}] "
                      f" - {r['material']}")
    else:
        print("\n[6/7] SAM segmentation SKIPPED (--no-sam)")

    # ─── Step 7: Visualization & Report ───────────────────────────────────
    print("\n[7/7] Generating figures and report...")
    plot_element_maps(maps_raw, maps_denoised, norm_maps, fig_dir)
    plot_nmf(nmf_res, fig_dir)
    plot_cvi(cvi_data, fig_dir)
    plot_risk_rules(cvi_data, fig_dir)
    plot_publication_figure(maps_raw, maps_denoised, nmf_res, cvi_data,
                            norm_maps, fig_dir)

    if segments_data and region_reports:
        plot_sam_results(segments_data, region_reports, cvi_data, norm_maps,
                         rgb_input, fig_dir)

    elapsed = time.time() - t0
    report = generate_risk_table(region_reports, cvi_data, nmf_res, elapsed, fig_dir)

    # Print summary
    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE in {elapsed:.0f}s")
    print(f"{'='*70}")
    print(f"  Output: {fig_dir}")
    print(f"  Figures: {5 + (2 if segments_data else 0)} visualizations")
    print(f"  Report: risk_report.txt + pipeline_summary.json")
    print(f"  NMF components: {nmf_res['K']}")
    print(f"  CVI: mean={cvi_data['cvi'].mean():.3f}, "
          f"critical={cvi_data['stats']['critical']:.1f}%")
    if region_reports:
        print(f"  SAM regions: {len(region_reports)}")
    print(f"{'='*70}")
