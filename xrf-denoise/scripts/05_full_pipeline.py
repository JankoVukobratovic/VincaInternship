"""
Phase 6: Full end-to-end pipeline.

    Denoise -> Element Maps -> NMF -> CVI -> SAM Segmentation -> Risk Report

Loads trained UNet1D model, denoises the raw datacube, extracts element maps
from the denoised data, runs NMF blind decomposition, computes CVI (Chemical
Vulnerability Index), runs SAM segmentation, and generates a restaurator report
with risk table.

Usage:
    py -3.11 scripts/05_full_pipeline.py [--no-sam] [--dataset prova1]
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# Parent dir for vulnerability_mapping imports
VINCA_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(VINCA_ROOT))

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
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

ELEMENTI = list(cfg.elements.keys())  # ['Ca', 'Ti', 'Fe', 'Cu', 'Pb_La']

RISK_CMAP = LinearSegmentedColormap.from_list('risk', [
    (0.0, '#1a9641'), (0.3, '#a6d96a'), (0.5, '#ffffbf'),
    (0.7, '#fdae61'), (1.0, '#d7191c'),
])

POZNATI_PIKOVI = {
    'K':     3.31, 'Ca':    3.69, 'Ti':    4.51,
    'Fe Ka': 6.40, 'Fe Kb': 7.06, 'Cu Ka': 8.05,
    'Zn Ka': 8.64, 'Pb La': 10.55, 'Pb Lb': 12.61,
    'Sr Ka': 14.16, 'Sn Ka': 25.27,
}

PRAVILA_RIZIKA = [
    {'id': 'R1', 'el_a': 'Ti',    'el_b': 'Ca',    'w': 1.00,
     'naziv': 'Termalna inkompatibilnost',
     'opis': 'TiO2 restauracija preko CaCO3 maltera',
     'mehanizam': 'Razliciti koef. termalne ekspanzije (TiO2 vs CaCO3)'},
    {'id': 'R2', 'el_a': 'Cu',    'el_b': 'Cu',    'w': 0.85,
     'naziv': 'Degradacija azurita',
     'opis': 'Azurit -> malahit transformacija',
     'mehanizam': 'Cu3(CO3)2(OH)2 -> Cu2CO3(OH)2 u vlazi'},
    {'id': 'R3', 'el_a': 'Pb_La', 'el_b': 'Pb_La', 'w': 0.70,
     'naziv': 'Potamnjivanje olovne bele',
     'opis': 'PbCO3 -> PbS u prisustvu sumpora',
     'mehanizam': '2PbCO3*Pb(OH)2 -> PbS'},
    {'id': 'R4', 'el_a': 'Ti',    'el_b': 'Cu',    'w': 0.90,
     'naziv': 'Zarobljena vlaga pod restauracijom',
     'opis': 'TiO2 blokira difuziju vlage iz Cu-pigmenta',
     'mehanizam': 'Nepropusni TiO2 ubrzava Cu degradaciju'},
    {'id': 'R5', 'el_a': 'Fe',    'el_b': 'Pb_La', 'w': 0.55,
     'naziv': 'Fe/Pb interfejs degradacija',
     'opis': 'Fe2O3 katalizuje oksidaciju Pb-pigmenta',
     'mehanizam': 'Fe2O3 -> PbO2 na granici slojeva'},
]


# ═════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def norm_percentil(mapa, q_lo=8, q_hi=99):
    bg   = np.percentile(mapa, q_lo)
    peak = np.percentile(mapa, q_hi)
    return np.clip((mapa - bg) / (peak - bg + 1e-10), 0, 1)


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
    diffs = np.diff(errors)
    diffs2 = np.diff(diffs)
    K_opt = K_list[np.argmax(diffs2) + 2]
    print(f"  Optimal K = {K_opt}")

    model = NMF(n_components=K_opt, init='nndsvda', max_iter=1000, random_state=42)
    W = model.fit_transform(D_trim)
    H = model.components_

    # Identify peaks in each component
    nazivi = []
    for k in range(K_opt):
        peaks_idx, _ = find_peaks(H[k], height=np.max(H[k]) * 0.1,
                                  distance=5, prominence=np.max(H[k]) * 0.05)
        peaks_kev = energy_trim[peaks_idx]
        elementi = []
        for kev in peaks_kev:
            best_el, best_d = None, 999
            for el, el_kev in POZNATI_PIKOVI.items():
                d = abs(kev - el_kev)
                if d < best_d:
                    best_d = d
                    best_el = el
            if best_d < 0.4:
                elementi.append(best_el)
        nazivi.append(" + ".join(elementi) if elementi else f"Comp. {k+1}")

    mape_nmf = W.reshape(cfg.rows, cfg.cols, K_opt)

    return {
        'W': W, 'H': H, 'K': K_opt,
        'energy': energy_trim, 'nazivi': nazivi,
        'mape': mape_nmf, 'errors': errors, 'K_range': K_list,
    }


def compute_cvi(norm_maps):
    """Compute CVI per pixel from normalized element maps."""
    cvi = np.zeros((cfg.rows, cfg.cols))
    dominant_risk = np.zeros((cfg.rows, cfg.cols), dtype=int)
    risk_maps = {}

    for pi, pravilo in enumerate(PRAVILA_RIZIKA):
        a = norm_maps[pravilo['el_a']]
        b = norm_maps[pravilo['el_b']]
        w = pravilo['w']

        if pravilo['el_a'] == pravilo['el_b']:
            risk = w * a
        else:
            risk = w * np.sqrt(a * b)

        risk = gaussian_filter(risk, sigma=1.0)
        risk = np.clip(risk, 0, 1)
        risk_maps[pravilo['id']] = risk

        mask = risk > cvi
        cvi[mask] = risk[mask]
        dominant_risk[mask] = pi

        pct_high = np.mean(risk > 0.5) * 100
        print(f"  {pravilo['id']} ({pravilo['naziv'][:40]}): "
              f"max={risk.max():.3f}, mean={risk.mean():.3f}, "
              f">0.5: {pct_high:.1f}%")

    zona_nizak = np.mean(cvi < 0.25) * 100
    zona_umeren = np.mean((cvi >= 0.25) & (cvi < 0.50)) * 100
    zona_povisen = np.mean((cvi >= 0.50) & (cvi < 0.75)) * 100
    zona_kritican = np.mean(cvi >= 0.75) * 100

    print(f"\n  Composite CVI: max={cvi.max():.3f}, mean={cvi.mean():.3f}")
    print(f"  Zones: low={zona_nizak:.1f}%, moderate={zona_umeren:.1f}%, "
          f"elevated={zona_povisen:.1f}%, critical={zona_kritican:.1f}%")

    return {
        'cvi': cvi,
        'dominant_risk': dominant_risk,
        'risk_maps': risk_maps,
        'stats': {
            'nizak': zona_nizak, 'umeren': zona_umeren,
            'povisen': zona_povisen, 'kritican': zona_kritican,
        },
    }


def run_sam_segmentation(norm_maps):
    """Run SAM automatic segmentation on false-color XRF image."""
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
    sam_checkpoint = VINCA_ROOT / 'modeli' / 'sam_vit_b_01ec64.pth'
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

    # Downscale masks to original resolution
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

    # Assign unassigned pixels to nearest segment
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

    el_desc = {
        'Ca': 'Krecni malter (intonaco)',
        'Ti': 'Moderna restauracija (TiO2)',
        'Fe': 'Okra/hematit (Fe-pigment)',
        'Cu': 'Azurit (Cu-pigment)',
        'Pb_La': 'Olovna bela (Pb-pigment)',
    }

    region_reports = []
    for seg in segment_info:
        mask = segments == seg['id']
        n_px = mask.sum()
        if n_px == 0:
            continue

        el_means = {el: float(norm_maps[el][mask].mean()) for el in ELEMENTI}
        cvi_vals = cvi[mask]

        dom_vals = dominant_risk[mask]
        dom_counts = np.bincount(dom_vals, minlength=len(PRAVILA_RIZIKA))
        dom_idx = dom_counts.argmax()

        dominant_el = max(el_means, key=el_means.get)

        region_reports.append({
            'id': seg['id'],
            'area_px': n_px,
            'area_pct': n_px / cfg.n_pixels * 100,
            'el_means': el_means,
            'dominant_el': dominant_el,
            'material': el_desc.get(dominant_el, '?'),
            'cvi_mean': float(cvi_vals.mean()),
            'cvi_max': float(cvi_vals.max()),
            'dominant_risk': PRAVILA_RIZIKA[dom_idx],
            'pct_elevated': float(np.mean(cvi_vals >= 0.5) * 100),
            'pct_critical': float(np.mean(cvi_vals >= 0.75) * 100),
        })

    region_reports.sort(key=lambda r: r['cvi_mean'], reverse=True)
    return region_reports


# ═════════════════════════════════════════════════════════════════════════════
#  VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════

def plot_element_maps(maps_raw, maps_denoised, norm_den, fig_dir):
    """Raw vs denoised element maps, with difference row."""
    n_el = len(ELEMENTI)
    fig, axes = plt.subplots(3, n_el, figsize=(5*n_el, 13))

    for j, el in enumerate(ELEMENTI):
        vmax = np.percentile(np.concatenate([
            maps_raw[el].ravel(), maps_denoised[el].ravel()
        ]), 99)

        axes[0, j].imshow(maps_raw[el], origin='upper', cmap='hot',
                          vmin=0, vmax=vmax, interpolation='bicubic')
        axes[0, j].set_title(f'{el} — Raw', fontsize=10, fontweight='bold')
        axes[0, j].axis('off')

        axes[1, j].imshow(maps_denoised[el], origin='upper', cmap='hot',
                          vmin=0, vmax=vmax, interpolation='bicubic')
        axes[1, j].set_title(f'{el} — Denoised', fontsize=10, fontweight='bold')
        axes[1, j].axis('off')

        # Difference: Raw - Denoised (what the model removed = noise estimate)
        diff = maps_raw[el].astype(np.float64) - maps_denoised[el].astype(np.float64)
        diff_abs = np.abs(diff)
        diff_vmax = np.percentile(diff_abs, 99)
        im = axes[2, j].imshow(diff, origin='upper', cmap='RdBu_r',
                               vmin=-diff_vmax, vmax=diff_vmax,
                               interpolation='bicubic')
        axes[2, j].set_title(f'{el} — Difference', fontsize=10, fontweight='bold')
        axes[2, j].axis('off')
        plt.colorbar(im, ax=axes[2, j], fraction=0.046, pad=0.04, shrink=0.8)

    fig.suptitle('Element Maps: Raw vs Denoised vs Difference (Raw $-$ Denoised)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_dir / '01_element_maps_raw_vs_denoised.png',
                dpi=180, bbox_inches='tight')
    plt.close()
    print("  Saved: 01_element_maps_raw_vs_denoised.png")


def plot_nmf(nmf_res, fig_dir):
    """NMF spectral signatures + spatial maps."""
    K = nmf_res['K']
    H = nmf_res['H']
    energy = nmf_res['energy']
    mape = nmf_res['mape']
    nazivi = nmf_res['nazivi']

    colors = plt.cm.Set1(np.linspace(0, 1, K))

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(20, 3.5 * K))
    gs = GridSpec(K, 2, width_ratios=[1.5, 1], hspace=0.4, wspace=0.15)

    for k in range(K):
        ax_sp = fig.add_subplot(gs[k, 0])
        ax_sp.fill_between(energy, H[k], alpha=0.3, color=colors[k])
        ax_sp.plot(energy, H[k], color=colors[k], linewidth=1.5)
        ax_sp.set_title(f'Component {k+1}: {nazivi[k]}',
                        fontsize=11, fontweight='bold', color=colors[k])
        ax_sp.set_xlim(0, 14)
        ax_sp.grid(True, alpha=0.2)
        ax_sp.set_ylabel('Intensity')
        if k == K - 1:
            ax_sp.set_xlabel('Energy (keV)')

        peaks_idx, _ = find_peaks(H[k], height=np.max(H[k]) * 0.1,
                                  distance=5, prominence=np.max(H[k]) * 0.05)
        for pi in peaks_idx:
            kev = energy[pi]
            best_el, best_d = '', 999
            for el, el_kev in POZNATI_PIKOVI.items():
                if abs(kev - el_kev) < best_d:
                    best_d = abs(kev - el_kev)
                    best_el = el
            if best_d < 0.4:
                ax_sp.annotate(best_el, xy=(kev, H[k][pi]), fontsize=9,
                               fontweight='bold', ha='center', va='bottom',
                               xytext=(0, 6), textcoords='offset points',
                               arrowprops=dict(arrowstyle='->', lw=0.8))

        ax_map = fig.add_subplot(gs[k, 1])
        m = mape[:, :, k]
        p99 = np.percentile(m, 99)
        im = ax_map.imshow(m / max(p99, 1e-10), origin='upper', aspect='equal',
                           cmap='hot', interpolation='bicubic', vmin=0, vmax=1)
        ax_map.axis('off')
        plt.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)

    fig.suptitle('NMF Blind Decomposition of Denoised Spectra',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.savefig(fig_dir / '02_nmf_components.png', dpi=180, bbox_inches='tight')
    plt.close()
    print("  Saved: 02_nmf_components.png")


def plot_cvi(cvi_data, fig_dir):
    """CVI composite map with zone classification."""
    cvi = cvi_data['cvi']
    stats = cvi_data['stats']

    fig, axes = plt.subplots(1, 2, figsize=(24, 9))

    im = axes[0].imshow(cvi, origin='upper', aspect='equal', cmap=RISK_CMAP,
                        interpolation='bicubic', vmin=0, vmax=1)
    axes[0].set_title('Chemical Vulnerability Index (CVI)', fontsize=13, fontweight='bold')
    axes[0].axis('off')
    cb = plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    cb.set_label('CVI score')
    axes[0].text(0.02, 0.03,
                 f"Low (<0.25): {stats['nizak']:.1f}%\n"
                 f"Moderate (0.25-0.50): {stats['umeren']:.1f}%\n"
                 f"Elevated (0.50-0.75): {stats['povisen']:.1f}%\n"
                 f"Critical (>0.75): {stats['kritican']:.1f}%",
                 transform=axes[0].transAxes, fontsize=10, color='white',
                 va='bottom', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    klasifikacija = np.zeros((cfg.rows, cfg.cols), dtype=int)
    klasifikacija[cvi >= 0.25] = 1
    klasifikacija[cvi >= 0.50] = 2
    klasifikacija[cvi >= 0.75] = 3

    zone_colors = ['#1a9641', '#ffffbf', '#fdae61', '#d7191c']
    zone_cmap = LinearSegmentedColormap.from_list('zone', zone_colors, N=4)

    axes[1].imshow(klasifikacija, origin='upper', aspect='equal', cmap=zone_cmap,
                   interpolation='nearest', vmin=0, vmax=3)
    axes[1].set_title('Risk Zone Classification', fontsize=13, fontweight='bold')
    axes[1].axis('off')
    patches = [mpatches.Patch(color=c, label=l) for c, l in
               zip(zone_colors, ['Low risk', 'Moderate risk',
                                  'Elevated risk', 'Critical risk'])]
    axes[1].legend(handles=patches, loc='lower left', fontsize=10,
                   frameon=True, facecolor='white', framealpha=0.9)

    fig.suptitle('Chemical Vulnerability Map (from denoised XRF data)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_dir / '03_cvi_map.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 03_cvi_map.png")


def plot_risk_rules(cvi_data, fig_dir):
    """Individual risk rule maps."""
    fig, axes = plt.subplots(2, 3, figsize=(22, 13))
    axes_flat = axes.flatten()

    for i, pravilo in enumerate(PRAVILA_RIZIKA):
        ax = axes_flat[i]
        r = cvi_data['risk_maps'][pravilo['id']]
        im = ax.imshow(r, origin='upper', aspect='equal', cmap=RISK_CMAP,
                       interpolation='bicubic', vmin=0, vmax=1)
        ax.set_title(f"{pravilo['id']}: {pravilo['naziv']}\n"
                     f"({pravilo['el_a']}/{pravilo['el_b']})",
                     fontsize=10, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    axes_flat[len(PRAVILA_RIZIKA)].axis('off')

    fig.suptitle('Individual Risk Rule Maps (from denoised data)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_dir / '04_risk_rules.png', dpi=180, bbox_inches='tight')
    plt.close()
    print("  Saved: 04_risk_rules.png")


def plot_publication_figure(maps_raw, maps_denoised, nmf_res, cvi_data,
                           norm_maps, fig_dir):
    """Publication-quality composite figure for the paper (Fig. 1).
    3 rows: (a) raw element maps, (b) denoised, (c) CVI + zones + overlay.
    NMF excluded — one component captures background (wooden support), not pigment.
    """
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(18, 11))
    gs = GridSpec(3, 5, hspace=0.08, wspace=0.12,
                  left=0.03, right=0.97, top=0.93, bottom=0.02,
                  height_ratios=[1, 1, 1.15])

    cvi = cvi_data['cvi']

    # --- Row 1: Raw element maps ---
    for j, el in enumerate(ELEMENTI):
        ax = fig.add_subplot(gs[0, j])
        vmax = np.percentile(maps_raw[el], 99)
        ax.imshow(maps_raw[el], origin='upper', cmap='hot', vmin=0, vmax=vmax,
                  interpolation='bicubic', aspect='equal')
        ax.set_title(f'{el}\nRaw', fontsize=9, fontweight='bold')
        ax.axis('off')
    fig.text(0.005, 0.84, '(a)', fontsize=12, fontweight='bold', va='center')

    # --- Row 2: Denoised element maps ---
    for j, el in enumerate(ELEMENTI):
        ax = fig.add_subplot(gs[1, j])
        vmax = np.percentile(maps_raw[el], 99)
        ax.imshow(maps_denoised[el], origin='upper', cmap='hot', vmin=0, vmax=vmax,
                  interpolation='bicubic', aspect='equal')
        ax.set_title(f'{el}\nDenoised', fontsize=9, fontweight='bold')
        ax.axis('off')
    fig.text(0.005, 0.55, '(b)', fontsize=12, fontweight='bold', va='center')

    # --- Row 3: CVI map + zone classification + overlay ---
    ax_cvi = fig.add_subplot(gs[2, 0:2])
    im = ax_cvi.imshow(cvi, origin='upper', cmap=RISK_CMAP, vmin=0, vmax=1,
                       interpolation='bicubic', aspect='equal')
    ax_cvi.set_title('Chemical Vulnerability Index', fontsize=10, fontweight='bold')
    ax_cvi.axis('off')
    plt.colorbar(im, ax=ax_cvi, fraction=0.035, pad=0.02, shrink=0.85)

    ax_zone = fig.add_subplot(gs[2, 2:4])
    klasifikacija = np.zeros((cfg.rows, cfg.cols), dtype=int)
    klasifikacija[cvi >= 0.25] = 1
    klasifikacija[cvi >= 0.50] = 2
    klasifikacija[cvi >= 0.75] = 3
    zone_colors = ['#1a9641', '#ffffbf', '#fdae61', '#d7191c']
    zone_cmap = LinearSegmentedColormap.from_list('zone', zone_colors, N=4)
    ax_zone.imshow(klasifikacija, origin='upper', cmap=zone_cmap,
                   interpolation='nearest', vmin=0, vmax=3, aspect='equal')
    ax_zone.set_title('Risk Zone Classification', fontsize=10, fontweight='bold')
    ax_zone.axis('off')
    patches = [mpatches.Patch(color=c, label=l) for c, l in
               zip(zone_colors, ['Low', 'Moderate', 'Elevated', 'Critical'])]
    ax_zone.legend(handles=patches, loc='lower left', fontsize=7,
                   frameon=True, facecolor='white', framealpha=0.9)

    ax_over = fig.add_subplot(gs[2, 4])
    rgb_fc = np.stack([norm_maps['Fe'], norm_maps['Cu'], norm_maps['Pb_La']], axis=2)
    rgb_fc = np.clip(rgb_fc, 0, 1)
    ax_over.imshow(rgb_fc, origin='upper', interpolation='bicubic', aspect='equal')
    cs = ax_over.contour(cvi, levels=[0.3, 0.5, 0.7],
                         colors=['yellow', 'orange', 'red'],
                         linewidths=[0.8, 1.2, 1.8], origin='upper')
    ax_over.clabel(cs, inline=True, fontsize=6, fmt='%.1f')
    ax_over.set_title('CVI Overlay', fontsize=10, fontweight='bold')
    ax_over.axis('off')
    fig.text(0.005, 0.25, '(c)', fontsize=12, fontweight='bold', va='center')

    fig.suptitle('XRF Spectral Denoising and Chemical Vulnerability Mapping',
                 fontsize=14, fontweight='bold', y=0.97)

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
        color = rng.random(3) * 0.7 + 0.3
        mask = segments == seg['id']
        seg_rgb[mask] = color

    fig, axes = plt.subplots(1, 3, figsize=(28, 8))
    axes[0].imshow(rgb_input, origin='upper', aspect='equal', interpolation='bicubic')
    axes[0].set_title('SAM Input\n(R=Fe, G=Cu, B=Pb)', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(seg_rgb, origin='upper', aspect='equal', interpolation='nearest')
    axes[1].set_title(f'SAM Segmentation\n({n_segments} regions)',
                      fontsize=12, fontweight='bold')
    axes[1].axis('off')

    # Region CVI map
    region_cvi_map = np.zeros((cfg.rows, cfg.cols))
    for r in region_reports:
        mask = segments == r['id']
        region_cvi_map[mask] = r['cvi_mean']

    im = axes[2].imshow(region_cvi_map, origin='upper', aspect='equal',
                        cmap=RISK_CMAP, interpolation='nearest', vmin=0, vmax=1)
    axes[2].set_title('CVI per SAM Region', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle('SAM Segmentation + Chemical Vulnerability Analysis',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_dir / '05_sam_segmentation.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 05_sam_segmentation.png")

    # --- Figure 6: Summary panel ---
    material_colors = {
        'Ca': [0.93, 0.87, 0.72], 'Ti': [0.96, 0.96, 0.94],
        'Fe': [0.68, 0.20, 0.03], 'Cu': [0.09, 0.27, 0.70],
        'Pb_La': [0.94, 0.91, 0.80],
    }
    mat_rgb = np.zeros((cfg.rows, cfg.cols, 3))
    for r in region_reports:
        mask = segments == r['id']
        mat_rgb[mask] = material_colors[r['dominant_el']]

    fig, axes = plt.subplots(2, 3, figsize=(28, 16))
    axes[0,0].imshow(rgb_input, origin='upper', aspect='equal', interpolation='bicubic')
    axes[0,0].set_title('XRF False-color', fontsize=11, fontweight='bold')
    axes[0,0].axis('off')

    axes[0,1].imshow(seg_rgb, origin='upper', aspect='equal', interpolation='nearest')
    axes[0,1].set_title(f'SAM ({n_segments} regions)', fontsize=11, fontweight='bold')
    axes[0,1].axis('off')

    axes[0,2].imshow(mat_rgb, origin='upper', aspect='equal', interpolation='nearest')
    axes[0,2].set_title('Dominant Material', fontsize=11, fontweight='bold')
    axes[0,2].axis('off')

    im = axes[1,0].imshow(cvi, origin='upper', aspect='equal', cmap=RISK_CMAP,
                          interpolation='bicubic', vmin=0, vmax=1)
    axes[1,0].set_title('CVI Pixel Map', fontsize=11, fontweight='bold')
    axes[1,0].axis('off')
    plt.colorbar(im, ax=axes[1,0], fraction=0.046, pad=0.04)

    im = axes[1,1].imshow(region_cvi_map, origin='upper', aspect='equal',
                          cmap=RISK_CMAP, interpolation='nearest', vmin=0, vmax=1)
    axes[1,1].set_title('CVI per SAM Region', fontsize=11, fontweight='bold')
    axes[1,1].axis('off')
    plt.colorbar(im, ax=axes[1,1], fraction=0.046, pad=0.04)

    # CVI overlay on XRF
    axes[1,2].imshow(rgb_input, origin='upper', aspect='equal', interpolation='bicubic')
    cs = axes[1,2].contour(cvi, levels=[0.3, 0.5, 0.7],
                           colors=['yellow', 'orange', 'red'],
                           linewidths=[1, 1.5, 2], origin='upper')
    axes[1,2].clabel(cs, inline=True, fontsize=8, fmt='%.1f')
    axes[1,2].set_title('CVI Overlay on XRF', fontsize=11, fontweight='bold')
    axes[1,2].axis('off')

    fig.suptitle('Full Pipeline Summary: Denoise + NMF + CVI + SAM',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_dir / '06_full_summary.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 06_full_summary.png")


def generate_risk_table(region_reports, cvi_data, nmf_res, elapsed, fig_dir):
    """Generate risk table and restaurator report as text + JSON."""
    cvi = cvi_data['cvi']

    # Text report
    lines = []
    lines.append("=" * 75)
    lines.append("  FRESCO CHEMICAL VULNERABILITY REPORT")
    lines.append("  Auto-generated: Denoise + NMF + CVI + SAM Pipeline")
    lines.append("=" * 75)
    lines.append("")
    lines.append("METHODOLOGY:")
    lines.append("  1. UNet1D denoising (Noise2Noise / Poisson splitting)")
    lines.append("  2. NMF blind spectral decomposition")
    lines.append("  3. Chemical Vulnerability Index (CVI) — 5 degradation rules")
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
        lines.append("  RISK TABLE — TOP REGIONS (sorted by CVI)")
        lines.append("=" * 75)
        lines.append("")
        lines.append(f"  {'#':>3} {'Region':>7} {'Area%':>6} {'CVI':>6} "
                     f"{'Max':>6} {'Level':>10} {'Material':>25} {'Risk':>30}")
        lines.append(f"  {'-'*97}")

        for i, r in enumerate(region_reports[:20]):
            level = ('CRITICAL' if r['cvi_mean'] >= 0.5 else
                     'MODERATE' if r['cvi_mean'] >= 0.25 else 'LOW')
            lines.append(
                f"  {i+1:3d} R{r['id']:>5d} {r['area_pct']:5.1f}% "
                f"{r['cvi_mean']:6.3f} {r['cvi_max']:6.3f} {level:>10} "
                f"{r['material']:>25} {r['dominant_risk']['id']}: "
                f"{r['dominant_risk']['naziv'][:24]}"
            )

        lines.append("")
        lines.append("=" * 75)
        lines.append("  RECOMMENDATIONS")
        lines.append("=" * 75)
        lines.append("")

        critical = [r for r in region_reports if r['cvi_mean'] >= 0.5]
        moderate = [r for r in region_reports if 0.25 <= r['cvi_mean'] < 0.5]

        if critical:
            lines.append(f"  URGENT: {len(critical)} regions require immediate attention:")
            for r in critical[:5]:
                lines.append(f"    - Region {r['id']}: {r['material']} "
                             f"(CVI={r['cvi_mean']:.3f}, "
                             f"{r['dominant_risk']['id']}: {r['dominant_risk']['naziv']})")
            lines.append("")

        if moderate:
            lines.append(f"  MONITOR: {len(moderate)} regions with moderate risk:")
            for r in moderate[:5]:
                lines.append(f"    - Region {r['id']}: {r['material']} "
                             f"(CVI={r['cvi_mean']:.3f})")
            lines.append("")

    lines.append("=" * 75)
    lines.append("  RISK RULES LEGEND")
    lines.append("=" * 75)
    for p in PRAVILA_RIZIKA:
        lines.append(f"  {p['id']}: {p['naziv']} (w={p['w']})")
        lines.append(f"      {p['opis']}")
        lines.append(f"      Mechanism: {p['mehanizam']}")
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
    print("\n[2/7] Denoising datacube with trained UNet1D...")
    model_path = cfg.abs_path(cfg.exp_a_dir) / "checkpoints" / "best_model.pt"
    if not model_path.exists():
        print(f"  ERROR: Trained model not found at {model_path}")
        print(f"  Run 03a_train_scratch.py first!")
        sys.exit(1)

    # Load global scale from training
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
    norm_maps = {el: norm_percentil(maps_denoised[el]) for el in ELEMENTI}
    print(f"  Elements: {', '.join(ELEMENTI)}")

    # ─── Step 4: NMF ──────────────────────────────────────────────────────
    print("\n[4/7] NMF blind decomposition on denoised spectra...")
    spectra_flat = cube_denoised.reshape(-1, cube_denoised.shape[-1])
    nmf_res = run_nmf(spectra_flat)

    # ─── Step 5: CVI ──────────────────────────────────────────────────────
    print("\n[5/7] Computing Chemical Vulnerability Index...")
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
                level = ('CRITICAL' if r['cvi_mean'] >= 0.5 else
                         'MODERATE' if r['cvi_mean'] >= 0.25 else 'LOW')
                print(f"    R{r['id']:2d}: CVI={r['cvi_mean']:.3f} [{level}] "
                      f"— {r['material']}")
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
    print(f"  Figures: {4 + (2 if segments_data else 0)} visualizations")
    print(f"  Report: risk_report.txt + pipeline_summary.json")
    print(f"  NMF components: {nmf_res['K']}")
    print(f"  CVI: mean={cvi_data['cvi'].mean():.3f}, "
          f"critical={cvi_data['stats']['kritican']:.1f}%")
    if region_reports:
        print(f"  SAM regions: {len(region_reports)}")
    print(f"{'='*70}")
