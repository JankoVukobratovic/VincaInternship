"""
05b_logratio_maps.py
Per-element log-ratio maps log(D10264 / D19511) for all three scans -
the geometry-tracking counterpart of the abundance-tracking difference
maps in 05_detector_diff.py (diff maps follow element abundance; ratio
maps follow geometry, because composition cancels pixel-wise to first
order).

For every scan and every reliable element line the script saves:

    logratio  = log(D1 / D2)             (masked where counts are low)
    smooth    = NaN-aware Gaussian smooth of logratio
                -> map of geometric non-uniformity
    residual  = logratio - smooth        -> canvas-topography proxy

plus a per-scan median geometry map: the per-element mean is removed
(dropping the detector-efficiency part of the ratio) and the median is
taken across elements, which suppresses element-specific noise and
leaves the shared spatial-geometry component.

Uses the npy cache built by 05/06. Outputs under results/logratio/<scan>/:

    logratio_<El>.npy / smooth_<El>.npy / residual_<El>.npy
    median_geometry.npy
    logratio_grid.png, smooth_grid.png, median_geometry.png

Run from the project root:
    python scripts/05b_logratio_maps.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

DETECTORS  = ["10264", "19511"]
CACHE_DIR  = os.path.join("results", "detector_diff", "_npy_cache")
OUTPUT_DIR = os.path.join("results", "logratio")

GRIDS = {"prova1": (120, 60), "prova2": (120, 60), "ruotato": (80, 45)}

# the 8 lines used in the ratio analysis (K/Zn excluded as unreliable)
ELEMENTS = ["Ca", "Ti", "Fe", "Cu", "PbLl", "PbLa", "PbLb", "PbLg"]

MIN_COUNTS = 10.0   # mask pixels where either detector is below this


def load_map(scan: str, det: str, el: str) -> np.ndarray:
    p = os.path.join(CACHE_DIR, f"{scan}_{det}_{el}.npy")
    if not os.path.exists(p):
        sys.exit(f"ERROR: cache missing: {p} - run scripts/06_efficiency_ratios.py first.")
    return np.load(p)


def nan_gaussian(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing that ignores NaNs (normalized convolution)."""
    valid = np.isfinite(img).astype(float)
    filled = np.where(np.isfinite(img), img, 0.0)
    num = gaussian_filter(filled, sigma)
    den = gaussian_filter(valid, sigma)
    with np.errstate(invalid="ignore"):
        out = num / den
    out[den < 1e-3] = np.nan
    return out


def plot_grid(maps: dict[str, np.ndarray], title: str, path: str,
              center_zero: bool = True):
    n = len(maps)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    any_map = next(iter(maps.values()))
    h, w = any_map.shape
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.4 * ncols, 4.4 * ncols * h / w * 0.9),
                             dpi=110, layout="constrained", squeeze=False)
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    for ax, (name, img) in zip(axes, maps.items()):
        if center_zero:
            am = np.nanpercentile(np.abs(img), 99)
            am = am if np.isfinite(am) and am > 0 else 1.0
            im = ax.imshow(img, cmap="RdBu_r", vmin=-am, vmax=am, aspect="equal")
        else:
            im = ax.imshow(img, cmap="magma", aspect="equal")
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        cb = plt.colorbar(im, ax=ax, fraction=0.03)
        cb.ax.tick_params(labelsize=6)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


if __name__ == "__main__":
    for scan, (width, height) in GRIDS.items():
        out_dir = os.path.join(OUTPUT_DIR, scan)
        os.makedirs(out_dir, exist_ok=True)
        sigma = max(2.0, width / 20)
        print(f"\n=== {scan}  ({width}x{height} px, smooth sigma={sigma:.1f} px) ===")

        logratios, smooths, residuals, centered = {}, {}, {}, []
        for el in ELEMENTS:
            d1 = load_map(scan, "10264", el)
            d2 = load_map(scan, "19511", el)
            with np.errstate(divide="ignore", invalid="ignore"):
                lr = np.log(d1 / d2)
            lr[(d1 < MIN_COUNTS) | (d2 < MIN_COUNTS)] = np.nan

            sm = nan_gaussian(lr, sigma)
            logratios[el] = lr
            smooths[el]   = sm
            residuals[el] = lr - sm
            centered.append(lr - np.nanmean(lr))

            np.save(os.path.join(out_dir, f"logratio_{el}.npy"), lr)
            np.save(os.path.join(out_dir, f"smooth_{el}.npy"), sm)
            np.save(os.path.join(out_dir, f"residual_{el}.npy"), lr - sm)
            print(f"  {el:5s}: mean log R = {np.nanmean(lr):+.4f}  "
                  f"(R = {np.exp(np.nanmean(lr)):.3f})  "
                  f"valid px = {np.isfinite(lr).mean() * 100:.0f}%")

        median_geom = np.nanmedian(np.stack(centered), axis=0)
        np.save(os.path.join(out_dir, "median_geometry.npy"), median_geom)

        # figures: per-element log-ratio (mean removed for display),
        # smooth geometric component, median geometry map
        plot_grid(
            {el: logratios[el] - np.nanmean(logratios[el]) for el in ELEMENTS},
            f"log(D1/D2) per element, mean removed — {scan}",
            os.path.join(out_dir, "logratio_grid.png"),
        )
        plot_grid(
            {el: smooths[el] - np.nanmean(smooths[el]) for el in ELEMENTS},
            f"Smooth component (geometric non-uniformity) — {scan}, σ={sigma:.0f} px",
            os.path.join(out_dir, "smooth_grid.png"),
        )
        fig, ax = plt.subplots(figsize=(8, 8 * height / width + 0.8),
                               dpi=130, layout="constrained")
        am = np.nanpercentile(np.abs(median_geom), 99)
        im = ax.imshow(median_geom, cmap="RdBu_r", vmin=-am, vmax=am, aspect="equal")
        ax.set_title(f"Median geometry map (element-median of centered log-ratios)\n"
                     f"{scan}", fontsize=11, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046 * height / width, pad=0.02,
                     label="Δ log(D1/D2)")
        p = os.path.join(out_dir, "median_geometry.png")
        fig.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {p}")

    print(f"\nDone -> {os.path.abspath(OUTPUT_DIR)}/")
