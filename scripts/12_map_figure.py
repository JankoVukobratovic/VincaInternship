"""
12_map_figure.py
===============================================================================
The map figure of the dual-detector paper: what the two channels see
differently, and where.

Four panels, all on the frontal (prova1) grid:

  a) an element map (Pb La, summed detectors) - what the instrument is
     normally used for, for scale and orientation;
  b) the smooth component of log(d10264/d19511) for Ca and for Pb La
     side by side - two lines at opposite ends of the response curve.
     Composition cancels pixel-wise to first order, so what is left is
     geometry; each panel carries its own colour scale because the Ca
     ratio is an order of magnitude noisier per pixel;
  c) the geometric non-uniformity (median over the reliable lines with
     the per-element mean removed, both frontal scans) - the flat-field
     of the ratio;
  d) the scatter-tail band that marks the acquisition artifact, which is
     the same spatial structure as (c) (r = 0.86, script 10).

Panel (d) is the point of the figure: the non-uniformity of the channel
ratio is not a detector quirk drawn on top of the painting, it follows
the acquisition geometry that also produces the artifact bands masked
out of the NMF input.

Input : results/logratio/prova1/*.npy            (script 05b)
        results/detector_diff/_npy_cache/*.npy   (script 06)
        xrf-denoise/data/processed/prova1_*_raw.npy
Output: results/detector_diff/map_figure.png

Run from the project root:
    python scripts/12_map_figure.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

LOGRATIO_DIR = os.path.join("results", "logratio", "prova1")
CACHE_DIR = os.path.join("results", "detector_diff", "_npy_cache")
CUBE_DIR = os.path.join("xrf-denoise", "data", "processed")
OUT = os.path.join("results", "detector_diff", "map_figure.png")

SCATTER_KEV = (12.95, 15.0)
CAL_SLOPE, CAL_INTERCEPT = 0.02916632052744066, -0.06901678838180736


def need(path):
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} missing - run scripts 05b/06 first.")
    return np.load(path)


def band_map(cube, lo, hi):
    energy = np.arange(cube.shape[2]) * CAL_SLOPE + CAL_INTERCEPT
    sel = (energy >= lo) & (energy <= hi)
    return cube[:, :, sel].sum(axis=2)


def show(ax, m, title, cmap="RdBu_r", lim=None, pct=True):
    data = 100.0 * (np.exp(m) - 1.0) if pct else m
    if lim is None:
        lim = np.nanpercentile(np.abs(data), 98)
    kw = dict(vmin=-lim, vmax=lim) if pct else dict(
        vmin=np.nanpercentile(data, 2), vmax=np.nanpercentile(data, 98))
    im = ax.imshow(data, origin="upper", aspect="equal", cmap=cmap,
                   interpolation="nearest", **kw)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


if __name__ == "__main__":
    pb = sum(need(os.path.join(CACHE_DIR, f"prova1_{d}_PbLa.npy"))
             for d in ("10264", "19511"))
    lr_ca = need(os.path.join(LOGRATIO_DIR, "smooth_Ca.npy"))
    lr_pb = need(os.path.join(LOGRATIO_DIR, "smooth_PbLa.npy"))
    # the paper's flat-field is the two frontal scans combined (script 10);
    # fall back to prova1 alone if 10 has not been run yet
    combined = os.path.join("results", "detector_diff",
                            "flatfield_combined.npy")
    if os.path.exists(combined):
        geom, geom_src = np.load(combined), "combined (script 10)"
    else:
        geom = need(os.path.join(LOGRATIO_DIR, "median_geometry.npy"))
        geom_src = "prova1 only - run script 10 for the paper's map"
    print(f"  flat-field source: {geom_src}")

    scatter = sum(
        band_map(need(os.path.join(CUBE_DIR, f"prova1_{d}_raw.npy")),
                 *SCATTER_KEV) for d in ("10264", "19511"))

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 5.6), layout="constrained")

    im0 = show(axes[0, 0], pb, "Pb La map (summed detectors)",
               cmap="inferno", pct=False)
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.03, pad=0.02,
                 label="net counts")

    im1 = show(axes[0, 1], lr_ca - np.nanmean(lr_ca),
               "ratio, Ca Ka (3.69 keV), smoothed")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.03, pad=0.02,
                 label="deviation (%)")
    im1b = show(axes[0, 2], lr_pb - np.nanmean(lr_pb),
                "ratio, Pb La (10.54 keV), smoothed")
    fig.colorbar(im1b, ax=axes[0, 2], fraction=0.03, pad=0.02,
                 label="deviation (%)")

    im2 = show(axes[1, 0], geom,
               "geometric non-uniformity (flat-field)")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.03, pad=0.02,
                 label="deviation (%)")

    im3 = show(axes[1, 1], scatter, "scatter-tail band (12.95-15 keV)",
               cmap="viridis", pct=False)
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.03, pad=0.02,
                 label="counts")

    # panel f: the two against each other, pixel by pixel
    ok = np.isfinite(geom)
    ax = axes[1, 2]
    ax.scatter(scatter[ok], 100.0 * (np.exp(geom[ok]) - 1.0), s=2,
               alpha=0.25, color="C0", edgecolors="none")
    r = float(np.corrcoef(scatter[ok], geom[ok])[0, 1])
    ax.set_title(f"non-uniformity vs artifact  (r = {r:.2f})", fontsize=10)
    ax.set_xlabel("scatter-band counts", fontsize=9)
    ax.set_ylabel("ratio deviation (%)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3)

    for ax, letter in zip(axes.ravel(), "abcdef"):
        ax.text(0.02, 0.96, f"({letter})", transform=ax.transAxes,
                fontsize=10, va="top", ha="left",
                color="white" if letter in "ad" else "black",
                bbox=dict(boxstyle="round,pad=0.15", fc="black" if
                          letter in "ad" else "white", ec="none", alpha=0.45))

    fig.savefig(OUT, dpi=200)
    print(f"Saved: {OUT}")
    print(f"  non-uniformity vs scatter artifact: r = {r:.3f}"
          f"  [{geom_src}]")
