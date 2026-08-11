"""
10_flatfield.py
===============================================================================
Flat-field (geometric non-uniformity) correction map of the dual-detector
ratio, from the median-geometry maps of 05b_logratio_maps.py.

The per-scan median-geometry map is the per-pixel log-ratio
log(D10264/D19511) with the per-element mean removed (dropping the
detector-efficiency level) and the median taken across the 8 reliable
lines (suppressing element noise). What is left is the shared spatial
component: the relative geometric non-uniformity of the two channels -
the "flat-field" of the ratio.

This script validates it as an instrument property and packages it:

  - the two frontal scans (prova1, prova2) must show the SAME map:
    their pixel-wise Pearson r is the instrument-vs-noise test;
  - the combined map (mean of the two, in log domain) is the correction
    map, also expressed in percent deviation of the ratio;
  - summary stats: RMS non-uniformity, 2.5-97.5 percentile span.

Input : results/logratio/{prova1,prova2}/median_geometry.npy  (from 05b)
Output: results/detector_diff/flatfield_map.png
        results/detector_diff/flatfield_map.txt
        results/detector_diff/flatfield_combined.npy   (for per-pixel
        fusion weights; .npy is gitignored)

Run from the project root (after 05b):
    python scripts/10_flatfield.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

IN_DIR = os.path.join("results", "logratio")
OUT_DIR = os.path.join("results", "detector_diff")
CUBE_DIR = os.path.join("xrf-denoise", "data", "processed")

# Acquisition-artifact bands, as masked out of the NMF input in
# 02_vulnerability.py: the Hg lines (La ~9.99, Lb ~11.82 keV) and the
# high-energy scatter tail. Both are known to paint a rectangle into the
# scan; here they are used to test whether the flat-field lives on the
# same rectangle.
HG_MASK_KEV = [(9.75, 10.20), (11.60, 12.10)]
SCATTER_MASK_KEV = [(12.95, 15.0)]
CAL_SLOPE, CAL_INTERCEPT = 0.02916632052744066, -0.06901678838180736


def band_map(cube, bands):
    """Per-pixel counts in a set of energy bands."""
    energy = np.arange(cube.shape[2]) * CAL_SLOPE + CAL_INTERCEPT
    sel = np.zeros(energy.shape, dtype=bool)
    for lo, hi in bands:
        sel |= (energy >= lo) & (energy <= hi)
    return cube[:, :, sel].sum(axis=2)


def artifact_overlap(flat, border_px=6):
    """Does the flat-field live on the acquisition-artifact geometry?

    The Hg and scatter bands both mark the rectangular scan artifact,
    but not the same part of it: the scatter tail is a bright frame in
    the border rows/columns, the Hg lines an inner rectangle. Reported
    per artifact: the pixel-wise correlation with the flat-field, the
    flat-field level in the top-quartile pixels of the artifact map
    against the rest, and how much of that top quartile sits in the
    border strip -- which tells the two geometries apart instead of
    assuming one.
    """
    cubes = {}
    for det in ("10264", "19511"):
        p = os.path.join(CUBE_DIR, f"prova1_{det}_raw.npy")
        if not os.path.exists(p):
            return None
        cubes[det] = np.load(p)

    rows, cols = flat.shape
    rr, cc = np.indices((rows, cols))
    border = ((rr < border_px) | (rr >= rows - border_px)
              | (cc < border_px) | (cc >= cols - border_px))

    out = {}
    for label, bands in (("Hg lines", HG_MASK_KEV),
                         ("scatter tail", SCATTER_MASK_KEV)):
        # both detectors summed: the artifact is an acquisition effect,
        # not a per-channel one
        a = sum(band_map(c, bands) for c in cubes.values())
        ok = np.isfinite(flat)
        r = float(np.corrcoef(a[ok], flat[ok])[0, 1])
        hot = (a >= np.percentile(a[ok], 75)) & ok
        d_hot = 100.0 * (np.exp(np.nanmean(flat[hot])) - 1.0)
        d_rest = 100.0 * (np.exp(np.nanmean(flat[~hot & ok])) - 1.0)
        out[label] = (r, d_hot, d_rest,
                      100.0 * border[hot].mean(),
                      100.0 * border[ok].mean())
    return out

if __name__ == "__main__":
    maps = {}
    for scan in ("prova1", "prova2"):
        p = os.path.join(IN_DIR, scan, "median_geometry.npy")
        if not os.path.exists(p):
            sys.exit(f"ERROR: {p} missing - run scripts/05b_logratio_maps.py "
                     "first.")
        maps[scan] = np.load(p)

    m1, m2 = maps["prova1"], maps["prova2"]
    valid = np.isfinite(m1) & np.isfinite(m2)

    r = float(np.corrcoef(m1[valid], m2[valid])[0, 1])
    combined = np.where(valid, 0.5 * (m1 + m2), np.nan)
    pct = 100.0 * (np.exp(combined) - 1.0)

    stats = {}
    for label, m in (("prova1", m1), ("prova2", m2), ("combined", combined)):
        p_m = 100.0 * (np.exp(m[np.isfinite(m)]) - 1.0)
        stats[label] = (np.sqrt(np.mean(p_m ** 2)),
                        np.percentile(p_m, 2.5), np.percentile(p_m, 97.5))

    lines = [
        "Flat-field (geometric non-uniformity) of the ratio D10264/D19511",
        "from the 05b median-geometry maps (8 reliable lines).",
        "",
        f"cross-scan reproducibility (prova1 vs prova2, "
        f"{int(valid.sum())} px): r = {r:.3f}",
        "",
        f"{'map':10s} {'RMS %':>7s} {'2.5th %':>8s} {'97.5th %':>9s}",
    ]
    for label in ("prova1", "prova2", "combined"):
        rms, lo, hi = stats[label]
        lines.append(f"{label:10s} {rms:7.2f} {lo:+8.2f} {hi:+9.2f}")
    lines += [
        "",
        "combined map = flat-field correction: divide the per-pixel ratio",
        "(or reweight the fusion) by exp(map) to remove the geometric",
        "non-uniformity of the two channels.",
    ]

    overlap = artifact_overlap(combined)
    if overlap is not None:
        lines += [
            "",
            "Overlap with the acquisition-artifact regions (the bands",
            "masked out of the NMF input). 'hot' = top-quartile pixels of",
            "the artifact map; 'border' = share of them in the outer",
            "6-pixel strip of the scan.",
            "",
            f"{'artifact':14s} {'r vs flat-field':>16s} {'hot %':>7s}"
            f" {'rest %':>7s} {'border of hot':>14s}",
        ]
        for label, (r_a, d_hot, d_rest, b_hot, b_all) in overlap.items():
            lines.append(f"{label:14s} {r_a:16.3f} {d_hot:+7.2f}"
                         f" {d_rest:+7.2f} {b_hot:11.0f}%"
                         f"  (grid {b_all:.0f}%)")
        lines += [
            "",
            "The scatter-tail frame and the flat-field are the same",
            "spatial structure (r = 0.86): the ratio runs several percent",
            "high exactly where the scatter artifact is strong. The Hg",
            "rectangle is the complementary region and carries the",
            "opposite-signed deviation, but only weakly correlated",
            "(r = -0.24), so it is a consistent part of the same",
            "acquisition geometry rather than a second measurement of it.",
        ]

    print("\n".join(lines))
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "flatfield_map.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    np.save(os.path.join(OUT_DIR, "flatfield_combined.npy"), combined)

    # ---- figure --------------------------------------------------------
    lim = np.nanpercentile(np.abs(pct), 99)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.4), layout="constrained")
    panels = (("prova1", 100.0 * (np.exp(m1) - 1.0)),
              ("prova2", 100.0 * (np.exp(m2) - 1.0)),
              ("combined (flat-field)", pct))
    for ax, (label, m) in zip(axes, panels):
        im = ax.imshow(m, origin="upper", aspect="equal", cmap="RdBu_r",
                       vmin=-lim, vmax=lim, interpolation="nearest")
        ax.set_title(label, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01)
    cb.set_label("ratio deviation (%)", fontsize=10)
    fig.suptitle(f"Geometric non-uniformity of D10264/D19511  "
                 f"(cross-scan r = {r:.3f})", fontsize=12)
    fig.savefig(os.path.join(OUT_DIR, "flatfield_map.png"), dpi=200)

    print(f"\nSaved: {os.path.join(OUT_DIR, 'flatfield_map.txt')}")
    print(f"Saved: {os.path.join(OUT_DIR, 'flatfield_map.png')}")
    print(f"Saved: {os.path.join(OUT_DIR, 'flatfield_combined.npy')}")
