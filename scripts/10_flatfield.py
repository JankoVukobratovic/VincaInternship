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
