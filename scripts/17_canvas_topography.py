"""
17_canvas_topography.py
===============================================================================
Canvas topography from the detector disagreement: per-pixel local surface
slope, in degrees, from a single frontal scan.

Principle: the tilt experiment measured how the detector ratio
R = D10264/D19511 responds to a change of surface orientation -- the
per-line response k_e = d(ln R)/d(theta) (script 08/overlap table:
+9.5% at Ca Ka down to +0.6% at Pb Lg for the 7.7 deg tilt). A LOCAL
tilt of the canvas surface (weave, relief, cupping) must move the local
ratio by the same law. The per-pixel log-ratio residuals of script 05b
(logratio minus its large-scale smooth component -- the flat-field) are
therefore an 8-fold repeated measurement of the local slope:

    residual_e(p) ~= k_e * theta(p)   =>   weighted LS over the 8 lines

theta(p) is the slope component along the axis probed by the tilt
experiment (the detector-separation direction); the orthogonal
component is invisible to this estimator.

Honest gate: the recovered theta map must REPRODUCE between the two
independent frontal scans (7 days apart). The cross-scan correlation of
theta and the per-line consistency decide whether this is real surface
structure or noise; both are reported, and a null result is a result.

Inputs : results/logratio/{prova1,prova2}/residual_<El>.npy   (05b)
         results/registration/overlap_ratios.csv              (08)
Outputs: results/detector_diff/canvas_topography.png
         results/detector_diff/canvas_topography.txt
         results/detector_diff/canvas_topography_combined.npy (deg,
         gitignored)

Run from the project root (after 05b):
    python scripts/17_canvas_topography.py
"""

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

IN_DIR = os.path.join("results", "logratio")
OUT_DIR = os.path.join("results", "detector_diff")
RATIOS = os.path.join("results", "registration", "overlap_ratios.csv")

LINES = ["Ca", "Ti", "Fe", "Cu", "PbLl", "PbLa", "PbLb", "PbLg"]
TILT_DEG = 7.7   # angle at which the per-line response was measured


def line_sensitivities():
    """k_e = d(ln R)/d(theta) in 1/deg, from the measured overlap tilt
    shift at 7.7 deg (linear small-angle approximation; the fitted
    stage-2 model is close to linear over this range)."""
    k = {}
    with open(RATIOS, newline="") as f:
        for row in csv.DictReader(f):
            el = row["element"]
            if el in LINES:
                shift = float(row["tilt_overlap_pct"]) / 100.0
                k[el] = np.log1p(shift) / TILT_DEG
    return k


def load_residuals(scan):
    res = {}
    for el in LINES:
        p = os.path.join(IN_DIR, scan, f"residual_{el}.npy")
        if not os.path.exists(p):
            sys.exit(f"ERROR: {p} missing - run scripts/05b_logratio_maps.py")
        res[el] = np.load(p)
    return res


if __name__ == "__main__":
    k = line_sensitivities()
    res1 = load_residuals("prova1")
    res2 = load_residuals("prova2")

    # Per-line noise of the residual maps from the scan pair
    sigma = {}
    for el in LINES:
        d = (res1[el] - res2[el])
        d = d[np.isfinite(d)]
        sigma[el] = np.std(d) / np.sqrt(2.0)

    lines_rep = [
        "Canvas topography from detector disagreement",
        "(per-pixel weighted LS over 8 lines; slope component along the",
        " tilt-experiment axis; sensitivities from the measured 7.7-deg",
        " overlap tilt shift)",
        "",
        f"{'line':6s} {'k (%/deg)':>10s} {'res noise %':>12s}"
        f" {'implied theta noise (deg)':>26s}",
    ]
    for el in LINES:
        th_noise = 100 * sigma[el] / (100 * k[el]) if k[el] != 0 else np.inf
        lines_rep.append(f"{el:6s} {100 * k[el]:10.3f} {100 * sigma[el]:12.2f}"
                         f" {th_noise:26.1f}")

    # Weighted least squares per pixel: theta = sum(w k res/k) ... i.e.
    # theta_hat = sum_e (k_e res_e / s_e^2) / sum_e (k_e^2 / s_e^2)
    def theta_map(res):
        num = np.zeros_like(res[LINES[0]])
        den = 0.0
        for el in LINES:
            w = 1.0 / sigma[el] ** 2
            num = num + w * k[el] * np.nan_to_num(res[el])
            den = den + w * k[el] ** 2
        return num / den

    th1, th2 = theta_map(res1), theta_map(res2)
    th_err = 1.0 / np.sqrt(sum(k[el] ** 2 / sigma[el] ** 2 for el in LINES))

    # Discriminator: geometry must scale ACROSS the 8 lines with k_e;
    # composition/thickness effects need not. Per-pixel chi2 of the
    # one-parameter (theta) fit, dof = 7.
    def chi2_map(res, th):
        c = np.zeros_like(th)
        for el in LINES:
            c = c + ((np.nan_to_num(res[el]) - k[el] * th)
                     / sigma[el]) ** 2
        return c / (len(LINES) - 1)

    chi1 = chi2_map(res1, th1)
    chi2m = chi2_map(res2, th2)

    valid = np.isfinite(th1) & np.isfinite(th2)
    r_cross = float(np.corrcoef(th1[valid], th2[valid])[0, 1])
    comb = 0.5 * (th1 + th2)

    # split the reproducible part from noise: RMS of the mean vs the diff
    rms1, rms2 = np.std(th1[valid]), np.std(th2[valid])
    rms_noise = np.std((th1 - th2)[valid]) / np.sqrt(2.0)
    rms_signal2 = max(0.0, 0.5 * (rms1 ** 2 + rms2 ** 2) - rms_noise ** 2)
    rms_signal = np.sqrt(rms_signal2)

    lines_rep += [
        "",
        f"per-pixel theta uncertainty (propagated): {th_err:.2f} deg",
        f"RMS theta, prova1 / prova2: {rms1:.2f} / {rms2:.2f} deg",
        f"RMS noise (from scan pair): {rms_noise:.2f} deg",
        f"RMS reproducible topographic signal: {rms_signal:.2f} deg",
        f"cross-scan reproducibility of theta: r = {r_cross:.3f}"
        f"  ({int(valid.sum())} px)",
        "",
        "Tilt-law consistency (per-pixel chi2/dof of the one-parameter",
        "fit across the 8 lines; ~1 = residuals scale like geometry,",
        ">> 1 = line-specific/composition effects dominate):",
        f"  median chi2/dof: prova1 {np.median(chi1[valid]):.2f},"
        f" prova2 {np.median(chi2m[valid]):.2f}",
        f"  pixels consistent with pure geometry (chi2/dof < 2):"
        f" {100 * np.mean(chi1[valid] < 2):.0f}% / "
        f"{100 * np.mean(chi2m[valid] < 2):.0f}%",
        "",
    ]
    if r_cross >= 0.3 and rms_signal > 0.5 * rms_noise:
        lines_rep += [
            "VERDICT: a reproducible surface-slope signal is present -- the",
            "detector disagreement measures canvas relief (one slope",
            "component) at the ~degree scale from a single scan.",
        ]
    else:
        lines_rep += [
            "VERDICT: no reproducible topographic signal above the noise --",
            f"the map sets an UPPER BOUND of ~{rms_noise:.1f} deg RMS on",
            "recoverable single-scan topography with this instrument.",
        ]

    print("\n".join(lines_rep))
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "canvas_topography.txt"), "w") as f:
        f.write("\n".join(lines_rep) + "\n")
    np.save(os.path.join(OUT_DIR, "canvas_topography_combined.npy"), comb)

    lim = np.nanpercentile(np.abs(comb), 98)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.4), layout="constrained")
    for ax, (label, m) in zip(axes, (("prova1", th1), ("prova2", th2),
                                     ("combined", comb))):
        im = ax.imshow(m, origin="upper", aspect="equal", cmap="RdBu_r",
                       vmin=-lim, vmax=lim, interpolation="nearest")
        ax.set_title(label, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01)
    cb.set_label("local slope along tilt axis (deg)", fontsize=10)
    fig.suptitle(f"Canvas topography from detector disagreement"
                 f"  (cross-scan r = {r_cross:.2f}, reproducible RMS = "
                 f"{rms_signal:.1f} deg)", fontsize=12)
    fig.savefig(os.path.join(OUT_DIR, "canvas_topography.png"), dpi=200)
    print(f"\nSaved: {os.path.join(OUT_DIR, 'canvas_topography.txt')}")
    print(f"Saved: {os.path.join(OUT_DIR, 'canvas_topography.png')}")
