"""
15_perpixel_fusion.py
===============================================================================
Per-pixel gain-matched fusion vs the global-scalar fusion of 09_fusion.py.

09_fusion.py showed that inverse-variance weighting with GLOBAL
per-element scalar weights gains only +0.7-0.9% mean SNR over plain
summing -- the channels are Poisson-limited and a scalar weight has
nothing to correct. But 10_flatfield.py measured a per-pixel flat-field
of the detector ratio D10264/D19511 (results/detector_diff/
flatfield_combined.npy, log-domain deviation from the global level,
RMS ~9%, reproducible across scans at r = 0.70): the two channels are
NOT proportional pixel-to-pixel. Hypothesis under test: per-pixel
gain-matching / weighting using this map beats the global-scalar
version.

Variants (identical protocol to 09: weights and gains estimated on
checkerboard A, ALL metrics on checkerboard B; cross-scan SNR and
Pearson r between prova1 and prova2):

  1. sum          : m_10264 + m_19511                       (baseline)
  2. wgt_global   : 09's inverse-variance weighting, global scalar
                    gain alpha and scalar weights. Reproduced here as a
                    sanity check -- it must match fusion_weighted.csv.
  3. wgt_pixgain  : detector 19511 gain-matched to 10264 PER PIXEL,
                    g(p) = alpha * exp(flat(p)), then combined with
                    per-element scalar inverse-variance weights
                    (estimated on checkerboard A from the per-pixel
                    gain-matched maps).
  4. wgt_pixpois  : same per-pixel gains, plus a per-pixel variance
                    model: under Poisson statistics the gain-matched
                    channel g*m2 has variance proportional to g at
                    matched signal level, so the 19511 weight is
                    modulated as w2(p) = w2 * exp(-flat(p)).
                    Algebraically (w1*m1 + w2*alpha*m2) /
                    (w1 + w2*exp(-flat)) -- the numerator is exactly
                    the global-weighted one, only the normalisation
                    varies per pixel; this is the per-pixel maximum-
                    likelihood fusion of two Poisson channels with the
                    measured relative efficiency field.

Why scalar weights + per-pixel gains (variant 3), rather than fitting a
per-pixel variance empirically? Two scans give one difference sample
per pixel: an empirical per-pixel variance has ~100% relative error and
fitting it would inject far more noise than the ~9% RMS effect it is
meant to capture. The flat-field gain, in contrast, is an averaged,
cross-scan-reproducible instrument property, so it is the only per-
pixel quantity we trust; the weight scale stays anchored on the
measured checkerboard-A noise. Variant 4 then adds the per-pixel
variance dependence ANALYTICALLY (parameter-free, from the Poisson
model) instead of fitting it.

Honest caveat, stated up front: the cross-scan SNR metric is largely
blind to a static gain field -- g(p) multiplies prova1 and prova2
identically, so it cancels to first order in the (prova1 - prova2)
noise term. Per-pixel gain-matching mainly removes a flat-field BIAS
from the fused map (an accuracy gain the metric cannot reward); any SNR
change must come from the per-pixel re-weighting, which is second order
in a 9%-RMS field. A null result is therefore a real possibility and is
reported as such.

NaN flat-field pixels fall back to the global gain (flat = 0).

Input : results/vulnerability_mapping/ablation_cube_{prova1,prova2}_
        {10264,19511}.npy (cube cache; rebuilt from MCA if missing)
        results/detector_diff/flatfield_combined.npy   (from 10)
        results/detector_diff/fusion_weighted.csv      (09, sanity ref)
Output: results/detector_diff/fusion_perpixel.txt
        results/detector_diff/fusion_perpixel.csv

Run from the project root:
    python scripts/15_perpixel_fusion.py
"""

import csv
import importlib
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

# Reuse 09's machinery directly (load_cube / extract_line_maps /
# snr_and_r / fidelity_vs_sum) so the protocol is identical by
# construction, not by re-implementation.
fus = importlib.import_module("09_fusion")

LINES = fus.LINES
DATASETS = fus.DATASETS
ROWS, COLS = fus.ROWS, fus.COLS

OUT_DIR = os.path.join("results", "detector_diff")
FLAT_PATH = os.path.join(OUT_DIR, "flatfield_combined.npy")
REF_CSV = os.path.join(OUT_DIR, "fusion_weighted.csv")

VARIANTS = ("sum", "wgt_global", "wgt_pixgain", "wgt_pixpois")


def estimate_weights(m1, g2, mask_A):
    """Per-element scalar inverse-variance weights from checkerboard A,
    exactly as in 09: noise = std(prova1 - prova2)/sqrt(2) of each
    (gain-matched) channel."""
    w = {}
    for label, mm in (("d1", m1), ("d2", g2)):
        diff = (mm["prova1"] - mm["prova2"])[mask_A]
        sig = np.std(diff) / np.sqrt(2.0)
        w[label] = 1.0 / sig ** 2 if sig > 0 else 0.0
    return w


def load_sanity_reference():
    """09's stored all_px rows: line -> (snr_sum, snr_weighted)."""
    if not os.path.exists(REF_CSV):
        return None
    ref = {}
    with open(REF_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["subset"] == "all_px":
                ref[row["line"]] = (float(row["snr_sum"]),
                                    float(row["snr_weighted"]))
    return ref


if __name__ == "__main__":
    print("=" * 70)
    print("  PER-PIXEL GAIN-MATCHED FUSION vs GLOBAL-SCALAR WEIGHTING")
    print("=" * 70)

    if not os.path.exists(FLAT_PATH):
        sys.exit(f"ERROR: {FLAT_PATH} missing - run scripts/10_flatfield.py"
                 " first.")
    flat = np.load(FLAT_PATH)
    if flat.shape != (ROWS, COLS):
        sys.exit(f"ERROR: flat-field shape {flat.shape} != ({ROWS},{COLS})")
    ff_nan = ~np.isfinite(flat)
    # NaN flat-field pixels: fall back to the global gain (flat = 0,
    # i.e. exp(flat) = 1, so g(p) = alpha there).
    flat_filled = np.where(ff_nan, 0.0, flat)
    exp_flat = np.exp(flat_filled)
    print(f"  flat-field: RMS {100.0 * np.sqrt(np.mean((exp_flat[~ff_nan] - 1.0) ** 2)):.1f}%"
          f", {int(ff_nan.sum())} NaN px fall back to the global gain")

    maps = {}                                # (dataset, det) -> {line: map}
    for ds in DATASETS:
        for det in fus.DETS:
            cube = fus.load_cube(ds, det)
            print(f"  [{ds}/{det}] extracting line maps...")
            maps[(ds, det)] = fus.extract_line_maps(cube)

    # identical checkerboard split to 09
    rc = np.add.outer(np.arange(ROWS), np.arange(COLS))
    mask_A = (rc % 2) == 0                   # weight/gain estimation
    mask_B = ~mask_A                         # evaluation (all metrics)

    results = []
    for key in LINES:
        m1 = {ds: maps[(ds, "10264")][key] for ds in DATASETS}
        m2 = {ds: maps[(ds, "19511")][key] for ds in DATASETS}

        # global gain alpha, exactly as in 09 (per element, both scans)
        num = sum(m1[ds].sum() for ds in DATASETS)
        den = sum(m2[ds].sum() for ds in DATASETS)
        alpha = num / den if den > 0 else 1.0

        # ---- variant 2: 09's global weighting (sanity check) ----------
        g2_glob = {ds: alpha * m2[ds] for ds in DATASETS}
        w_g = estimate_weights(m1, g2_glob, mask_A)

        # ---- variants 3/4: per-pixel gain field ------------------------
        # g(p) = alpha * exp(flat(p)); flat is the log-domain deviation
        # of the local ratio D10264/D19511 from its global level, so
        # multiplying m2 by it matches detector 19511 to 10264 per pixel.
        g2_pix = {ds: alpha * exp_flat * m2[ds] for ds in DATASETS}
        w_p = estimate_weights(m1, g2_pix, mask_A)

        # variant 4 per-pixel weight modulation (Poisson variance model):
        # w2(p) = w_p["d2"] * exp(-flat(p)); w1 stays scalar because a
        # common per-pixel factor cancels in the normalised combination.
        w2_pix = w_p["d2"] * np.exp(-flat_filled)

        variants = {
            "sum": {ds: m1[ds] + m2[ds] for ds in DATASETS},
            "wgt_global": {
                ds: (w_g["d1"] * m1[ds] + w_g["d2"] * g2_glob[ds])
                / (w_g["d1"] + w_g["d2"]) for ds in DATASETS},
            "wgt_pixgain": {
                ds: (w_p["d1"] * m1[ds] + w_p["d2"] * g2_pix[ds])
                / (w_p["d1"] + w_p["d2"]) for ds in DATASETS},
            "wgt_pixpois": {
                ds: (w_p["d1"] * m1[ds] + w2_pix * g2_pix[ds])
                / (w_p["d1"] + w2_pix) for ds in DATASETS},
        }

        row = {"line": key,
               "n_px": int(mask_B.sum()),
               "alpha": alpha,
               "w_share_10264_global": w_g["d1"] / (w_g["d1"] + w_g["d2"]),
               "w_share_10264_pix": w_p["d1"] / (w_p["d1"] + w_p["d2"])}
        for name, vv in variants.items():
            snr, r = fus.snr_and_r(vv["prova1"], vv["prova2"], mask_B)
            cv_ratio, r_sum = fus.fidelity_vs_sum(vv, variants["sum"],
                                                  mask_B)
            row[f"snr_{name}"] = snr
            row[f"r_{name}"] = r
            row[f"cv_ratio_{name}"] = cv_ratio
            row[f"r_vs_sum_{name}"] = r_sum
        for name in VARIANTS[1:]:
            row[f"gain_{name}_pct"] = 100.0 * (row[f"snr_{name}"]
                                               / row["snr_sum"] - 1.0)
        results.append(row)

    # ---- sanity check against 09's stored numbers ----------------------
    ref = load_sanity_reference()
    sanity_lines = ["", "Sanity check: variant 'wgt_global' vs 09's stored"
                        " fusion_weighted.csv (all_px)"]
    if ref is None:
        sanity_lines.append("  (reference CSV missing - skipped)")
        sanity_ok = None
    else:
        max_dev = 0.0
        sanity_lines.append(f"{'line':6s} {'SNR sum (09)':>13s}"
                            f" {'here':>9s} {'SNR wgt (09)':>13s}"
                            f" {'here':>9s}")
        for row in results:
            r_sum_ref, r_wgt_ref = ref[row["line"]]
            max_dev = max(max_dev,
                          abs(row["snr_sum"] - r_sum_ref),
                          abs(row["snr_wgt_global"] - r_wgt_ref))
            sanity_lines.append(
                f"{row['line']:6s} {r_sum_ref:13.4f} {row['snr_sum']:9.4f}"
                f" {r_wgt_ref:13.4f} {row['snr_wgt_global']:9.4f}")
        sanity_ok = max_dev < 5e-3
        sanity_lines.append(
            f"  max |deviation| = {max_dev:.2e}"
            f"  -> {'OK (matches within rounding)' if sanity_ok else 'MISMATCH'}")

    # ---- report ---------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    lines_out = [
        "Per-pixel gain-matched fusion vs global-scalar inverse-variance"
        " weighting",
        "(gains/weights from checkerboard A, all metrics on checkerboard B;",
        " cross-scan SNR and Pearson r, prova1 vs prova2, "
        f"{results[0]['n_px']} px)",
        "",
        "variants: sum = plain sum; wgt_global = 09's scalar weighting;",
        " wgt_pixgain = per-pixel gain alpha*exp(flatfield), scalar weights;",
        " wgt_pixpois = same gains + Poisson per-pixel weight"
        " w2(p) ~ exp(-flat)",
        "",
        f"{'line':6s} {'SNR sum':>9s} {'SNR glob':>9s} {'gain':>7s}"
        f" {'SNR pixg':>9s} {'gain':>7s} {'SNR pixp':>9s} {'gain':>7s}"
        f" {'r sum':>7s} {'w10264':>7s}",
    ]
    for row in results:
        lines_out.append(
            f"{row['line']:6s} {row['snr_sum']:9.2f}"
            f" {row['snr_wgt_global']:9.2f}"
            f" {row['gain_wgt_global_pct']:+6.1f}%"
            f" {row['snr_wgt_pixgain']:9.2f}"
            f" {row['gain_wgt_pixgain_pct']:+6.1f}%"
            f" {row['snr_wgt_pixpois']:9.2f}"
            f" {row['gain_wgt_pixpois_pct']:+6.1f}%"
            f" {row['r_sum']:7.4f}"
            f" {row['w_share_10264_pix']:7.2f}")

    summary = {}
    for name in VARIANTS[1:]:
        g = [row[f"gain_{name}_pct"] for row in results]
        summary[name] = (float(np.mean(g)), float(np.median(g)))
    lines_out += [
        "",
        f"{'variant':14s} {'mean gain':>10s} {'median gain':>12s}"
        "   (SNR vs plain sum)",
    ]
    for name in VARIANTS[1:]:
        mg, dg = summary[name]
        lines_out.append(f"{name:14s} {mg:+9.2f}% {dg:+11.2f}%")

    # per-line verdict: does either per-pixel variant beat the global one?
    n_pix_beats = sum(1 for row in results
                      if max(row["gain_wgt_pixgain_pct"],
                             row["gain_wgt_pixpois_pct"])
                      > row["gain_wgt_global_pct"])
    best_pix = max(summary["wgt_pixgain"][0], summary["wgt_pixpois"][0])
    delta = best_pix - summary["wgt_global"][0]
    lines_out += [
        "",
        f"Per-pixel beats global on {n_pix_beats}/{len(results)} lines;"
        f" best per-pixel mean gain {best_pix:+.2f}% vs global"
        f" {summary['wgt_global'][0]:+.2f}% (delta {delta:+.2f} pp).",
    ]
    if delta > 0.5:
        lines_out.append(
            "Verdict: per-pixel weighting gives a real improvement over the"
            " global scalar version.")
    elif delta > 0.05:
        lines_out.append(
            "Verdict: per-pixel weighting is marginally ahead of the global"
            " scalar version - a small but not decisive improvement.")
    else:
        lines_out.append(
            "Verdict: NULL RESULT - per-pixel weighting does not beat the"
            " global scalar version on cross-scan SNR.")
    lines_out += [
        "",
        "Note: the cross-scan SNR metric is blind to a static gain field",
        "(g(p) multiplies prova1 and prova2 identically and cancels in the",
        "difference), so the accuracy benefit of flat-field correction --",
        "removing a ~9% RMS spatial bias from the fused map -- is real but",
        "invisible here; only the second-order re-weighting effect can move",
        "this SNR.",
    ]
    lines_out += sanity_lines

    print("\n".join(lines_out))
    txt_path = os.path.join(OUT_DIR, "fusion_perpixel.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines_out) + "\n")

    csv_path = os.path.join(OUT_DIR, "fusion_perpixel.csv")
    cols = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=cols)
        wtr.writeheader()
        wtr.writerows(results)

    print(f"\nSaved: {txt_path}")
    print(f"Saved: {csv_path}")
    if sanity_ok is False:
        sys.exit("ERROR: sanity check failed - wgt_global does not"
                 " reproduce 09's stored numbers.")
