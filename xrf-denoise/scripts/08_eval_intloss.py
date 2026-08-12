"""
Benchmark the integral-anchor fusion (07 --integral-loss-weight, tag
``intloss``) with exactly the conventions of the main repo's
scripts/09_fusion.py, without touching that script or any of the main
run's outputs.

All machinery is imported from 09_fusion via importlib: the same
net-intensity integrator over the same raw cubes, checkerboard A for
weight estimation / checkerboard B for metrics, cross-scan SNR and
Pearson r between prova1 and prova2, the cv_ratio / r_vs_sum fidelity
guards, and the bias column (absolute level of the learned map against
the inverse-variance combination of the raw maps -- the one metric the
scale-invariant SNR cannot see). The held-out subset comes from the
main run's fused_heldout_px.json; the split is deterministic in the
seed, so the intloss run holds out the same pixels.

Input : xrf-denoise/data/processed/fused_{prova1,prova2}_intloss.npy
        xrf-denoise/data/processed/fused_heldout_px.json
        (raw cubes and caches as in 09_fusion)
Output: results/detector_diff/fusion_intloss.txt
        results/detector_diff/fusion_intloss.csv

Run from anywhere:
    python xrf-denoise/scripts/08_eval_intloss.py
"""

import csv
import importlib
import json
import os
import sys

import numpy as np

XD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(XD_ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))

fus = importlib.import_module("09_fusion")   # chdirs to the main root

DATASETS = fus.DATASETS
LINES = fus.LINES
ROWS, COLS = fus.ROWS, fus.COLS
FUSED_DIR = fus.FUSED_DIR
OUT_DIR = fus.OUT_DIR
TAG = "intloss"


def load_intloss_maps():
    """Line maps of the intloss-fused cubes + the held-out pixel mask."""
    fused = {}
    for ds in DATASETS:
        p = os.path.join(FUSED_DIR, f"fused_{ds}_{TAG}.npy")
        if not os.path.exists(p):
            sys.exit(f"ERROR: {p} missing - run xrf-denoise/scripts/"
                     "07_train_cross_detector.py --tag intloss "
                     "--integral-loss-weight ... first.")
        print(f"  [{ds}/fused_{TAG}] extracting line maps...")
        fused[ds] = fus.extract_line_maps(np.load(p).astype(np.float64))

    hp = os.path.join(FUSED_DIR, "fused_heldout_px.json")
    with open(hp) as f:
        rec = json.load(f)
    heldout = np.zeros(ROWS * COLS, dtype=bool)
    heldout[np.asarray(rec["val_indices"], dtype=int)] = True
    return fused, heldout.reshape(ROWS, COLS)


if __name__ == "__main__":
    print("=" * 70)
    print("  FUSION BENCHMARK (INTEGRAL-ANCHOR RUN): SUM vs WGT vs LEARNED")
    print("=" * 70)

    maps = {}
    for ds in DATASETS:
        for det in fus.DETS:
            cube = fus.load_cube(ds, det)
            print(f"  [{ds}/{det}] extracting line maps...")
            maps[(ds, det)] = fus.extract_line_maps(cube)

    fused_maps, heldout_mask = load_intloss_maps()

    rc = np.add.outer(np.arange(ROWS), np.arange(COLS))
    mask_A = (rc % 2) == 0                         # weight estimation
    mask_B = ~mask_A                               # evaluation

    subsets = {"all_px": mask_B,
               "heldout_px": mask_B & heldout_mask}

    results = []
    for key in LINES:
        m1 = {ds: maps[(ds, "10264")][key] for ds in DATASETS}
        m2 = {ds: maps[(ds, "19511")][key] for ds in DATASETS}

        num = sum(m1[ds].sum() for ds in DATASETS)
        den = sum(m2[ds].sum() for ds in DATASETS)
        alpha = num / den if den > 0 else 1.0
        g2 = {ds: alpha * m2[ds] for ds in DATASETS}

        w = {}
        for label, mm in (("d1", m1), ("d2", g2)):
            diff = (mm["prova1"] - mm["prova2"])[mask_A]
            sig = np.std(diff) / np.sqrt(2.0)
            w[label] = 1.0 / sig ** 2 if sig > 0 else 0.0

        variants = {
            "det10264": m1,
            "det19511": g2,
            "sum": {ds: m1[ds] + m2[ds] for ds in DATASETS},
            "weighted": {ds: (w["d1"] * m1[ds] + w["d2"] * g2[ds])
                         / (w["d1"] + w["d2"]) for ds in DATASETS},
            "learned": {ds: fused_maps[ds][key] for ds in DATASETS},
        }

        for subset, px in subsets.items():
            row = {"subset": subset,
                   "n_px": int(px.sum()),
                   "line": key,
                   "alpha": alpha,
                   "w_share_10264": w["d1"] / (w["d1"] + w["d2"])}
            for name, vv in variants.items():
                snr, r = fus.snr_and_r(vv["prova1"], vv["prova2"], px)
                cv_ratio, r_sum = fus.fidelity_vs_sum(vv, variants["sum"],
                                                      px)
                row[f"snr_{name}"] = snr
                row[f"r_{name}"] = r
                row[f"cv_ratio_{name}"] = cv_ratio
                row[f"r_vs_sum_{name}"] = r_sum
            row["snr_gain_pct"] = 100.0 * (row["snr_weighted"]
                                           / row["snr_sum"] - 1.0)
            row["snr_gain_learned_pct"] = 100.0 * (
                row["snr_learned"] / row["snr_sum"] - 1.0)
            row["bias_learned_pct"] = fus.level_bias(
                variants["learned"], m1, m2, key, px)
            results.append(row)

    # ---- report --------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    lines_out = [
        "Fusion benchmark, integral-anchor run (07 --integral-loss-weight,"
        " tag intloss):",
        "summing vs inverse-variance weighting vs learned (N2N +"
        " integrated-line-intensity anchor)",
        "(weights from checkerboard A, metrics on checkerboard B;",
        " cross-scan SNR and Pearson r, prova1 vs prova2)",
    ]
    for subset in subsets:
        sub_rows = [r for r in results if r["subset"] == subset]
        note = {
            "all_px": "all pixels",
            "heldout_px": "pixels the network never saw"
                          " (prova1 validation blocks)",
        }[subset]
        head = (f"{'line':6s} {'SNR d10264':>10s} {'SNR d19511':>10s}"
                f" {'SNR sum':>9s} {'SNR wgt':>9s} {'gain':>7s}"
                f" {'SNR lrn':>9s} {'gain':>7s}"
                f" {'cv lrn':>7s} {'bias':>8s}"
                f" {'r sum':>7s} {'w10264':>7s}")
        lines_out += ["",
                      f"[{subset}]  {note}, {sub_rows[0]['n_px']} px",
                      head]
        for row in sub_rows:
            lines_out.append(
                f"{row['line']:6s} {row['snr_det10264']:10.2f}"
                f" {row['snr_det19511']:10.2f} {row['snr_sum']:9.2f}"
                f" {row['snr_weighted']:9.2f}"
                f" {row['snr_gain_pct']:+6.1f}%"
                f" {row['snr_learned']:9.2f}"
                f" {row['snr_gain_learned_pct']:+6.1f}%"
                f" {row['cv_ratio_learned']:7.3f}"
                f" {row['bias_learned_pct']:+7.1f}%"
                f" {row['r_sum']:7.4f} {row['w_share_10264']:7.2f}")
        mg = np.mean([r["snr_gain_pct"] for r in sub_rows])
        ml = np.mean([r["snr_gain_learned_pct"] for r in sub_rows])
        mcv = np.mean([r["cv_ratio_learned"] for r in sub_rows])
        lines_out += [
            f"  mean SNR gain over summing: weighted {mg:+.1f}%,"
            f" learned {ml:+.1f}%",
            f"  learned map fidelity: mean cv ratio {mcv:.3f}"
            " (1.0 = same spatial contrast as the summed map)"]
    b = [abs(r["bias_learned_pct"]) for r in results
         if r["subset"] == "heldout_px"]
    lines_out += [
        "",
        "cv ratio guards the SNR column: a network that blurs the map"
        " would raise SNR while cv ratio falls below 1.",
        "",
        "bias is the absolute level of the learned map against the same"
        " inverse-variance",
        "combination of the raw maps. SNR, cv and r are all"
        " scale-invariant and cannot see it.",
        f"Held-out bias: worst line {max(b):.0f}%, median {np.median(b):.0f}%"
        " (main run for comparison: worst 33%, median 7%,"
        " fusion_weighted.txt).",
    ]

    print("\n".join(lines_out))
    with open(os.path.join(OUT_DIR, "fusion_intloss.txt"), "w") as f:
        f.write("\n".join(lines_out) + "\n")

    cols = list(results[0].keys())
    with open(os.path.join(OUT_DIR, "fusion_intloss.csv"), "w",
              newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=cols)
        wtr.writeheader()
        wtr.writerows(results)

    print(f"\nSaved: {os.path.join(OUT_DIR, 'fusion_intloss.txt')}")
    print(f"Saved: {os.path.join(OUT_DIR, 'fusion_intloss.csv')}")
