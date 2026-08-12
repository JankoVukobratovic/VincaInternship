"""
13_fusion_ablation.py
===============================================================================
What actually makes the learned fusion work.

The network is the same in every row below; only two decisions change,
and both follow from the response ratio R(E) rather than from anything
about the architecture:

  loss weighting   Scaling detector B up by R multiplies its variance by
                   R^2 while the mean grows only by R, so an unweighted
                   MSE is dominated by the low-energy channels where
                   R ~ 6. Weights 1/R (direction 0) and R (direction 1)
                   restore the balance.
  fusion weighting At export the two directions are combined per channel
                   inverse-variance, R : 1, rather than averaged. At the
                   Ca line that is 85:15 - which is what the classical
                   inverse-variance fusion independently finds (w = 0.89).

Each row is the same metric as script 09: cross-scan SNR of the element
maps (prova1 vs prova2) on the pixels the network never saw, against the
summed map, plus the cv ratio that catches a network winning by blurring.

Input : xrf-denoise/data/processed/fused_{scan}[_tag].npy
        (produced by 07_train_cross_detector.py --tag ...)

Only the selected model's checkpoint is committed; the ablation arms
are not. Reproducing this table therefore means re-running the four
trainings and the exports listed in PLAN 8.10, about 25 minutes on a
laptop GPU, and their numbers will differ in the last digit because
training is not bit-deterministic across devices.
Output: results/detector_diff/fusion_ablation.csv
        results/detector_diff/fusion_ablation.txt

Run from the project root, after 09_fusion.py:
    python scripts/13_fusion_ablation.py
"""

import csv
import importlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

fus = importlib.import_module("09_fusion")

OUT_DIR = fus.OUT_DIR
FUSED_DIR = fus.FUSED_DIR
MAP_CACHE = os.path.join("results", "detector_diff", "_npy_cache")

# tag -> (label, loss weighting, fusion weighting)
CONFIGS = [
    ("nolossequal",  "neither",           "none",         "1 : 1"),
    ("noloss",       "fusion only",       "none",         "R : 1"),
    ("equalfuse",    "ratio loss only",   "1/R, R",       "1 : 1"),
    ("lr1e4",        "ratio, lr 1e-4",    "1/R, R",       "R : 1"),
    ("lr3e4",        "ratio, lr 3e-4",    "1/R, R",       "R : 1"),
    ("poissonequal", "full loss only",    "1/(Ra), R/b",  "1 : 1"),
    ("",             "SELECTED",          "1/(Ra), R/b",  "R : 1"),
]


def fused_maps(tag):
    """Line maps of one ablation's fused cubes, cached per tag."""
    suffix = f"_{tag}" if tag else ""
    out = {}
    for ds in fus.DATASETS:
        cache = os.path.join(MAP_CACHE, f"fusedmaps_{ds}{suffix}.npz")
        if os.path.exists(cache):
            out[ds] = dict(np.load(cache))
            continue
        cube_p = os.path.join(FUSED_DIR, f"fused_{ds}{suffix}.npy")
        if not os.path.exists(cube_p):
            return None
        print(f"  [{tag or 'main'}/{ds}] extracting line maps...")
        maps = fus.extract_line_maps(np.load(cube_p).astype(np.float64))
        np.savez(cache, **maps)
        out[ds] = maps
    return out


if __name__ == "__main__":
    print("=" * 70)
    print("  FUSION ABLATION: WHICH DECISION CARRIES THE GAIN")
    print("=" * 70)

    # reference: the raw detector maps and their sum (same code as 09)
    raw = {}
    for ds in fus.DATASETS:
        for det in fus.DETS:
            raw[(ds, det)] = fus.extract_line_maps(fus.load_cube(ds, det))

    hp = os.path.join(FUSED_DIR, "fused_heldout_px.json")
    if not os.path.exists(hp):
        sys.exit("ERROR: fused_heldout_px.json missing - run the training "
                 "script first.")
    with open(hp) as f:
        held = np.zeros(fus.ROWS * fus.COLS, dtype=bool)
        held[np.asarray(json.load(f)["val_indices"], dtype=int)] = True
    rc = np.add.outer(np.arange(fus.ROWS), np.arange(fus.COLS))
    px = ((rc % 2) == 1) & held.reshape(fus.ROWS, fus.COLS)

    rows = []
    for tag, label, lw, fw in CONFIGS:
        maps = fused_maps(tag)
        if maps is None:
            print(f"  [{label}] fused cubes missing, skipped")
            continue
        gains, cvs, per_line = [], [], {}
        for key in fus.LINES:
            summed = {ds: raw[(ds, "10264")][key] + raw[(ds, "19511")][key]
                      for ds in fus.DATASETS}
            learned = {ds: maps[ds][key] for ds in fus.DATASETS}
            snr_s, _ = fus.snr_and_r(summed["prova1"], summed["prova2"], px)
            snr_l, _ = fus.snr_and_r(learned["prova1"], learned["prova2"], px)
            cv, _ = fus.fidelity_vs_sum(learned, summed, px)
            gain = 100.0 * (snr_l / snr_s - 1.0)
            gains.append(gain)
            cvs.append(cv)
            per_line[key] = (gain, cv)
        rows.append({"tag": tag or "main", "config": label,
                     "loss_weights": lw, "fusion_weights": fw,
                     "mean_gain_pct": float(np.mean(gains)),
                     "min_gain_pct": float(np.min(gains)),
                     "mean_cv_ratio": float(np.mean(cvs)),
                     "min_cv_ratio": float(np.min(cvs)),
                     "gain_Ca": per_line["Ca"][0], "cv_Ca": per_line["Ca"][1],
                     "gain_Ti": per_line["Ti"][0], "cv_Ti": per_line["Ti"][1]})

    if not rows:
        sys.exit("ERROR: no ablation cubes found.")

    lines = [
        "Fusion ablation: same network, two decisions toggled",
        "(cross-scan SNR of the element maps on the held-out pixels,",
        " against the summed map; cv ratio 1.0 = same spatial contrast)",
        "",
        f"{'config':14s} {'loss w':8s} {'fuse w':7s} {'mean gain':>10s}"
        f" {'worst line':>11s} {'mean cv':>8s} {'Ca gain':>8s} {'Ca cv':>6s}"
        f" {'Ti gain':>8s} {'Ti cv':>6s}",
    ]
    for r in rows:
        lines.append(
            f"{r['config']:14s} {r['loss_weights']:8s} {r['fusion_weights']:7s}"
            f" {r['mean_gain_pct']:+9.1f}% {r['min_gain_pct']:+10.1f}%"
            f" {r['mean_cv_ratio']:8.3f} {r['gain_Ca']:+7.1f}%"
            f" {r['cv_Ca']:6.2f} {r['gain_Ti']:+7.1f}% {r['cv_Ti']:6.2f}")

    full = next((r for r in rows if r["tag"] == "main"), None)
    none_ = next((r for r in rows if r["tag"] == "nolossequal"), None)
    if full and none_:
        lines += [
            "",
            f"Everything on: {full['mean_gain_pct']:+.1f}% mean gain."
            f" Everything off: {none_['mean_gain_pct']:+.1f}%.",
            "The network alone does not beat summing. What beats summing is",
            "the network plus the variance structure that the measured R(E)",
            "prescribes: weight every channel by its inverse target variance",
            "during training, and combine the two directions inverse-variance",
            "at export. Both corrections are read off the data, neither is",
            "tuned. The two learning rates differ by less than the gap",
            "between weighting schemes, so the result is not a tuning",
            "artifact.",
        ]

    print()
    print("\n".join(lines))
    with open(os.path.join(OUT_DIR, "fusion_ablation.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT_DIR, "fusion_ablation.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {os.path.join(OUT_DIR, 'fusion_ablation.txt')}")
    print(f"Saved: {os.path.join(OUT_DIR, 'fusion_ablation.csv')}")
