"""
09_fusion.py
===============================================================================
Energy-weighted fusion of the two detector channels vs. simple summing.

The per-element efficiency imbalance (Table 1: R = 5.8 at Ca Ka down to
0.63 at Pb Lg) means the low-efficiency channel contributes mostly noise
at some lines. Simple summing ignores that; inverse-variance weighting
with per-element weights derived from the measured noise should not.

Method
------
For every reliable line and both detectors (10264, 19511), net-intensity
maps are extracted from prova1 and prova2 with the same integrator as
the rest of the pipeline. Pixels are split into two checkerboards:
weights are estimated on one (A) and all metrics are evaluated on the
other (B), so the comparison is not self-fitted.

  noise_d  : std over A of (map_p1 - map_p2) / sqrt(2), gain-matched
  weight_d : 1 / noise_d^2
  sum      : map_10264 + map_19511          (what "summing" does)
  weighted : (w1*m_10264 + w2*a*m_19511) / (w1 + w2)

Metrics on B, per element and per variant (det1, det2, sum, weighted):
  SNR = mean((v_p1 + v_p2)/2) / ( std(v_p1 - v_p2) / sqrt(2) )
  r   = Pearson correlation of v_p1 vs v_p2

Input : Resources/aurora-antico1-{prova1,prova2}/{10264,19511}/None_N.mca
        (cubes cached as .npy after the first run)
Output: results/detector_diff/fusion_weighted.csv
        results/detector_diff/fusion_weighted.txt

Run from the project root:
    python scripts/09_fusion.py
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

import xrf_core                                   # noqa: E402
vuln = importlib.import_module("02_vulnerability")

ROWS, COLS = vuln.ROWS, vuln.COLS
TOTAL = ROWS * COLS
DETS = ("10264", "19511")
DATASETS = ("prova1", "prova2")

# The reliable lines of the dual-detector analysis (overlap_ratios.csv)
LINES = ["Ca", "Ti", "Fe", "Cu", "PbLl", "PbLa", "PbLb", "PbLg"]

OUT_DIR = os.path.join("results", "detector_diff")
CUBE_CACHE = os.path.join("results", "vulnerability_mapping")

_ELEMENTS_JSON = xrf_core.load_elements()


def load_cube(dataset, det):
    """Raw counts cube (ROWS, COLS, n_ch). Cached as .npy (gitignored)."""
    cache = os.path.join(CUBE_CACHE, f"ablation_cube_{dataset}_{det}.npy")
    if os.path.exists(cache):
        print(f"  [{dataset}/{det}] cube from cache")
        return np.load(cache)

    if dataset == "prova1" and det == "10264":
        p = os.path.join("xrf-denoise", "data", "processed", f"{det}_raw.npy")
        if os.path.exists(p):
            cube = np.load(p).astype(np.float64)
            np.save(cache, cube)
            return cube

    folder = os.path.join(
        vuln.resolve_dataset_dir(f"aurora-antico1-{dataset}"), det)
    print(f"  [{dataset}/{det}] parsing {TOTAL} MCA files...")
    probe = xrf_core.parse_mca_file(os.path.join(folder, "None_1.mca"))
    cube = np.zeros((ROWS, COLS, len(probe["counts"])), dtype=np.float64)
    for i in range(1, TOTAL + 1):
        data = xrf_core.parse_mca_file(os.path.join(folder, f"None_{i}.mca"))
        cube[(i - 1) // COLS, (i - 1) % COLS] = data["counts"]
        if i % 2000 == 0:
            print(f"    {i}/{TOTAL}", flush=True)
    np.save(cache, cube)
    return cube


def extract_line_maps(cube):
    """Net-intensity maps for the reliable lines, pipeline integrator."""
    n_ch = cube.shape[2]
    en = xrf_core.energy_axis(n_ch, vuln._SLOPE, vuln._INTERCEPT)
    maps = {}
    for key in LINES:
        cfg_el = _ELEMENTS_JSON[key]
        m = np.zeros((ROWS, COLS))
        for r in range(ROWS):
            for c in range(COLS):
                m[r, c] = xrf_core.integrate(cube[r, c], en, key, cfg_el,
                                             "fixed_hw")
        maps[key] = m
    return maps


def snr_and_r(v1, v2, mask):
    a, b = v1[mask], v2[mask]
    noise = np.std(a - b) / np.sqrt(2.0)
    signal = np.mean(0.5 * (a + b))
    r = float(np.corrcoef(a, b)[0, 1])
    return (signal / noise if noise > 0 else np.inf), r


if __name__ == "__main__":
    print("=" * 70)
    print("  ENERGY-WEIGHTED FUSION vs SIMPLE SUMMING")
    print("=" * 70)

    maps = {}                                      # (dataset, det) -> {line: map}
    for ds in DATASETS:
        for det in DETS:
            cube = load_cube(ds, det)
            print(f"  [{ds}/{det}] extracting line maps...")
            maps[(ds, det)] = extract_line_maps(cube)

    rc = np.add.outer(np.arange(ROWS), np.arange(COLS))
    mask_A = (rc % 2) == 0                         # weight estimation
    mask_B = ~mask_A                               # evaluation

    results = []
    for key in LINES:
        m1 = {ds: maps[(ds, "10264")][key] for ds in DATASETS}
        m2 = {ds: maps[(ds, "19511")][key] for ds in DATASETS}

        # gain-match 19511 to the 10264 scale (per element, both scans)
        num = sum(m1[ds].sum() for ds in DATASETS)
        den = sum(m2[ds].sum() for ds in DATASETS)
        alpha = num / den if den > 0 else 1.0
        g2 = {ds: alpha * m2[ds] for ds in DATASETS}

        # weights from checkerboard A only
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
        }

        row = {"line": key,
               "alpha": alpha,
               "w_share_10264": w["d1"] / (w["d1"] + w["d2"])}
        for name, vv in variants.items():
            snr, r = snr_and_r(vv["prova1"], vv["prova2"], mask_B)
            row[f"snr_{name}"] = snr
            row[f"r_{name}"] = r
        row["snr_gain_pct"] = 100.0 * (row["snr_weighted"]
                                       / row["snr_sum"] - 1.0)
        results.append(row)

    # ---- report --------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    lines_out = [
        "Energy-weighted fusion vs simple summing",
        "(weights from checkerboard A, metrics on checkerboard B;",
        " cross-scan SNR and Pearson r, prova1 vs prova2)",
        "",
        f"{'line':6s} {'SNR d10264':>10s} {'SNR d19511':>10s}"
        f" {'SNR sum':>9s} {'SNR wgt':>9s} {'gain':>7s}"
        f" {'r sum':>7s} {'r wgt':>7s} {'w10264':>7s}",
    ]
    for row in results:
        lines_out.append(
            f"{row['line']:6s} {row['snr_det10264']:10.2f}"
            f" {row['snr_det19511']:10.2f} {row['snr_sum']:9.2f}"
            f" {row['snr_weighted']:9.2f} {row['snr_gain_pct']:+6.1f}%"
            f" {row['r_sum']:7.4f} {row['r_weighted']:7.4f}"
            f" {row['w_share_10264']:7.2f}")
    mean_gain = np.mean([r["snr_gain_pct"] for r in results])
    lines_out += ["", f"mean SNR gain over simple summing: {mean_gain:+.1f}%"]

    print("\n".join(lines_out))
    with open(os.path.join(OUT_DIR, "fusion_weighted.txt"), "w") as f:
        f.write("\n".join(lines_out) + "\n")

    cols = list(results[0].keys())
    with open(os.path.join(OUT_DIR, "fusion_weighted.csv"), "w",
              newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=cols)
        wtr.writeheader()
        wtr.writerows(results)

    print(f"\nSaved: {os.path.join(OUT_DIR, 'fusion_weighted.txt')}")
    print(f"Saved: {os.path.join(OUT_DIR, 'fusion_weighted.csv')}")
