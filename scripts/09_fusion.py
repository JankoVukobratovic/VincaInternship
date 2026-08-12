"""
09_fusion.py
===============================================================================
Three levels of dual-detector fusion, on one metric: simple summing,
inverse-variance weighting, and the learned Noise2Noise fusion.

The per-element efficiency imbalance (Table 1: R = 5.8 at Ca Ka down to
0.63 at Pb Lg) means the low-efficiency channel contributes mostly noise
at some lines. Simple summing ignores that; inverse-variance weighting
with per-element weights derived from the measured noise should not. For
pure Poisson counting with proportional efficiencies the plain sum is
already a sufficient statistic (PLAN 8.5), so a weighting that only
matches it is itself a result: the channels are Poisson-limited. Only
the learned variant, which uses spectral structure rather than a scalar
weight, can go beyond that.

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
  learned  : maps of the fused cubes from the N2N network, if exported
             by xrf-denoise/scripts/07_train_cross_detector.py

Metrics per element and per variant:
  SNR = mean((v_p1 + v_p2)/2) / ( std(v_p1 - v_p2) / sqrt(2) )
  r   = Pearson correlation of v_p1 vs v_p2
  cv_ratio, r_vs_sum : spatial contrast and shape of the variant's map
             against the summed map -- a denoiser that simply blurs the
             image also raises SNR, and these two columns catch it.
  bias_pct : absolute level of the learned map against what the same
             inverse-variance combination of the raw maps would give.
             SNR, cv_ratio and r_vs_sum are all scale-invariant, so none
             of them sees a systematic error in the intensity itself;
             this column does, and it matters because element-to-element
             ratios are what pigment identification uses.

Subsets: "all_px" is checkerboard B over the whole grid. When the
network's held-out pixel list is present, every variant is also
evaluated on "heldout_px" (checkerboard B restricted to the prova1
validation blocks); that is the row to quote for the learned variant.
Those pixels carry no gradient, though the validation loss on them did
select the stopping epoch. prova2, the other half of every SNR pair, is
never seen by the network in either subset.

Input : Resources/aurora-antico1-{prova1,prova2}/{10264,19511}/None_N.mca
        (cubes cached as .npy after the first run)
        xrf-denoise/data/processed/fused_{prova1,prova2}.npy   (optional)
        xrf-denoise/data/processed/fused_heldout_px.json       (optional)
Output: results/detector_diff/fusion_weighted.csv
        results/detector_diff/fusion_weighted.txt

Run from the project root:
    python scripts/09_fusion.py
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
FUSED_DIR = os.path.join("xrf-denoise", "data", "processed")

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


def fidelity_vs_sum(variant, reference, mask):
    """Spatial contrast and shape of a variant's map against the sum.

    Both maps are the prova1/prova2 average. cv_ratio compares the
    coefficient of variation, so it is insensitive to the overall gain
    difference between variants but drops below 1 as soon as a variant
    flattens real structure; r_vs_sum checks the structure is the same
    structure.
    """
    v = 0.5 * (variant["prova1"] + variant["prova2"])[mask]
    s = 0.5 * (reference["prova1"] + reference["prova2"])[mask]
    cv_v = np.std(v) / np.mean(v) if np.mean(v) > 0 else np.nan
    cv_s = np.std(s) / np.mean(s) if np.mean(s) > 0 else np.nan
    return float(cv_v / cv_s), float(np.corrcoef(v, s)[0, 1])


def ratio_at(kev):
    """Handoff-2 R(E) at one line energy (1.0 if the curve is absent)."""
    p = os.path.join(OUT_DIR, "handoff2_ratio_curve.csv")
    if not os.path.exists(p):
        return 1.0
    with open(p, newline="") as f:
        rows = [(float(r["kev"]), float(r["R"])) for r in csv.DictReader(f)]
    ks = np.array([r[0] for r in rows])
    vs = np.array([r[1] for r in rows])
    return float(np.interp(kev, ks, vs))


def level_bias(learned, m1, m2, key, mask):
    """Learned level against the inverse-variance combination of the raw
    maps, both in detector-A scale. Zero means the fusion is calibrated."""
    r = ratio_at(_ELEMENTS_JSON[key]["kev"])
    wa = r / (r + 1.0)
    expect = {ds: wa * m1[ds] + (1.0 - wa) * (r * m2[ds])
              for ds in DATASETS}
    got = np.mean(0.5 * (learned["prova1"] + learned["prova2"])[mask])
    ref = np.mean(0.5 * (expect["prova1"] + expect["prova2"])[mask])
    return 100.0 * (got / ref - 1.0) if ref > 0 else np.nan


def load_fused_maps():
    """Line maps of the N2N-fused cubes, plus the held-out pixel mask."""
    paths = {ds: os.path.join(FUSED_DIR, f"fused_{ds}.npy")
             for ds in DATASETS}
    if not all(os.path.exists(p) for p in paths.values()):
        return None, None
    fused = {}
    for ds, p in paths.items():
        print(f"  [{ds}/fused] extracting line maps...")
        fused[ds] = extract_line_maps(np.load(p).astype(np.float64))

    heldout = None
    hp = os.path.join(FUSED_DIR, "fused_heldout_px.json")
    if os.path.exists(hp):
        with open(hp) as f:
            rec = json.load(f)
        heldout = np.zeros(ROWS * COLS, dtype=bool)
        heldout[np.asarray(rec["val_indices"], dtype=int)] = True
        heldout = heldout.reshape(ROWS, COLS)
    return fused, heldout


if __name__ == "__main__":
    print("=" * 70)
    print("  FUSION BENCHMARK: SUM vs WEIGHTED vs LEARNED")
    print("=" * 70)

    maps = {}                                      # (dataset, det) -> {line: map}
    for ds in DATASETS:
        for det in DETS:
            cube = load_cube(ds, det)
            print(f"  [{ds}/{det}] extracting line maps...")
            maps[(ds, det)] = extract_line_maps(cube)

    fused_maps, heldout_mask = load_fused_maps()
    if fused_maps is None:
        print("  (no fused cubes yet - run xrf-denoise/scripts/"
              "07_train_cross_detector.py for the learned variant)")

    rc = np.add.outer(np.arange(ROWS), np.arange(COLS))
    mask_A = (rc % 2) == 0                         # weight estimation
    mask_B = ~mask_A                               # evaluation

    subsets = {"all_px": mask_B}
    if heldout_mask is not None:
        subsets["heldout_px"] = mask_B & heldout_mask

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
        if fused_maps is not None:
            variants["learned"] = {ds: fused_maps[ds][key]
                                   for ds in DATASETS}

        for subset, px in subsets.items():
            row = {"subset": subset,
                   "n_px": int(px.sum()),
                   "line": key,
                   "alpha": alpha,
                   "w_share_10264": w["d1"] / (w["d1"] + w["d2"])}
            for name, vv in variants.items():
                snr, r = snr_and_r(vv["prova1"], vv["prova2"], px)
                cv_ratio, r_sum = fidelity_vs_sum(vv, variants["sum"], px)
                row[f"snr_{name}"] = snr
                row[f"r_{name}"] = r
                row[f"cv_ratio_{name}"] = cv_ratio
                row[f"r_vs_sum_{name}"] = r_sum
            row["snr_gain_pct"] = 100.0 * (row["snr_weighted"]
                                           / row["snr_sum"] - 1.0)
            if "learned" in variants:
                row["snr_gain_learned_pct"] = 100.0 * (
                    row["snr_learned"] / row["snr_sum"] - 1.0)
                row["bias_learned_pct"] = level_bias(
                    variants["learned"], m1, m2, key, px)
            results.append(row)

    # ---- report --------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    has_learned = fused_maps is not None
    lines_out = [
        "Fusion benchmark: summing vs inverse-variance weighting"
        + (" vs learned (N2N)" if has_learned else ""),
        "(weights from checkerboard A, metrics on checkerboard B;",
        " cross-scan SNR and Pearson r, prova1 vs prova2)",
    ]
    for subset in subsets:
        sub_rows = [r for r in results if r["subset"] == subset]
        note = {
            "all_px": "all pixels",
            "heldout_px": "pixels the network never saw"
                          " (prova1 validation blocks)",
        }.get(subset, subset)
        head = (f"{'line':6s} {'SNR d10264':>10s} {'SNR d19511':>10s}"
                f" {'SNR sum':>9s} {'SNR wgt':>9s} {'gain':>7s}")
        if has_learned:
            head += (f" {'SNR lrn':>9s} {'gain':>7s}"
                     f" {'cv lrn':>7s} {'bias':>8s}")
        head += f" {'r sum':>7s} {'w10264':>7s}"
        lines_out += ["",
                      f"[{subset}]  {note}, {sub_rows[0]['n_px']} px",
                      head]
        for row in sub_rows:
            line = (f"{row['line']:6s} {row['snr_det10264']:10.2f}"
                    f" {row['snr_det19511']:10.2f} {row['snr_sum']:9.2f}"
                    f" {row['snr_weighted']:9.2f}"
                    f" {row['snr_gain_pct']:+6.1f}%")
            if has_learned:
                line += (f" {row['snr_learned']:9.2f}"
                         f" {row['snr_gain_learned_pct']:+6.1f}%"
                         f" {row['cv_ratio_learned']:7.3f}"
                         f" {row['bias_learned_pct']:+7.1f}%")
            line += f" {row['r_sum']:7.4f} {row['w_share_10264']:7.2f}"
            lines_out.append(line)
        mg = np.mean([r["snr_gain_pct"] for r in sub_rows])
        lines_out.append(f"  mean SNR gain over summing:"
                         f" weighted {mg:+.1f}%")
        if has_learned:
            ml = np.mean([r["snr_gain_learned_pct"] for r in sub_rows])
            mcv = np.mean([r["cv_ratio_learned"] for r in sub_rows])
            lines_out[-1] += f", learned {ml:+.1f}%"
            lines_out.append(
                f"  learned map fidelity: mean cv ratio {mcv:.3f}"
                " (1.0 = same spatial contrast as the summed map)")
    if has_learned:
        b = [abs(r["bias_learned_pct"]) for r in results
             if r["subset"] == subset]
        lines_out += [
            "",
            "cv ratio guards the SNR column: a network that blurs the map"
            " would raise SNR while cv ratio falls below 1.",
            "",
            "bias is the absolute level of the learned map against the same"
            " inverse-variance",
            "combination of the raw maps. SNR, cv and r are all"
            " scale-invariant and cannot see it.",
            f"Worst line {max(b):.0f}%, median {np.median(b):.0f}%: the fused"
            " maps carry a per-line gain",
            "error and need calibrating against the summed maps before"
            " element-to-element",
            "ratios are read off them.",
        ]

    print("\n".join(lines_out))
    with open(os.path.join(OUT_DIR, "fusion_weighted.txt"), "w") as f:
        f.write("\n".join(lines_out) + "\n")

    cols = list(results[0].keys())
    with open(os.path.join(OUT_DIR, "fusion_weighted.csv"), "w",
              newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=cols)
        wtr.writeheader()
        wtr.writerows(results)

    # ---- figure --------------------------------------------------------
    import matplotlib                                    # noqa: E402
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt                      # noqa: E402

    subset = "heldout_px" if "heldout_px" in subsets else "all_px"
    sub = [r for r in results if r["subset"] == subset]
    x = np.arange(len(sub))
    labels = [r["line"] for r in sub]

    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(11, 4.0), layout="constrained",
        gridspec_kw={"width_ratios": [1.35, 1]})

    width = 0.26
    series = [("sum", "summed", "0.55"), ("weighted", "inverse-variance", "C0")]
    if has_learned:
        series.append(("learned", "learned (N2N)", "C3"))
    for i, (key, label, colour) in enumerate(series):
        axa.bar(x + (i - (len(series) - 1) / 2) * width,
                [r[f"snr_{key}"] for r in sub], width, label=label,
                color=colour)
    axa.set_xticks(x)
    axa.set_xticklabels(labels)
    axa.set_ylabel("cross-scan SNR")
    axa.set_xlabel("emission line")
    axa.set_title(f"Fusion levels ({subset.replace('_', ' ')})")
    axa.legend(fontsize=8)
    axa.grid(alpha=0.3, axis="y")

    axb.axhline(0, color="gray", lw=0.8)
    axb.plot(x, [r["snr_gain_pct"] for r in sub], "o-", color="C0",
             label="inverse-variance")
    if has_learned:
        axb.plot(x, [r["snr_gain_learned_pct"] for r in sub], "D-",
                 color="C3", label="learned (N2N)")
        for i, r in enumerate(sub):
            if r["cv_ratio_learned"] < 0.85 or r["cv_ratio_learned"] > 1.15:
                axb.annotate(f"cv {r['cv_ratio_learned']:.2f}", (i, 0),
                             xytext=(0, -22), textcoords="offset points",
                             ha="center", fontsize=7, color="0.35")
    axb.set_xticks(x)
    axb.set_xticklabels(labels)
    axb.set_ylabel("SNR gain over summing (%)")
    axb.set_xlabel("emission line")
    axb.set_title("Gain over the summed map")
    axb.legend(fontsize=8)
    axb.grid(alpha=0.3)

    fig_path = os.path.join(OUT_DIR, "fusion_benchmark.png")
    fig.savefig(fig_path, dpi=200)

    print(f"\nSaved: {os.path.join(OUT_DIR, 'fusion_weighted.txt')}")
    print(f"Saved: {os.path.join(OUT_DIR, 'fusion_weighted.csv')}")
    print(f"Saved: {fig_path}")
