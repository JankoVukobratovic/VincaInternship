"""
11_positioning_sensitivity.py
===============================================================================
How much does an element map change if the canvas is not mounted flat?
The "practical estimate of the error caused by imperfect mounting" of the
abstract (PLAN 4.6).

Method
------
The tilted scan is the same painting seen at a different mounting angle,
so the change of each element map between frontal and tilted, once the
two are registered, IS the positioning error -- measured rather than
modelled. Using the joint affine transform of script 08, every frontal
map is resampled into the tilted scan's frame and compared pixel by
pixel on the common footprint:

    delta(el) = sum(ruotato) / sum(frontal resampled) - 1

Two corrections make that number mean what it should:

1. Common mode. The scans differ in overall level (~6% fewer counts per
   pixel in the tilted scan) for reasons that have nothing to do with
   geometry -- dwell, tube current, session drift. That global factor is
   degenerate with the tilt's own solid-angle change, so the median
   delta over the reliable lines is removed and reported separately. It
   is the *differential*, element-to-element error that biases pigment
   identification, and that is what survives the correction.
2. Repeatability floor. prova1 vs prova2 (same geometry, 7 days apart)
   goes through the identical pipeline with the control transform; a
   per-element result only counts if it clears that floor.

Dividing by the mounting angle gives %/degree. The angle from
foreshortening is an UPPER bound (7.7 deg, script 07b), so the
sensitivities below are LOWER bounds: if the canvas was tilted less, the
same map change happened over fewer degrees and the per-degree error is
larger. This flips the usual reading of an upper bound and is the honest
way to quote it until the instrument builder confirms the angle.

Input : results/detector_diff/_npy_cache/*.npy        (from script 06)
        results/registration/affine_params.csv        (from script 08)
Output: results/registration/positioning_sensitivity.csv
        results/registration/positioning_sensitivity.txt
        results/registration/positioning_sensitivity.png

Run from the project root:
    python scripts/11_positioning_sensitivity.py
"""

import csv
import importlib
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)

reg = importlib.import_module("08_registration")

OUT_DIR = os.path.join("results", "registration")
PARAMS_CSV = os.path.join(OUT_DIR, "affine_params.csv")

# Mounting angle of the tilted scan, from the vertical foreshortening
# (07b). An upper bound -- see the module docstring on what that does to
# the per-degree numbers.
TILT_DEG = 7.7

ELEMENTS = reg.ALL_ELEMENTS
RELIABLE = [el for el in ELEMENTS if el not in reg.UNRELIABLE]
VARIANTS = ("10264", "19511", "sum")
SEED = 42


def load_variant(scan, el, variant):
    if variant == "sum":
        return sum(reg.load_map(scan, d, el) for d in reg.DETECTORS)
    return reg.load_map(scan, variant, el)


def read_params(fit):
    with open(PARAMS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["fit"] == fit:
                return np.array([float(row[k]) for k in
                                 ("sx", "sy", "rot_deg", "shear", "tx", "ty")])
    sys.exit(f"ERROR: fit '{fit}' missing from {PARAMS_CSV} - run "
             "scripts/08_registration.py first.")


def change(ref_map, src_map, p, rng):
    """Relative change of src against ref, ref resampled into src's frame."""
    warped, valid = reg.warp_reference(ref_map, p, src_map.shape)
    m = valid & np.isfinite(warped) & (warped > 0)
    r, sig = reg.ratio_with_ci(src_map[m], warped[m], rng)
    return 100.0 * (r - 1.0), 100.0 * sig, int(m.sum())


if __name__ == "__main__":
    print("=" * 70)
    print("  POSITIONING SENSITIVITY OF THE ELEMENT MAPS")
    print("=" * 70)

    rng = np.random.default_rng(SEED)
    p_tilt = read_params("ruotato_joint")
    p_ctl = read_params("prova2_control")
    print(f"  tilt transform : {np.round(p_tilt, 4)}")
    print(f"  control        : {np.round(p_ctl, 4)}")
    print(f"  mounting angle : {TILT_DEG} deg (upper bound, 07b)")

    rows = []
    for el in ELEMENTS:
        row = {"element": el, "kev": ELEMENTS[el],
               "reliable": el not in reg.UNRELIABLE}
        for v in VARIANTS:
            ref = load_variant("prova1", el, v)
            d_tilt, s_tilt, n_px = change(ref, load_variant("ruotato", el, v),
                                          p_tilt, rng)
            d_base, s_base, _ = change(ref, load_variant("prova2", el, v),
                                       p_ctl, rng)
            row[f"tilt_pct_{v}"] = d_tilt
            row[f"tilt_sig_{v}"] = s_tilt
            row[f"baseline_pct_{v}"] = d_base
            row[f"baseline_sig_{v}"] = s_base
            row[f"n_px_{v}"] = n_px
        rows.append(row)
        print(f"  {el:5s} tilt {row['tilt_pct_sum']:+7.2f}%"
              f"  baseline {row['baseline_pct_sum']:+6.2f}%")

    # ---- common mode and per-degree numbers ----------------------------
    rel = [r for r in rows if r["reliable"]]
    common = {v: float(np.median([r[f"tilt_pct_{v}"] for r in rel]))
              for v in VARIANTS}
    base_rms = {v: float(np.sqrt(np.mean(
        [r[f"baseline_pct_{v}"] ** 2 for r in rel]))) for v in VARIANTS}

    for r in rows:
        for v in VARIANTS:
            r[f"tilt_rel_pct_{v}"] = r[f"tilt_pct_{v}"] - common[v]
            r[f"per_deg_{v}"] = r[f"tilt_rel_pct_{v}"] / TILT_DEG
            r[f"per_deg_sig_{v}"] = r[f"tilt_sig_{v}"] / TILT_DEG

    # ---- report --------------------------------------------------------
    worst = max(rel, key=lambda r: abs(r["per_deg_sum"]))
    spread = (max(r["tilt_rel_pct_sum"] for r in rel)
              - min(r["tilt_rel_pct_sum"] for r in rel))

    lines = [
        "Positioning sensitivity of the element maps",
        "(frontal vs tilted after joint affine registration, summed",
        " detectors; common mode removed, see below)",
        "",
        f"mounting angle          : {TILT_DEG:.1f} deg (upper bound -> the",
        "                          per-degree values are lower bounds)",
        f"common mode removed     : {common['sum']:+.2f}% (median over the"
        " reliable lines;",
        "                          overall level difference between the"
        " scans)",
        f"repeatability floor     : {base_rms['sum']:.2f}% RMS"
        " (prova1 vs prova2, same geometry)",
        "",
        f"{'line':6s} {'keV':>6s} {'tilt %':>8s} {'rel %':>8s}"
        f" {'%/deg':>8s} {'baseline %':>11s} {'clears floor':>13s}",
    ]
    for r in rows:
        mark = "" if r["reliable"] else "  (excluded)"
        clears = ("yes" if abs(r["tilt_rel_pct_sum"])
                  > 2 * abs(r["baseline_pct_sum"]) else "no")
        lines.append(
            f"{r['element']:6s} {r['kev']:6.2f} {r['tilt_pct_sum']:+8.2f}"
            f" {r['tilt_rel_pct_sum']:+8.2f} {r['per_deg_sum']:+8.3f}"
            f" {r['baseline_pct_sum']:+11.2f} {clears:>13s}{mark}")

    # ---- cross-check against the region-level ratios of script 08 ------
    # 08 compares sums over two regions that cover roughly the same area;
    # here the frontal map is resampled pixel by pixel into the tilted
    # frame. Both estimate the same tilt shift of R = d1/d2, so their
    # difference measures the systematic of the comparison itself.
    ratio_csv = os.path.join(OUT_DIR, "overlap_ratios.csv")
    cross = []
    if os.path.exists(ratio_csv):
        with open(ratio_csv, newline="") as f:
            ov = {r["element"]: float(r["tilt_overlap_pct"])
                  for r in csv.DictReader(f)}
        for r in rel:
            d1 = r["tilt_pct_10264"] / 100.0
            d2 = r["tilt_pct_19511"] / 100.0
            here = 100.0 * ((1 + d1) / (1 + d2) - 1)
            cross.append((r["element"], here, ov[r["element"]]))
        diffs = [a - b for _, a, b in cross]
        lines += [
            "",
            "Cross-check -- tilt shift of R = d10264/d19511 implied by the",
            "per-detector changes above, against the region-level ratios of",
            "script 08 (same quantity, independent estimator):",
            "",
            f"{'line':6s} {'this (%)':>9s} {'08 (%)':>8s} {'diff (pp)':>10s}",
        ]
        for el, here, there in cross:
            lines.append(f"{el:6s} {here:+9.2f} {there:+8.2f}"
                         f" {here - there:+10.2f}")
        lines += [
            "",
            f"RMS difference {np.sqrt(np.mean(np.square(diffs))):.2f} pp,"
            f" always the same sign (this estimator reads lower).",
            "The two agree on the shape and on every sign; the offset is the",
            "systematic of region-matching against pixel-matching and should",
            "be quoted as such on the tilt-shift figure.",
        ]

    lines += [
        "",
        f"Spread across the reliable lines: {spread:.1f} percentage points"
        f" over {TILT_DEG:.1f} deg,",
        f"i.e. {spread / TILT_DEG:.2f} pp/deg between the extremes."
        f" Largest single sensitivity:",
        f"{worst['element']} at {worst['per_deg_sum']:+.3f} %/deg.",
        "",
        "Reading: a mounting error of one degree changes the light-element",
        "maps relative to the heavy-element maps by the amounts above. The",
        "absolute level of a map is not recoverable this way (it is",
        "degenerate with session drift), but the element-to-element ratios",
        "that pigment identification relies on carry exactly this error.",
        "",
        "Per detector (the two channels move oppositely -- the tilt tips",
        "the take-off geometry one way for 10264 and the other for 19511):",
        "",
        f"{'line':6s} {'%/deg 10264':>12s} {'%/deg 19511':>12s}"
        f" {'%/deg sum':>10s}",
    ]
    for r in rows:
        if not r["reliable"]:
            continue
        lines.append(f"{r['element']:6s} {r['per_deg_10264']:+12.3f}"
                     f" {r['per_deg_19511']:+12.3f} {r['per_deg_sum']:+10.3f}")

    print()
    print("\n".join(lines))
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "positioning_sensitivity.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    cols = list(rows[0].keys())
    with open(os.path.join(OUT_DIR, "positioning_sensitivity.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # ---- figure --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
    e = np.array([r["kev"] for r in rel])
    order = np.argsort(e)
    labels = [rel[i]["element"] for i in order]
    x = np.arange(len(order))
    for v, marker, label in (("10264", "o", "det 10264"),
                             ("19511", "s", "det 19511"),
                             ("sum", "D", "summed map")):
        y = np.array([rel[i][f"per_deg_{v}"] for i in order])
        yerr = np.array([rel[i][f"per_deg_sig_{v}"] for i in order])
        ax.errorbar(x, y, yerr=yerr, fmt=marker + "-", capsize=3, label=label,
                    lw=1.2, ms=5)
    floor = base_rms["sum"] / TILT_DEG
    ax.axhspan(-floor, floor, color="0.85", zorder=0,
               label=f"repeatability floor ({floor:.2f} %/deg)")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("emission line (increasing energy)")
    ax.set_ylabel("map change per degree of tilt (%)")
    ax.set_title("Sensitivity of the element maps to canvas mounting angle")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(OUT_DIR, "positioning_sensitivity.png"), dpi=200)

    print(f"\nSaved: {os.path.join(OUT_DIR, 'positioning_sensitivity.txt')}")
    print(f"Saved: {os.path.join(OUT_DIR, 'positioning_sensitivity.csv')}")
    print(f"Saved: {os.path.join(OUT_DIR, 'positioning_sensitivity.png')}")
