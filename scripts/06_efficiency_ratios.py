"""
06_efficiency_ratios.py
Per-element detector efficiency ratios R = det10264 / det19511 with
bootstrap uncertainties, for the two frontal scans (prova1, prova2) and
the tilted scan (ruotato).

The prova1<->prova2 spread is the repeatability baseline (same geometry,
7 days apart); the ruotato shift relative to the frontal mean is the
geometric signal induced by tilting the canvas. K is noise-level and Zn
is sparse and correction-dependent -- both are reported but should be
excluded from any model fit.

Uses (and if needed builds) the same npy cache as 05_detector_diff.py.
Outputs:

    results/detector_diff/efficiency_ratios.csv
    results/detector_diff/ratio_vs_energy.png

Run from the project root:
    python scripts/06_efficiency_ratios.py
"""

import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from xrf_core import (
    apply_corrections,
    build_calibration,
    load_elements,
    process_folder,
)

DETECTORS  = ["10264", "19511"]
OUTPUT_DIR = os.path.join("results", "detector_diff")
CACHE_DIR  = os.path.join(OUTPUT_DIR, "_npy_cache")

# scan label -> (accepted dataset folder names, width, height)
SCANS = {
    "prova1":  (["aurora-antico1-prova1"], 120, 60),
    "prova2":  (["aurora-antico1-prova2"], 120, 60),
    "ruotato": (["antico1-prova4-ruotato", "aurora-antico1-ruotato"], 80, 45),
}

N_BOOT = 2000
SEED   = 42

# excluded from fits: K is at noise level, Zn is sparse and inherits the
# Cu Kbeta correction
UNRELIABLE = {"K", "Zn"}


def resolve_scan_dir(names: list[str]) -> str | None:
    for root in ("Resources", "."):
        for name in names:
            p = os.path.join(root, name)
            if all(os.path.isdir(os.path.join(p, det)) for det in DETECTORS):
                return p
    return None


def ratio_with_ci(a: np.ndarray, b: np.ndarray, rng) -> tuple[float, float]:
    """Global sum ratio a/b with a pixel-bootstrap standard error."""
    if b.sum() <= 0:
        return np.nan, np.nan
    r = a.sum() / b.sum()
    idx  = rng.integers(0, a.size, size=(N_BOOT, a.size))
    rb   = b[idx].sum(axis=1)
    boot = a[idx].sum(axis=1) / np.where(rb > 0, rb, np.nan)
    return r, float(np.nanstd(boot))


if __name__ == "__main__":
    slope, intercept = build_calibration()
    elements = load_elements()
    el_keys  = list(elements.keys())
    rng      = np.random.default_rng(SEED)

    # corrected per-element pixel vectors: cubes[scan][det][el]
    cubes: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for scan, (names, width, height) in SCANS.items():
        scan_dir = resolve_scan_dir(names)
        cubes[scan] = {}
        for det in DETECTORS:
            label  = f"{scan}_{det}"
            folder = os.path.join(scan_dir or names[0], det)
            cube, keys = process_folder(
                folder, width, height, label, elements,
                "fixed_hw", slope, intercept, CACHE_DIR,
            )
            cube_c = apply_corrections(cube, keys, label)
            cubes[scan][det] = {
                k: cube_c[i].ravel() for i, k in enumerate(keys)
            }

    # ratios per element per scan, ordered by line energy
    els  = sorted(el_keys, key=lambda k: elements[k]["kev"])
    rows = []
    for el in els:
        r = {
            scan: ratio_with_ci(
                cubes[scan]["10264"][el], cubes[scan]["19511"][el], rng
            )
            for scan in SCANS
        }
        (r1, s1), (r2, s2), (rr, sr) = r["prova1"], r["prova2"], r["ruotato"]
        rf    = 0.5 * (r1 + r2)
        base  = abs(r1 - r2) / rf * 100
        tilt  = (rr - rf) / rf * 100
        sigma = np.sqrt(sr**2 + 0.25 * (s1**2 + s2**2))
        nsig  = abs(rr - rf) / sigma if sigma > 0 else np.nan
        rows.append({
            "element": el, "kev": elements[el]["kev"],
            "R_prova1": r1, "sig_prova1": s1,
            "R_prova2": r2, "sig_prova2": s2,
            "R_ruotato": rr, "sig_ruotato": sr,
            "baseline_pct": base, "tilt_shift_pct": tilt,
            "significance_sigma": nsig,
            "reliable": el not in UNRELIABLE,
        })

    # console table
    hdr = (f"{'el':6}{'keV':>6}  {'R_p1':>8}{'R_p2':>8}{'R_ruo':>8}  "
           f"{'base%':>7}{'tilt%':>7}{'tilt/sig':>9}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for w in rows:
        flag = "" if w["reliable"] else "  (excluded from fits)"
        print(f"{w['element']:6}{w['kev']:6.2f}  "
              f"{w['R_prova1']:8.4f}{w['R_prova2']:8.4f}{w['R_ruotato']:8.4f}  "
              f"{w['baseline_pct']:7.2f}{w['tilt_shift_pct']:+7.2f}"
              f"{w['significance_sigma']:9.1f}{flag}")

    # CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "efficiency_ratios.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # figure: R vs line energy, frontal mean vs tilted
    good = [w for w in rows if w["reliable"]]
    kev  = [w["kev"] for w in good]
    rf_v = [0.5 * (w["R_prova1"] + w["R_prova2"]) for w in good]
    rf_e = [0.5 * np.hypot(w["sig_prova1"], w["sig_prova2"]) for w in good]
    rr_v = [w["R_ruotato"] for w in good]
    rr_e = [w["sig_ruotato"] for w in good]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(kev, rf_v, yerr=rf_e, fmt="o-", capsize=3,
                label="frontal (prova1/prova2 mean)")
    ax.errorbar(kev, rr_v, yerr=rr_e, fmt="s--", capsize=3,
                label="tilted (ruotato)")
    for w in good:
        ax.annotate(w["element"], (w["kev"], w["R_ruotato"]),
                    textcoords="offset points", xytext=(4, -10), fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("line energy (keV)")
    ax.set_ylabel("R = det10264 / det19511")
    ax.set_title("Detector ratio vs energy: frontal vs tilted canvas")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "ratio_vs_energy.png")
    fig.savefig(fig_path, dpi=200)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {fig_path}")
