"""
02_deterministic_baseline.py - the deterministic restoration baseline
that any learned model must beat (MVP item 2).

Restores the REAL tilted scan (ruotato, 45x80) into the frontal frame
(60x120) by inverting the measured degradation operator:

  (a) warp-only      inverse affine (ruotato -> frontal, joint fit of
                     script 08), bilinear resampling
  (b) warp + gain    (a) divided by the per-element tilt gain of
                     script 11: gain = 1 + per_deg_sum * 7.7 / 100
                     (common-mode-free level change at the nominal
                     mounting angle)
  (c) noise floor    prova1 vs prova2 on the same footprint - the
                     repeatability ceiling no restoration can exceed

Truth = mean of the two frontal scans (prova1, prova2), detector-summed
maps, scored on the footprint of the warped tilted scan with the
harness in neurips-restore/src/eval.py: Pearson r, local-window SSIM,
level bias %, and the contrast guard cv_ratio.

Outputs:
    neurips-restore/results/deterministic_baseline.csv
    neurips-restore/results/deterministic_baseline.txt
    neurips-restore/results/deterministic_baseline.png   (Ca, PbLa)

Run from the repo root:
    python neurips-restore/scripts/02_deterministic_baseline.py
"""

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import eval as ev

OUT_DIR = os.path.join(ev.REPO_ROOT, "neurips-restore", "results")
CSV_PATH = os.path.join(OUT_DIR, "deterministic_baseline.csv")
TXT_PATH = os.path.join(OUT_DIR, "deterministic_baseline.txt")
PNG_PATH = os.path.join(OUT_DIR, "deterministic_baseline.png")

FIG_LINES = ("Ca", "PbLa")
CANDIDATES = ("warp_only", "warp_gain", "noise_floor")
CAND_LABEL = {
    "warp_only":   "warp only",
    "warp_gain":   "warp + gain",
    "noise_floor": "noise floor (p1 vs p2)",
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    p_affine = ev.read_affine("ruotato_joint")
    gains = ev.read_tilt_gains(ev.NOMINAL_TILT_DEG)
    print("Affine ruotato -> frontal [sx, sy, rot, shear, tx, ty]:")
    print("  " + np.array2string(np.round(p_affine, 4), separator=", "))
    print(f"Tilt gains at {ev.NOMINAL_TILT_DEG} deg (divided out in candidate b):")
    print("  " + "  ".join(f"{el}={gains[el]:.4f}" for el in ev.LINES))

    rows = []            # flat rows for the csv
    per_line = {}        # el -> {candidate -> metrics}
    fig_data = {}        # el -> (truth, warp_only, warp_gain, mask)

    for el in ev.LINES:
        truth = 0.5 * (ev.detsum_map("prova1", el) + ev.detsum_map("prova2", el))
        ruo = ev.detsum_map("ruotato", el)

        warped, mask = ev.warp_to_frontal(ruo, p_affine, truth.shape)
        corrected = warped / gains[el]

        # one shared SSIM data range per line so the three rows are comparable
        drange = float(truth[mask].max() - truth[mask].min())

        scores = {
            "warp_only":   ev.score_pair(warped, truth, mask, data_range=drange),
            "warp_gain":   ev.score_pair(corrected, truth, mask, data_range=drange),
            "noise_floor": ev.noise_floor(el, mask, data_range=drange),
        }
        per_line[el] = scores
        for cand in CANDIDATES:
            rows.append({"element": el, "candidate": cand, **scores[cand]})

        if el in FIG_LINES:
            fig_data[el] = (truth, warped, corrected, mask)

        print(f"  {el:5s} footprint {int(mask.sum()):4d} px   "
              f"r: {scores['warp_only']['r']:.3f} -> {scores['warp_gain']['r']:.3f}"
              f" (floor {scores['noise_floor']['r']:.3f})   "
              f"bias%: {scores['warp_only']['bias_pct']:+.2f} -> "
              f"{scores['warp_gain']['bias_pct']:+.2f}"
              f" (floor {scores['noise_floor']['bias_pct']:+.2f})")

    # ---- csv ---------------------------------------------------------
    with open(CSV_PATH, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["element", "candidate", "r", "ssim",
                                           "bias_pct", "cv_ratio", "n_px"])
        w.writeheader()
        w.writerows(rows)

    # ---- txt report ----------------------------------------------------
    lines = [
        "Deterministic restoration baseline - REAL ruotato -> frontal",
        "",
        "Restoration: inverse joint affine (script 08) + per-element tilt",
        f"gain of script 11 at the nominal {ev.NOMINAL_TILT_DEG} deg"
        " (candidate 'warp + gain').",
        "Truth = mean(prova1, prova2), detector-summed maps; all metrics on",
        "the footprint of the warped tilted scan"
        f" ({int(fig_data[FIG_LINES[0]][3].sum())} px of the 60x120 frame).",
        "Noise floor = prova2 scored against prova1 on the same footprint",
        "(independent noise in both maps - a candidate scored against the",
        " less-noisy mean can legitimately sit slightly above it in r/SSIM).",
        "",
        f"{'line':6s} {'candidate':<24s} {'r':>7s} {'SSIM':>7s}"
        f" {'bias %':>8s} {'cv_ratio':>9s}",
        "-" * 64,
    ]
    for el in ev.LINES:
        for cand in CANDIDATES:
            s = per_line[el][cand]
            lines.append(f"{el:6s} {CAND_LABEL[cand]:<24s} {s['r']:7.4f}"
                         f" {s['ssim']:7.4f} {s['bias_pct']:+8.2f}"
                         f" {s['cv_ratio']:9.4f}")
        lines.append("")

    # gap to the noise floor (headroom a learned model can claim)
    lines += [
        "Headroom to the noise floor (floor minus warp+gain; positive =",
        "room a learned model can still claim):",
        "",
        f"{'line':6s} {'dr':>8s} {'dSSIM':>8s} {'|bias|-|floor bias|':>21s}",
        "-" * 48,
    ]
    for el in ev.LINES:
        b = per_line[el]["warp_gain"]
        f = per_line[el]["noise_floor"]
        lines.append(f"{el:6s} {f['r'] - b['r']:+8.4f} {f['ssim'] - b['ssim']:+8.4f}"
                     f" {abs(b['bias_pct']) - abs(f['bias_pct']):+18.2f} pp")

    worse = [el for el in ev.LINES
             if abs(per_line[el]['warp_gain']['bias_pct'])
             > abs(per_line[el]['warp_only']['bias_pct'])]
    lines += [
        "",
        "Gain correction increases |bias| on: " + (", ".join(worse) if worse else "(none)"),
    ]

    with open(TXT_PATH, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))

    # ---- figure: truth | warp only | warp + gain (Ca, PbLa) -----------
    fig, axes = plt.subplots(len(FIG_LINES), 3,
                             figsize=(12.5, 2.4 * len(FIG_LINES) + 0.8),
                             dpi=120, layout="constrained")
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("0.82")  # outside-footprint pixels in recessive gray
    for i, el in enumerate(FIG_LINES):
        truth, warped, corrected, mask = fig_data[el]
        vmin, vmax = np.percentile(truth[mask], [1, 99])
        panels = [
            (np.where(mask, truth, np.nan), f"{el} - truth (mean prova1/2)"),
            (warped, f"{el} - warped-back ruotato"),
            (corrected, f"{el} - warp + gain (÷{ev.read_tilt_gains()[el]:.4f})"),
        ]
        for j, (img, title) in enumerate(panels):
            ax = axes[i, j]
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        cb = fig.colorbar(im, ax=axes[i, :], fraction=0.025, pad=0.01)
        cb.set_label("counts / s", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle("Deterministic baseline - real tilted scan restored to the "
                 "frontal frame (shared color scale per row; gray = outside "
                 "footprint)", fontsize=11, fontweight="bold")
    fig.savefig(PNG_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    for p in (CSV_PATH, TXT_PATH, PNG_PATH):
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()
