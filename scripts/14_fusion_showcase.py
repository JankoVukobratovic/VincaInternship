"""
14_fusion_showcase.py
===============================================================================
Conference showcase figure: the learned Noise2Noise fusion against simple
summing, on the two lines where the improvement is worth showing.

09_fusion.py tabulates the cross-scan SNR of every fusion variant; this
script makes the headline visible. For Pb Ll (+69% SNR on held-out
pixels) and Fe Ka (+29%) it shows, per variant, the two independent scan
maps (prova1, prova2) and the cross-scan noise map (prova1 - prova2) /
sqrt(2). The two scans are back-to-back acquisitions of the same canvas,
so their difference contains no signal: whatever survives in the third
column is pure noise, and the learned rows visibly calm down while the
signal columns keep the same structure.

Fairness of the comparison:
  * within a line, all four signal panels share one color scale and the
    two noise panels share one symmetric diverging scale;
  * the learned cubes live in the detector-A scale (about half the
    summed level), so for display only they are gain-matched to the
    summed maps with a single scalar per line -- the SNR is
    scale-invariant and is computed on the un-scaled maps;
  * the annotated SNR uses the same definition and the same pixels as
    fusion_weighted.txt (checkerboard B restricted to the network's
    held-out prova1 validation blocks), so the numbers on the figure are
    the numbers in the paper;
  * cv_ratio (spatial contrast of the learned map against the summed
    map, 09_fusion.py fidelity_vs_sum) is printed and written to the txt
    as the guard against a blur artificially inflating the SNR.

Input : results/vulnerability_mapping/ablation_cube_{prova1,prova2}_{det}.npy
        xrf-denoise/data/processed/fused_{prova1,prova2}.npy
        xrf-denoise/data/processed/fused_heldout_px.json
Output: results/detector_diff/fusion_showcase.png
        results/detector_diff/fusion_showcase.txt

Run from the project root:
    python scripts/14_fusion_showcase.py
"""

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
DETS = ("10264", "19511")
DATASETS = ("prova1", "prova2")

# headline line first; pretty labels are used inside the figure only
SHOW_LINES = [("PbLl", "Pb L$\\ell$"), ("Fe", "Fe K$\\alpha$")]

OUT_DIR = os.path.join("results", "detector_diff")
CUBE_CACHE = os.path.join("results", "vulnerability_mapping")
FUSED_DIR = os.path.join("xrf-denoise", "data", "processed")

_ELEMENTS_JSON = xrf_core.load_elements()


def extract_line_maps(cube, keys):
    """Net-intensity maps with the pipeline integrator (as 09_fusion.py)."""
    n_ch = cube.shape[2]
    en = xrf_core.energy_axis(n_ch, vuln._SLOPE, vuln._INTERCEPT)
    maps = {}
    for key in keys:
        cfg_el = _ELEMENTS_JSON[key]
        m = np.zeros((ROWS, COLS))
        for r in range(ROWS):
            for c in range(COLS):
                m[r, c] = xrf_core.integrate(cube[r, c], en, key, cfg_el,
                                             "fixed_hw")
        maps[key] = m
    return maps


def snr(v1, v2, mask):
    """Cross-scan SNR, same definition as 09_fusion.py."""
    a, b = v1[mask], v2[mask]
    noise = np.std(a - b) / np.sqrt(2.0)
    return float(np.mean(0.5 * (a + b)) / noise) if noise > 0 else np.inf


def cv_ratio_vs_sum(variant, reference, mask):
    """Spatial-contrast guard, same as 09_fusion.py fidelity_vs_sum."""
    v = 0.5 * (variant["prova1"] + variant["prova2"])[mask]
    s = 0.5 * (reference["prova1"] + reference["prova2"])[mask]
    cv_v = np.std(v) / np.mean(v) if np.mean(v) > 0 else np.nan
    cv_s = np.std(s) / np.mean(s) if np.mean(s) > 0 else np.nan
    return float(cv_v / cv_s)


if __name__ == "__main__":
    print("=" * 70)
    print("  FUSION SHOWCASE: SUMMED vs LEARNED, Pb Ll AND Fe")
    print("=" * 70)

    keys = [k for k, _ in SHOW_LINES]

    # ---- maps ------------------------------------------------------------
    raw = {}                                       # (dataset, det) -> {line: map}
    for ds in DATASETS:
        for det in DETS:
            p = os.path.join(CUBE_CACHE, f"ablation_cube_{ds}_{det}.npy")
            print(f"  [{ds}/{det}] extracting line maps...")
            raw[(ds, det)] = extract_line_maps(np.load(p), keys)

    fused = {}                                     # dataset -> {line: map}
    for ds in DATASETS:
        p = os.path.join(FUSED_DIR, f"fused_{ds}.npy")
        if not os.path.exists(p):
            sys.exit(f"ERROR: {p} missing - run xrf-denoise/scripts/"
                     "07_train_cross_detector.py --export-only first.")
        print(f"  [{ds}/fused] extracting line maps...")
        fused[ds] = extract_line_maps(np.load(p).astype(np.float64), keys)

    # ---- evaluation pixels (identical to 09_fusion.py) --------------------
    rc = np.add.outer(np.arange(ROWS), np.arange(COLS))
    mask_B = (rc % 2) == 1                         # evaluation checkerboard

    with open(os.path.join(FUSED_DIR, "fused_heldout_px.json")) as f:
        rec = json.load(f)
    heldout = np.zeros(ROWS * COLS, dtype=bool)
    heldout[np.asarray(rec["val_indices"], dtype=int)] = True
    heldout = heldout.reshape(ROWS, COLS)

    subsets = {"all_px": mask_B, "heldout_px": mask_B & heldout}

    # ---- metrics -----------------------------------------------------------
    stats = {}                                     # (line, variant) -> dict
    variants_by_line = {}
    for key in keys:
        v_sum = {ds: raw[(ds, "10264")][key] + raw[(ds, "19511")][key]
                 for ds in DATASETS}
        v_lrn = {ds: fused[ds][key] for ds in DATASETS}
        variants_by_line[key] = {"sum": v_sum, "learned": v_lrn}
        for name, vv in (("sum", v_sum), ("learned", v_lrn)):
            row = {}
            for subset, px in subsets.items():
                row[f"snr_{subset}"] = snr(vv["prova1"], vv["prova2"], px)
                row[f"cv_{subset}"] = cv_ratio_vs_sum(vv, v_sum, px)
            stats[(key, name)] = row

    for key in keys:
        for subset in subsets:
            s0 = stats[(key, "sum")][f"snr_{subset}"]
            s1 = stats[(key, "learned")][f"snr_{subset}"]
            stats[(key, "learned")][f"gain_{subset}"] = 100 * (s1 / s0 - 1)

    print()
    print(f"{'line':6s} {'variant':8s} {'SNR all':>8s} {'SNR held':>9s}"
          f" {'gain held':>10s} {'cv all':>7s} {'cv held':>8s}")
    for key in keys:
        for name in ("sum", "learned"):
            s = stats[(key, name)]
            gain = (f"{s['gain_heldout_px']:+7.1f}%"
                    if name == "learned" else f"{'-':>8s}")
            print(f"{key:6s} {name:8s} {s['snr_all_px']:8.2f}"
                  f" {s['snr_heldout_px']:9.2f} {gain:>10s}"
                  f" {s['cv_all_px']:7.3f} {s['cv_heldout_px']:8.3f}")

    # ---- figure ------------------------------------------------------------
    import matplotlib                                    # noqa: E402
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt                      # noqa: E402

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "figure.facecolor": "white",
    })

    fig = plt.figure(figsize=(13.2, 8.6), layout="constrained")
    gs = fig.add_gridspec(4, 5, width_ratios=[1, 1, 0.045, 1, 0.045],
                          hspace=0.04, wspace=0.03)

    col_titles = ["scan 1 (prova1)", "scan 2 (prova2)",
                  "cross-scan noise  (scan1 $-$ scan2)/$\\sqrt{2}$"]
    disp_gain = {}

    for li, (key, label) in enumerate(SHOW_LINES):
        vv = variants_by_line[key]
        r0 = 2 * li

        # display-only gain match of the learned maps to the summed level
        g = (np.mean(vv["sum"]["prova1"] + vv["sum"]["prova2"])
             / np.mean(vv["learned"]["prova1"] + vv["learned"]["prova2"]))
        disp_gain[key] = float(g)
        disp = {
            "sum": {ds: vv["sum"][ds] for ds in DATASETS},
            "learned": {ds: g * vv["learned"][ds] for ds in DATASETS},
        }
        noise = {name: (m["prova1"] - m["prova2"]) / np.sqrt(2.0)
                 for name, m in disp.items()}

        # shared scales within the line block (P99-clipped display)
        pool_sig = np.concatenate([disp[n][ds].ravel()
                                   for n in ("sum", "learned")
                                   for ds in DATASETS])
        vmax = np.percentile(pool_sig, 99)
        pool_noise = np.concatenate([np.abs(noise[n]).ravel()
                                     for n in ("sum", "learned")])
        nlim = np.percentile(pool_noise, 99)

        ims, imn = None, None
        for vi, name in enumerate(("sum", "learned")):
            row = r0 + vi
            axes_row = [fig.add_subplot(gs[row, c]) for c in (0, 1, 3)]
            for ax, ds in zip(axes_row[:2], DATASETS):
                ims = ax.imshow(disp[name][ds], cmap="inferno",
                                vmin=0, vmax=vmax, interpolation="nearest")
            imn = axes_row[2].imshow(noise[name], cmap="RdBu_r",
                                     vmin=-nlim, vmax=nlim,
                                     interpolation="nearest")
            for ax in axes_row:
                ax.set_xticks([])
                ax.set_yticks([])
            if row == 0:
                for ax, t in zip(axes_row, col_titles):
                    ax.set_title(t, fontsize=11.5)

            vname = "summed" if name == "sum" else "learned (N2N)"
            axes_row[0].set_ylabel(f"{label}\n{vname}", fontsize=12,
                                   multialignment="center")

            s = stats[(key, name)]
            txt = f"SNR {s['snr_heldout_px']:.1f}"
            if name == "learned":
                txt += f"  ({s['gain_heldout_px']:+.0f}%)"
            axes_row[2].text(
                0.985, 0.955, txt, transform=axes_row[2].transAxes,
                ha="right", va="top", fontsize=12.5,
                fontweight="bold" if name == "learned" else "normal",
                color="0.15",
                bbox=dict(facecolor="white", alpha=0.85,
                          edgecolor="0.6", boxstyle="round,pad=0.25"))

        cax_s = fig.add_subplot(gs[r0:r0 + 2, 2])
        cb = fig.colorbar(ims, cax=cax_s)
        cb.set_label("net counts", fontsize=9)
        cb.ax.tick_params(labelsize=8)
        cax_n = fig.add_subplot(gs[r0:r0 + 2, 4])
        cb = fig.colorbar(imn, cax=cax_n)
        cb.ax.tick_params(labelsize=8)

    fig.suptitle("Dual-detector fusion: learned Noise2Noise vs simple"
                 " summing", fontsize=15)
    cv_note = ", ".join(
        f"{lb.replace('$\\ell$', 'l').replace('$\\alpha$', 'a')}"
        f" {stats[(k, 'learned')]['cv_heldout_px']:.2f}"
        for k, lb in SHOW_LINES)
    fig.text(0.01, -0.015,
             "Cross-scan SNR on held-out pixels (never seen in training);"
             " scans 1 and 2 are independent acquisitions, so their"
             " difference is pure noise.\n"
             "Signal maps P99-clipped, one color scale per line; learned"
             " maps gain-matched to the summed level for display only"
             " (SNR is scale-invariant)."
             f" Spatial-contrast ratio vs summed map: {cv_note}"
             " (1.0 = no blurring).",
             fontsize=9, color="0.35", va="top")

    os.makedirs(OUT_DIR, exist_ok=True)
    fig_path = os.path.join(OUT_DIR, "fusion_showcase.png")
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")

    # ---- txt report --------------------------------------------------------
    lines_out = [
        "Fusion showcase: simple summing vs learned N2N fusion",
        "(companion numbers of fusion_showcase.png; same SNR definition,",
        " same checkerboard-B pixels as fusion_weighted.txt)",
        "",
        "SNR = mean((map_p1 + map_p2)/2) / (std(map_p1 - map_p2)/sqrt(2))",
        "cv_ratio = spatial contrast of the variant's average map vs the",
        "           summed map (fidelity guard: < 1 would mean blurring)",
        "",
    ]
    for subset, px in subsets.items():
        note = {"all_px": "all pixels",
                "heldout_px": "pixels the network never saw"
                              " (prova1 validation blocks)"}[subset]
        lines_out += [f"[{subset}]  {note}, {int(px.sum())} px",
                      f"{'line':6s} {'variant':9s} {'SNR':>7s} {'gain':>8s}"
                      f" {'cv_ratio':>9s}"]
        for key in keys:
            for name in ("sum", "learned"):
                s = stats[(key, name)]
                gain = (f"{s[f'gain_{subset}']:+7.1f}%"
                        if name == "learned" else f"{'-':>8s}")
                lines_out.append(
                    f"{key:6s} {name:9s} {s[f'snr_{subset}']:7.2f}"
                    f" {gain:>8s} {s[f'cv_{subset}']:9.3f}")
        lines_out.append("")
    lines_out += [
        "display-only gain factors (learned maps are in detector-A scale;",
        "multiplied by this scalar so both rows share one color scale):",
    ]
    for key in keys:
        lines_out.append(f"  {key:6s} x {disp_gain[key]:.3f}")
    lines_out += [
        "",
        "The figure quotes the heldout_px SNR values -- the row of",
        "fusion_weighted.txt that the paper quotes.",
    ]
    txt_path = os.path.join(OUT_DIR, "fusion_showcase.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines_out) + "\n")

    print(f"\nSaved: {fig_path}")
    print(f"Saved: {txt_path}")
