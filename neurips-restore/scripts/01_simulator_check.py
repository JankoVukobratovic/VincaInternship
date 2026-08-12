"""
01_simulator_check.py
=====================
Fidelity gate for the measured-physics forward simulator (MVP item 1).

Simulates the tilted scan from the frontal prova1 (detector-summed
maps, 8 lines) with src/forward_model.py and compares against the REAL
ruotato scan:

  - convention self-test: the affine_transform-based warp must equal a
    direct map_coordinates re-implementation of script 08's convention;
  - per-element Pearson r (simulated vs real tilted) on the valid
    footprint, both noise-free and noisy, with the prova1-prova2
    correlation on the overlap region as the noise-floor ceiling;
  - level check: median(sim/real) per element;
  - noise check: calibrated k vs a detector-split estimate on the real
    ruotato (validated on prova1), plus a high-frequency (Immerkaer)
    roughness comparison real vs simulated;
  - round-trip test of the deterministic inverse;
  - figure results/simulator_check.png (Ca, Fe, PbLa: real | sim |
    difference, shared scales) and report results/simulator_check.txt.

Run from the repository root:
    python neurips-restore/scripts/01_simulator_check.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve, map_coordinates

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))
import forward_model as fm

OUT_DIR = os.path.join(fm.REPO_ROOT, "neurips-restore", "results")
PNG_PATH = os.path.join(OUT_DIR, "simulator_check.png")
TXT_PATH = os.path.join(OUT_DIR, "simulator_check.txt")
SEED = 42
FIG_ELEMENTS = ("Ca", "Fe", "PbLa")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else np.nan


def robust_sigma(x: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def reference_warp_08(ref: np.ndarray) -> np.ndarray:
    """Script-08-style warp of a frontal map into the tilted frame,
    via map_coordinates -- independent re-implementation used to verify
    the affine_transform convention in forward_model."""
    A, t = fm.affine_ruotato_to_frontal()
    hs, ws = fm.TILTED_SHAPE
    hr, wr = fm.FRONTAL_SHAPE
    xs, ys = np.meshgrid(np.arange(ws), np.arange(hs))
    v = np.stack([xs.ravel() - (ws - 1) / 2, ys.ravel() - (hs - 1) / 2])
    x1, y1 = A @ v
    x1 = x1 + (wr - 1) / 2 + t[0]
    y1 = y1 + (hr - 1) / 2 + t[1]
    warped = map_coordinates(ref, [y1, x1], order=1, mode="constant",
                             cval=np.nan)
    return warped.reshape(hs, ws)


def split_k(scan: str, el: str) -> tuple[float, float]:
    """Noise scale k of the detector-SUMMED map of one scan, estimated
    from the two detectors alone (no repeat scan needed).

    With R = sum(d1)/sum(d2) and e = d1 - R*d2 the painting structure
    cancels (up to the ratio flat-field, which inflates the estimate,
    so this is an upper bound); assuming Var = k*counts in each channel,
    z = e / sqrt(d1 + R^2 d2) has std sqrt(k) and the summed map obeys
    Var(d1+d2) = k*(d1+d2).  Returns (k, median summed signal).
    """
    d1 = np.load(os.path.join(fm.CACHE_DIR, f"{scan}_10264_{el}.npy"))
    d2 = np.load(os.path.join(fm.CACHE_DIR, f"{scan}_19511_{el}.npy"))
    R = d1.sum() / d2.sum()
    e = (d1 - R * d2).ravel()
    v = (d1 + R * R * d2).ravel()
    ok = v > 0
    z = e[ok] / np.sqrt(v[ok])
    return robust_sigma(z) ** 2, float(np.median((d1 + d2)[ok.reshape(d1.shape)]))


def hf_sigma(img: np.ndarray) -> float:
    """Robust Immerkaer high-frequency noise estimate (counts).
    Texture leaks in, so it is comparable BETWEEN images, not absolute."""
    kernel = np.array([[1.0, -2.0, 1.0],
                       [-2.0, 4.0, -2.0],
                       [1.0, -2.0, 1.0]])
    m = np.asarray(img, dtype=float)
    fin = np.isfinite(m)
    mm = np.where(fin, m, 0.0)
    resp = convolve(mm, kernel, mode="constant")[2:-2, 2:-2]
    good = convolve(fin.astype(float), np.abs(kernel), mode="constant"
                    )[2:-2, 2:-2] >= 15.9   # all 8 neighbours finite
    return robust_sigma(resp[good]) / 6.0


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    lines_out = []

    def emit(s=""):
        print(s)
        lines_out.append(s)

    emit("=" * 74)
    emit("  FORWARD SIMULATOR FIDELITY CHECK  (prova1 -> sim tilted vs ruotato)")
    emit("=" * 74)

    p = fm.load_affine_params()
    gains = fm.tilt_gains()
    ks = fm.calibrate_noise()
    emit(f"affine (ruotato->prova1, script 08 joint): "
         f"sx={p[0]:.4f} sy={p[1]:.4f} rot={p[2]:.3f} deg "
         f"shear={p[3]:.4f} tx={p[4]:.3f} ty={p[5]:.3f}")
    emit(f"angle = {fm.REF_ANGLE_DEG} deg; gains from tilt_pct_sum "
         f"(script 11, common mode included); noise Var = k*counts")
    emit("")

    frontal = fm.load_summed_maps("prova1")
    frontal2 = fm.load_summed_maps("prova2")
    real = fm.load_summed_maps("ruotato")

    # ---- 1. convention self-test ---------------------------------------
    ref = reference_warp_08(frontal["Ca"])
    mine = fm.warp_frontal_to_tilted(frontal["Ca"])
    both = np.isfinite(ref) & np.isfinite(mine)
    dmax = float(np.abs(ref[both] - mine[both]).max())
    emit("[1] warp convention self-test (affine_transform vs script-08 "
         "map_coordinates)")
    emit(f"    finite pixels agree: {both.sum()}/{ref.size}, "
         f"max |diff| = {dmax:.3e}  -> "
         + ("PASS" if dmax < 1e-6 else "FAIL"))
    valid = fm.tilted_valid_mask()
    emit(f"    tilted-frame pixels seeing the frontal frame: "
         f"{valid.sum()}/{valid.size} ({100 * valid.mean():.1f}%)")
    emit("")

    # ---- 2. simulate ----------------------------------------------------
    sim_nf = fm.forward(frontal, add_noise=False)
    sim = fm.forward(frontal, rng=SEED, add_noise=True,
                     input_noise="measured")

    # noise-floor ceiling: prova1 vs prova2 on the ruotato overlap region
    # (frontal pixels covered by the tilted footprint, computed from the
    # same affine as script 08's overlap_mask_prova1.npy)
    ovl = fm.frontal_footprint_mask()

    # ---- 3. fidelity table ----------------------------------------------
    emit("[2] per-element fidelity on the valid footprint")
    emit(f"    {'line':6s} {'r_nf':>7s} {'r_noisy':>8s} {'r_floor':>8s} "
         f"{'med(sim/real)':>14s} {'gain':>7s} {'n_px':>6s}")
    r_nf, r_noisy, levels = {}, {}, {}
    for el in fm.ELEMENTS:
        m = valid & np.isfinite(sim_nf[el]) & np.isfinite(real[el])
        r_nf[el] = pearson(sim_nf[el][m], real[el][m])
        r_noisy[el] = pearson(sim[el][m], real[el][m])
        pos = m & (real[el] > 0)
        levels[el] = float(np.median(sim_nf[el][pos] / real[el][pos]))
        rfloor = pearson(frontal[el][ovl], frontal2[el][ovl])
        emit(f"    {el:6s} {r_nf[el]:7.4f} {r_noisy[el]:8.4f} "
             f"{rfloor:8.4f} {levels[el]:14.4f} {gains[el]:7.4f} "
             f"{int(m.sum()):6d}")
    emit("    r_nf/r_noisy: noise-free/noisy simulation vs real ruotato;")
    emit("    r_floor: prova1 vs prova2 (same geometry) on the overlap "
         "region = ceiling.")
    emit("")

    # ---- 4. noise check ---------------------------------------------------
    emit("[3] noise check  (model Var = k * counts, detector-summed)")
    emit(f"    {'line':6s} {'k_pair':>7s} {'k_spl_p1':>9s} {'k_spl_ruo':>10s} "
         f"{'sd_pred':>8s} {'sd_real':>8s} {'HF_real':>8s} {'HF_sim':>7s} "
         f"{'HF_simnf':>8s}")
    for el in fm.ELEMENTS:
        k_p1, _ = split_k("prova1", el)
        k_ruo, med_ruo = split_k("ruotato", el)
        sd_pred = np.sqrt(ks[el] * med_ruo)      # simulator noise @ median
        sd_real = np.sqrt(k_ruo * med_ruo)       # split estimate  @ median
        emit(f"    {el:6s} {ks[el]:7.2f} {k_p1:9.2f} {k_ruo:10.2f} "
             f"{sd_pred:8.1f} {sd_real:8.1f} {hf_sigma(real[el]):8.1f} "
             f"{hf_sigma(sim[el]):7.1f} {hf_sigma(sim_nf[el]):8.1f}")
    emit("    k_pair: prova1-prova2 calibration (used by the simulator);")
    emit("    k_spl_*: detector-split estimate (upper bound - ratio "
         "flat-field leaks in);")
    emit("    sd_pred/sd_real: noise std at the median ruotato signal, "
         "counts;")
    emit("    HF_*: Immerkaer high-frequency sigma (texture leaks in; "
         "compare between columns).")
    emit("")

    # ---- 5. round trip of the deterministic inverse ----------------------
    emit("[4] round trip: inverse(forward(prova1, noise-free)) vs prova1")
    rec = fm.inverse(sim_nf)
    fmask = fm.frontal_footprint_mask()
    emit(f"    frontal pixels inside the tilted footprint: "
         f"{fmask.sum()}/{fmask.size} ({100 * fmask.mean():.1f}%)")
    for el in fm.ELEMENTS:
        m = fmask & np.isfinite(rec[el])
        pos = m & (frontal[el] > 0)
        r = pearson(rec[el][m], frontal[el][m])
        lv = float(np.median(rec[el][pos] / frontal[el][pos]))
        emit(f"    {el:6s} r = {r:7.4f}   median(rec/orig) = {lv:.4f}")
    emit("    (two bilinear warps smooth the map; r < 1 here is "
         "interpolation, not bias)")
    emit("")

    # ---- 6. honest notes --------------------------------------------------
    emit("[5] honest notes")
    emit("  - NOT modelled: flat-field of the detector ratio (per-pixel gain")
    emit("    structure), per-pixel topography / local incidence angle (the")
    emit("    tilt gains are global per line), scatter/background structure,")
    emit("    intra-scan dwell drift.")
    emit("  - The warp is the measured 7.7 deg registration for every")
    emit("    angle_deg; only the gains scale with angle.")
    emit("  - Calibration circularity: the affine and the tilt gains were")
    emit("    fitted on this same prova1/ruotato pair (scripts 08/11), so")
    emit("    the LEVEL check validates that the pieces compose correctly,")
    emit("    not out-of-sample prediction; the r and noise checks are the")
    emit("    informative part of the gate.")
    emit("  - Noise k is calibrated on the frontal pair and assumed to")
    emit("    transfer to the ruotato session (levels agree to a few %).")
    emit("")

    # ---- 7. figure --------------------------------------------------------
    fig, axes = plt.subplots(len(FIG_ELEMENTS), 3,
                             figsize=(12.5, 2.9 * len(FIG_ELEMENTS)),
                             dpi=110, layout="constrained")
    for i, el in enumerate(FIG_ELEMENTS):
        re_m = np.where(valid, real[el], np.nan)
        si_m = np.where(valid, sim[el], np.nan)
        diff = si_m - re_m
        vmin = float(np.nanpercentile(np.stack([re_m, si_m]), 1))
        vmax = float(np.nanpercentile(np.stack([re_m, si_m]), 99))
        am = float(np.nanpercentile(np.abs(diff), 99)) or 1.0
        panels = [
            (re_m, f"{el} - real ruotato", "magma", vmin, vmax),
            (si_m, f"{el} - simulated (r={r_noisy[el]:.3f})",
             "magma", vmin, vmax),
            (diff, f"{el} - sim minus real", "RdBu_r", -am, am),
        ]
        for j, (img, title, cmap, lo, hi) in enumerate(panels):
            ax = axes[i, j]
            im = ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, aspect="equal")
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=7)
            if j == 0:
                ax.set_ylabel("net counts", fontsize=8)
    fig.suptitle(
        "Forward simulator fidelity - simulated tilted scan (from prova1) "
        "vs real ruotato\n"
        "warp: measured affine (script 08); gains: measured tilt response "
        "(script 11); noise: prova1-prova2 calibrated",
        fontsize=11, fontweight="bold")
    fig.savefig(PNG_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- 8. verdict and report -------------------------------------------
    n_r = sum(1 for el in fm.ELEMENTS if r_noisy[el] >= 0.8)
    n_lv = sum(1 for el in fm.ELEMENTS if abs(levels[el] - 1) <= 0.05)
    worst = min(fm.ELEMENTS, key=lambda e: r_noisy[e])
    emit("[6] gate summary")
    emit(f"    lines with r(sim, real) >= 0.8 : {n_r}/8")
    emit(f"    lines with |median(sim/real)-1| <= 5% : {n_lv}/8")
    emit(f"    worst line by r: {worst} (r = {r_noisy[worst]:.4f})")
    emit("")

    with open(TXT_PATH, "w") as fh:
        fh.write("\n".join(lines_out) + "\n")
    print(f"Saved: {PNG_PATH}")
    print(f"Saved: {TXT_PATH}")
