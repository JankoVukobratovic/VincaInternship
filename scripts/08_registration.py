"""
08_registration.py
Affine registration of the tilted scan (ruotato) onto the frontal scan
(prova1), generalizing the single-element approach of compare_Ti.py to a
multi-element objective.

Produces the vertical-foreshortening factor f = s_x / s_y of the
ruotato -> prova1 mapping (handoff 1 for Person A: with equal motor
pitch in both scans, f = cos(tilt angle)), plus the overlap-region
efficiency ratios of PLAN.md amendment 8.6 (frontal R recomputed only
on the canvas region the ruotato scan actually covers, which removes
the "different framing" objection).

Control: prova2 is registered onto prova1 with the same machinery; its
deviation from the identity transform is the pipeline noise floor.

Uses the npy cache built by 05/06. Outputs:

    results/registration/affine_params.csv
    results/registration/handoff1_foreshortening.md      <-- handoff 1
    results/registration/overlap_mask_prova1.npy
    results/registration/overlap_ratios.csv
    results/registration/registration_check.png

Run from the project root:
    python scripts/08_registration.py
"""

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

DETECTORS  = ["10264", "19511"]
CACHE_DIR  = os.path.join("results", "detector_diff", "_npy_cache")
OUTPUT_DIR = os.path.join("results", "registration")

# scan label -> (width, height)
GRIDS = {"prova1": (120, 60), "prova2": (120, 60), "ruotato": (80, 45)}

# registration features: detector-summed maps; Pb = sum of the 4 L lines
REG_ELEMENTS = ["Ca", "Ti", "Fe", "Cu", "Pb"]
PB_LINES     = ["PbLl", "PbLa", "PbLb", "PbLg"]

# ratio table elements (same set as 06)
ALL_ELEMENTS = {
    "K": 3.3138, "Ca": 3.69, "Ti": 4.51, "Fe": 6.40, "Cu": 8.04,
    "Zn": 8.64, "PbLl": 9.185, "PbLa": 10.54, "PbLb": 12.61, "PbLg": 14.77,
}
UNRELIABLE = {"K", "Zn"}

N_BOOT = 2000
SEED   = 42
MIN_VALID_PX = 300


def load_map(scan: str, det: str, el: str) -> np.ndarray:
    p = os.path.join(CACHE_DIR, f"{scan}_{det}_{el}.npy")
    if not os.path.exists(p):
        sys.exit(f"ERROR: cache missing: {p} — run scripts/06_efficiency_ratios.py first.")
    return np.load(p)


def feature_map(scan: str, el: str) -> np.ndarray:
    """Detector-summed, asinh-compressed map for registration."""
    if el == "Pb":
        m = sum(load_map(scan, det, l) for det in DETECTORS for l in PB_LINES)
    else:
        m = sum(load_map(scan, det, el) for det in DETECTORS)
    scale = np.median(m[m > 0]) if (m > 0).any() else 1.0
    return np.arcsinh(m / max(scale, 1e-9))


def affine_matrix(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """params [sx, sy, rot_deg, shear, tx, ty] -> (2x2 matrix A, offset t).

    Maps centered ruotato (x, y) to centered prova1 (x, y):
        v1 = A @ vr + t
    """
    sx, sy, rot, shear, tx, ty = p
    th = np.deg2rad(rot)
    R  = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    S  = np.array([[sx, shear], [0.0, sy]])
    return R @ S, np.array([tx, ty])


def warp_reference(ref: np.ndarray, p: np.ndarray,
                   src_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Sample the reference (prova1-frame) map at the transformed positions
    of every source (ruotato-frame) pixel. Returns (warped, valid_mask)."""
    hs, ws = src_shape
    hr, wr = ref.shape
    A, t = affine_matrix(p)

    xs, ys = np.meshgrid(np.arange(ws), np.arange(hs))
    v = np.stack([xs.ravel() - (ws - 1) / 2, ys.ravel() - (hs - 1) / 2])
    x1, y1 = A @ v
    x1 = x1 + (wr - 1) / 2 + t[0]
    y1 = y1 + (hr - 1) / 2 + t[1]

    valid = (x1 >= 0) & (x1 <= wr - 1) & (y1 >= 0) & (y1 <= hr - 1)
    warped = map_coordinates(ref, [y1, x1], order=1, mode="constant", cval=np.nan)
    return warped.reshape(hs, ws), valid.reshape(hs, ws)


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def objective(p: np.ndarray, refs: list[np.ndarray], srcs: list[np.ndarray]) -> float:
    """Negative mean NCC across feature maps."""
    total = 0.0
    for ref, src in zip(refs, srcs):
        warped, valid = warp_reference(ref, p, src.shape)
        m = valid & np.isfinite(warped)
        if m.sum() < MIN_VALID_PX:
            return 1.0
        total += ncc(warped[m], src[m])
    return -total / len(refs)


def register(refs: list[np.ndarray], srcs: list[np.ndarray],
             p0: np.ndarray | None = None, coarse: bool = True) -> np.ndarray:
    """Coarse grid search over scale/translation, then Powell refinement."""
    if p0 is None:
        best, best_val = None, np.inf
        scales_x = [1.0, 1.25, 1.5] if coarse else [1.0]
        scales_y = [1.0, 1.17, 1.33, 1.5] if coarse else [1.0]
        for sx in scales_x:
            for sy in scales_y:
                for tx in range(-16, 17, 4):
                    for ty in range(-10, 11, 4):
                        p = np.array([sx, sy, 0.0, 0.0, float(tx), float(ty)])
                        val = objective(p, refs, srcs)
                        if val < best_val:
                            best, best_val = p, val
        p0 = best
    res = minimize(
        objective, p0, args=(refs, srcs), method="Powell",
        bounds=[(0.5, 2.0), (0.5, 2.0), (-10, 10), (-0.3, 0.3), (-40, 40), (-25, 25)],
        options={"xtol": 1e-4, "ftol": 1e-6, "maxiter": 4000},
    )
    return res.x


def footprint_mask(p: np.ndarray, src_shape: tuple[int, int],
                   ref_shape: tuple[int, int]) -> np.ndarray:
    """Boolean mask on the reference grid: pixels inside the transformed
    source-frame rectangle."""
    hs, ws = src_shape
    hr, wr = ref_shape
    A, t = affine_matrix(p)
    corners = np.array([[0, 0], [ws - 1, 0], [ws - 1, hs - 1], [0, hs - 1]], float)
    cc = (A @ (corners - [(ws - 1) / 2, (hs - 1) / 2]).T).T \
        + [(wr - 1) / 2 + t[0], (hr - 1) / 2 + t[1]]
    poly = MplPath(cc)
    xs, ys = np.meshgrid(np.arange(wr), np.arange(hr))
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1)
    return poly.contains_points(pts).reshape(hr, wr)


def ratio_with_ci(a: np.ndarray, b: np.ndarray, rng) -> tuple[float, float]:
    """Global-sum ratio with paired pixel-bootstrap SE (same as 06)."""
    if b.sum() <= 0:
        return np.nan, np.nan
    r = a.sum() / b.sum()
    idx  = rng.integers(0, a.size, size=(N_BOOT, a.size))
    rb   = b[idx].sum(axis=1)
    boot = a[idx].sum(axis=1) / np.where(rb > 0, rb, np.nan)
    return r, float(np.nanstd(boot))


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    feats = {
        scan: [feature_map(scan, el) for el in REG_ELEMENTS]
        for scan in GRIDS
    }

    # ── 1. joint registration: ruotato -> prova1 ────────────────
    print("Registering ruotato -> prova1 (joint, 5 elements)...")
    p_ruo = register(feats["prova1"], feats["ruotato"])
    print(f"  params [sx, sy, rot, shear, tx, ty] = {np.round(p_ruo, 4)}")
    print(f"  mean NCC = {-objective(p_ruo, feats['prova1'], feats['ruotato']):.4f}")

    # per-element refits (warm start) -> spread = uncertainty
    per_el = {}
    for el, ref, src in zip(REG_ELEMENTS, feats["prova1"], feats["ruotato"]):
        pe = register([ref], [src], p0=p_ruo.copy())
        per_el[el] = pe
        print(f"    {el:3s}: sx={pe[0]:.4f}  sy={pe[1]:.4f}  f=sx/sy={pe[0]/pe[1]:.4f}")

    # ── 2. control: prova2 -> prova1 (expect identity) ──────────
    print("Registering prova2 -> prova1 (control)...")
    p_ctl = register(feats["prova1"], feats["prova2"],
                     p0=np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]))
    print(f"  params = {np.round(p_ctl, 4)}")

    # ── 3. foreshortening factor and implied angle ──────────────
    f_joint = p_ruo[0] / p_ruo[1]
    f_els   = np.array([per_el[el][0] / per_el[el][1] for el in REG_ELEMENTS])
    f_sigma = float(f_els.std(ddof=1))
    f_ctl   = p_ctl[0] / p_ctl[1]

    def implied_angle(f: float) -> float:
        return float(np.degrees(np.arccos(min(f, 1.0))))

    alpha       = implied_angle(f_joint)
    alpha_lo    = implied_angle(min(f_joint + f_sigma, 1.0))
    alpha_hi    = implied_angle(f_joint - f_sigma)

    # ── 4. overlap-region efficiency ratios (amendment 8.6) ─────
    print("Computing overlap-region efficiency ratios...")
    mask = footprint_mask(p_ruo, feats["ruotato"][0].shape, feats["prova1"][0].shape)
    np.save(os.path.join(OUTPUT_DIR, "overlap_mask_prova1.npy"), mask)
    print(f"  overlap covers {mask.mean() * 100:.1f}% of the frontal frame")

    rows = []
    for el, kev in ALL_ELEMENTS.items():
        r = {}
        for scan in ("prova1", "prova2"):
            d1 = load_map(scan, "10264", el)
            d2 = load_map(scan, "19511", el)
            r[f"{scan}_full"]    = ratio_with_ci(d1.ravel(), d2.ravel(), rng)
            r[f"{scan}_overlap"] = ratio_with_ci(d1[mask], d2[mask], rng)
        d1 = load_map("ruotato", "10264", el)
        d2 = load_map("ruotato", "19511", el)
        r["ruotato"] = ratio_with_ci(d1.ravel(), d2.ravel(), rng)

        rf_full = 0.5 * (r["prova1_full"][0] + r["prova2_full"][0])
        rf_ovl  = 0.5 * (r["prova1_overlap"][0] + r["prova2_overlap"][0])
        rr      = r["ruotato"][0]
        (r1o, s1o), (r2o, s2o) = r["prova1_overlap"], r["prova2_overlap"]
        sr        = r["ruotato"][1]
        sigma_ovl = np.sqrt(sr**2 + 0.25 * (s1o**2 + s2o**2))
        rows.append({
            "element": el, "kev": kev,
            "R_frontal_full":    rf_full,
            "R_frontal_overlap": rf_ovl,
            "sig_frontal_overlap": 0.5 * np.hypot(s1o, s2o),
            "R_ruotato": rr, "sig_ruotato": sr,
            "tilt_full_pct":    (rr - rf_full) / rf_full * 100,
            "tilt_overlap_pct": (rr - rf_ovl) / rf_ovl * 100,
            "baseline_overlap_pct": abs(r1o - r2o) / rf_ovl * 100,
            "significance_overlap_sigma":
                abs(rr - rf_ovl) / sigma_ovl if sigma_ovl > 0 else np.nan,
            "reliable": el not in UNRELIABLE,
        })

    csv_path = os.path.join(OUTPUT_DIR, "overlap_ratios.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    hdr = (f"{'el':6}{'keV':>6}  {'R_full':>8}{'R_ovl':>8}{'R_ruo':>8}"
           f"{'tilt_full%':>11}{'tilt_ovl%':>10}{'base_ovl%':>10}{'sig':>6}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for w_ in rows:
        flag = "" if w_["reliable"] else "  (unreliable)"
        print(f"{w_['element']:6}{w_['kev']:6.2f}  "
              f"{w_['R_frontal_full']:8.4f}{w_['R_frontal_overlap']:8.4f}"
              f"{w_['R_ruotato']:8.4f}{w_['tilt_full_pct']:+11.2f}"
              f"{w_['tilt_overlap_pct']:+10.2f}"
              f"{w_['baseline_overlap_pct']:10.2f}"
              f"{w_['significance_overlap_sigma']:6.1f}{flag}")

    # ── 5. affine parameter CSV ─────────────────────────────────
    par_path = os.path.join(OUTPUT_DIR, "affine_params.csv")
    with open(par_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fit", "sx", "sy", "rot_deg", "shear", "tx", "ty", "f_sx_over_sy"])
        w.writerow(["ruotato_joint", *np.round(p_ruo, 6), round(f_joint, 6)])
        for el in REG_ELEMENTS:
            pe = per_el[el]
            w.writerow([f"ruotato_{el}", *np.round(pe, 6), round(pe[0] / pe[1], 6)])
        w.writerow(["prova2_control", *np.round(p_ctl, 6), round(f_ctl, 6)])

    # ── 6. handoff 1 ────────────────────────────────────────────
    handoff = os.path.join(OUTPUT_DIR, "handoff1_foreshortening.md")
    with open(handoff, "w") as fh:
        fh.write(f"""# Handoff 1 (B -> A): vertical foreshortening from registration

**f = s_x / s_y (ruotato -> prova1) = {f_joint:.4f} +/- {f_sigma:.4f}**

- per-element spread ({', '.join(REG_ELEMENTS)}): {np.array2string(np.round(f_els, 4), separator=', ')}
- control (prova2 -> prova1, expected 1.0): f = {f_ctl:.4f} -> pipeline
  noise floor |f-1| = {abs(f_ctl - 1):.4f}
- joint affine [sx, sy, rot_deg, shear, tx, ty] = {np.array2string(np.round(p_ruo, 4), separator=', ')}

## Implied tilt angle

Assuming equal motor pitch (mm/px) in both scans and pure forward tilt,
f = cos(alpha):

**alpha = {alpha:.1f} deg  (range {alpha_lo:.1f}-{alpha_hi:.1f} deg from the element spread)**

## Caveats

- If the ruotato scan used a different step size, the *isotropic* part
  of the scale is absorbed by (sx, sy) jointly and f is unaffected; an
  *anisotropic* pitch difference would contaminate f directly. Cross-check
  against Ridolfi's number (PLAN §3.1.2) before using alpha in the fit.
- The registration measures |cos| only; it cannot distinguish tilt
  forward from backward.
- Small angles are at the edge of resolution: the control run puts the
  noise floor at |f-1| ~ {abs(f_ctl - 1):.3f}, i.e. angles below
  ~{implied_angle(1 - abs(f_ctl - 1)):.0f} deg are not distinguishable from zero
  by foreshortening alone.
""")

    # ── 7. QA figure ────────────────────────────────────────────
    fig, axes = plt.subplots(len(REG_ELEMENTS), 3,
                             figsize=(12, 2.6 * len(REG_ELEMENTS)),
                             dpi=110, layout="constrained")
    for i, el in enumerate(REG_ELEMENTS):
        src = feats["ruotato"][i]
        warped, valid = warp_reference(feats["prova1"][i], p_ruo, src.shape)
        resid = np.where(valid, src - warped, np.nan)
        for j, (img, title) in enumerate([
            (src, f"{el} — ruotato"),
            (np.where(valid, warped, np.nan), f"{el} — prova1 warped"),
            (resid, f"{el} — residual"),
        ]):
            ax = axes[i, j]
            if j < 2:
                im = ax.imshow(img, cmap="magma", aspect="equal")
            else:
                am = np.nanpercentile(np.abs(img), 99) or 1.0
                im = ax.imshow(img, cmap="RdBu_r", vmin=-am, vmax=am, aspect="equal")
            ax.set_title(title, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.03)
    fig.suptitle(
        f"Registration QA — ruotato vs warped prova1   "
        f"(f = {f_joint:.3f} ± {f_sigma:.3f}, implied tilt {alpha:.1f}°)",
        fontweight="bold")
    qa_path = os.path.join(OUTPUT_DIR, "registration_check.png")
    fig.savefig(qa_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"\nf = sx/sy = {f_joint:.4f} +- {f_sigma:.4f}   "
          f"(control: {f_ctl:.4f})   implied tilt = {alpha:.1f}°")
    for path in (par_path, csv_path, handoff, qa_path):
        print(f"Saved: {path}")
