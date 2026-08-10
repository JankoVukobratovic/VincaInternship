"""
07b_tilt_angle.py
Estimate the canvas tilt angle of the ruotato scan from anisotropic
image scaling.

A forward tilt by theta compresses the scanned image vertically by
cos(theta) while leaving the horizontal scale unchanged. Registering
the tilted scan onto a frontal scan with independent x/y scales gives

    sy / sx = 1 / cos(theta)

and the unknown step sizes of the two scans cancel in the ratio.

Registration: detector-summed, log-compressed, z-scored element maps
(Ca, Ti, Fe, Cu, Pb); five-parameter warp (two scales, two shifts,
one rotation); masked NCC objective maximized jointly over the five
element pairs by differential evolution plus a Nelder-Mead polish.

Checks:
  * control prova1 <-> prova2 (same geometry) must give sy/sx = 1;
    its deviation from 1 is the method's precision floor, and doubles
    as a measured positioning/calibration drift between the two
    frontal sessions;
  * triangle: the ruotato -> prova1 and ruotato -> prova2 anisotropies
    must differ by the control anisotropy;
  * per-element refits from the joint optimum give the statistical
    spread;
  * the warped-overlay figure shows whether the alignment is genuine.

Input : results/detector_diff/_npy_cache/  (from script 06)
Output: results/detector_diff/tilt_angle.png
        results/detector_diff/tilt_angle.txt

Run from the project root:
    python scripts/07b_tilt_angle.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import differential_evolution, minimize

OUTPUT_DIR = os.path.join("results", "detector_diff")
CACHE      = os.path.join(OUTPUT_DIR, "_npy_cache")

PB_LINES = ["PbLl", "PbLa", "PbLb", "PbLg"]
ELS      = ["Ca", "Ti", "Fe", "Cu", "Pb"]
SEED     = 42


def load_map(scan, el):
    """Detector-summed, log-compressed, z-scored element map."""
    names = PB_LINES if el == "Pb" else [el]
    m = None
    for det in ("10264", "19511"):
        for name in names:
            a = np.load(os.path.join(CACHE, f"{scan}_{det}_{name}.npy"))
            m = a if m is None else m + a
    m = np.log1p(np.maximum(m, 0))
    return (m - m.mean()) / (m.std() + 1e-12)


# params = (sy, sx, ty, tx, rot_deg); frontal pixel (x,y) samples the
# tilted map at rot/scale/shifted coordinates
def warp_sample(tilted, grid_y, grid_x, params):
    sy, sx, ty, tx, rot = params
    r  = np.radians(rot)
    dx = grid_x - tx
    dy = grid_y - ty
    u  = ( np.cos(r) * dx + np.sin(r) * dy) / sx
    v  = (-np.sin(r) * dx + np.cos(r) * dy) / sy
    valid = ((v >= 0) & (v <= tilted.shape[0] - 1)
             & (u >= 0) & (u <= tilted.shape[1] - 1))
    return v, u, valid


def ncc(pairs, grid_y, grid_x, params, min_overlap):
    """Mean masked NCC over (frontal, tilted) map pairs."""
    total = 0.0
    for frontal, tilted in pairs:
        v, u, valid = warp_sample(tilted, grid_y, grid_x, params)
        n = int(valid.sum())
        if n < min_overlap:
            total += -1.0 + 0.5 * n / min_overlap   # graded penalty
            continue
        w = map_coordinates(tilted, [v[valid], u[valid]], order=1)
        fv = frontal[valid]
        fv = fv - fv.mean()
        w  = w - w.mean()
        denom = np.sqrt((fv * fv).sum() * (w * w).sum())
        total += (fv * w).sum() / denom if denom > 0 else -1.0
    return total / len(pairs)


def register(pairs, bounds, min_overlap):
    shape = pairs[0][0].shape
    gy, gx = np.mgrid[0:shape[0], 0:shape[1]].astype(float)

    def cost(p):
        return -ncc(pairs, gy, gx, p, min_overlap)

    de = differential_evolution(
        cost, bounds, seed=SEED, maxiter=250, popsize=24,
        tol=1e-8, polish=True,
    )
    nm = minimize(cost, de.x, method="Nelder-Mead",
                  options={"xatol": 1e-5, "fatol": 1e-9, "maxiter": 4000})
    best = nm if nm.fun < de.fun else de
    return best.x, -best.fun


def report(tag, params, score, lines):
    sy, sx, ty, tx, rot = params
    q = sx / sy
    theta = np.degrees(np.arccos(np.clip(q, 0.0, 1.0)))
    line = (f"{tag:10}  sy={sy:7.4f} sx={sx:7.4f}  sy/sx={sy/sx:7.4f}  "
            f"ty={ty:+6.2f} tx={tx:+6.2f} rot={rot:+5.2f}  "
            f"ncc={score:6.4f}  ->  theta={theta:5.2f} deg")
    print(line)
    lines.append(line)
    return theta, sy / sx


if __name__ == "__main__":
    frontal = {el: load_map("prova1", el) for el in ELS}
    fron2   = {el: load_map("prova2", el) for el in ELS}
    tilted  = {el: load_map("ruotato", el) for el in ELS}

    min_ov = int(0.5 * tilted["Pb"].size)
    lines  = []

    def say(text=""):
        print(text)
        lines.append(text)

    # ---- control: prova1 <-> prova2, must give sy/sx = 1 --------------
    ctrl_bounds = [(0.9, 1.1), (0.9, 1.1), (-8, 8), (-8, 8), (-3, 3)]
    ctrl_pairs  = [(frontal[el], fron2[el]) for el in ELS]
    p_ctrl, s_ctrl = register(ctrl_pairs, ctrl_bounds,
                              int(0.5 * frontal["Pb"].size))
    say("control (prova1 vs prova2, same geometry):")
    report("control", p_ctrl, s_ctrl, lines)

    # ---- joint registration ruotato -> prova1 -------------------------
    bounds = [(0.7, 1.9), (0.7, 1.9), (-15, 45), (-15, 105), (-5, 5)]
    pairs  = [(frontal[el], tilted[el]) for el in ELS]
    p_joint, s_joint = register(pairs, bounds, min_ov)
    say()
    say("joint (all elements), ruotato -> prova1:")
    report("joint", p_joint, s_joint, lines)

    # ---- triangle check: ruotato -> prova2 ----------------------------
    pairs2 = [(fron2[el], tilted[el]) for el in ELS]
    p_joint2, s_joint2 = register(pairs2, bounds, min_ov)
    say()
    say("triangle check, ruotato -> prova2:")
    report("joint2", p_joint2, s_joint2, lines)

    # ---- per-element refits from the joint optimum --------------------
    say()
    say("per-element refits (ruotato -> prova1):")
    thetas = []
    shape = frontal["Pb"].shape
    gy, gx = np.mgrid[0:shape[0], 0:shape[1]].astype(float)
    for el in ELS:
        pr = [(frontal[el], tilted[el])]

        def cost(p, pr=pr):
            return -ncc(pr, gy, gx, p, min_ov)

        nm = minimize(cost, p_joint, method="Nelder-Mead",
                      options={"xatol": 1e-5, "fatol": 1e-9,
                               "maxiter": 4000})
        th, _ = report(el, nm.x, -nm.fun, lines)
        thetas.append(th)

    thetas     = np.array(thetas)
    theta_mean = thetas.mean()
    theta_std  = thetas.std(ddof=1)
    q_ctrl     = p_ctrl[1] / p_ctrl[0]
    drift      = abs(1.0 / q_ctrl - 1.0)
    # a scale-ratio error dq maps to dtheta = dq / tan(theta)
    sys_deg    = np.degrees(drift / np.tan(np.radians(theta_mean)))

    say()
    say(f"control sy/sx deviation from 1: {100 * drift:.3f}%"
        "  (frontal session drift = precision floor)")
    say(f"tilt angle: {theta_mean:.1f}"
        f" +- {theta_std:.1f} (element spread)"
        f" +- {sys_deg:.1f} (registration floor) deg")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    txt_path = os.path.join(OUTPUT_DIR, "tilt_angle.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---- overlay figure: does the warp actually line up? --------------
    v, u, valid = warp_sample(tilted["Pb"], gy, gx, p_joint)
    warped = np.full(shape, np.nan)
    warped[valid] = map_coordinates(
        tilted["Pb"], [v[valid], u[valid]], order=1)

    fig, axes = plt.subplots(3, 1, figsize=(9, 9))
    axes[0].imshow(frontal["Pb"], cmap="magma")
    axes[0].set_title("prova1 Pb (frontal)")
    axes[1].imshow(tilted["Pb"], cmap="magma")
    axes[1].set_title("ruotato Pb (tilted, native 80x45)")
    axes[2].imshow(warped, cmap="magma")
    axes[2].set_title(
        f"ruotato Pb warped onto prova1 frame (best fit,"
        f" {theta_mean:.1f}\N{DEGREE SIGN} tilt)")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "tilt_angle.png")
    fig.savefig(fig_path, dpi=150)

    print(f"\nSaved: {txt_path}")
    print(f"Saved: {fig_path}")
