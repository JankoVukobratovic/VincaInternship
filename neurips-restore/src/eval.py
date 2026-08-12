"""
eval.py — metric harness for the neurips-restore MVP (item 2).

Import-friendly functions to score a restored frontal-frame element map
against the frontal truth on the footprint actually covered by the
warped tilted scan, plus the prova1<->prova2 noise floor that bounds
what any restoration can reach.

Metrics per element line (all evaluated on a boolean footprint mask):

    pearson r      linear correlation of the masked pixels
    ssim           local-window SSIM (uniform 7x7 window), averaged over
                   windows fully inside the mask
    bias_pct       absolute level bias in percent:
                   (median(restored / truth) - 1) * 100
    cv_ratio       contrast guard: CV(restored) / CV(truth) with
                   CV = std / mean on the mask  (<1 = blurring /
                   contrast loss, >1 = contrast inflation)

Also provides the affine machinery to warp the tilted ("ruotato") scan
back into the frontal frame. Convention follows scripts/08_registration.py:
the fitted parameters [sx, sy, rot_deg, shear, tx, ty] map CENTERED
ruotato pixel coordinates to CENTERED frontal (prova1) coordinates,

    v_frontal = A @ v_ruotato + t,   A = R(rot) @ [[sx, shear], [0, sy]].

Restoration goes the other way: for every frontal pixel we invert the
affine to find its source location in the ruotato grid and sample there
(bilinear). Pixels whose source falls outside the ruotato grid form the
complement of the footprint mask.

Data: detector-summed per-element maps from
results/detector_diff/_npy_cache/{scan}_{det}_{el}.npy.

All paths are anchored to the repo root via __file__, so the module
works regardless of the caller's cwd.
"""

import csv
import os

import numpy as np
from scipy.ndimage import map_coordinates, uniform_filter

# --------------------------------------------------------------------------
# paths and constants
# --------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(REPO_ROOT, "results", "detector_diff", "_npy_cache")
REG_DIR   = os.path.join(REPO_ROOT, "results", "registration")

AFFINE_CSV      = os.path.join(REG_DIR, "affine_params.csv")
SENSITIVITY_CSV = os.path.join(REG_DIR, "positioning_sensitivity.csv")

DETECTORS = ("10264", "19511")
LINES = ("Ca", "Ti", "Fe", "Cu", "PbLl", "PbLa", "PbLb", "PbLg")

# scan label -> (height, width) of the stored maps
SHAPES = {"prova1": (60, 120), "prova2": (60, 120), "ruotato": (45, 80)}

# nominal mounting angle of the tilted scan (deg); the %/deg tilt gains
# of script 11 were measured over exactly this angle, so gain(el) =
# 1 + per_deg_sum(el) * NOMINAL_TILT_DEG / 100 reproduces the measured
# common-mode-free level change of the tilted maps.
NOMINAL_TILT_DEG = 7.7


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------

def load_map(scan: str, det: str, el: str) -> np.ndarray:
    """One per-element map from the npy cache (CPS units, (h, w))."""
    path = os.path.join(CACHE_DIR, f"{scan}_{det}_{el}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"cache missing: {path} - run scripts/06_efficiency_ratios.py first")
    return np.load(path)


def detsum_map(scan: str, el: str) -> np.ndarray:
    """Detector-summed map (the MVP working representation)."""
    return sum(load_map(scan, det, el) for det in DETECTORS)


def read_affine(fit: str = "ruotato_joint") -> np.ndarray:
    """[sx, sy, rot_deg, shear, tx, ty] of one fit from affine_params.csv."""
    with open(AFFINE_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["fit"] == fit:
                return np.array([float(row[k]) for k in
                                 ("sx", "sy", "rot_deg", "shear", "tx", "ty")])
    raise KeyError(f"fit '{fit}' not found in {AFFINE_CSV}")


def read_tilt_gains(tilt_deg: float = NOMINAL_TILT_DEG,
                    variant: str = "sum") -> dict:
    """Per-element multiplicative tilt gain at the given angle.

    positioning_sensitivity.csv (script 11) measured
    delta(el) = ruotato / frontal - 1 on registered pixels; per_deg_*
    is that change per degree AFTER removing the common mode (the
    element-independent session/level drift, which is not a geometry
    effect). The forward tilt therefore multiplies a frontal map by

        gain(el) = 1 + per_deg(el) * tilt_deg / 100

    and restoration divides it out.
    """
    gains = {}
    with open(SENSITIVITY_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            gains[row["element"]] = 1.0 + float(row[f"per_deg_{variant}"]) * tilt_deg / 100.0
    return gains


# --------------------------------------------------------------------------
# affine warp: ruotato frame -> frontal frame
# --------------------------------------------------------------------------

def affine_matrix(p: np.ndarray) -> tuple:
    """params [sx, sy, rot_deg, shear, tx, ty] -> (A, t), centered coords.

    Same convention as scripts/08_registration.py: v_frontal = A @ v_ruotato + t.
    """
    sx, sy, rot, shear, tx, ty = p
    th = np.deg2rad(rot)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    S = np.array([[sx, shear], [0.0, sy]])
    return R @ S, np.array([tx, ty])


def warp_to_frontal(src: np.ndarray, p: np.ndarray,
                    ref_shape: tuple = SHAPES["prova1"]) -> tuple:
    """Warp a ruotato-frame map into the frontal frame (inverse affine).

    For each frontal pixel (x1, y1) the source location in the ruotato
    grid is  v_r = A^-1 @ (v_1 - t)  (centered coordinates); the map is
    sampled there bilinearly. Returns (restored, footprint):
      restored  (h_ref, w_ref) float, NaN outside the footprint
      footprint (h_ref, w_ref) bool, True where the source location lies
                inside the ruotato grid
    """
    hs, ws = src.shape
    hr, wr = ref_shape
    A, t = affine_matrix(p)
    Ainv = np.linalg.inv(A)

    x1, y1 = np.meshgrid(np.arange(wr, dtype=float), np.arange(hr, dtype=float))
    v1 = np.stack([x1.ravel() - (wr - 1) / 2 - t[0],
                   y1.ravel() - (hr - 1) / 2 - t[1]])
    xr, yr = Ainv @ v1
    xr = xr + (ws - 1) / 2
    yr = yr + (hs - 1) / 2

    footprint = (xr >= 0) & (xr <= ws - 1) & (yr >= 0) & (yr <= hs - 1)
    sampled = map_coordinates(src, [yr, xr], order=1, mode="constant", cval=np.nan)
    restored = np.where(footprint, sampled, np.nan).reshape(hr, wr)
    return restored, footprint.reshape(hr, wr)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def _masked_pair(restored: np.ndarray, truth: np.ndarray,
                 mask: np.ndarray) -> tuple:
    m = mask & np.isfinite(restored) & np.isfinite(truth)
    return restored[m], truth[m], m


def pearson_r(restored: np.ndarray, truth: np.ndarray,
              mask: np.ndarray) -> float:
    a, b, _ = _masked_pair(restored, truth, mask)
    if a.size < 2:
        return np.nan
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else np.nan


def ssim(restored: np.ndarray, truth: np.ndarray, mask: np.ndarray,
         win: int = 7, data_range: float = None) -> float:
    """Mean local-window SSIM over windows fully inside the mask.

    Uniform (box) window of size win x win, standard SSIM constants
    C1 = (0.01 L)^2, C2 = (0.03 L)^2 with L = data_range (defaults to
    max - min of the truth on the mask). Pixels outside the mask are
    zero-filled but never contribute: only window centers whose full
    win x win neighbourhood lies inside the mask (and inside the image)
    enter the average.
    """
    m = mask & np.isfinite(restored) & np.isfinite(truth)
    if data_range is None:
        tv = truth[m]
        data_range = float(tv.max() - tv.min()) if tv.size else 1.0
    if data_range <= 0:
        return np.nan
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    a = np.where(m, restored, 0.0).astype(float)
    b = np.where(m, truth, 0.0).astype(float)

    mu_a = uniform_filter(a, win)
    mu_b = uniform_filter(b, win)
    var_a = uniform_filter(a * a, win) - mu_a ** 2
    var_b = uniform_filter(b * b, win) - mu_b ** 2
    cov = uniform_filter(a * b, win) - mu_a * mu_b

    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / \
        ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))

    # windows fully inside the mask ...
    interior = uniform_filter(m.astype(float), win) > 1.0 - 1e-6
    # ... and fully inside the image (uniform_filter reflects at edges)
    r = win // 2
    edge = np.zeros_like(interior)
    edge[r:interior.shape[0] - r, r:interior.shape[1] - r] = True
    interior &= edge

    return float(s[interior].mean()) if interior.any() else np.nan


def bias_pct(restored: np.ndarray, truth: np.ndarray,
             mask: np.ndarray) -> float:
    """Level bias in percent: (median(restored/truth) - 1) * 100."""
    a, b, _ = _masked_pair(restored, truth, mask)
    ok = b > 0
    if not ok.any():
        return np.nan
    return float((np.median(a[ok] / b[ok]) - 1.0) * 100.0)


def cv_ratio(restored: np.ndarray, truth: np.ndarray,
             mask: np.ndarray) -> float:
    """Contrast guard: CV(restored) / CV(truth), CV = std/mean on mask."""
    a, b, _ = _masked_pair(restored, truth, mask)
    if a.size < 2 or a.mean() == 0 or b.mean() == 0:
        return np.nan
    cva = a.std() / a.mean()
    cvb = b.std() / b.mean()
    return float(cva / cvb) if cvb > 0 else np.nan


def score_pair(restored: np.ndarray, truth: np.ndarray, mask: np.ndarray,
               win: int = 7, data_range: float = None) -> dict:
    """All four metrics for one (restored, truth) pair on one footprint."""
    return {
        "r":        pearson_r(restored, truth, mask),
        "ssim":     ssim(restored, truth, mask, win=win, data_range=data_range),
        "bias_pct": bias_pct(restored, truth, mask),
        "cv_ratio": cv_ratio(restored, truth, mask),
        "n_px":     int((mask & np.isfinite(restored) & np.isfinite(truth)).sum()),
    }


def noise_floor(el: str, mask: np.ndarray, win: int = 7,
                data_range: float = None) -> dict:
    """prova1 vs prova2 on the same footprint: the repeatability ceiling.

    Scores prova2 as 'restored' against prova1 as 'truth' (detector-summed
    maps, raw grids - the control registration of script 08 puts their
    misalignment at ~0.15 px, so no resampling is applied). No restoration
    of the tilted scan can beat two frontal scans of the same geometry.
    """
    p1 = detsum_map("prova1", el)
    p2 = detsum_map("prova2", el)
    return score_pair(p2, p1, mask, win=win, data_range=data_range)
