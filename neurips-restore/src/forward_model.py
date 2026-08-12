"""
forward_model.py
================
Measured-physics forward simulator for the NeurIPS restoration project:
frontal per-element MA-XRF maps (60 rows x 120 cols, scan "prova1"
convention) -> simulated tilted measurement (45 rows x 80 cols, scan
"ruotato" convention), plus the deterministic inverse operator.

Every ingredient is MEASURED, not modelled:

1. Geometry -- the affine registration of script 08
   (results/registration/affine_params.csv, row "ruotato_joint").
   Convention of the stored parameters p = [sx, sy, rot_deg, shear,
   tx, ty]: with A = R(rot) @ [[sx, shear], [0, sy]] they map CENTERED
   ruotato pixel coordinates (x, y) to CENTERED prova1 pixel
   coordinates,

       v_prova1 = A @ v_ruotato + [tx, ty]        (x = column, y = row)

   i.e. the stored transform answers "where in the frontal frame does
   this tilted pixel look?".  That is exactly the output->input mapping
   scipy.ndimage.affine_transform wants for the FORWARD simulation
   (output = tilted grid, input = frontal map); the inverse operator
   uses A^-1.  affine_transform works in (row, col) order, so the 2x2
   matrix is re-ordered accordingly (see _matrix_offset).
   Note: the fitted scales are ~1 (sx=0.9984, sy=0.9989); the vertical
   foreshortening expected from a 7.7 deg tilt (sy/sx = 1/cos ~ 1.009)
   is below the registration noise floor (|f-1| ~ 0.004, see
   handoff1_foreshortening.md), so the measured warp is essentially a
   rotation (+1.44 deg) plus a translation onto the sub-region of the
   frontal frame that the tilted scan covers.  The warp does NOT change
   with angle_deg here -- only the per-line gains do.

2. Per-element tilt gain -- the measured per-line tilt response of
   script 11 (results/registration/positioning_sensitivity.csv, column
   "tilt_pct_sum" = 100 * (sum ruotato / sum warped-prova1 - 1) on the
   common footprint, detector-summed maps).  Forward gain per element:

       g_el(angle) = 1 + (tilt_pct_sum/100) * (angle_deg / 7.7)

   tilt_pct_sum is the TOTAL measured level response at the reference
   7.7 deg mounting, which includes the -0.40% session common mode
   (degenerate with the tilt's solid-angle change, see script 11); the
   common-mode-removed slopes are in column "per_deg_sum" if a purely
   differential gain is ever needed.  The linear scaling in angle is an
   extrapolation; at the reference angle the gains reproduce the
   measurement exactly.

3. Noise -- calibrated from the measured prova1-prova2 difference
   (same geometry, 7 days apart; detector-summed maps = 10264 + 19511).

   NOISE MODEL.  The maps are NET counts (fitted line areas), so the
   noise is Poisson-consistent -- variance proportional to signal --
   but super-Poissonian, because peak fitting and background
   subtraction add variance:  Var(m) = k_el * m.  k_el is calibrated
   per element from the pair: with g = sum(m1)/sum(m2) removing the
   per-element session gain, z = (m1 - g*m2) / sqrt(2 * sbar),
   sbar = (m1 + g*m2)/2, on pixels with sbar > 0, then

       k_el = (1.4826 * MAD(z))^2        (robust; measured k ~ 4-9)

   Simulation: after warp and gain, zero-mean Gaussian noise with
   variance  k_el * s * max(1 - g_el * w_eff, 0)  is added, where
   w_eff is the per-pixel sum of squared bilinear interpolation
   weights ( ((1-fy)^2+fy^2) * ((1-fx)^2+fx^2), in [0.25, 1] ).  The
   factor (1 - g_el * w_eff) compensates for the noise the MEASURED
   input map already carries through the warp (carried variance
   ~ g_el * w_eff * k_el * s), so the total simulated variance matches
   the calibrated k_el * s of a real acquisition.  For clean/denoised
   inputs pass input_noise="none" to add the full k_el * s.  Gaussian
   rather than integer Poisson because net counts are continuous and
   the calibrated variance is ~4-9x the count level; the result is
   clipped at 0, matching the non-negative net-count maps the fits
   produce.

What the simulator does NOT model: the flat-field of the detector
ratio (per-pixel gain structure), per-pixel topography / local
incidence angle (gains are global per line), scatter background
structure, and intra-scan dwell drift.

Public API
----------
    forward(maps, angle_deg=7.7, rng=None, add_noise=True,
            input_noise="measured")            frontal dict -> tilted dict
    inverse(maps_tilted, angle_deg=7.7)        tilted dict -> frontal dict
                                               (NaN outside the footprint)
    tilt_gains(angle_deg=7.7)                  per-element gain dict
    calibrate_noise()                          per-element k dict
    load_summed_maps(scan)                     detector-summed maps from
                                               the npy cache
    warp_frontal_to_tilted(img) /              bare geometric warps
    warp_tilted_to_frontal(img)
    tilted_valid_mask(), frontal_footprint_mask()

`maps` are dicts {element: 2D float array}; elements from ELEMENTS.
All paths are resolved relative to the repository root, so this module
works from any working directory.
"""

import csv
import os

import numpy as np
from scipy.ndimage import affine_transform

# --------------------------------------------------------------------------
# constants and paths
# --------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

AFFINE_CSV = os.path.join(REPO_ROOT, "results", "registration",
                          "affine_params.csv")
SENS_CSV = os.path.join(REPO_ROOT, "results", "registration",
                        "positioning_sensitivity.csv")
CACHE_DIR = os.path.join(REPO_ROOT, "results", "detector_diff",
                         "_npy_cache")

FRONTAL_SHAPE = (60, 120)   # (rows, cols) -- prova1 / prova2
TILTED_SHAPE = (45, 80)     # (rows, cols) -- ruotato
REF_ANGLE_DEG = 7.7         # mounting angle the gains were measured at

ELEMENTS = ("Ca", "Ti", "Fe", "Cu", "PbLl", "PbLa", "PbLb", "PbLg")
DETECTORS = ("10264", "19511")

# module-level caches (all lazily filled)
_AFFINE_P = None
_TILT_PCT = None
_NOISE_K = {}


# --------------------------------------------------------------------------
# measured-parameter loading
# --------------------------------------------------------------------------

def load_affine_params(fit: str = "ruotato_joint") -> np.ndarray:
    """[sx, sy, rot_deg, shear, tx, ty] of the requested fit (script 08).

    Convention: A = R(rot) @ [[sx, shear], [0, sy]] maps centered
    ruotato (x, y) to centered prova1 (x, y): v1 = A @ vr + [tx, ty].
    """
    global _AFFINE_P
    if _AFFINE_P is not None and fit == "ruotato_joint":
        return _AFFINE_P
    with open(AFFINE_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["fit"] == fit:
                p = np.array([float(row[k]) for k in
                              ("sx", "sy", "rot_deg", "shear", "tx", "ty")])
                if fit == "ruotato_joint":
                    _AFFINE_P = p
                return p
    raise FileNotFoundError(
        f"fit '{fit}' not found in {AFFINE_CSV} - run scripts/08_registration.py")


def affine_ruotato_to_frontal(p: np.ndarray | None = None
                              ) -> tuple[np.ndarray, np.ndarray]:
    """(A, t) with v_prova1_centered = A @ v_ruotato_centered + t, (x, y)."""
    if p is None:
        p = load_affine_params()
    sx, sy, rot, shear, tx, ty = p
    th = np.deg2rad(rot)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    S = np.array([[sx, shear], [0.0, sy]])
    return R @ S, np.array([tx, ty])


def tilt_gains(angle_deg: float = REF_ANGLE_DEG,
               elements=ELEMENTS) -> dict:
    """Per-element multiplicative gain frontal -> tilted at angle_deg.

    1 + (tilt_pct_sum / 100) * (angle_deg / 7.7); tilt_pct_sum is the
    measured detector-summed level response at the 7.7 deg reference
    (script 11), session common mode included.
    """
    global _TILT_PCT
    if _TILT_PCT is None:
        _TILT_PCT = {}
        with open(SENS_CSV, newline="") as fh:
            for row in csv.DictReader(fh):
                _TILT_PCT[row["element"]] = float(row["tilt_pct_sum"])
    return {el: 1.0 + (_TILT_PCT[el] / 100.0) * (angle_deg / REF_ANGLE_DEG)
            for el in elements}


def load_summed_maps(scan: str, elements=ELEMENTS) -> dict:
    """Detector-summed per-element maps of a scan from the npy cache."""
    out = {}
    for el in elements:
        out[el] = sum(
            np.load(os.path.join(CACHE_DIR, f"{scan}_{det}_{el}.npy"))
            for det in DETECTORS)
    return out


def calibrate_noise(elements=ELEMENTS) -> dict:
    """Per-element k of the noise model Var(m) = k * m (see module doc).

    Calibrated from the prova1-prova2 pair (detector-summed maps) with
    the per-element session gain removed; robust MAD estimator.
    """
    missing = [el for el in elements if el not in _NOISE_K]
    for el in missing:
        m1 = load_summed_maps("prova1", [el])[el]
        m2 = load_summed_maps("prova2", [el])[el]
        g = m1.sum() / m2.sum()
        d = (m1 - g * m2).ravel()
        s = (0.5 * (m1 + g * m2)).ravel()
        ok = s > 0
        z = d[ok] / np.sqrt(2.0 * s[ok])
        mad = np.median(np.abs(z - np.median(z)))
        _NOISE_K[el] = float((1.4826 * mad) ** 2)
    return {el: _NOISE_K[el] for el in elements}


# --------------------------------------------------------------------------
# geometry: warps via scipy.ndimage.affine_transform
# --------------------------------------------------------------------------

def _centers():
    hr, wr = TILTED_SHAPE
    hf, wf = FRONTAL_SHAPE
    return ((wr - 1) / 2.0, (hr - 1) / 2.0,   # cxr, cyr (ruotato)
            (wf - 1) / 2.0, (hf - 1) / 2.0)   # cxf, cyf (frontal)


def _matrix_offset(direction: str) -> tuple[np.ndarray, np.ndarray]:
    """(matrix, offset) for scipy.ndimage.affine_transform.

    affine_transform computes  input_coord = matrix @ output_coord +
    offset  in (row, col) order.  "forward": output = tilted grid,
    input = frontal map (uses A directly).  "inverse": output = frontal
    grid, input = tilted map (uses A^-1).
    """
    A, t = affine_ruotato_to_frontal()
    cxr, cyr, cxf, cyf = _centers()
    if direction == "forward":
        # x_f = A00 (x_r - cxr) + A01 (y_r - cyr) + cxf + tx  (same for y_f)
        M = np.array([[A[1, 1], A[1, 0]],
                      [A[0, 1], A[0, 0]]])
        off = np.array([
            cyf + t[1] - A[1, 0] * cxr - A[1, 1] * cyr,
            cxf + t[0] - A[0, 0] * cxr - A[0, 1] * cyr])
        return M, off
    if direction == "inverse":
        B = np.linalg.inv(A)
        # v_r = B @ (v_f_centered - t) + center_r
        M = np.array([[B[1, 1], B[1, 0]],
                      [B[0, 1], B[0, 0]]])
        off = np.array([
            cyr - B[1, 0] * (cxf + t[0]) - B[1, 1] * (cyf + t[1]),
            cxr - B[0, 0] * (cxf + t[0]) - B[0, 1] * (cyf + t[1])])
        return M, off
    raise ValueError("direction must be 'forward' or 'inverse'")


def _warp_coords(direction: str) -> tuple[np.ndarray, np.ndarray]:
    """Float input coordinates (rows_in, cols_in) for every output pixel."""
    M, off = _matrix_offset(direction)
    h, w = TILTED_SHAPE if direction == "forward" else FRONTAL_SHAPE
    rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    rows_in = M[0, 0] * rr + M[0, 1] * cc + off[0]
    cols_in = M[1, 0] * rr + M[1, 1] * cc + off[1]
    return rows_in, cols_in


def _valid_mask(direction: str) -> np.ndarray:
    """Output pixels whose sampling position lies inside the input frame."""
    hin, win = FRONTAL_SHAPE if direction == "forward" else TILTED_SHAPE
    rows_in, cols_in = _warp_coords(direction)
    return ((rows_in >= 0) & (rows_in <= hin - 1)
            & (cols_in >= 0) & (cols_in <= win - 1))


def _weff(direction: str = "forward") -> np.ndarray:
    """Per-pixel sum of squared bilinear weights of the warp, in [0.25, 1].

    A warped noisy image carries variance ~ w_eff * Var_in per pixel.
    """
    rows_in, cols_in = _warp_coords(direction)
    fy = rows_in - np.floor(rows_in)
    fx = cols_in - np.floor(cols_in)
    return (((1 - fy) ** 2 + fy ** 2) * ((1 - fx) ** 2 + fx ** 2))


def _warp(img: np.ndarray, direction: str, cval: float = np.nan) -> np.ndarray:
    """Bilinear affine warp; out-of-footprint pixels become cval."""
    out_shape = TILTED_SHAPE if direction == "forward" else FRONTAL_SHAPE
    M, off = _matrix_offset(direction)
    return affine_transform(np.asarray(img, dtype=float), M, offset=off,
                            output_shape=out_shape, order=1,
                            mode="constant", cval=cval)


def warp_frontal_to_tilted(img: np.ndarray) -> np.ndarray:
    """Bare geometric warp frontal (60x120) -> tilted (45x80), no gain."""
    return _warp(img, "forward")


def warp_tilted_to_frontal(img: np.ndarray) -> np.ndarray:
    """Bare geometric warp tilted (45x80) -> frontal (60x120), NaN outside."""
    return _warp(img, "inverse")


def tilted_valid_mask() -> np.ndarray:
    """Tilted-grid pixels that see the frontal frame (should be all)."""
    return _valid_mask("forward")


def frontal_footprint_mask() -> np.ndarray:
    """Frontal-grid pixels covered by the tilted scan's footprint."""
    return _valid_mask("inverse")


# --------------------------------------------------------------------------
# public operators
# --------------------------------------------------------------------------

def forward(maps: dict, angle_deg: float = REF_ANGLE_DEG,
            rng=None, add_noise: bool = True,
            input_noise: str = "measured") -> dict:
    """Simulate the tilted measurement of frontal per-element maps.

    Parameters
    ----------
    maps : dict {element: (60, 120) array}, frontal net-count maps.
    angle_deg : mounting angle; scales the per-element gains linearly
        (the geometric warp is the measured 7.7 deg registration and is
        not re-scaled -- its foreshortening is below the measurement
        noise floor anyway, see module docstring).
    rng : np.random.Generator, int seed, or None (fresh entropy).
    add_noise : add the calibrated Poisson-consistent noise.
    input_noise : "measured" (default) if `maps` are real measured maps
        that already carry the calibrated noise level -- the added
        variance is then reduced by the warp-carried part; "none" if
        `maps` are clean/denoised -- the full k*s variance is added.

    Returns dict {element: (45, 80) array} of simulated tilted maps.
    """
    if input_noise not in ("measured", "none"):
        raise ValueError("input_noise must be 'measured' or 'none'")
    if rng is None or isinstance(rng, (int, np.integer)):
        rng = np.random.default_rng(rng)
    gains = tilt_gains(angle_deg, elements=list(maps))
    ks = calibrate_noise(elements=list(maps)) if add_noise else {}
    weff = _weff("forward") if (add_noise and input_noise == "measured") \
        else None

    out = {}
    for el, m in maps.items():
        if m.shape != FRONTAL_SHAPE:
            raise ValueError(f"{el}: expected {FRONTAL_SHAPE}, got {m.shape}")
        sim = _warp(m, "forward") * gains[el]
        if add_noise:
            var = ks[el] * np.clip(sim, 0.0, None)
            if input_noise == "measured":
                var = var * np.clip(1.0 - gains[el] * weff, 0.0, None)
            sim = sim + rng.normal(size=sim.shape) * np.sqrt(var)
            sim = np.clip(sim, 0.0, None)   # net counts are non-negative
        out[el] = sim
    return out


def inverse(maps_tilted: dict, angle_deg: float = REF_ANGLE_DEG) -> dict:
    """Deterministic inverse: divide gains, warp back to the frontal frame.

    Returns dict {element: (60, 120) array}; pixels outside the tilted
    scan's footprint are NaN (the tilted scan only covers part of the
    frontal frame).  This is the physics-only baseline restoration.
    """
    gains = tilt_gains(angle_deg, elements=list(maps_tilted))
    out = {}
    for el, m in maps_tilted.items():
        if m.shape != TILTED_SHAPE:
            raise ValueError(f"{el}: expected {TILTED_SHAPE}, got {m.shape}")
        out[el] = _warp(np.asarray(m, dtype=float) / gains[el], "inverse")
    return out
