"""
datagen.py - infinite generator of physics-simulated training pairs for
the learned restoration (MVP item 3).

Training data comes from the prova1 maps ONLY (prova2 and the real
ruotato scan are reserved for the honest final test).  One sample:

    1. take the 8 prova1 detector-summed maps (optionally flipped -
       a flipped painting is an equally valid painting - and optionally
       dose-scaled to simulate a low-dose acquisition)
    2. simulate the tilted measurement at a random angle in [4, 25]
       deg: either the validated forward() (bilinear warp; v1 style)
       or forward_sharp() (cubic sampling at the exact warp positions;
       the v2 default - see its docstring for why the bilinear forward
       over-blurs the training inputs relative to a real acquisition)
    3. optionally zero out random rectangular blocks in the TILTED
       frame (acquisition dropout across all channels) and record them
       in the validity mask
    4. inverse(tilted, angle): the deterministic physics restoration
       (identical operator at train and test time)
    5. network input  = normalized inverse + validity + angle channel
       network target = the normalized frontal maps of step 1

so the network learns only the RESIDUAL restoration on top of physics:
contrast/detail lost to warp resampling + the added acquisition noise
(+ inpainting of dropout blocks).

Spatial holdout: 4 fixed 15x30 blocks of the frontal frame
(VAL_BLOCKS, chosen for footprint coverage 70-100%) are NEVER used in
the training loss; they are the early-stopping validation region.  The
holdout mask is flipped together with the map content, so those prova1
pixels stay out of the loss under augmentation too.

Normalization: fixed per-element scales = P99 of the prova1 maps
(norm_scales), applied identically at train and test time; the absolute
level is preserved because the model only adds residuals on top of the
physics inversion.

Numpy only - torch stays in the model/training code.
"""

import numpy as np
from scipy.ndimage import map_coordinates

try:
    import forward_model as fm
except ImportError:  # pragma: no cover - package-style import
    from . import forward_model as fm

ELEMENTS = fm.ELEMENTS
FRONTAL_SHAPE = fm.FRONTAL_SHAPE          # (60, 120)
TILTED_SHAPE = fm.TILTED_SHAPE            # (45, 80)
ANGLE_NORM = 25.0                         # angle channel = angle / ANGLE_NORM
ANGLE_RANGE = (4.0, 25.0)
DOSE_RANGE = (0.35, 1.0)                  # log-uniform when applied
P_DOSE = 0.25                             # probability of a low-dose sample
P_FLIP = 0.5                              # per-axis flip probability
P_BLOCKS = 0.3                            # probability of dropout blocks
BLOCK_SIDE = (8, 17)                      # block height/width range (px)

# fixed validation blocks (r0, c0), each 15 x 30, frontal frame; chosen
# once from the footprint coverage map (70.0 / 93.3 / 100.0 / 70.7 %)
VAL_BLOCKS = ((15, 30), (15, 90), (30, 60), (45, 90))
VAL_BLOCK_SHAPE = (15, 30)

INPUT_STYLES = ("bilinear", "cubic")
FRESH_FRAC = 0.25                         # fresh-noise fraction, cubic style

_STACK = None
_SCALES = None
_FOOTPRINT = None
_HOLDOUT = None
_FWD_COORDS = None


# --------------------------------------------------------------------------
# fixed assets
# --------------------------------------------------------------------------

def prova1_stack() -> np.ndarray:
    """(8, 60, 120) float64 stack of the prova1 detector-summed maps."""
    global _STACK
    if _STACK is None:
        maps = fm.load_summed_maps("prova1")
        _STACK = np.stack([maps[el] for el in ELEMENTS]).astype(np.float64)
    return _STACK


def norm_scales() -> np.ndarray:
    """(8,) per-element normalization scales: P99 of prova1."""
    global _SCALES
    if _SCALES is None:
        _SCALES = np.percentile(prova1_stack(), 99, axis=(1, 2))
    return _SCALES


def footprint() -> np.ndarray:
    """(60, 120) bool: frontal pixels covered by the tilted footprint."""
    global _FOOTPRINT
    if _FOOTPRINT is None:
        _FOOTPRINT = fm.frontal_footprint_mask()
    return _FOOTPRINT


def holdout_mask() -> np.ndarray:
    """(60, 120) bool: the fixed spatial-holdout blocks (True = held out)."""
    global _HOLDOUT
    if _HOLDOUT is None:
        m = np.zeros(FRONTAL_SHAPE, dtype=bool)
        bh, bw = VAL_BLOCK_SHAPE
        for r0, c0 in VAL_BLOCKS:
            m[r0:r0 + bh, c0:c0 + bw] = True
        _HOLDOUT = m
    return _HOLDOUT


# --------------------------------------------------------------------------
# sharp-acquisition forward variant (training inputs only)
# --------------------------------------------------------------------------

def _forward_coords():
    """(yf, xf): frontal position each tilted pixel samples (public affine).

    Same geometry as forward_model._warp_coords('forward'), rebuilt here
    from the public affine accessors (forward_model is read-only).
    """
    global _FWD_COORDS
    if _FWD_COORDS is None:
        A, t = fm.affine_ruotato_to_frontal()
        hr, wr = TILTED_SHAPE
        hf, wf = FRONTAL_SHAPE
        yy, xx = np.meshgrid(np.arange(hr), np.arange(wr), indexing="ij")
        xr = xx - (wr - 1) / 2.0
        yr = yy - (hr - 1) / 2.0
        xf = A[0, 0] * xr + A[0, 1] * yr + t[0] + (wf - 1) / 2.0
        yf = A[1, 0] * xr + A[1, 1] * yr + t[1] + (hf - 1) / 2.0
        _FWD_COORDS = (yf, xf)
    return _FWD_COORDS


def forward_sharp(maps_f: dict, angle_deg: float, rng,
                  fresh_frac: float = FRESH_FRAC) -> dict:
    """Sharp-acquisition variant of the forward simulator (training only).

    Rationale (v1 -> v2 iteration).  The validated forward() emulates the
    tilted acquisition by BILINEARLY warping the frontal map, but a real
    tilted scan is a direct measurement of the painting on the tilted
    grid and carries no resampling blur (MVP-2 check [3]: HF_sim <
    HF_real on all 8 lines).  Training inputs built with forward() thus
    carry TWO bilinear blurs (forward + inverse) where the real
    restoration input carries ONE; a net trained on them learns to
    over-sharpen (v1 finding: cv_ratio overshoot up to 1.05 on the real
    test and r below baseline).  Measured at 7.7 deg, the deterministic
    inverse of forward() gives cv_ratio 0.90-0.96 while the real ruotato
    gives 0.93-0.96; sampling the frontal map with a CUBIC SPLINE at the
    exact warp positions reproduces the real contrast loss (mean
    |cv_sim - cv_real| 0.012 vs 0.023 bilinear, closer on 6/8 lines).

    Noise: gains as in forward(); added variance k*s*(max(1-g, 0) +
    fresh_frac).  The cubic sample carries ~ the full k*s of the source
    map, and the extra fresh fraction both breaks the input/target noise
    correlation (a denoising incentive) and reflects the measured
    sd_real/sd_pred >= 1 of MVP-2 [3] on Ca/Ti/Cu/PbLa/PbLb.
    """
    yf, xf = _forward_coords()
    gains = fm.tilt_gains(angle_deg, elements=list(maps_f))
    ks = fm.calibrate_noise(elements=list(maps_f))
    out = {}
    for el, m in maps_f.items():
        g = gains[el]
        T = map_coordinates(np.asarray(m, dtype=float), [yf, xf],
                            order=3, mode="nearest") * g
        var = ks[el] * np.clip(T, 0.0, None) * (max(1.0 - g, 0.0)
                                                + fresh_frac)
        T = T + rng.normal(size=T.shape) * np.sqrt(var)
        out[el] = np.clip(T, 0.0, None)
    return out


# --------------------------------------------------------------------------
# network-input construction (shared by training and the real test)
# --------------------------------------------------------------------------

def build_input(inv_maps: dict, angle_deg: float,
                validity: np.ndarray | None = None) -> np.ndarray:
    """(10, 60, 120) float32 network input from a deterministic inverse.

    inv_maps : dict {element: (60, 120)} from forward_model.inverse
               (NaN outside the footprint -> zeroed here)
    validity : float mask in [0, 1]; defaults to the footprint mask.
    """
    scales = norm_scales()
    x = np.zeros((10,) + FRONTAL_SHAPE, dtype=np.float32)
    for i, el in enumerate(ELEMENTS):
        x[i] = np.nan_to_num(inv_maps[el], nan=0.0) / scales[i]
    if validity is None:
        validity = footprint().astype(np.float32)
    x[8] = validity
    x[9] = angle_deg / ANGLE_NORM
    return x


# --------------------------------------------------------------------------
# sample generation
# --------------------------------------------------------------------------

def sample(rng: np.random.Generator,
           angle: float | None = None,
           dose: float | None = None,
           flip: tuple | None = None,
           blocks: list | None = None,
           input_style: str = "cubic",
           fresh_frac: float = FRESH_FRAC):
    """One simulated training pair; None arguments are randomized.

    angle  : tilt angle in deg (default: uniform in ANGLE_RANGE)
    dose   : count-scale factor applied BEFORE the simulated noise
             (default: 1.0 w.p. 1-P_DOSE, else log-uniform DOSE_RANGE)
    flip   : (flip_ud, flip_lr) booleans (default: each w.p. P_FLIP)
    blocks : list of (r0, c0, h, w) dropout rectangles in the TILTED
             frame, zeroed across all channels (default: [] w.p.
             1-P_BLOCKS, else 1-2 random rectangles)
    input_style : "cubic" (sharp-acquisition forward, see forward_sharp;
             matches the real one-warp restoration input) or "bilinear"
             (the validated forward(); carries a second resampling blur
             the real input does not have - kept for the v1 ablation)
    fresh_frac : fresh-noise fraction of the cubic style

    Returns (x, y, loss_mask, val_mask, meta):
        x         (10, 60, 120) float32  network input
        y         (8, 60, 120)  float32  normalized frontal target
        loss_mask (60, 120)     bool     footprint minus holdout blocks
        val_mask  (60, 120)     bool     footprint AND holdout blocks
        meta      dict                   angle/dose/flip/blocks
    """
    if angle is None:
        angle = float(rng.uniform(*ANGLE_RANGE))
    if dose is None:
        dose = 1.0
        if rng.random() < P_DOSE:
            lo, hi = np.log(DOSE_RANGE)
            dose = float(np.exp(rng.uniform(lo, hi)))
    if flip is None:
        flip = (bool(rng.random() < P_FLIP), bool(rng.random() < P_FLIP))

    stack = prova1_stack() * dose
    hold = holdout_mask()
    if flip[0]:
        stack = stack[:, ::-1, :]
        hold = hold[::-1, :]
    if flip[1]:
        stack = stack[:, :, ::-1]
        hold = hold[:, ::-1]
    maps_f = {el: np.ascontiguousarray(stack[i])
              for i, el in enumerate(ELEMENTS)}

    if input_style == "bilinear":
        tilted = fm.forward(maps_f, angle_deg=angle, rng=rng,
                            add_noise=True, input_noise="measured")
    elif input_style == "cubic":
        tilted = forward_sharp(maps_f, angle, rng, fresh_frac=fresh_frac)
    else:
        raise ValueError(f"input_style must be one of {INPUT_STYLES}")

    if blocks is None:
        blocks = []
        if rng.random() < P_BLOCKS:
            th, tw = TILTED_SHAPE
            for _ in range(int(rng.integers(1, 3))):
                h = int(rng.integers(*BLOCK_SIDE))
                w = int(rng.integers(*BLOCK_SIDE))
                r0 = int(rng.integers(0, th - h + 1))
                c0 = int(rng.integers(0, tw - w + 1))
                blocks.append((r0, c0, h, w))
    v_tilt = np.ones(TILTED_SHAPE)
    for (r0, c0, h, w) in blocks:
        v_tilt[r0:r0 + h, c0:c0 + w] = 0.0
        for el in ELEMENTS:
            tilted[el][r0:r0 + h, c0:c0 + w] = 0.0

    inv = fm.inverse(tilted, angle_deg=angle)
    v_frontal = np.nan_to_num(fm.warp_tilted_to_frontal(v_tilt), nan=0.0)

    x = build_input(inv, angle, validity=v_frontal.astype(np.float32))
    y = (stack / norm_scales()[:, None, None]).astype(np.float32)

    fp = footprint()
    loss_mask = fp & ~hold
    val_mask = fp & hold
    meta = {"angle": angle, "dose": dose, "flip": flip, "blocks": blocks,
            "style": input_style}
    return x, y, loss_mask, val_mask, meta


def make_batch(rng: np.random.Generator, n: int, **kwargs):
    """Stack n samples: (x, y, loss_mask, val_mask, metas) as arrays."""
    xs, ys, lms, vms, metas = [], [], [], [], []
    for _ in range(n):
        x, y, lm, vm, meta = sample(rng, **kwargs)
        xs.append(x)
        ys.append(y)
        lms.append(lm)
        vms.append(vm)
        metas.append(meta)
    return (np.stack(xs), np.stack(ys),
            np.stack(lms), np.stack(vms), metas)


def generator(seed: int = 0, batch: int = 8, **kwargs):
    """Infinite batch generator (the advertised entry point)."""
    rng = np.random.default_rng(seed)
    while True:
        yield make_batch(rng, batch, **kwargs)
