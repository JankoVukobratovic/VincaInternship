"""perturb.py - THE shared contract: a parametrized (perturbable)
version of the measured forward simulator.

OWNERSHIP: TEAM (physics code).  What is here is a working first
version so the pipeline runs end-to-end - rewrite or extend it freely.
The contract that must survive a rewrite: the SimKnobs fields, the
sample()/forward_perturbed()/jittered() signatures, and the identity
`forward_perturbed(NOMINAL) == datagen.forward_sharp` bit-for-bit.
`python neurips_submission/main.py --stage verify` checks all of it.

All three workpackages speak through `SimKnobs`:

    WP1  draws knobs WITHIN calibration uncertainty (jittered)  -> UQ
    WP2  sets knobs BEYOND it (named defects)                   -> audit
    WP3  keeps knobs nominal and sweeps the degradation instead

Design decisions baked in here (do not change without team sign-off):
  - The perturbation applies ONLY to the training-data simulator; the
    deterministic restoration operator `forward_model.inverse` stays
    nominal, exactly as at test time.  This models the realistic
    situation "the simulator is wrong, the instrument pipeline is what
    it is".
  - `sample()` mirrors `datagen.sample` (augmentation, dropout blocks,
    masks) with only the tilted-measurement simulation swapped for the
    perturbed forward.  Keep the two in sync.
  - `blur_mode="bilinear"` reproduces the v1 resampling-blur defect
    (order-1 sampling at the same warp coordinates) - the very mistake
    that motivated this paper, now available as a controlled knob.
"""

from dataclasses import dataclass, asdict

import numpy as np
from scipy.ndimage import map_coordinates

from common import core

fm, dg = core.fm, core.dg
ELEMENTS = core.ELEMENTS


@dataclass(frozen=True)
class SimKnobs:
    """One concrete (possibly wrong) belief about the instrument."""
    noise_k_scale: float = 1.0        # multiplies the calibrated k per line
    gain_scale: float = 1.0           # g' = 1 + gain_scale * (g - 1)
    angle_bias_deg: float = 0.0       # simulator's tilt belief minus truth
    blur_mode: str = "cubic"          # 'cubic' (v2, matches reality) or
                                      # 'bilinear' (the v1 defect)
    warp_shift_px: tuple = (0.0, 0.0)  # (dy, dx) systematic registration err
    warp_rot_deg: float = 0.0         # registration rotation error about the
                                      # footprint centre, deg
    gain_pct_offset: tuple = (0.0,) * len(ELEMENTS)
                                      # per-line additive error on the measured
                                      # tilt_pct_sum, in percent, ELEMENTS order
    noise_k_line_scale: tuple = (1.0,) * len(ELEMENTS)
                                      # per-line multiplier on the calibrated k
                                      # (WP4 round-2 knob: the PPC of round 1
                                      # showed the residual noise misfit is
                                      # line-dependent), ELEMENTS order
    fresh_frac: float = dg.FRESH_FRAC
    label: str = "nominal"

    def to_meta(self) -> dict:
        d = asdict(self)
        d["warp_shift_px"] = f"{self.warp_shift_px[0]}/{self.warp_shift_px[1]}"
        d["gain_pct_offset"] = "/".join(f"{v:.4f}" for v in self.gain_pct_offset)
        d["noise_k_line_scale"] = "/".join(f"{v:.4f}"
                                           for v in self.noise_k_line_scale)
        return d


NOMINAL = SimKnobs()


def forward_perturbed(maps_f: dict, angle_deg: float, rng,
                      knobs: SimKnobs = NOMINAL) -> dict:
    """Tilted measurement simulated under a perturbed instrument belief.

    Nominal knobs + cubic blur == datagen.forward_sharp to numerical
    identity (same coordinates, gains, noise law).
    """
    yf, xf = dg._forward_coords()
    if knobs.warp_rot_deg:
        # rotate the sampling pattern about the footprint centre: the
        # registration's rotation is wrong by warp_rot_deg
        th = np.deg2rad(knobs.warp_rot_deg)
        cy, cx = float(yf.mean()), float(xf.mean())
        y0, x0 = yf - cy, xf - cx
        yf = cy + np.cos(th) * y0 - np.sin(th) * x0
        xf = cx + np.sin(th) * y0 + np.cos(th) * x0
    dy, dx = knobs.warp_shift_px
    if dy or dx:
        yf, xf = yf + dy, xf + dx
    belief = angle_deg + knobs.angle_bias_deg
    gains = fm.tilt_gains(belief, elements=list(maps_f))
    ks = fm.calibrate_noise(elements=list(maps_f))
    order = 3 if knobs.blur_mode == "cubic" else 1
    out = {}
    for el, m in maps_f.items():
        # measured gain + per-line offset on the measured percent response,
        # both extrapolated linearly in the believed angle, then the
        # global slope factor
        off = (knobs.gain_pct_offset[ELEMENTS.index(el)] / 100.0
               * (belief / fm.REF_ANGLE_DEG))
        g = 1.0 + knobs.gain_scale * (gains[el] - 1.0 + off)
        T = map_coordinates(np.asarray(m, dtype=float), [yf, xf],
                            order=order, mode="nearest") * g
        var = (knobs.noise_k_scale
               * knobs.noise_k_line_scale[ELEMENTS.index(el)]
               * ks[el] * np.clip(T, 0.0, None)
               * (max(1.0 - g, 0.0) + knobs.fresh_frac))
        T = T + rng.normal(size=T.shape) * np.sqrt(var)
        out[el] = np.clip(T, 0.0, None)
    return out


def sample(rng: np.random.Generator, knobs: SimKnobs = NOMINAL,
           angle: float | None = None, dose: float | None = None,
           flip: tuple | None = None, blocks: list | None = None):
    """One training pair from the PERTURBED simulator.

    Mirrors datagen.sample (see its docstring for the argument
    semantics); returns the same (x, y, loss_mask, val_mask, meta).
    """
    if angle is None:
        angle = float(rng.uniform(*dg.ANGLE_RANGE))
    if dose is None:
        dose = 1.0
        if rng.random() < dg.P_DOSE:
            lo, hi = np.log(dg.DOSE_RANGE)
            dose = float(np.exp(rng.uniform(lo, hi)))
    if flip is None:
        flip = (bool(rng.random() < dg.P_FLIP),
                bool(rng.random() < dg.P_FLIP))

    stack = dg.prova1_stack() * dose
    hold = dg.holdout_mask()
    if flip[0]:
        stack, hold = stack[:, ::-1, :], hold[::-1, :]
    if flip[1]:
        stack, hold = stack[:, :, ::-1], hold[:, ::-1]
    maps_f = {el: np.ascontiguousarray(stack[i])
              for i, el in enumerate(ELEMENTS)}

    tilted = forward_perturbed(maps_f, angle, rng, knobs)

    if blocks is None:
        blocks = []
        if rng.random() < dg.P_BLOCKS:
            th, tw = core.TILTED_SHAPE
            for _ in range(int(rng.integers(1, 3))):
                h = int(rng.integers(*dg.BLOCK_SIDE))
                w = int(rng.integers(*dg.BLOCK_SIDE))
                blocks.append((int(rng.integers(0, th - h + 1)),
                               int(rng.integers(0, tw - w + 1)), h, w))
    v_tilt = np.ones(core.TILTED_SHAPE)
    for (r0, c0, h, w) in blocks:
        v_tilt[r0:r0 + h, c0:c0 + w] = 0.0
        for el in ELEMENTS:
            tilted[el][r0:r0 + h, c0:c0 + w] = 0.0

    # test-time operator: NOMINAL inverse, never perturbed (see module doc)
    inv = fm.inverse(tilted, angle_deg=angle)
    v_frontal = np.nan_to_num(fm.warp_tilted_to_frontal(v_tilt), nan=0.0)

    x = dg.build_input(inv, angle, validity=v_frontal.astype(np.float32))
    y = (stack / dg.norm_scales()[:, None, None]).astype(np.float32)

    fp = dg.footprint()
    meta = {"angle": angle, "dose": dose, "flip": flip, "blocks": blocks,
            "knobs": knobs.label}
    return x, y, fp & ~hold, fp & hold, meta


def jittered(rng: np.random.Generator, spec: dict,
             label: str = "jitter") -> SimKnobs:
    """Random knobs WITHIN calibration uncertainty (WP1 ensemble draw).

    The draws happen in a FIXED order so that adding optional knobs to
    the spec never changes the rng stream of existing specs (the WP1
    members and the WP2 nulls must stay bit-reproducible): the optional
    per-line noise draw happens last and ONLY if its sd is set.
    """
    nk = float(np.exp(rng.normal(0.0, spec["noise_k_log_sd"])))
    gs = float(rng.normal(1.0, spec["gain_scale_sd"]))
    ab = float(rng.normal(0.0, spec["angle_bias_sd_deg"]))
    ws = (float(rng.normal(0.0, spec["warp_shift_sd_px"])),
          float(rng.normal(0.0, spec["warp_shift_sd_px"])))
    wr = float(rng.normal(0.0, spec.get("warp_rot_sd_deg", 0.0)))
    go = tuple(float(v) for v in rng.normal(
        0.0, spec.get("gain_pct_offset_sd", 0.0), len(ELEMENTS)))
    line_sd = spec.get("noise_k_line_log_sd", 0.0)
    nl = tuple(float(v) for v in np.exp(
        rng.normal(0.0, line_sd, len(ELEMENTS)))) if line_sd > 0 \
        else (1.0,) * len(ELEMENTS)
    return SimKnobs(noise_k_scale=nk, gain_scale=gs, angle_bias_deg=ab,
                    warp_shift_px=ws, warp_rot_deg=wr, gain_pct_offset=go,
                    noise_k_line_scale=nl, label=label)
