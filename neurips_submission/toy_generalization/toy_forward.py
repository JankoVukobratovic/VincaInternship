"""
toy_forward.py

Fully synthetic "ground truth field" generator plus a tiny forward simulator
with three physical-style knobs. This is a stand-in for the real X-ray
fluorescence pipeline's "painting + measured forward simulator" pair, built
from scratch with no connection to any real instrument or dataset.

NOTE on the calibration sigmas below: since there is no real instrument here,
the three "calibration sigma" values (gain_scale sd, warp_shift sd per axis,
noise_scale log-sd) are ARBITRARY CHOICES made by the author for this toy
demo, not measurements of anything. They only need to be small enough that
"nominal" and "jittered" simulators are close, and large enough that a
20-30x out-of-calibration defect is clearly distinguishable. Any similarly
scaled choice would work equally well for the purpose of this sanity check.
"""

from dataclasses import dataclass, replace
import numpy as np
from scipy.ndimage import shift as ndi_shift

GRID = 32  # field size (GRID x GRID)
K0 = 0.08  # fixed constant in the Poisson-like variance law: Var = noise_scale * K0 * clip(signal, 0, None)

# Calibration sigmas (arbitrary, synthetic; see module docstring)
GAIN_SCALE_SD = 0.10      # sd of multiplicative gain_scale around nominal 1.0
WARP_SHIFT_SD = 0.15      # sd (pixels) of each of dy, dx around nominal 0.0
NOISE_SCALE_LOGSD = 0.20  # sd of log(noise_scale) around nominal log(1.0) = 0


@dataclass
class SimKnobs:
    gain_scale: float = 1.0
    warp_shift: tuple = (0.0, 0.0)  # (dy, dx) in pixels
    noise_scale: float = 1.0


def nominal_knobs() -> SimKnobs:
    return SimKnobs(gain_scale=1.0, warp_shift=(0.0, 0.0), noise_scale=1.0)


def jitter_knobs(rng: np.random.Generator, base: SimKnobs = None) -> SimKnobs:
    """Draw a knob vector within calibration uncertainty of `base` (default: nominal)."""
    if base is None:
        base = nominal_knobs()
    gain = base.gain_scale * float(np.exp(rng.normal(0.0, GAIN_SCALE_SD)))
    # use exp(normal) so gain_scale stays positive; sd chosen small so this
    # is close to base.gain_scale * (1 + N(0, GAIN_SCALE_SD)) to first order
    dy = base.warp_shift[0] + rng.normal(0.0, WARP_SHIFT_SD)
    dx = base.warp_shift[1] + rng.normal(0.0, WARP_SHIFT_SD)
    noise = base.noise_scale * float(np.exp(rng.normal(0.0, NOISE_SCALE_LOGSD)))
    return SimKnobs(gain_scale=gain, warp_shift=(dy, dx), noise_scale=noise)


def random_field(rng: np.random.Generator, size: int = GRID) -> np.ndarray:
    """Synthetic 'ground truth field' (stand-in for a painting patch): sum of
    3-6 random 2D Gaussians plus small smooth low-frequency noise, clipped
    to be non-negative."""
    n_blobs = rng.integers(3, 7)  # 3..6 inclusive
    yy, xx = np.mgrid[0:size, 0:size]
    field = np.zeros((size, size), dtype=np.float64)
    for _ in range(n_blobs):
        cy = rng.uniform(0.15, 0.85) * size
        cx = rng.uniform(0.15, 0.85) * size
        sy = rng.uniform(2.0, 7.0)
        sx = rng.uniform(2.0, 7.0)
        amp = rng.uniform(0.5, 2.0)
        field += amp * np.exp(-(((yy - cy) ** 2) / (2 * sy ** 2) + ((xx - cx) ** 2) / (2 * sx ** 2)))

    # small smooth low-frequency noise: coarse random grid upsampled by
    # nearest-neighbour-free bilinear-ish interpolation via zoom
    coarse = rng.normal(0.0, 1.0, size=(4, 4))
    from scipy.ndimage import zoom
    low_freq = zoom(coarse, size / 4.0, order=3)
    low_freq = low_freq[:size, :size]
    field += 0.15 * low_freq

    field = np.clip(field, 0.0, None)
    return field.astype(np.float64)


def forward(field: np.ndarray, knobs: SimKnobs, rng: np.random.Generator) -> np.ndarray:
    """Degrade a clean field into a synthetic 'observation' using the three
    knobs: sub-pixel warp shift, multiplicative gain, and a Poisson-like
    additive noise whose variance scales with noise_scale and the clipped
    signal level."""
    warped = ndi_shift(field, shift=knobs.warp_shift, order=1, mode="nearest")
    signal = knobs.gain_scale * warped
    var = knobs.noise_scale * K0 * np.clip(signal, 0.0, None)
    noise = rng.normal(0.0, 1.0, size=signal.shape) * np.sqrt(var)
    obs = signal + noise
    obs = np.clip(obs, 0.0, None)
    return obs
