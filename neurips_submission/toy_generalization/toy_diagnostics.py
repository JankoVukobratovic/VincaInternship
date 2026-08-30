"""
toy_diagnostics.py

Blind diagnostic battery (3 simple summary statistics comparing a
"measurement" to a "simulation" of the same underlying field) plus a
calibration-uncertainty-aware null distribution, and a whitened distance
used both for defect naming and for rejection ABC. This mirrors the real
pipeline's lesson that the null distribution must itself be built from
pairs that are BOTH simulated within calibration uncertainty (never a
zero-noise vs noisy comparison), otherwise everything looks "broken".
"""

import numpy as np

from toy_forward import random_field, forward, jitter_knobs

STAT_NAMES = ["level_ratio", "shift_offset", "noise_ratio"]

# which statistic is designed to be sensitive to which knob family
STAT_TO_FAMILY = {
    "level_ratio": "gain",
    "shift_offset": "warp",
    "noise_ratio": "noise",
}

Z_THRESHOLD = 3.0


def stat_level_ratio(meas: np.ndarray, sim: np.ndarray) -> float:
    """Mean ratio: sensitive to a broken gain_scale."""
    return float(np.mean(meas) / (np.mean(sim) + 1e-8))


def stat_shift_offset(meas: np.ndarray, sim: np.ndarray) -> float:
    """Sub-pixel offset magnitude via FFT cross-correlation: sensitive to a
    broken warp_shift. Written from scratch: both images are lightly
    Gaussian-smoothed first (the signal here is low-frequency blobs, so
    this suppresses the per-pixel measurement noise that would otherwise
    dominate a phase-normalised correlation); the integer-pixel peak of the
    (un-normalised) cross-power spectrum is then refined to sub-pixel with
    a 1-D parabolic fit around the peak on each axis."""
    from scipy.ndimage import gaussian_filter

    a = gaussian_filter(meas, sigma=1.2)
    b = gaussian_filter(sim, sigma=1.2)
    a = a - a.mean()
    b = b - b.mean()
    Fa = np.fft.fft2(a)
    Fb = np.fft.fft2(b)
    R = Fa * np.conj(Fb)  # cross-power spectrum, not phase-normalised
    corr = np.fft.ifft2(R).real
    H, W = corr.shape
    py, px = np.unravel_index(np.argmax(corr), corr.shape)

    def parabolic_refine(c_m1, c_0, c_p1):
        denom = (c_m1 - 2 * c_0 + c_p1)
        if abs(denom) < 1e-12:
            return 0.0
        return 0.5 * (c_m1 - c_p1) / denom

    dy_sub = parabolic_refine(corr[(py - 1) % H, px], corr[py, px], corr[(py + 1) % H, px])
    dx_sub = parabolic_refine(corr[py, (px - 1) % W], corr[py, px], corr[py, (px + 1) % W])

    dy = py + dy_sub
    dx = px + dx_sub
    if dy > H // 2:
        dy -= H
    if dx > W // 2:
        dx -= W
    return float(np.hypot(dy, dx))


def _high_freq_power(img: np.ndarray) -> float:
    F = np.fft.fft2(img)
    P = np.abs(F) ** 2
    H, W = img.shape
    fy = np.fft.fftfreq(H)
    fx = np.fft.fftfreq(W)
    FY, FX = np.meshgrid(fy, fx, indexing="ij")
    freq_mag = np.sqrt(FY ** 2 + FX ** 2)
    thresh = np.quantile(freq_mag, 0.75)
    mask = freq_mag >= thresh
    return float(P[mask].sum())


def stat_noise_ratio(meas: np.ndarray, sim: np.ndarray) -> float:
    """Ratio of high-frequency power (top quartile of spatial frequency):
    sensitive to a broken noise_scale."""
    return _high_freq_power(meas) / (_high_freq_power(sim) + 1e-8)


def compute_stats(meas: np.ndarray, sim: np.ndarray) -> dict:
    return {
        "level_ratio": stat_level_ratio(meas, sim),
        "shift_offset": stat_shift_offset(meas, sim),
        "noise_ratio": stat_noise_ratio(meas, sim),
    }


def build_null(rng: np.random.Generator, n_pairs: int = 400) -> dict:
    """Null distribution built from pairs where BOTH sides are simulated
    with knobs independently drawn within calibration uncertainty of
    nominal (never a noise-free vs noisy comparison)."""
    values = {name: [] for name in STAT_NAMES}
    for _ in range(n_pairs):
        field = random_field(rng)
        k1 = jitter_knobs(rng)
        k2 = jitter_knobs(rng)
        a = forward(field, k1, rng)
        b = forward(field, k2, rng)
        s = compute_stats(a, b)
        for name in STAT_NAMES:
            values[name].append(s[name])

    null = {}
    for name in STAT_NAMES:
        arr = np.array(values[name])
        null[name] = {"mean": float(arr.mean()), "std": float(arr.std() + 1e-8)}
    return null


def zscores(stats: dict, null: dict) -> dict:
    return {name: (stats[name] - null[name]["mean"]) / null[name]["std"] for name in STAT_NAMES}


def whitened_distance(stats: dict, null: dict) -> float:
    z = zscores(stats, null)
    return float(np.sqrt(sum(z[name] ** 2 for name in STAT_NAMES)))


def diagnose(stats: dict, null: dict, threshold: float = Z_THRESHOLD):
    """Pre-registered rule: the statistic with the largest |z|-score, if it
    exceeds `threshold`, names the broken knob family. Returns
    (family_or_none, best_stat_name, best_abs_z)."""
    z = zscores(stats, null)
    best_name = max(STAT_NAMES, key=lambda n: abs(z[n]))
    best_abs_z = abs(z[best_name])
    if best_abs_z > threshold:
        return STAT_TO_FAMILY[best_name], best_name, best_abs_z
    return "none", best_name, best_abs_z
