"""
Poisson (binomial) splitting for Noise2Noise training pair generation.

Core methodological contribution: from a single photon-counting spectrum,
generate two independent noisy observations by splitting each count
via Binomial(n, 0.5). Both halves share the same expected signal
but have independent noise, enabling self-supervised denoising.
"""

import numpy as np


def poisson_split(
    spectrum: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a photon-counting spectrum into two independent noisy observations.

    For each channel with count n, draw k ~ Binomial(n, 0.5).
    Return (k, n - k) as two sub-spectra.

    Properties:
        - split_a + split_b == original (exact, always)
        - E[split_a] = E[split_b] = spectrum / 2
        - Var[split_a] = spectrum / 4 (binomial variance np(1-p))
        - Cov(split_a[i], split_b[i]) = 0 (conditional independence)

    Parameters
    ----------
    spectrum : np.ndarray, shape (..., C)
        Integer photon counts per channel. Can be 1D (single spectrum)
        or 2D (batch of spectra).
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    split_a, split_b : np.ndarray, same shape as input, float32
    """
    if rng is None:
        rng = np.random.default_rng()

    counts = np.round(spectrum).astype(np.int64)
    counts = np.maximum(counts, 0)

    split_a = rng.binomial(counts, 0.5)
    split_b = counts - split_a

    return split_a.astype(np.float32), split_b.astype(np.float32)
