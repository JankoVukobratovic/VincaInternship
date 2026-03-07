"""Cross-detector validation: the key evidence that denoising works."""

import numpy as np
from scipy.stats import pearsonr


def datacube_to_element_map(
    datacube: np.ndarray,
    kev_center: float,
    cal_slope: float,
    cal_intercept: float,
    half_width_kev: float = 0.3,
) -> np.ndarray:
    """Extract elemental map by integrating around emission line."""
    C = datacube.shape[-1]
    ch_center = int(round((kev_center - cal_intercept) / cal_slope))
    half_ch = max(1, int(round(half_width_kev / cal_slope)))
    lo = max(0, ch_center - half_ch)
    hi = min(C, ch_center + half_ch + 1)
    return datacube[..., lo:hi].sum(axis=-1)


def cross_detector_validation(
    denoised_a: np.ndarray,
    raw_a: np.ndarray,
    raw_b: np.ndarray,
    elements: dict,
    cal_slope: float,
    cal_intercept: float,
    denoised_b: np.ndarray | None = None,
) -> dict:
    """
    Validate denoising using detector B as independent witness.

    For each element:
      - r(denoised_A, raw_B) should be > r(raw_A, raw_B)
      - If true, denoising extracted real signal, not noise artifacts.
      - r(denoised_A, denoised_B) serves as negative control:
        if it drops below r(denoised_A, raw_B), the model may be over-smoothing.

    Parameters
    ----------
    denoised_a : np.ndarray, shape (H, W, C) — denoised detector A
    raw_a : np.ndarray — raw detector A
    raw_b : np.ndarray — raw detector B (independent witness)
    elements : dict — {name: {'kev': float}}
    cal_slope, cal_intercept : float
    denoised_b : np.ndarray or None — denoised detector B (for negative control)

    Returns
    -------
    dict with per-element results
    """
    results = {}
    for el, info in elements.items():
        kev = info['kev']

        map_denoised_a = datacube_to_element_map(denoised_a, kev, cal_slope, cal_intercept)
        map_raw_a = datacube_to_element_map(raw_a, kev, cal_slope, cal_intercept)
        map_raw_b = datacube_to_element_map(raw_b, kev, cal_slope, cal_intercept)

        r_raw, _ = pearsonr(map_raw_a.ravel(), map_raw_b.ravel())
        r_denoised, _ = pearsonr(map_denoised_a.ravel(), map_raw_b.ravel())
        improvement = r_denoised - r_raw

        entry = {
            'r_raw_vs_B': float(r_raw),
            'r_denoised_vs_B': float(r_denoised),
            'improvement': float(improvement),
        }

        if denoised_b is not None:
            map_denoised_b = datacube_to_element_map(denoised_b, kev, cal_slope, cal_intercept)
            r_both_denoised, _ = pearsonr(map_denoised_a.ravel(), map_denoised_b.ravel())
            entry['r_denoised_A_vs_denoised_B'] = float(r_both_denoised)

        results[el] = entry

    return results
