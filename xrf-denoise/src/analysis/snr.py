"""Per-element SNR computation for XRF datacubes."""

import numpy as np
from ..config import Config


def compute_element_snr(
    datacube: np.ndarray,
    element_kev: float,
    cal_slope: float,
    cal_intercept: float,
    peak_half_width_kev: float = 0.3,
    bg_offset_kev: float = 0.8,
    bg_width_kev: float = 0.3,
) -> np.ndarray:
    """
    Compute SNR for a given element at each pixel.

    SNR = (peak_integral - background_estimate) / sqrt(peak_integral)

    Parameters
    ----------
    datacube : np.ndarray, shape (H, W, C) or (N, C)
    element_kev : float
        Center energy of emission line in keV.
    peak_half_width_kev, bg_offset_kev, bg_width_kev : float
        Parameters for peak and background windows.

    Returns
    -------
    snr_map : np.ndarray, same spatial shape as input
    """
    original_shape = datacube.shape[:-1]
    C = datacube.shape[-1]
    flat = datacube.reshape(-1, C)

    def kev_to_ch(kev):
        return int(round((kev - cal_intercept) / cal_slope))

    # Peak window
    pk_center = kev_to_ch(element_kev)
    pk_half = max(1, int(round(peak_half_width_kev / cal_slope)))
    pk_lo = max(0, pk_center - pk_half)
    pk_hi = min(C, pk_center + pk_half + 1)

    # Background windows (flanking the peak)
    bg_half = max(1, int(round(bg_width_kev / cal_slope)))
    bg_offset = int(round(bg_offset_kev / cal_slope))

    bg_lo_l = max(0, pk_center - bg_offset - bg_half)
    bg_hi_l = max(0, pk_center - bg_offset + bg_half + 1)
    bg_lo_r = min(C, pk_center + bg_offset - bg_half)
    bg_hi_r = min(C, pk_center + bg_offset + bg_half + 1)

    peak_integral = flat[:, pk_lo:pk_hi].sum(axis=1)
    bg_left = flat[:, bg_lo_l:bg_hi_l].sum(axis=1)
    bg_right = flat[:, bg_lo_r:bg_hi_r].sum(axis=1)

    # Scale background to match peak window width
    bg_width_channels = (bg_hi_l - bg_lo_l) + (bg_hi_r - bg_lo_r)
    peak_width_channels = pk_hi - pk_lo
    bg_estimate = (bg_left + bg_right) * peak_width_channels / max(bg_width_channels, 1)

    net_signal = peak_integral - bg_estimate
    noise = np.sqrt(np.maximum(peak_integral, 1))
    snr = net_signal / noise

    return snr.reshape(original_shape)


def compute_all_element_snr(
    datacube: np.ndarray,
    cfg: Config,
) -> dict[str, np.ndarray]:
    """Compute SNR maps for all configured elements."""
    results = {}
    for el, info in cfg.elements.items():
        snr = compute_element_snr(
            datacube, info['kev'],
            cfg.cal_slope, cfg.cal_intercept,
        )
        results[el] = snr
    return results
