"""Spatial registration of two detector datacubes."""

import numpy as np
from scipy.signal import correlate2d
from scipy.ndimage import shift as ndi_shift


def compute_element_map(cube: np.ndarray, channel_center: int,
                        half_width: int = 10) -> np.ndarray:
    """Integrate counts around a channel for registration reference."""
    lo = max(0, channel_center - half_width)
    hi = min(cube.shape[2], channel_center + half_width + 1)
    return cube[:, :, lo:hi].sum(axis=2)


def find_shift(map_a: np.ndarray, map_b: np.ndarray) -> tuple[float, float]:
    """
    Find the (row, col) shift to align map_b to map_a using cross-correlation.

    Returns
    -------
    shift_row, shift_col : float
        Sub-pixel shift to apply to map_b.
    """
    # Normalize
    a = (map_a - map_a.mean()) / (map_a.std() + 1e-10)
    b = (map_b - map_b.mean()) / (map_b.std() + 1e-10)

    corr = correlate2d(a, b, mode='full')
    peak = np.unravel_index(corr.argmax(), corr.shape)
    center = np.array(corr.shape) // 2
    shift_yx = np.array(peak) - center
    return float(shift_yx[0]), float(shift_yx[1])


def register_scans(
    cube_a: np.ndarray,
    cube_b: np.ndarray,
    reference_channel: int = 220,
    half_width: int = 10,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """
    Align cube_b to cube_a using cross-correlation on a reference element map.

    Parameters
    ----------
    cube_a, cube_b : np.ndarray, shape (H, W, C)
    reference_channel : int
        Center channel for the reference element (default ~Fe Ka at ch 220).
    half_width : int
        Integration half-width in channels.

    Returns
    -------
    cube_a_aligned, cube_b_aligned : np.ndarray
        Aligned datacubes (same shape, potentially cropped).
    shift_vector : tuple (dy, dx)
    """
    map_a = compute_element_map(cube_a, reference_channel, half_width)
    map_b = compute_element_map(cube_b, reference_channel, half_width)

    dy, dx = find_shift(map_a, map_b)
    print(f"  Detected shift: dy={dy:.1f}, dx={dx:.1f} pixels")

    # If shift is zero or very small, skip alignment
    if abs(dy) < 0.5 and abs(dx) < 0.5:
        print("  Shift negligible — skipping alignment")
        return cube_a.copy(), cube_b.copy(), (dy, dx)

    # Apply shift to cube_b (channel by channel)
    cube_b_shifted = np.zeros_like(cube_b)
    for ch in range(cube_b.shape[2]):
        cube_b_shifted[:, :, ch] = ndi_shift(
            cube_b[:, :, ch], [dy, dx], order=1, mode='constant', cval=0
        )

    # Crop to valid overlap region
    margin_y = int(np.ceil(abs(dy)))
    margin_x = int(np.ceil(abs(dx)))
    if margin_y > 0 or margin_x > 0:
        sy = slice(margin_y, -margin_y if margin_y > 0 else None)
        sx = slice(margin_x, -margin_x if margin_x > 0 else None)
        cube_a_out = cube_a[sy, sx, :]
        cube_b_out = cube_b_shifted[sy, sx, :]
    else:
        cube_a_out = cube_a.copy()
        cube_b_out = cube_b_shifted

    return cube_a_out, cube_b_out, (dy, dx)
