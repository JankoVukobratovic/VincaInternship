"""classical.py - non-learned inpainting controls (WP3's key comparison).

Without these the paper cannot claim the learned prior is NEEDED - a
reviewer will (rightly) ask whether plain interpolation fills the hole
just as well.  All methods fill the hole in the TILTED frame (where the
data was lost), then go through the same nominal physics inverse as
every other candidate, so the comparison isolates the fill quality.

Implemented: nearest-neighbour fill, biharmonic fill.
TODO(WP3): OpenCV Navier-Stokes / Telea (cv2.inpaint, needs
opencv-python-headless) and any stronger classical method you deem
fair; keep the same signature and add it to CANDIDATES.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt

from common import core

fm = core.fm
ELEMENTS = core.ELEMENTS


def nearest_fill(tilted: dict, v_tilt: np.ndarray) -> dict:
    """Fill missing pixels with the nearest valid pixel's value."""
    missing = v_tilt < 0.5
    if not missing.any():
        return {el: m.copy() for el, m in tilted.items()}
    idx = distance_transform_edt(missing, return_distances=False,
                                 return_indices=True)
    out = {}
    for el, m in tilted.items():
        f = m.copy()
        f[missing] = m[tuple(i[missing] for i in idx)]
        out[el] = f
    return out


def biharmonic_fill(tilted: dict, v_tilt: np.ndarray) -> dict:
    """Biharmonic (smooth PDE) inpainting of the missing block."""
    try:
        from skimage.restoration import inpaint_biharmonic
    except ImportError:  # older/newer skimage layout
        from skimage.restoration.inpaint import inpaint_biharmonic
    missing = v_tilt < 0.5
    if not missing.any():
        return {el: m.copy() for el, m in tilted.items()}
    out = {}
    for el, m in tilted.items():
        scale = max(float(m.max()), 1e-9)
        out[el] = inpaint_biharmonic(m / scale, missing) * scale
    return out


def opencv_fill(tilted: dict, v_tilt: np.ndarray,
                method: str = "ns", radius: int = 3) -> dict:
    """cv2.inpaint control (INPAINT_NS or INPAINT_TELEA).

    Each map is scaled to [0, 1] -> uint8 for cv2 and the scaling is
    undone afterwards; only the masked (v_tilt < 0.5) pixels change,
    valid pixels are copied back verbatim so the 8-bit round trip does
    not touch them.
    """
    import cv2
    missing = v_tilt < 0.5
    if not missing.any():
        return {el: m.copy() for el, m in tilted.items()}
    flag = cv2.INPAINT_NS if method == "ns" else cv2.INPAINT_TELEA
    mask = missing.astype(np.uint8)
    out = {}
    for el, m in tilted.items():
        m = np.asarray(m, dtype=float)
        lo, hi = float(m[~missing].min()), float(m[~missing].max())
        scale = max(hi - lo, 1e-9)
        u8 = np.clip((m - lo) / scale * 255.0, 0, 255).astype(np.uint8)
        f = (cv2.inpaint(u8, mask, radius, flag).astype(float) / 255.0
             * scale + lo)
        f[~missing] = m[~missing]
        out[el] = f
    return out


def telea_fill(tilted: dict, v_tilt: np.ndarray) -> dict:
    return opencv_fill(tilted, v_tilt, method="telea")


def ns_fill(tilted: dict, v_tilt: np.ndarray) -> dict:
    return opencv_fill(tilted, v_tilt, method="ns")


# name -> fill function; WP3 extends this dict, everything downstream
# (grid loop, regime map) picks the additions up automatically
CANDIDATES = {
    "nearest": nearest_fill,
    "biharmonic": biharmonic_fill,
    "telea": telea_fill,
    "ns": ns_fill,
}


def classical_restorations(tilted: dict, v_tilt: np.ndarray,
                           angle_deg: float) -> dict:
    """{method: frontal maps} - fill in the tilted frame, invert nominally."""
    out = {}
    for name, fill in CANDIDATES.items():
        out[f"classical_{name}"] = fm.inverse(fill(tilted, v_tilt),
                                              angle_deg=angle_deg)
    return out
