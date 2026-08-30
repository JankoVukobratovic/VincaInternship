"""
xrf_core.py
══════════════════════════════════════════════════════════════════════
Unified XRF analysis engine.

Element definitions live in elements.json - to add a new element, add
one JSON entry, e.g.

    "Mn": { "name": "Mn Kα", "kev": 5.895, "hw": 0.30, "color": "#884400" }

and it will be integrated, cached and rendered automatically. Elements
sharing a "display_group" (e.g. all Pb L-lines) are summed into a single
combined map for display. A few elements receive special treatment
(K uses a tight-sideband integrator; Zn and S get overlap corrections).

PUBLIC API
──────────
    run_scan(
        datasets,           # dict[label -> folder_path]  e.g. {"prova1": "dir/det/"}
        width, height,      # scan grid dimensions
        output_dir,         # root folder for all output
        integrator,         # "fixed_hw" | "dynamic_trapezoid"
        mode,               # "standard" | "compare_scans" | "compare_detectors" | "outlier"
        detectors,          # list of detector IDs used when building cache key/paths
        elements_json,      # path to elements.json  (default: "elements.json")
        calibration,        # list of [channel, keV] pairs (default: built-in)
        cache_dir,          # override cache location (default: <output_dir>/_npy_cache)
        individual_maps,    # bool – also save one PNG per element (default False)
        stacked_spectrum,   # bool – also save summed spectrum plot (default False)
    ) -> dict               # {"cubes": {label: cube}, "disp_cubes": {label: disp_cube}, ...}

MODES
──────
    standard         – process each dataset independently, save element maps
    compare_scans    – like standard, then diff between first two dataset labels
    compare_detectors– single scan in two detector folders; sum + diff of detectors
    outlier          – read existing cache, find top-N outlier pixels between two datasets

INTEGRATORS
────────────
    fixed_hw          – half-window around peak center, linear background subtraction
                        K Kα always uses the tight-sideband variant regardless
    dynamic_trapezoid – walk outward from peak center until counts stop rising,
                        integrate with trapezoid rule (no explicit background removal)
"""

from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import linregress

# ══════════════════════════════════════════════════════════════════════
#  DEFAULTS
# ══════════════════════════════════════════════════════════════════════

_DEFAULT_CAL = np.array([
    [219, 6.4], [278, 8.0], [363, 10.5], [436, 12.6], [869, 25.3]
], dtype=float)

_DEFAULT_ELEMENTS_JSON = os.path.join(os.path.dirname(__file__), "elements.json")

# Correction constants
_CU_KB_KA_RATIO    = 0.17   # Cu Kβ / Cu Kα  (standard)
_PB_LA_LB_RATIO_0  = 1.40   # initial Pb Lα / Pb Lβ estimate


# ══════════════════════════════════════════════════════════════════════
#  ELEMENT LOADING
# ══════════════════════════════════════════════════════════════════════

def load_elements(path: str = _DEFAULT_ELEMENTS_JSON) -> dict:
    """
    Load element definitions from a JSON file.

    Returns a dict keyed by element symbol. Private keys (starting with '_')
    are filtered out. Each value is guaranteed to have at least:
        name, kev, hw, color
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    elements = {
        k: v for k, v in raw.items()
        if not k.startswith("_") and isinstance(v, dict) and "kev" in v
    }

    # Build LinearSegmentedColormap from hex color string
    for key, cfg in elements.items():
        if "color" in cfg and isinstance(cfg["color"], str):
            cfg["cmap"] = LinearSegmentedColormap.from_list(
                key, ["#000000", cfg["color"]]
            )

    return elements


def load_annotation_lines(path: str = _DEFAULT_ELEMENTS_JSON) -> list[dict]:
    """Return the _annotation_lines list from the JSON, or [] if absent."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return raw.get("_annotation_lines", [])


# ══════════════════════════════════════════════════════════════════════
#  CALIBRATION
# ══════════════════════════════════════════════════════════════════════

def build_calibration(cal_points: np.ndarray | None = None) -> tuple[float, float]:
    """
    Linear regression over (channel, keV) pairs.
    Returns (slope, intercept) in keV/channel.
    """
    pts = np.array(cal_points) if cal_points is not None else _DEFAULT_CAL
    slope, intercept, *_ = linregress(pts[:, 0], pts[:, 1])
    return float(slope), float(intercept)


def energy_axis(n_channels: int, slope: float, intercept: float) -> np.ndarray:
    return np.arange(n_channels) * slope + intercept


# ══════════════════════════════════════════════════════════════════════
#  MCA PARSING
# ══════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=512)
def parse_mca_file(filepath: str) -> dict:
    """
    Parse a .mca file. Returns {"counts": ndarray, "time": float}.
    Results are cached in memory so repeated reads within a session are free.
    """
    meta, counts, in_data = {}, [], False
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == "<<DATA>>":
                in_data = True
                continue
            if line == "<<END>>":
                break
            if in_data:
                try:
                    counts.append(int(line))
                except ValueError:
                    pass
            elif " - " in line:
                k, v = line.split(" - ", 1)
                meta[k.strip()] = v.strip()

    return {
        "counts": np.array(counts, dtype=np.float64),
        "time":   float(meta.get("REAL_TIME", 1.0)),
    }


# ══════════════════════════════════════════════════════════════════════
#  INTEGRATORS
# ══════════════════════════════════════════════════════════════════════

def _integrate_fixed_hw(
    counts: np.ndarray,
    energy: np.ndarray,
    target_kev: float,
    hw: float = 0.30,
    bg_hw: float = 0.25,
) -> float:
    """
    Net peak signal using a fixed half-window with linear background subtraction.

    Peak window  : [center − hw,  center + hw]
    BG sidebands : bg_hw wide, immediately outside the peak window on each side.
    Returns max(0, net_integral).
    """
    slope = energy[1] - energy[0]  # keV per channel (uniform axis)

    idx     = int(np.argmin(np.abs(energy - target_kev)))
    half_ch = max(1, int(round(hw   / slope)))
    bg_ch   = max(1, int(round(bg_hw / slope)))

    lo = max(0,               idx - half_ch)
    hi = min(len(counts) - 1, idx + half_ch)

    bg_lo_l = max(0,               lo - bg_ch)
    bg_lo_r = lo
    bg_hi_l = hi + 1
    bg_hi_r = min(len(counts) - 1, hi + 1 + bg_ch)

    bg_left  = counts[bg_lo_l:bg_lo_r].mean() if bg_lo_r > bg_lo_l else counts[lo]
    bg_right = counts[bg_hi_l:bg_hi_r].mean() if bg_hi_r > bg_hi_l else counts[hi]

    n_peak  = hi - lo + 1
    bg_line = np.linspace(bg_left, bg_right, n_peak)

    return max(0.0, float((counts[lo:hi + 1] - bg_line).sum()))


def _integrate_dynamic_trapezoid(
    counts: np.ndarray,
    energy: np.ndarray,
    target_kev: float,
    max_wing_kev: float = 1.0,
) -> float:
    """
    Dynamic peak area via trapezoid integration.

    Walks outward from the peak center while counts are still rising.
    Stops when counts decrease or when max_wing_kev is exceeded.
    Returns the trapezoid integral of the found region (no BG subtraction).
    Note: divides by acquisition time externally if needed – raw counts here.
    """
    idx = int(np.argmin(np.abs(energy - target_kev)))

    l = idx
    while l > 0 and counts[l - 1] < counts[l]:
        if energy[idx] - energy[l] > max_wing_kev:
            break
        l -= 1

    r = idx
    while r < len(counts) - 1 and counts[r + 1] < counts[r]:
        if energy[r] - energy[idx] > max_wing_kev:
            break
        r += 1

    return float(np.trapezoid(counts[l:r + 1], x=energy[l:r + 1]))


def _integrate_k_sideband(counts: np.ndarray, energy: np.ndarray) -> float:
    """
    Special tight-sideband integrator for K Kα (3.3138 keV).

    Uses narrow sideband windows in the clean valley between the Ar Kα
    (2.957 keV) tail and the Ca Kα (3.69 keV) rising edge:
        BG left  : 3.23 – 3.28 keV
        BG right : 3.35 – 3.42 keV
        Peak     : ±2 channels around 3.3138 keV
    """
    k_ch  = int(np.argmin(np.abs(energy - 3.3138)))
    pk_lo = k_ch - 2
    pk_hi = k_ch + 2

    ch_ll = int(np.argmin(np.abs(energy - 3.23)))
    ch_lr = int(np.argmin(np.abs(energy - 3.28)))
    ch_rl = int(np.argmin(np.abs(energy - 3.35)))
    ch_rr = int(np.argmin(np.abs(energy - 3.42)))

    bg_left  = counts[ch_ll:ch_lr + 1].mean() if ch_lr >= ch_ll else counts[pk_lo]
    bg_right = counts[ch_rl:ch_rr + 1].mean() if ch_rr >= ch_rl else counts[pk_hi]
    bg_per_ch = (bg_left + bg_right) / 2.0

    n   = pk_hi - pk_lo + 1
    net = float(counts[pk_lo:pk_hi + 1].sum()) - bg_per_ch * n
    return max(0.0, net)


def integrate(
    counts: np.ndarray,
    energy: np.ndarray,
    key: str,
    cfg: dict,
    integrator: str = "fixed_hw",
) -> float:
    """
    Dispatch to the correct integrator for a single element / pixel.

    K always uses the tight-sideband method regardless of `integrator`.
    """
    if key == "K":
        return _integrate_k_sideband(counts, energy)
    if integrator == "dynamic_trapezoid":
        return _integrate_dynamic_trapezoid(counts, energy, cfg["kev"])
    # default: fixed_hw
    return _integrate_fixed_hw(counts, energy, cfg["kev"], cfg.get("hw", 0.30))


# ══════════════════════════════════════════════════════════════════════
#  SCAN PROCESSING
# ══════════════════════════════════════════════════════════════════════

def process_folder(
    folder:     str,
    width:      int,
    height:     int,
    label:      str,
    elements:   dict,
    integrator: str,
    slope:      float,
    intercept:  float,
    cache_dir:  str,
) -> tuple[np.ndarray, list[str]]:
    """
    Read all None_N.mca files in `folder`, integrate every element for
    every pixel, return a 3-D cube of shape (n_elements, height, width).

    Results are cached as .npy files in cache_dir so subsequent runs skip
    the expensive file-reading loop.
    """
    el_keys     = list(elements.keys())
    total       = width * height
    os.makedirs(cache_dir, exist_ok=True)

    cache_paths = {
        k: os.path.join(cache_dir, f"{label}_{k}.npy")
        for k in el_keys
    }
    if all(os.path.exists(p) for p in cache_paths.values()):
        print(f"  [{label}] Loading from cache...")
        cube = np.stack([np.load(cache_paths[k]) for k in el_keys], axis=0)
        return cube, el_keys

    cube   = np.zeros((len(el_keys), height, width), dtype=np.float64)
    _ecache: dict[int, np.ndarray] = {}   # energy axis cache by n_channels
    n_read = 0

    for i in range(1, total + 1):
        path = os.path.join(folder, f"None_{i}.mca")
        if not os.path.exists(path):
            continue
        n_read += 1

        data   = parse_mca_file(path)
        counts = data["counts"]
        n_ch   = len(counts)

        if n_ch not in _ecache:
            _ecache[n_ch] = energy_axis(n_ch, slope, intercept)
        en = _ecache[n_ch]

        row, col = (i - 1) // width, (i - 1) % width

        for ei, key in enumerate(el_keys):
            cube[ei, row, col] = integrate(counts, en, key, elements[key], integrator)

        if i % 500 == 0 or i == total:
            print(f"  [{label}] {i}/{total}  ({100 * i // total}%)")

    if n_read == 0:
        # Never cache an all-zero cube - that silently poisons every
        # downstream script the next time it loads "from cache".
        raise FileNotFoundError(
            f"[{label}] no MCA files found in '{folder}' - check that the "
            "raw data are in place (see README: Setup / Raw MCA data)"
        )

    for ei, k in enumerate(el_keys):
        np.save(cache_paths[k], cube[ei])
    print(f"  [{label}] Cache saved.")
    return cube, el_keys


# ══════════════════════════════════════════════════════════════════════
#  CORRECTIONS
# ══════════════════════════════════════════════════════════════════════

def apply_corrections(cube: np.ndarray, el_keys: list[str], label: str = "") -> np.ndarray:
    """
    In-place-safe correction pipeline (operates on a copy).

      1. Zn  : subtract Cu Kβ contribution  (Kβ/Kα ≈ 0.17)
      2. S   : subtract Pb Mα contribution  (ratio estimated from data)
         Pb Mα (2.346 keV) and S Kα (2.308 keV) overlap in the S window.

    The As/Pb Lα window is left uncorrected here; the Pb-combined and
    corrected-As channels are built later in build_display_cube().
    """
    cube = cube.copy()
    ku   = {k: i for i, k in enumerate(el_keys)}

    # ── Zn − Cu Kβ ───────────────────────────────────────────
    if "Cu" in ku and "Zn" in ku:
        cube[ku["Zn"]] = np.maximum(
            0, cube[ku["Zn"]] - _CU_KB_KA_RATIO * cube[ku["Cu"]]
        )

    # ── S − Pb Mα ────────────────────────────────────────────
    pb_lb_key = "PbLb" if "PbLb" in ku else ("Pb" if "Pb" in ku else None)
    if "S" in ku and pb_lb_key:
        s_flat  = cube[ku["S"]].flatten()
        pb_flat = cube[ku[pb_lb_key]].flatten()

        # Estimate ratio from pixels where S signal is lowest
        mask = s_flat <= np.percentile(s_flat, 25)
        if mask.sum() > 50 and pb_flat[mask].std() > 1:
            slope_est, _, r, *_ = linregress(pb_flat[mask], s_flat[mask])
            ratio_s = max(0.0, min(2.0, slope_est))
            print(f"  [{label}] Pb Mα/Lβ from data: {ratio_s:.4f}  (r={r:.3f})")
        else:
            ratio_s = 0.005
            print(f"  [{label}] Pb Mα/Lβ = default: {ratio_s:.4f}")

        cube[ku["S"]] = np.maximum(0, cube[ku["S"]] - ratio_s * cube[ku[pb_lb_key]])

    return cube


def build_display_cube(
    cube:     np.ndarray,
    el_keys:  list[str],
    elements: dict,
    label:    str = "",
) -> tuple[np.ndarray, list[str]]:
    """
    Collapse the raw processing cube into a display cube:

        elements without a display_group  →  passthrough (already corrected)
        elements sharing a display_group  →  summed into one combined channel
                                             named after the group (e.g. all
                                             Pb L-lines → "Pb")
        "As" (if a dedicated window exists) →  As-window minus the estimated
                                             Pb Lα contribution

    The Pb Lα/Lβ ratio for the As correction is estimated from the data
    using pixels where the As-window signal is lowest (likely pure-Pb
    pixels). Falls back to _PB_LA_LB_RATIO_0 if the data are insufficient.
    """
    ku = {k: i for i, k in enumerate(el_keys)}

    # group name -> member keys, in el_keys order
    groups: dict[str, list[str]] = {}
    for k in el_keys:
        g = elements.get(k, {}).get("display_group")
        if g:
            groups.setdefault(g, []).append(k)
    grouped = {m for members in groups.values() for m in members}

    disp, disp_keys = [], []

    # Passthrough channels (keep acquisition order; As handled separately)
    for k in el_keys:
        if k in grouped or k == "As":
            continue
        disp.append(cube[ku[k]])
        disp_keys.append(k)

    # Combined group channels
    for g, members in groups.items():
        disp.append(sum(cube[ku[m]] for m in members))
        disp_keys.append(g)
        print(f"  [{label}] {g} = combined {' + '.join(members)}")

    # Corrected As channel - only when an explicit As window was integrated
    if "As" in ku:
        as_raw    = cube[ku["As"]]
        pb_lb_key = "PbLb" if "PbLb" in ku else ("Pb" if "Pb" in ku else None)
        pb_lb     = cube[ku[pb_lb_key]] if pb_lb_key else np.zeros_like(cube[0])

        as_f, pb_f = as_raw.flatten(), pb_lb.flatten()
        mask = as_f <= np.percentile(as_f, 25)
        if mask.sum() > 30 and pb_f[mask].std() > 1:
            s, _, r, *_ = linregress(pb_f[mask], as_f[mask])
            ratio = max(0.5, min(2.5, s))
            print(f"  [{label}] As: Pb Lα/Lβ ratio = {ratio:.3f}  (r={r:.3f})")
        else:
            ratio = _PB_LA_LB_RATIO_0
            print(f"  [{label}] As: Pb Lα/Lβ ratio = default ({ratio:.2f})")
        disp.append(np.maximum(0, as_raw - ratio * pb_lb))
        disp_keys.append("As")

    return np.stack(disp, axis=0), disp_keys


# ══════════════════════════════════════════════════════════════════════
#  COLORMAPS FOR DISPLAY
# ══════════════════════════════════════════════════════════════════════

def _display_cmap(key: str, elements: dict) -> Any:
    """Return the colormap for a display-cube key."""
    _override = {
        "Pb": LinearSegmentedColormap.from_list("Pb", ["#000000", "#CC66FF"]),
        "As": LinearSegmentedColormap.from_list("As", ["#000000", "#FF8800"]),
    }
    if key in _override:
        return _override[key]
    if key in elements and "cmap" in elements[key]:
        return elements[key]["cmap"]
    return "viridis"


def _display_name(key: str, elements: dict) -> str:
    _override = {
        "Pb": "Pb (Lα+Lβ+Ll+Lγ)",
        "As": "As Kα (corr.)",
    }
    if key in _override:
        return _override[key]
    if key in elements:
        return elements[key]["name"]
    return key


# ══════════════════════════════════════════════════════════════════════
#  RENDERING
# ══════════════════════════════════════════════════════════════════════

def _imshow_ax(
    ax, img: np.ndarray, cmap, is_diff: bool, w: int, h: int,
    title: str, cbar_label: str
):
    """Helper: imshow + attached colorbar on a single axes (square pixels)."""
    if is_diff:
        am = np.percentile(np.abs(img), 99) or 1.0
        vmin, vmax = -am, am
    else:
        vmin = 0
        vmax = np.percentile(img, 99) or 1.0

    im = ax.imshow(img, cmap=cmap, aspect="equal", origin="upper",
                   interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10, fontweight="bold", color="black", pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    # Scale colorbar height to the image aspect so it matches the panel
    cbar = plt.colorbar(im, ax=ax, fraction=0.046 * h / w, pad=0.02)
    cbar.set_label(cbar_label, fontsize=7, color="black")
    cbar.ax.tick_params(labelsize=6, colors="black")
    cbar.outline.set_edgecolor("#CCCCCC")


def render_grid(
    cube:      np.ndarray,
    keys:      list[str],
    elements:  dict,
    width:     int,
    height:    int,
    save_path: str,
    title:     str = "",
    is_diff:   bool = False,
    n_cols:    int = 4,
):
    """Save a grid of all element maps to a single PNG."""
    n  = len(keys)
    nc = n_cols
    nr = math.ceil(n / nc)

    panel_w = 5.2
    panel_h = panel_w * height / width + 0.55          # image + title strip
    fig, axes = plt.subplots(
        nr, nc, figsize=(panel_w * nc, panel_h * nr + (0.6 if title else 0)),
        dpi=110, layout="constrained",
    )
    fig.patch.set_facecolor("white")
    axes = np.array(axes).flatten()

    cbar_label = "Δ counts" if is_diff else "net counts"

    for idx, key in enumerate(keys):
        _imshow_ax(
            axes[idx], cube[idx],
            cmap      = "RdBu_r" if is_diff else
                        (elements[key]["cmap"] if key in elements
                         else _display_cmap(key, elements)),
            is_diff   = is_diff,
            w=width, h=height,
            title     = _display_name(key, elements),
            cbar_label= cbar_label,
        )

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", color="black")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {os.path.relpath(save_path)}")


def render_display_grid(
    cube:      np.ndarray,
    keys:      list[str],
    elements:  dict,
    width:     int,
    height:    int,
    save_path: str,
    title:     str = "",
    is_diff:   bool = False,
    n_cols:    int = 4,
):
    """
    Like render_grid but uses the display-cube naming/colormaps
    (combined Pb, corrected As).
    """
    n  = len(keys)
    nr = math.ceil(n / n_cols)

    panel_w = 5.2
    panel_h = panel_w * height / width + 0.55
    fig, axes = plt.subplots(
        nr, n_cols, figsize=(panel_w * n_cols, panel_h * nr + (0.6 if title else 0)),
        dpi=110, layout="constrained",
    )
    fig.patch.set_facecolor("white")
    axes = np.array(axes).flatten()

    cbar_label = "Δ counts" if is_diff else "net counts"

    for idx, key in enumerate(keys):
        cmap = ("RdBu_r" if is_diff
                else _display_cmap(key, elements))
        _imshow_ax(
            axes[idx], cube[idx],
            cmap=cmap, is_diff=is_diff,
            w=width, h=height,
            title=_display_name(key, elements),
            cbar_label=cbar_label,
        )

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", color="black")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {os.path.relpath(save_path)}")


def render_individual(
    cube:      np.ndarray,
    keys:      list[str],
    elements:  dict,
    width:     int,
    height:    int,
    out_dir:   str,
    prefix:    str = "",
    is_diff:   bool = False,
):
    """Save one PNG per element."""
    os.makedirs(out_dir, exist_ok=True)
    cbar_label = "Δ counts" if is_diff else "net counts"

    for idx, key in enumerate(keys):
        cmap = "RdBu_r" if is_diff else _display_cmap(key, elements)
        fig, ax = plt.subplots(
            figsize=(7, 7 * height / width + 0.6), dpi=130, layout="constrained",
        )
        fig.patch.set_facecolor("white")
        _imshow_ax(
            ax, cube[idx], cmap=cmap, is_diff=is_diff,
            w=width, h=height,
            title=_display_name(key, elements),
            cbar_label=cbar_label,
        )
        fname = f"{prefix}{key}.png"
        save_path = os.path.join(out_dir, fname)
        plt.savefig(save_path, dpi=140, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved: {os.path.relpath(save_path)}")


def render_comparison_strips(
    cubes:       list[np.ndarray],
    keys:        list[str],
    elements:    dict,
    width:       int,
    height:      int,
    strip_names: list[str],
    save_path:   str,
    title:       str = "",
):
    """
    Side-by-side comparison layout (from pomocne_metode render_comparisons).

    Each row = one element. Each column = one dataset / strip.
    Last strip is assumed to be a difference map (uses diverging colormap
    and symmetric scale). All others share a common min/max per element.
    """
    n_scans   = len(cubes)
    n_elem    = len(keys)

    panel_w = 4.6
    panel_h = panel_w * height / width + 0.35
    fig, axes = plt.subplots(
        n_elem, n_scans,
        figsize=(panel_w * n_scans + 1.4, panel_h * n_elem + (0.6 if title else 0)),
        dpi=100, squeeze=False, layout="constrained",
    )
    fig.patch.set_facecolor("white")

    for row_i, key in enumerate(keys):
        all_others = [c[row_i] for c in cubes[:-1]]
        g_min = float(np.nanmin(all_others))
        g_max = float(np.nanmax(all_others))
        # symmetric per-element scale so weak elements stay readable
        diff_abs = max(float(np.percentile(np.abs(cubes[-1][row_i]), 99)), 1e-6)

        for col_i in range(n_scans):
            ax   = axes[row_i, col_i]
            img  = cubes[col_i][row_i]
            is_diff_col = (col_i == n_scans - 1)

            if is_diff_col:
                vmin, vmax = -diff_abs, diff_abs
                cmap = "RdBu_r"
            else:
                vmin, vmax = g_min, g_max
                cmap = _display_cmap(key, elements)

            im = ax.imshow(img, cmap=cmap, aspect="equal", origin="upper",
                           interpolation="nearest", vmin=vmin, vmax=vmax)

            if row_i == 0:
                ax.set_title(strip_names[col_i], fontsize=13, fontweight="bold")
            if col_i == 0:
                ax.set_ylabel(
                    f"{_display_name(key, elements)}\n({elements[key]['kev']} keV)"
                    if key in elements else _display_name(key, elements),
                    fontsize=10, fontweight="bold", rotation=0,
                    ha="right", va="center", labelpad=42,
                )

            cbar = plt.colorbar(im, ax=ax, fraction=0.046 * height / width, pad=0.02)
            cbar.ax.tick_params(labelsize=7)
            cbar.outline.set_edgecolor("#CCCCCC")
            ax.set_xticks([])
            ax.set_yticks([])

    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {os.path.relpath(save_path)}")


def plot_stacked_spectrum(
    folder:        str,
    width:         int,
    height:        int,
    det_label:     str,
    slope:         float,
    intercept:     float,
    save_path:     str,
    annotation_lines: list[dict] | None = None,
):
    """
    Sum all MCA spectra in a folder and plot on a log scale with
    annotated element lines.
    """
    stacked = np.zeros(1024)
    n_files = 0

    for i in range(1, width * height + 1):
        path = os.path.join(folder, f"None_{i}.mca")
        if not os.path.exists(path):
            continue
        counts_raw = []
        in_data = False
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line == "<<DATA>>":  in_data = True;  continue
                if line == "<<END>>":   break
                if in_data:
                    try: counts_raw.append(int(line))
                    except: pass
        if len(counts_raw) >= 1024:
            stacked[:1024] += counts_raw[:1024]
            n_files += 1

    en = energy_axis(1024, slope, intercept)

    fig, ax = plt.subplots(figsize=(14, 5), dpi=110, layout="constrained")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F8F8F8")
    ax.semilogy(en, np.maximum(stacked, 1), color="#1155AA", linewidth=0.7, alpha=0.85)

    if annotation_lines:
        ax.set_ylim(top=max(stacked.max(), 1) * 8)   # headroom for line labels
        ymax = stacked.max() * 2
        for ann in annotation_lines:
            kev, lbl, col = ann["kev"], ann["label"], ann.get("color", "#888888")
            if kev > en[-1]:
                continue
            ax.axvline(kev, color=col, linewidth=1.0, alpha=0.7, linestyle="--")
            idx   = int(np.argmin(np.abs(en - kev)))
            y_pos = max(stacked[max(0, idx - 3):idx + 4].max() * 1.8, ymax * 0.02)
            ax.text(kev, y_pos, lbl, color=col, fontsize=7, ha="center",
                    va="bottom", rotation=90, clip_on=True)

    ax.set_xlim(1, 16)
    ax.set_xlabel("Energy (keV)", fontsize=9, color="black")
    ax.set_ylabel("Counts (log scale)", fontsize=9, color="black")
    ax.tick_params(colors="black", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#AAAAAA")
    ax.set_title(
        f"Summed spectrum - detector {det_label}  ({n_files} pixels)",
        fontsize=11, fontweight="bold", color="black"
    )
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {os.path.relpath(save_path)}")


# ══════════════════════════════════════════════════════════════════════
#  OUTLIER ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def _parse_mca_counts(path: str) -> np.ndarray:
    """Lightweight MCA reader (counts only, no metadata)."""
    counts, in_data = [], False
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if ln == "<<DATA>>": in_data = True;  continue
            if ln == "<<END>>":  break
            if in_data:
                try: counts.append(int(ln))
                except: pass
    return np.array(counts, dtype=np.float64)


def _plot_outlier_pair(
    pixel_i:   int,
    width:     int,
    det:       str,
    elem_key:  str,
    diff_val:  float,
    sig1:      float,
    sig2:      float,
    path1:     str,
    path2:     str,
    save_path: str,
    slope:     float,
    intercept: float,
    annotation_lines: list[dict],
):
    row = (pixel_i - 1) // width + 1
    col = (pixel_i - 1) %  width + 1

    c1 = _parse_mca_counts(path1) if os.path.exists(path1) else np.zeros(1024)
    c2 = _parse_mca_counts(path2) if os.path.exists(path2) else np.zeros(1024)
    n  = max(len(c1), len(c2), 1024)
    en = energy_axis(n, slope, intercept)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), dpi=100)
    fig.patch.set_facecolor("#111111")

    for ax, counts, prova_lbl, sig in zip(axes, [c1, c2], ["prova1", "prova2"], [sig1, sig2]):
        ax.set_facecolor("#1A1A1A")
        ax.plot(en[:len(counts)], counts, color="#AADDFF", linewidth=0.6, alpha=0.9)
        ymax = counts.max() if counts.max() > 0 else 1.0

        for ann in annotation_lines:
            kev, lbl, col = ann["kev"], ann["label"], ann.get("color", "#888888")
            if kev > en[-1]: continue
            ax.axvline(kev, color=col, linewidth=1.0, alpha=0.75, linestyle="--")
            idx = int(np.argmin(np.abs(en - kev)))
            y   = max(counts[max(0, idx - 3):idx + 4].max() * 1.5, ymax * 0.05)
            ax.text(kev, y, lbl, color=col, fontsize=6, ha="center",
                    va="bottom", rotation=90, clip_on=True)

        ax.set_xlim(1.5, 15)
        ax.set_xlabel("Energy (keV)", color="white", fontsize=8)
        ax.set_ylabel("Counts",       color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#555555")
        ax.set_title(
            f"{prova_lbl}  |  Det {det}  |  Pixel {pixel_i} (r={row}, c={col})\n"
            f"{elem_key}: {sig:.0f} counts",
            color="white", fontsize=8,
        )

    fig.suptitle(
        f"OUTLIER: {elem_key}  |  Det {det}  |  Pixel {pixel_i}\n"
        f"Δ(prova1−prova2) = {diff_val:+.0f}  (prova1={sig1:.0f}, prova2={sig2:.0f})",
        color="#FF6666", fontsize=9, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


def run_outlier_analysis(
    cache_dir:   str,
    dir1:        str,
    dir2:        str,
    detectors:   list[str],
    elements:    dict,
    width:       int,
    height:      int,
    output_dir:  str,
    slope:       float,
    intercept:   float,
    annotation_lines: list[dict],
    top_n:       int   = 10,
    prag_pct:    float = 99.0,
):
    """Isolate pixels where prova1 − prova2 is extreme (per element)."""
    os.makedirs(output_dir, exist_ok=True)
    summary = []

    for det in detectors:
        print(f"=== Detector {det} ===")
        for elem in elements:
            p1 = os.path.join(cache_dir, f"prova1_{det}_{elem}.npy")
            p2 = os.path.join(cache_dir, f"prova2_{det}_{elem}.npy")
            if not (os.path.exists(p1) and os.path.exists(p2)):
                print(f"  [{elem}] cache missing – skipping")
                continue

            m1   = np.load(p1)
            m2   = np.load(p2)
            diff = (m1 - m2).flatten()

            prag        = np.percentile(np.abs(diff), prag_pct)
            outlier_idx = np.where(np.abs(diff) > prag)[0]
            outlier_idx = outlier_idx[np.argsort(-np.abs(diff[outlier_idx]))][:top_n]

            print(
                f"  [{elem}]  mean_diff={diff.mean():+.1f}  std={diff.std():.1f}  "
                f"prag={prag:.0f}  outliers={len(np.where(np.abs(diff)>prag)[0])}/{width*height}"
            )

            out_dir = os.path.join(output_dir, f"{elem}_{det}")
            os.makedirs(out_dir, exist_ok=True)

            for rank, flat_i in enumerate(outlier_idx):
                pixel_i = int(flat_i) + 1
                sig1    = float(m1.flatten()[flat_i])
                sig2    = float(m2.flatten()[flat_i])
                dv      = float(diff[flat_i])

                path1 = os.path.join(dir1, det, f"None_{pixel_i}.mca")
                path2 = os.path.join(dir2, det, f"None_{pixel_i}.mca")
                save  = os.path.join(out_dir, f"rank{rank+1:02d}_pixel{pixel_i}.png")

                _plot_outlier_pair(
                    pixel_i, width, det, elem, dv, sig1, sig2,
                    path1, path2, save, slope, intercept, annotation_lines,
                )
                summary.append((elem, det, pixel_i, dv, sig1, sig2))

            print(f"         Saved {len(outlier_idx)} spectra -> {os.path.relpath(out_dir)}/")

    # ── Histogram of differences ────────────────────────────
    n_elem = len(elements)
    nc     = 4
    nr     = math.ceil(n_elem * len(detectors) / nc)

    fig, axes = plt.subplots(
        math.ceil(n_elem / nc), nc * len(detectors),
        figsize=(5 * nc * len(detectors), 3.5 * math.ceil(n_elem / nc) + 1),
        dpi=90,
    )
    fig.patch.set_facecolor("white")
    axes = np.array(axes).flatten()
    ax_i = 0

    for elem in elements:
        for det in detectors:
            p1 = os.path.join(cache_dir, f"prova1_{det}_{elem}.npy")
            p2 = os.path.join(cache_dir, f"prova2_{det}_{elem}.npy")
            if not (os.path.exists(p1) and os.path.exists(p2)):
                if ax_i < len(axes): axes[ax_i].set_visible(False)
                ax_i += 1; continue

            diff = np.load(p1).flatten() - np.load(p2).flatten()
            prag = np.percentile(np.abs(diff), prag_pct)
            ax = axes[ax_i]
            ax.hist(diff, bins=80, color="#4488FF", alpha=0.75, edgecolor="none")
            ax.axvline( prag, color="red",   linewidth=1.2, linestyle="--")
            ax.axvline(-prag, color="red",   linewidth=1.2, linestyle="--")
            ax.axvline(0,     color="black", linewidth=0.8)
            ax.set_title(f"{elem}  det{det}", fontsize=8, fontweight="bold")
            ax.set_xlabel("Δ counts (p1−p2)", fontsize=7)
            ax.tick_params(labelsize=6)
            ax_i += 1

    for j in range(ax_i, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Distribution of prova1 − prova2 differences\n"
        "Red dashed = threshold (99th percentile of |diff|)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    hist_path = os.path.join(output_dir, "histogram_differences.png")
    plt.savefig(hist_path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {os.path.relpath(hist_path)}")

    # ── Text summary ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  OUTLIER ANALYSIS COMPLETE")
    print(f"  Total isolated spectra: {len(summary)}")
    print(f"{'='*60}")
    print("\nTop outliers (largest |Δ|):")
    for elem, det, pix, dv, s1, s2 in sorted(summary, key=lambda x: -abs(x[3]))[:20]:
        print(f"  {elem:6s} det{det}  pixel={pix:5d}  Δ={dv:+8.0f}  (p1={s1:.0f}, p2={s2:.0f})")

    return summary


# ══════════════════════════════════════════════════════════════════════
#  PUBLIC API  –  run_scan()
# ══════════════════════════════════════════════════════════════════════

def run_scan(
    datasets:          dict[str, str],
    width:             int,
    height:            int,
    output_dir:        str,
    mode:              str  = "standard",
    integrator:        str  = "fixed_hw",
    detectors:         list[str] | None = None,
    elements_json:     str  = _DEFAULT_ELEMENTS_JSON,
    calibration:       list | None = None,
    cache_dir:         str  | None = None,
    individual_maps:   bool = False,
    stacked_spectrum:  bool = False,
) -> dict:
    """
    Main entry point.

    Parameters
    ──────────
    datasets : dict[label -> folder_path]
        Each label becomes a named dataset. The exact semantics depend on mode:
          "standard"          – each entry is processed and saved independently
          "compare_scans"     – first two labels are diffed (e.g. prova1 vs prova2)
          "compare_detectors" – each label is a detector folder for the SAME scan;
                                sum and difference maps are also produced
          "outlier"           – exactly two labels; reads from cache and runs
                                outlier detection. Expects labels "prova1"/"prova2"
                                as keys, with folder values = top-level scan dirs
                                (e.g. "aurora-antico1-prova1"), not detector subdirs.

    width, height : int
        Scan grid dimensions in pixels.

    output_dir : str
        Root output directory. Sub-folders are created automatically.

    mode : str
        One of "standard" | "compare_scans" | "compare_detectors" | "outlier".

    integrator : str
        "fixed_hw" (default) or "dynamic_trapezoid".
        K Kα always uses the tight-sideband method regardless.

    detectors : list[str] | None
        Detector IDs (used for cache key naming in compare_* modes).
        Required for "compare_detectors" and "outlier" modes.

    elements_json : str
        Path to the elements JSON file.

    calibration : list of [channel, keV] pairs | None
        Override the default 5-point calibration.

    cache_dir : str | None
        Override the .npy cache location (default: <output_dir>/_npy_cache).

    individual_maps : bool
        Also save one PNG per element alongside the grid.

    stacked_spectrum : bool
        Also save a summed MCA spectrum plot (useful for ruotato / single scans).

    Returns
    ───────
    dict with keys:
        "cubes"      : {label: (raw_corrected_cube, el_keys)}
        "disp_cubes" : {label: (display_cube, disp_keys)}
        "elements"   : loaded element dict
        "slope"      : calibration slope
        "intercept"  : calibration intercept
    """

    # ── Setup ──────────────────────────────────────────────────
    slope, intercept = build_calibration(calibration)
    elements         = load_elements(elements_json)
    ann_lines        = load_annotation_lines(elements_json)
    _cache_dir       = cache_dir or os.path.join(output_dir, "_npy_cache")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  XRF run_scan")
    print(f"  mode={mode}  integrator={integrator}")
    print(f"  grid={width}×{height}  elements={list(elements.keys())}")
    print(f"{'='*60}\n")

    result = {"cubes": {}, "disp_cubes": {}, "elements": elements,
              "slope": slope, "intercept": intercept}

    # ── OUTLIER mode ───────────────────────────────────────────
    if mode == "outlier":
        labels = list(datasets.keys())
        assert len(labels) == 2, "outlier mode requires exactly 2 dataset labels"
        dir1, dir2 = datasets[labels[0]], datasets[labels[1]]
        dets = detectors or []
        run_outlier_analysis(
            cache_dir=_cache_dir, dir1=dir1, dir2=dir2,
            detectors=dets, elements=elements,
            width=width, height=height,
            output_dir=os.path.join(output_dir, "outlier"),
            slope=slope, intercept=intercept,
            annotation_lines=ann_lines,
        )
        return result

    # ── COMPARE_DETECTORS mode ─────────────────────────────────
    if mode == "compare_detectors":
        dets = detectors or list(datasets.keys())
        for det in dets:
            folder = datasets[det]
            label  = f"det_{det}"
            print(f"\n--- Processing detector {det} ---")
            cube, keys = process_folder(
                folder, width, height, label, elements, integrator,
                slope, intercept, _cache_dir,
            )
            cube_c = apply_corrections(cube, keys, label)
            disp_c, dkeys = build_display_cube(cube_c, keys, elements, label)
            result["cubes"][det]      = (cube_c, keys)
            result["disp_cubes"][det] = (disp_c, dkeys)

            # Per-detector map
            render_display_grid(
                disp_c, dkeys, elements, width, height,
                save_path=os.path.join(output_dir, f"elements_det{det}.png"),
                title=f"Element maps - detector {det}  ({width}×{height} px)",
            )
            if stacked_spectrum:
                plot_stacked_spectrum(
                    folder, width, height, det, slope, intercept,
                    save_path=os.path.join(output_dir, "spectra", f"stacked_{det}.png"),
                    annotation_lines=ann_lines,
                )
            if individual_maps:
                render_individual(
                    disp_c, dkeys, elements, width, height,
                    out_dir=os.path.join(output_dir, "individual", det),
                    prefix=f"det{det}_",
                )

        # Detector sum
        if len(dets) == 2:
            d1, d2    = dets[0], dets[1]
            disp_sum  = result["disp_cubes"][d1][0] + result["disp_cubes"][d2][0]
            disp_diff = result["disp_cubes"][d1][0] - result["disp_cubes"][d2][0]
            dk        = result["disp_cubes"][d1][1]

            render_display_grid(
                disp_sum, dk, elements, width, height,
                save_path=os.path.join(output_dir, "elements_sum_detectors.png"),
                title=f"Element maps - detectors {d1} + {d2} summed  (improved SNR)",
            )
            render_display_grid(
                disp_diff, dk, elements, width, height,
                save_path=os.path.join(output_dir, "diff_detectors.png"),
                title=f"Detector difference: {d1} − {d2}\nRED = more in {d1}  |  BLUE = more in {d2}",
                is_diff=True,
            )
            if individual_maps:
                render_individual(disp_sum,  dk, elements, width, height,
                    out_dir=os.path.join(output_dir, "individual", "sum"),  prefix="sum_")
                render_individual(disp_diff, dk, elements, width, height,
                    out_dir=os.path.join(output_dir, "individual", "diff"), prefix="diff_", is_diff=True)

        return result

    # ── STANDARD / COMPARE_SCANS modes ────────────────────────
    labels = list(datasets.keys())

    for label, folder in datasets.items():
        print(f"\n--- Processing: {label} ---")
        cube, keys = process_folder(
            folder, width, height, label, elements, integrator,
            slope, intercept, _cache_dir,
        )
        print(f"  Applying corrections...")
        cube_c = apply_corrections(cube, keys, label)
        disp_c, dkeys = build_display_cube(cube_c, keys, elements, label)

        result["cubes"][label]      = (cube_c, keys)
        result["disp_cubes"][label] = (disp_c, dkeys)

        render_display_grid(
            disp_c, dkeys, elements, width, height,
            save_path=os.path.join(output_dir, label, "element_maps.png"),
            title=f"Element maps - {label}  ({width}×{height} px, {integrator})\n"
                  "background-subtracted net peak intensities, spectral-overlap corrections applied",
        )
        if individual_maps:
            render_individual(
                disp_c, dkeys, elements, width, height,
                out_dir=os.path.join(output_dir, label, "individual"),
            )
        if stacked_spectrum:
            plot_stacked_spectrum(
                folder, width, height, label, slope, intercept,
                save_path=os.path.join(output_dir, label, "stacked_spectrum.png"),
                annotation_lines=ann_lines,
            )

    # ── Diff (compare_scans) ───────────────────────────────────
    if mode == "compare_scans" and len(labels) >= 2:
        l1, l2 = labels[0], labels[1]
        d1 = result["disp_cubes"][l1][0]
        d2 = result["disp_cubes"][l2][0]
        dk = result["disp_cubes"][l1][1]
        diff = d1 - d2

        render_display_grid(
            diff, dk, elements, width, height,
            save_path=os.path.join(output_dir, "compare", f"diff_{l1}_vs_{l2}.png"),
            title=f"Difference: {l1} − {l2}\nRED = more in {l1}  |  BLUE = more in {l2}",
            is_diff=True,
        )
        # Comparison strips (prova1 | prova2 | diff)
        render_comparison_strips(
            cubes=[d1, d2, diff], keys=dk,
            elements=elements, width=width, height=height,
            strip_names=[l1, l2, f"{l1}−{l2}"],
            save_path=os.path.join(output_dir, "compare", f"strips_{l1}_vs_{l2}.png"),
            title=f"Comparison: {l1} vs {l2}",
        )
        if individual_maps:
            render_individual(diff, dk, elements, width, height,
                out_dir=os.path.join(output_dir, "compare", "individual_diff"),
                prefix="diff_", is_diff=True)

    print(f"\n{'='*60}")
    print(f"  All output saved to: {os.path.abspath(output_dir)}/")
    print(f"{'='*60}\n")
    return result