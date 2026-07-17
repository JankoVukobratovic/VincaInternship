"""
compare_Ti.py
Ti Kα comparison map: prova1 vs ruotato scan, detector 10264.
Output: results_rotated/Ti_prova1_vs_ruotato.png

Requires the ruotato Ti cache (results_rotated/_npy_cache/) and either
the prova1 Ti cache or the raw prova1 MCA files under Resources/.
Run from the project root: python scripts/compare_Ti.py
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.stats import linregress

# ── Calibration ─────────────────────────────────────────────────
_CAL = np.array([[219, 6.4], [278, 8.0], [363, 10.5], [436, 12.6], [869, 25.3]])
_SLOPE, _INTERCEPT, *_ = linregress(_CAL[:, 0], _CAL[:, 1])

# ── Parameters ──────────────────────────────────────────────────
TI_KEV = 4.51
TI_HW  = 0.30
BG_HW  = 0.25

DET    = "10264"
W1, H1 = 120, 60   # prova1 grid
WR, HR = 80,  45   # ruotato grid

def _resolve(*candidates):
    for p in candidates:
        if os.path.isdir(p):
            return p
    return candidates[0]

PROVA1_DIR  = _resolve(f"Resources/aurora-antico1-prova1/{DET}",
                       f"aurora-antico1-prova1/{DET}")
PROVA1_NPY  = f"results_rotated/_npy_cache/prova1_{DET}_Ti.npy"
RUOTATO_NPY = f"results_rotated/_npy_cache/ruotato_{DET}_Ti.npy"
OUT_PATH    = "results_rotated/Ti_prova1_vs_ruotato.png"


def bg_subtracted_integral(counts, energy, target_kev, hw=0.30, bg_hw=0.25):
    """Net peak area: fixed half-window with linear background subtraction."""
    idx     = int(np.argmin(np.abs(energy - target_kev)))
    half_ch = max(1, int(round(hw    / _SLOPE)))
    bg_ch   = max(1, int(round(bg_hw / _SLOPE)))
    lo = max(0,               idx - half_ch)
    hi = min(len(counts) - 1, idx + half_ch)
    bg_lo_l = max(0,               lo - bg_ch)
    bg_hi_r = min(len(counts) - 1, hi + 1 + bg_ch)
    bg_left  = counts[bg_lo_l:lo].mean()    if lo      > bg_lo_l else counts[lo]
    bg_right = counts[hi + 1:bg_hi_r].mean() if bg_hi_r > hi + 1  else counts[hi]
    bg_line  = np.linspace(bg_left, bg_right, hi - lo + 1)
    return max(0.0, float((counts[lo:hi + 1] - bg_line).sum()))


# ── 1. prova1 Ti map (cache, else integrate from MCA files) ─────
if os.path.exists(PROVA1_NPY):
    print("prova1 Ti: loading from cache...")
    ti_prova1 = np.load(PROVA1_NPY)
elif os.path.isdir(PROVA1_DIR):
    print(f"prova1 Ti: integrating {W1 * H1} MCA files...")
    ti_prova1 = np.zeros((H1, W1), dtype=np.float64)
    _ec = {}
    for i in range(1, W1 * H1 + 1):
        path = os.path.join(PROVA1_DIR, f"None_{i}.mca")
        if not os.path.exists(path):
            continue
        counts_list, in_data = [], False
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line == "<<DATA>>":
                    in_data = True
                    continue
                if line == "<<END>>":
                    break
                if in_data:
                    try:
                        counts_list.append(int(line))
                    except ValueError:
                        pass
        counts = np.array(counts_list, dtype=np.float64)
        n_ch = len(counts)
        if n_ch not in _ec:
            _ec[n_ch] = np.arange(n_ch) * _SLOPE + _INTERCEPT
        row, col = (i - 1) // W1, (i - 1) % W1
        ti_prova1[row, col] = bg_subtracted_integral(
            counts, _ec[n_ch], TI_KEV, TI_HW, BG_HW)
        if i % 1000 == 0 or i == W1 * H1:
            print(f"  {i}/{W1 * H1}")
    os.makedirs(os.path.dirname(PROVA1_NPY), exist_ok=True)
    np.save(PROVA1_NPY, ti_prova1)
    print("  Cache saved.")
else:
    sys.exit(f"ERROR: neither {PROVA1_NPY} nor raw data in {PROVA1_DIR} found "
             "(see README: Setup / Raw MCA data).")

# ── 2. ruotato Ti map (cache only) ──────────────────────────────
if not os.path.exists(RUOTATO_NPY):
    sys.exit(f"ERROR: ruotato Ti cache not found: {RUOTATO_NPY}")
print("ruotato Ti: loading from cache...")
ti_ruotato = np.load(RUOTATO_NPY)   # (45, 80)

# ── 3. Resample prova1 to the ruotato grid ──────────────────────
ti_prova1_dn = zoom(ti_prova1, (HR / H1, WR / W1), order=1)   # (45, 80)
print(f"prova1 downsampled: {ti_prova1.shape} -> {ti_prova1_dn.shape}")

# ── 4. Difference map ───────────────────────────────────────────
diff = ti_prova1_dn - ti_ruotato

fig, ax = plt.subplots(figsize=(8, 5), dpi=140, layout="constrained")
fig.patch.set_facecolor("white")

am = np.percentile(np.abs(diff), 99) or 1.0
im = ax.imshow(diff, cmap="RdBu_r", aspect="equal", origin="upper",
               interpolation="nearest", vmin=-am, vmax=am)
ax.set_title(f"Ti Kα difference: prova1 − ruotato  (detector {DET})",
             fontsize=11, fontweight="bold", pad=6)
ax.set_xticks([])
ax.set_yticks([])
cbar = plt.colorbar(im, ax=ax, fraction=0.046 * HR / WR, pad=0.02)
cbar.set_label("Δ net counts", fontsize=8)
cbar.ax.tick_params(labelsize=7)
cbar.outline.set_edgecolor("#CCCCCC")

os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nSaved: {os.path.abspath(OUT_PATH)}")
