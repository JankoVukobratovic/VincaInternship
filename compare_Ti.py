"""
compare_Ti.py
Uporedna Ti mapa: prova1 vs ruotato, detektor 10264
Izlaz: rezultati_ruotato/Ti_prova1_vs_ruotato.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import zoom
from scipy.stats import linregress

# ── Kalibracija ─────────────────────────────────────────────────
_CAL = np.array([[219,6.4],[278,8.0],[363,10.5],[436,12.6],[869,25.3]])
_SLOPE, _INTERCEPT, *_ = linregress(_CAL[:,0], _CAL[:,1])

# ── Parametri ───────────────────────────────────────────────────
TI_KEV  = 4.51
TI_HW   = 0.30
BG_HW   = 0.25

DET     = "10264"
W1, H1  = 120, 60   # prova1
WR, HR  = 80,  45   # ruotato

PROVA1_DIR  = f"aurora-antico1-prova1/{DET}"
RUOTATO_NPY = f"rezultati_ruotato/_npy_cache/ruotato_{DET}_Ti.npy"
IZLAZ       = "rezultati_ruotato/Ti_prova1_vs_ruotato.png"

TI_CMAP = LinearSegmentedColormap.from_list("Ti", ["#000000", "#FF9966"])


def bg_subtracted_integral(counts, energy, target_kev, hw=0.30, bg_hw=0.25):
    idx     = int(np.argmin(np.abs(energy - target_kev)))
    half_ch = max(1, int(round(hw   / _SLOPE)))
    bg_ch   = max(1, int(round(bg_hw / _SLOPE)))
    lo = max(0,               idx - half_ch)
    hi = min(len(counts) - 1, idx + half_ch)
    bg_lo_l = max(0,               lo - bg_ch)
    bg_hi_r = min(len(counts) - 1, hi + 1 + bg_ch)
    bg_left  = counts[bg_lo_l:lo].mean()   if lo   > bg_lo_l else counts[lo]
    bg_right = counts[hi+1:bg_hi_r].mean() if bg_hi_r > hi+1  else counts[hi]
    n_peak   = hi - lo + 1
    bg_line  = np.linspace(bg_left, bg_right, n_peak)
    return max(0.0, float((counts[lo:hi+1] - bg_line).sum()))


# ── 1. Učitaj prova1 Ti (iz MCA fajlova) ────────────────────────
PROVA1_NPY = f"rezultati_ruotato/_npy_cache/prova1_{DET}_Ti.npy"

if os.path.exists(PROVA1_NPY):
    print("Prova1 Ti: učitavam iz cache-a...")
    ti_prova1 = np.load(PROVA1_NPY)
else:
    print(f"Prova1 Ti: čitam {W1*H1} MCA fajlova...")
    ti_prova1 = np.zeros((H1, W1), dtype=np.float64)
    _ec = {}
    for i in range(1, W1*H1 + 1):
        path = os.path.join(PROVA1_DIR, f"None_{i}.mca")
        if not os.path.exists(path):
            continue
        counts_list, in_data = [], False
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line == "<<DATA>>": in_data = True; continue
                if line == "<<END>>": break
                if in_data:
                    try: counts_list.append(int(line))
                    except: pass
        counts = np.array(counts_list, dtype=np.float64)
        n_ch = len(counts)
        if n_ch not in _ec:
            _ec[n_ch] = np.arange(n_ch) * _SLOPE + _INTERCEPT
        energy = _ec[n_ch]
        row, col = (i-1) // W1, (i-1) % W1
        ti_prova1[row, col] = bg_subtracted_integral(counts, energy, TI_KEV, TI_HW, BG_HW)
        if i % 1000 == 0 or i == W1*H1:
            print(f"  {i}/{W1*H1}")
    np.save(PROVA1_NPY, ti_prova1)
    print("  Cache sačuvan.")


# ── 2. Učitaj ruotato Ti (iz NPY cache-a) ───────────────────────
print("Ruotato Ti: učitavam iz cache-a...")
ti_ruotato = np.load(RUOTATO_NPY)   # (45, 80)


# ── 3. Resizuj prova1 → ruotato dimenzije za oduzimanje ─────────
zoom_y = HR / H1   # 45/60 = 0.75
zoom_x = WR / W1   # 80/120 = 0.667
ti_prova1_dn = zoom(ti_prova1, (zoom_y, zoom_x), order=1)   # (45, 80)
print(f"Prova1 downsampled: {ti_prova1.shape} → {ti_prova1_dn.shape}")


# ── 4. Razlika ───────────────────────────────────────────────────
diff = ti_prova1_dn - ti_ruotato


# ── 5. Crtaj ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
fig.patch.set_facecolor("white")

am = np.percentile(np.abs(diff), 99) or 1.0
im = ax.imshow(diff, cmap="RdBu_r", aspect="auto", origin="upper",
               interpolation="nearest", vmin=-am, vmax=am)
ax.set_title(f"Ti Kα  –  Razlika: Prova1 − Ruotato  |  det {DET}",
             fontsize=11, fontweight="bold", color="black", pad=6)
ax.set_xticks([]); ax.set_yticks([])
cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cbar.set_label("net counts", fontsize=8, color="black")
cbar.ax.tick_params(labelsize=7, colors="black")

plt.tight_layout()
os.makedirs(os.path.dirname(IZLAZ) or ".", exist_ok=True)
plt.savefig(IZLAZ, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nSačuvano: {os.path.abspath(IZLAZ)}")
