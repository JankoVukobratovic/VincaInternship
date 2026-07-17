"""
generate_signals.py
──────────────────────────────────────────────────────────────
Render one annotated spectrum plot per MCA file of prova1
(both detectors: 10264 and 19511).

Output:
  signals_prova1/10264/None_N.png
  signals_prova1/19511/None_N.png

Per-pixel peak detection:
  an element is marked "present" in a pixel when the integrated window
  intensity exceeds 1.5 x (5th percentile of that element's map)
  — i.e. clearly above the background level.

Requires the raw MCA files under Resources/ and the element-map cache
(any of the known cache layouts). Run from the project root:
    python scripts/generate_signals.py
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import linregress

# ── Calibration ───────────────────────────────────────────────
CAL = np.array([[219, 6.4], [278, 8.0], [363, 10.5], [436, 12.6], [869, 25.3]])
SLOPE, INTERCEPT, *_ = linregress(CAL[:, 0], CAL[:, 1])

# ── Elements with exact line energies ─────────────────────────
# kev      : exact fluorescence-line energy
# code_key : element key used in the .npy cache file names
ELEMENTS = [
    {"label": "K Kα",     "kev": 3.3138,  "code_key": "K",  "color": "#FFD700"},
    {"label": "Ca Kα",    "kev": 3.6917,  "code_key": "Ca", "color": "#FFFFFF"},
    {"label": "Ti Kα",    "kev": 4.5109,  "code_key": "Ti", "color": "#FF9966"},
    {"label": "Fe Kα",    "kev": 6.4038,  "code_key": "Fe", "color": "#FF3300"},
    {"label": "Cu Kα",    "kev": 8.0478,  "code_key": "Cu", "color": "#00FFAA"},
    {"label": "Zn Kα",    "kev": 8.6389,  "code_key": "Zn", "color": "#66CCFF"},
    {"label": "Pb Lα",    "kev": 10.545,  "code_key": "PbLa", "color": "#FF8800"},
    {"label": "Pb Lβ",    "kev": 12.6137, "code_key": "PbLb", "color": "#CC66FF"},
]

DETECTORS = ["10264", "19511"]
PROVA     = "prova1"
PROVA_DIR = next(
    (p for p in (os.path.join("Resources", "aurora-antico1-prova1"),
                 "aurora-antico1-prova1")
     if os.path.isdir(p)),
    os.path.join("Resources", "aurora-antico1-prova1"),
)
W, H      = 120, 60
TOTAL     = W * H
OUT_BASE  = "signals_prova1"

if not os.path.isdir(PROVA_DIR):
    sys.exit(f"ERROR: raw MCA data not found in {PROVA_DIR} "
             "(see README: Setup / Raw MCA data).")

# ── Element-map cache (threshold = 1.5 x 5th percentile) ──────
_ALIASES = {"PbLa": ["PbLa", "Pb_La"], "PbLb": ["PbLb", "Pb_Lb"]}

def _find_map(det: str, key: str):
    for name in _ALIASES.get(key, [key]):
        for path in (
            os.path.join("results", det, "_npy_cache", f"{PROVA}_{name}.npy"),
            os.path.join("results", "_npy_cache", PROVA, f"{det}_{name}.npy"),
            os.path.join("results", "_npy_cache", f"{PROVA}_{det}_{name}.npy"),
        ):
            if os.path.exists(path):
                return np.load(path)
    return None


bg_thresh = {}   # (det, code_key) -> presence threshold
npy_maps  = {}   # (det, code_key) -> 2D map (60x120) or None

for det in DETECTORS:
    for el in ELEMENTS:
        key = el["code_key"]
        m = _find_map(det, key)
        npy_maps[(det, key)]  = m
        bg_thresh[(det, key)] = (1.5 * np.percentile(m, 5)
                                 if m is not None else float("inf"))

# ── MCA parsing ────────────────────────────────────────────────
def parse_mca(filepath):
    counts, in_data = [], False
    real_time = 3.0
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
            elif line.startswith("REAL_TIME"):
                try:
                    real_time = float(line.split(" - ", 1)[1])
                except Exception:
                    pass
    return np.array(counts, dtype=np.float64), real_time


# ── Main loop ──────────────────────────────────────────────────
_energy_cache = {}

fig, ax = plt.subplots(figsize=(10, 4))

for det in DETECTORS:
    out_dir = os.path.join(OUT_BASE, det)
    os.makedirs(out_dir, exist_ok=True)

    for i in range(1, TOTAL + 1):
        mca_path = os.path.join(PROVA_DIR, det, f"None_{i}.mca")
        if not os.path.exists(mca_path):
            continue

        counts, real_time = parse_mca(mca_path)
        n_ch = len(counts)
        if n_ch not in _energy_cache:
            _energy_cache[n_ch] = np.arange(n_ch) * SLOPE + INTERCEPT
        energy = _energy_cache[n_ch]

        row = (i - 1) // W
        col = (i - 1) % W

        # ── Spectrum ───────────────────────────────────────────
        ax.clear()
        ax.set_facecolor("#111111")
        fig.patch.set_facecolor("#1A1A1A")

        ax.plot(energy, counts, color="#AADDFF", linewidth=0.6, alpha=0.9)

        # ── Element markers ────────────────────────────────────
        ymax = counts.max() if counts.max() > 0 else 1.0
        present_els = []

        for el in ELEMENTS:
            key   = el["code_key"]
            m     = npy_maps.get((det, key))
            thr   = bg_thresh.get((det, key), float("inf"))
            val   = m[row, col] if m is not None else 0.0

            if val >= thr:
                present_els.append(el)
                ax.axvline(el["kev"], color=el["color"], linewidth=1.2,
                           alpha=0.9, linestyle="-")
                ax.text(el["kev"], ymax * 0.97, el["label"],
                        color=el["color"], fontsize=6, ha="center", va="top",
                        rotation=90, clip_on=True)
            else:
                # absent: faint dashed reference line
                ax.axvline(el["kev"], color=el["color"], linewidth=0.5,
                           alpha=0.35, linestyle="--")

        ax.set_xlim(0, 20)
        ax.set_xlabel("Energy (keV)", color="white", fontsize=8)
        ax.set_ylabel("Counts", color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#555555")

        ax.set_title(
            f"prova1  |  detector {det}  |  pixel {i} "
            f"(row {row + 1}, col {col + 1})",
            color="white", fontsize=8, pad=4,
        )

        if present_els:
            handles = [Line2D([0], [0], color=el["color"], linewidth=1.5,
                              label=el["label"]) for el in present_els]
            ax.legend(handles=handles, loc="upper right", fontsize=6,
                      facecolor="#333333", edgecolor="#555555",
                      labelcolor="white", framealpha=0.7)

        fig.savefig(os.path.join(out_dir, f"None_{i}.png"), dpi=72,
                    bbox_inches="tight", facecolor=fig.get_facecolor())

        if i % 500 == 0 or i == TOTAL:
            print(f"  [{det}] {i}/{TOTAL} ({100 * i // TOTAL}%)", flush=True)

plt.close(fig)
print("\nAll plots saved to:", os.path.abspath(OUT_BASE))
