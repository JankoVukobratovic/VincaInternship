"""
generate_prova1_signals.py
──────────────────────────────────────────────────────────────
Za svaki MCA fajl iz prova1 (oba detektora: 10264 i 19511)
generise spektar plot i oznacava prisutne elemente.

Izlaz:
  prova1_signal/10264/None_N.png
  prova1_signal/19511/None_N.png

Peak detekcija po pikselu:
  element se obelezava ako je intenzitet u prozoru za dati piksel
  veci od 1.5 × (5-ti percentil mape za taj element) = pozadina.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ── Kalibracija ───────────────────────────────────────────────
CAL = np.array([[219, 6.4], [278, 8.0], [363, 10.5], [436, 12.6], [869, 25.3]])
SLOPE, INTERCEPT, *_ = linregress(CAL[:, 0], CAL[:, 1])

# ── Elementi sa tacknim energijama linija ──────────────────────
# kev_true  = tacna energija fluorescencione linije
# code_key  = kljuc elementa u analiza_novi ELEMENT_MAP (za npy ucitavanje)
# note      = opciona napomena (overlap itd.)
ELEMENTS = [
    {"label": "K Kα",   "kev": 3.3138,  "code_key": "K",  "color": "#FFD700"},
    {"label": "Ca Kα",  "kev": 3.6917,  "code_key": "Ca", "color": "#FFFFFF"},
    {"label": "Ti Kα",  "kev": 4.5109,  "code_key": "Ti", "color": "#FF9966"},
    {"label": "Fe Kα",  "kev": 6.4038,  "code_key": "Fe", "color": "#FF3300"},
    {"label": "Cu Kα",  "kev": 8.0478,  "code_key": "Cu", "color": "#00FFAA"},
    {"label": "Zn Kα",  "kev": 8.6389,  "code_key": "Zn", "color": "#66CCFF"},
    {"label": "As/Pb Lα","kev": 10.545, "code_key": "As", "color": "#FF8800"},
    {"label": "Pb Lβ",  "kev": 12.6137, "code_key": "Pb", "color": "#CC66FF"},
]

# ── Ucitaj NPY cache (prag = 1.5 × 5-ti percentil) ───────────
NPY_DIR    = "rezultati_novi/_npy_cache"
DETEKTORI  = ["10264", "19511"]
PROVA      = "prova1"

bg_thresh = {}   # bg_thresh[(det, code_key)] = prag
npy_maps  = {}   # npy_maps[(det, code_key)]  = 2D array (60×120)

for det in DETEKTORI:
    for el in ELEMENTS:
        key = el["code_key"]
        path = os.path.join(NPY_DIR, f"{PROVA}_{det}_{key}.npy")
        if os.path.exists(path):
            m = np.load(path)
            npy_maps[(det, key)]  = m
            bg_thresh[(det, key)] = 1.5 * np.percentile(m, 5)
        else:
            npy_maps[(det, key)]  = None
            bg_thresh[(det, key)] = float("inf")

# ── Parsiranje MCA fajla ───────────────────────────────────────
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


# ── Glavna petlja ──────────────────────────────────────────────
PROVA_DIR = "aurora-antico1-prova1"
W, H      = 120, 60
TOTAL     = W * H
OUT_BASE  = "prova1_signal"

# Unapred kreiraj energy os (ista za sve fajlove iste duzine)
_energy_cache = {}

fig, ax = plt.subplots(figsize=(10, 4))

for det in DETEKTORI:
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

        # ── Nacrtaj spektar ────────────────────────────────────
        ax.clear()
        ax.set_facecolor("#111111")
        fig.patch.set_facecolor("#1A1A1A")

        ax.plot(energy, counts, color="#AADDFF", linewidth=0.6, alpha=0.9)

        # ── Obelezi elemente ───────────────────────────────────
        ymax = counts.max() if counts.max() > 0 else 1.0

        for el in ELEMENTS:
            key   = el["code_key"]
            ekev  = el["kev"]
            color = el["color"]
            label = el["label"]

            # Da li je element prisutan u ovom pikselu?
            m     = npy_maps.get((det, key))
            thr   = bg_thresh.get((det, key), float("inf"))
            val   = m[row, col] if m is not None else 0.0
            present = (val >= thr)

            if present:
                ax.axvline(ekev, color=color, linewidth=1.2, alpha=0.9,
                           linestyle="-")
                ax.text(ekev, ymax * 0.97, label,
                        color=color, fontsize=6, ha="center", va="top",
                        rotation=90, clip_on=True)
            else:
                # Prikazuje se kao slaba isprekidana linija
                ax.axvline(ekev, color=color, linewidth=0.5, alpha=0.35,
                           linestyle="--")

        ax.set_xlim(0, 20)
        ax.set_xlabel("Energija (keV)", color="white", fontsize=8)
        ax.set_ylabel("Counts", color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#555555")

        ax.set_title(
            f"Prova1 | Detektor {det} | Piksel {i} (red={row+1}, kol={col+1})",
            color="white", fontsize=8, pad=4
        )

        # Legenda samo sa prisutnim elementima
        present_labels = [
            el["label"] for el in ELEMENTS
            if npy_maps.get((det, el["code_key"])) is not None
            and npy_maps[(det, el["code_key"])][row, col] >= bg_thresh.get((det, el["code_key"]), float("inf"))
        ]
        if present_labels:
            from matplotlib.lines import Line2D
            handles = [
                Line2D([0], [0], color=el["color"], linewidth=1.5, label=el["label"])
                for el in ELEMENTS
                if el["label"] in present_labels
            ]
            ax.legend(handles=handles, loc="upper right", fontsize=6,
                      facecolor="#333333", edgecolor="#555555",
                      labelcolor="white", framealpha=0.7)

        out_path = os.path.join(out_dir, f"None_{i}.png")
        fig.savefig(out_path, dpi=72, bbox_inches="tight",
                    facecolor=fig.get_facecolor())

        if i % 500 == 0 or i == TOTAL:
            print(f"  [{det}] {i}/{TOTAL} ({100*i//TOTAL}%)", flush=True)

plt.close(fig)
print("\nSvi plotovi sacuvani u:", os.path.abspath(OUT_BASE))
