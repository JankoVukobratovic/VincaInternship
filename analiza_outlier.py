"""
analiza_outlier.py
──────────────────────────────────────────────────────────────────────
Izoluje spektre gde je razlika prova1 − prova2 drasticna po pikselu.

Za svaki element:
  - Racuna diff[piksel] = prova1_signal − prova2_signal
  - Ova razlika treba da bude ~0 (konstantna) ako je uzorak isti
  - Pikseli gde |diff| > prag (99. percentil) se smatraju outlierima
  - Za top-10 outlier piksela: plota MCA spektar iz prova1 i prova2
    sa anotiranim elementima

Izlaz: rezultati_korigovani/outlier/<element>/piksel_N.png
"""

import os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ── Kalibracija ────────────────────────────────────────────────
CAL = np.array([[219,6.4],[278,8.0],[363,10.5],[436,12.6],[869,25.3]])
SLOPE, INTERCEPT, *_ = linregress(CAL[:,0], CAL[:,1])

ELEMENTS = {
    "S":  2.31, "K":  3.3138, "Ca": 3.69, "Ti": 4.51,
    "Fe": 6.40, "Cu": 8.04,  "Zn": 8.64, "Pb": 12.61,
}
ELEMENT_COLORS = {
    "S": "#FFFF44", "K": "#FFD700", "Ca": "#FFFFFF", "Ti": "#FF9966",
    "Fe": "#FF2200", "Cu": "#00FFAA", "Zn": "#66CCFF",
    "Pb Lα": "#FF8800", "Pb Lβ": "#CC66FF", "Pb Ll": "#BB88FF", "Pb Lγ": "#9933AA",
    "Ar": "#888888",
}
ALL_LINES = [
    ("Ar",    2.957, "#888888"),
    ("S",     2.31,  "#FFFF44"),
    ("K",     3.314, "#FFD700"),
    ("Ca",    3.69,  "#FFFFFF"),
    ("Ti",    4.51,  "#FF9966"),
    ("Cr",    5.415, "#00DD55"),
    ("Fe",    6.40,  "#FF2200"),
    ("Cu",    8.05,  "#00FFAA"),
    ("Zn",    8.64,  "#66CCFF"),
    ("Pb Ll", 9.185, "#BB88FF"),
    ("Pb Lα", 10.54, "#FF8800"),
    ("Pb Lβ", 12.61, "#CC66FF"),
    ("Pb Lγ", 14.77, "#9933AA"),
]

CACHE   = "rezultati_korigovani/_npy_cache"
DIR1    = "aurora-antico1-prova1"
DIR2    = "aurora-antico1-prova2"
DETEKTORI = ["10264", "19511"]
W, H    = 120, 60
IZLAZ   = "rezultati_korigovani/outlier"
TOP_N   = 10       # koliko outlier spektara po elementu
PRAG_PERCENTIL = 99.0

os.makedirs(IZLAZ, exist_ok=True)


# ── Parsiranje MCA ─────────────────────────────────────────────
def parse_mca(path):
    counts, ind = [], False
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if ln == "<<DATA>>": ind = True; continue
            if ln == "<<END>>": break
            if ind:
                try: counts.append(int(ln))
                except: pass
    return np.array(counts, dtype=np.float64)


# ── Plot spektra za jedan piksel ───────────────────────────────
def plot_spectrum_pair(pixel_i, det, elem_key, diff_val, sig1, sig2,
                       path1, path2, save_path):
    row = int((pixel_i - 1) // W) + 1
    col = int((pixel_i - 1) % W) + 1

    c1 = parse_mca(path1) if os.path.exists(path1) else np.zeros(1024)
    c2 = parse_mca(path2) if os.path.exists(path2) else np.zeros(1024)
    n  = max(len(c1), len(c2), 1024)
    e  = np.arange(n) * SLOPE + INTERCEPT

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), dpi=100)
    fig.patch.set_facecolor("#111111")

    for ax, counts, prova_lbl in zip(axes, [c1, c2], ["prova1", "prova2"]):
        ax.set_facecolor("#1A1A1A")
        en = e[:len(counts)]
        ax.plot(en, counts, color="#AADDFF", linewidth=0.6, alpha=0.9)

        ymax = counts.max() if counts.max() > 0 else 1.0
        for lbl, kev, col in ALL_LINES:
            if kev > en[-1]: continue
            ax.axvline(kev, color=col, linewidth=1.0, alpha=0.75, linestyle="--")
            idx = int(np.argmin(np.abs(en - kev)))
            y   = max(counts[max(0,idx-3):idx+4].max() * 1.5, ymax * 0.05)
            ax.text(kev, y, lbl, color=col, fontsize=6, ha="center",
                    va="bottom", rotation=90, clip_on=True)

        ax.set_xlim(1.5, 15)
        ax.set_xlabel("Energija (keV)", color="white", fontsize=8)
        ax.set_ylabel("Counts", color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#555555")
        ax.set_title(
            f"{prova_lbl}  |  Det {det}  |  Piksel {pixel_i} (r={row}, k={col})\n"
            f"{elem_key}: {(sig1 if prova_lbl=='prova1' else sig2):.0f} counts",
            color="white", fontsize=8)

    fig.suptitle(
        f"OUTLIER: {elem_key}  |  Det {det}  |  Piksel {pixel_i}\n"
        f"Δ(prova1−prova2) = {diff_val:+.0f} counts  "
        f"(prova1={sig1:.0f}, prova2={sig2:.0f})",
        color="#FF6666", fontsize=9, fontweight="bold"
    )
    plt.tight_layout(rect=[0,0,1,0.88])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()


# ── Glavna petlja ──────────────────────────────────────────────
print("Outlier analiza: prova1 vs prova2")
print(f"Parametri: top-{TOP_N} piksela po elementu, prag={PRAG_PERCENTIL}. percentil\n")

summary = []   # (element, det, pixel_i, diff)

for det in DETEKTORI:
    print(f"=== Detektor {det} ===")
    for elem in ELEMENTS:
        p1 = os.path.join(CACHE, f"prova1_{det}_{elem}.npy")
        p2 = os.path.join(CACHE, f"prova2_{det}_{elem}.npy")
        if not os.path.exists(p1) or not os.path.exists(p2):
            print(f"  [{elem}] cache ne postoji — preskacam")
            continue

        m1 = np.load(p1)   # shape (H, W)
        m2 = np.load(p2)
        diff = (m1 - m2).flatten()   # prova1 − prova2

        prag = np.percentile(np.abs(diff), PRAG_PERCENTIL)
        outlier_idx = np.where(np.abs(diff) > prag)[0]

        # Sortiraj po |diff| opadajuce, uzmi top-N
        outlier_idx = outlier_idx[np.argsort(-np.abs(diff[outlier_idx]))][:TOP_N]

        print(f"  [{elem}]  mean_diff={diff.mean():+.1f}  std={diff.std():.1f}  "
              f"prag={prag:.0f}  outlieri={len(np.where(np.abs(diff)>prag)[0])}/7200")

        out_dir = os.path.join(IZLAZ, f"{elem}_{det}")
        os.makedirs(out_dir, exist_ok=True)

        for rank, flat_i in enumerate(outlier_idx):
            pixel_i = flat_i + 1   # 1-indeksirano
            row_i   = flat_i // W
            col_i   = flat_i  % W

            sig1 = float(m1.flatten()[flat_i])
            sig2 = float(m2.flatten()[flat_i])
            dv   = diff[flat_i]

            path1 = os.path.join(DIR1, det, f"None_{pixel_i}.mca")
            path2 = os.path.join(DIR2, det, f"None_{pixel_i}.mca")

            save = os.path.join(out_dir, f"rank{rank+1:02d}_piksel{pixel_i}.png")
            plot_spectrum_pair(pixel_i, det, elem, dv, sig1, sig2,
                               path1, path2, save)
            summary.append((elem, det, pixel_i, dv, sig1, sig2))

        print(f"         Sacuvano {len(outlier_idx)} spektara u {os.path.relpath(out_dir)}/")


# ── Sumarni plot: |diff| histogram po elementu ─────────────────
print("\n--- Generisanje sumarnih histograma ---")
n_elem = len(ELEMENTS)
nc = 4
nr = math.ceil(n_elem * 2 / nc)   # po 2 detektora

fig, axes = plt.subplots(math.ceil(n_elem / nc), nc * 2,
                         figsize=(5 * nc * 2, 3.5 * math.ceil(n_elem / nc) + 1),
                         dpi=90)
fig.patch.set_facecolor("white")
axes = np.array(axes).flatten()

ax_i = 0
for elem in ELEMENTS:
    for det in DETEKTORI:
        p1 = os.path.join(CACHE, f"prova1_{det}_{elem}.npy")
        p2 = os.path.join(CACHE, f"prova2_{det}_{elem}.npy")
        if not os.path.exists(p1) or not os.path.exists(p2):
            if ax_i < len(axes): axes[ax_i].set_visible(False)
            ax_i += 1; continue

        m1 = np.load(p1).flatten()
        m2 = np.load(p2).flatten()
        diff = m1 - m2

        ax = axes[ax_i]
        prag = np.percentile(np.abs(diff), PRAG_PERCENTIL)
        ax.hist(diff, bins=80, color="#4488FF", alpha=0.75, edgecolor="none")
        ax.axvline( prag, color="red",  linewidth=1.2, linestyle="--", label=f"+prag {prag:.0f}")
        ax.axvline(-prag, color="red",  linewidth=1.2, linestyle="--")
        ax.axvline(0,     color="black", linewidth=0.8)
        ax.set_title(f"{elem}  det{det}", fontsize=8, fontweight="bold")
        ax.set_xlabel("Δ counts (p1−p2)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax_i += 1

for j in range(ax_i, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Distribucija razlika prova1 − prova2 po elementu i detektoru\n"
             "Crvena isprekidana linija = prag (99. percentil |razlike|)",
             fontsize=11, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
hist_path = os.path.join(IZLAZ, "histogram_razlika.png")
plt.savefig(hist_path, dpi=110, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  Sacuvano: {os.path.relpath(hist_path)}")


# ── Sumarni tekstualni izvestaj ────────────────────────────────
print(f"\n{'='*60}")
print(f"  OUTLIER ANALIZA ZAVRSENA")
print(f"  Ukupno izolovanih spektara: {len(summary)}")
print(f"  Sve slike u: {os.path.abspath(IZLAZ)}/")
print(f"{'='*60}")
print("\nTop outlieri (|Δ| najveci):")
summary_sorted = sorted(summary, key=lambda x: abs(x[3]), reverse=True)[:20]
for elem, det, pix, dv, s1, s2 in summary_sorted:
    print(f"  {elem:3s} det{det}  piksel={pix:5d}  Δ={dv:+8.0f}  (p1={s1:.0f}, p2={s2:.0f})")
