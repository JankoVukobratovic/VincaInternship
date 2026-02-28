"""
poredj_prova1_prova2.py
─────────────────────────────────────────────────────────────────────────────
Poređenje dva XRF skena iste freske:
  aurora-antico1-prova1  vs  aurora-antico1-prova2

Učitava keširane numpy mape (prosek detektora 10264 + 19511)
i generiše 5 analitičkih slika + README.md u rezultati/prova1_vs_prova2/.

Pokretanje:
    python3 poredj_prova1_prova2.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr

# ─── Putanje ──────────────────────────────────────────────────────────────────
NPY1    = os.path.join('rezultati', '_npy_cache', 'prova1')
NPY2    = os.path.join('rezultati', '_npy_cache', 'prova2')
IZLAZ   = os.path.join('rezultati', 'prova1_vs_prova2')
os.makedirs(IZLAZ, exist_ok=True)

ROWS, COLS = 60, 120

# ─── Elementi (bez Pb_Lb koji je redundantan) ─────────────────────────────────
ELEMENTI = ['Ca', 'Ti', 'Fe', 'Cu', 'Pb_La', 'Sn']

NASLOVI = {
    'Ca':    'Ca – Kalcijum\n(intonaco)',
    'Ti':    'Ti – Titanijum\n(moderna rest.)',
    'Fe':    'Fe – Gvožđe\n(okra/hematit)',
    'Cu':    'Cu – Bakar\n(azurit)',
    'Pb_La': 'Pb – Olovo\n(olovna bela)',
    'Sn':    'Sn – Kalaj\n(Pb-Sn žuta)',
}

BOJE_CMAPA = {
    'Ca':    'YlOrBr',
    'Ti':    'gray',
    'Fe':    'Reds',
    'Cu':    'Blues',
    'Pb_La': 'Purples',
    'Sn':    'YlOrRd',
}

# ─── Učitavanje mapa (prosek oba detektora) ───────────────────────────────────

def ucitaj_mape(npy_dir):
    mape = {}
    for el in ELEMENTI:
        m10 = np.load(os.path.join(npy_dir, f'10264_{el}.npy'))
        m19 = np.load(os.path.join(npy_dir, f'19511_{el}.npy'))
        mape[el] = (m10 + m19) / 2.0
    return mape

print("Učitavam prova1...")
mape1 = ucitaj_mape(NPY1)
print("Učitavam prova2...")
mape2 = ucitaj_mape(NPY2)
print("Podaci učitani.\n")


# ─── Normalizacija (percentilna) ──────────────────────────────────────────────

def norm_p1p99(mapa):
    lo  = np.percentile(mapa, 1)
    hi  = np.percentile(mapa, 99)
    return np.clip((mapa - lo) / (hi - lo + 1e-10), 0, 1)

norm1 = {el: norm_p1p99(mape1[el]) for el in ELEMENTI}
norm2 = {el: norm_p1p99(mape2[el]) for el in ELEMENTI}


# ══════════════════════════════════════════════════════════════════════════════
#  SLIKA 1 – Paralela: sve mape prova1 vs prova2
# ══════════════════════════════════════════════════════════════════════════════

print("Crtam sliku 1: Paralela svih elementnih mapa...")

N = len(ELEMENTI)
fig, axes = plt.subplots(N, 2, figsize=(18, 4 * N))

for r, el in enumerate(ELEMENTI):
    cmap = BOJE_CMAPA[el]

    im1 = axes[r, 0].imshow(norm1[el], origin='upper', aspect='equal',
                             cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
    axes[r, 0].set_title(f"prova1  –  {NASLOVI[el]}", fontsize=10, fontweight='bold')
    axes[r, 0].axis('off')
    plt.colorbar(im1, ax=axes[r, 0], fraction=0.03, pad=0.02)

    im2 = axes[r, 1].imshow(norm2[el], origin='upper', aspect='equal',
                             cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
    axes[r, 1].set_title(f"prova2  –  {NASLOVI[el]}", fontsize=10, fontweight='bold')
    axes[r, 1].axis('off')
    plt.colorbar(im2, ax=axes[r, 1], fraction=0.03, pad=0.02)

fig.suptitle("Poređenje elementnih mapa: prova1 vs prova2\n"
             "(p1–p99 normalizacija, prosek detektora 10264+19511)",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IZLAZ, '1_mape_paralela.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Sačuvano: 1_mape_paralela.png")


# ══════════════════════════════════════════════════════════════════════════════
#  SLIKA 2 – Mape razlika  (prova2 − prova1) po elementu
# ══════════════════════════════════════════════════════════════════════════════

print("Crtam sliku 2: Mape razlika (prova2 − prova1)...")

fig, axes = plt.subplots(2, 3, figsize=(21, 11))
axes_f = axes.flatten()

for i, el in enumerate(ELEMENTI):
    razlika = norm2[el] - norm1[el]   # ∈ [−1, +1]

    im = axes_f[i].imshow(razlika, origin='upper', aspect='equal',
                           cmap='RdBu_r', vmin=-0.6, vmax=0.6,
                           interpolation='nearest')
    axes_f[i].set_title(f"{NASLOVI[el]}\nprova2 − prova1", fontsize=10, fontweight='bold')
    axes_f[i].axis('off')
    plt.colorbar(im, ax=axes_f[i], fraction=0.03, pad=0.02,
                 label='Δ (prova2−prova1)')

    # RMS razlika u uglu
    rms = np.sqrt(np.mean(razlika ** 2))
    axes_f[i].text(0.02, 0.03, f"RMS Δ = {rms:.3f}",
                   transform=axes_f[i].transAxes, fontsize=8,
                   color='black', va='bottom',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

fig.suptitle("Razlike u distribuciji pigmenata: prova2 − prova1\n"
             "Plavo = jači signal u prova1  |  Crveno = jači signal u prova2",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IZLAZ, '2_razlike.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Sačuvano: 2_razlike.png")


# ══════════════════════════════════════════════════════════════════════════════
#  SLIKA 3 – Aditivni RGB kompoziti oba dataset-a side-by-side
# ══════════════════════════════════════════════════════════════════════════════

print("Crtam sliku 3: Aditivni kompoziti oba dataseta...")

# Dodeljivanje boja elementima za aditivni kompozit
BOJE_RGB = {
    'Ca':    np.array([0.85, 0.75, 0.55]),   # krem
    'Ti':    np.array([0.95, 0.95, 0.95]),   # bela
    'Fe':    np.array([0.90, 0.15, 0.05]),   # crvena
    'Cu':    np.array([0.05, 0.35, 0.90]),   # plava
    'Pb_La': np.array([0.70, 0.65, 0.90]),   # lila-bela
    'Sn':    np.array([0.95, 0.80, 0.05]),   # zlatna
}

def pravi_kompozit(norme, boje):
    kompozit = np.zeros((ROWS, COLS, 3))
    for el in ELEMENTI:
        n = gaussian_filter(norme[el], sigma=0.5)
        kompozit += n[:, :, np.newaxis] * boje[el]
    return np.clip(kompozit / kompozit.max(), 0, 1)

komp1 = pravi_kompozit(norm1, BOJE_RGB)
komp2 = pravi_kompozit(norm2, BOJE_RGB)

fig, axes = plt.subplots(1, 2, figsize=(24, 9))

axes[0].imshow(komp1, origin='upper', aspect='equal', interpolation='bicubic')
axes[0].set_title("prova1 – Aditivni kompozit\n(svi elementi, iste boje)", fontsize=12, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(komp2, origin='upper', aspect='equal', interpolation='bicubic')
axes[1].set_title("prova2 – Aditivni kompozit\n(svi elementi, iste boje)", fontsize=12, fontweight='bold')
axes[1].axis('off')

# Legenda
patches = [
    mpatches.Patch(color=BOJE_RGB[el], label=f"{el}  – {NASLOVI[el].split(chr(10))[0]}")
    for el in ELEMENTI
]
fig.legend(handles=patches, loc='lower center', ncol=6, fontsize=9,
           frameon=True, bbox_to_anchor=(0.5, -0.03),
           title="Boje elemenata", title_fontsize=9)

fig.suptitle("Aditivni kompozit: prova1 vs prova2  (iste boje – direktno uporedivo)",
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(os.path.join(IZLAZ, '3_kompoziti.png'), dpi=160, bbox_inches='tight')
plt.close()
print("  Sačuvano: 3_kompoziti.png")


# ══════════════════════════════════════════════════════════════════════════════
#  SLIKA 4 – Rekonstrukcije freske side-by-side
# ══════════════════════════════════════════════════════════════════════════════

print("Crtam sliku 4: Rekonstrukcije freske side-by-side...")

PIGMENTI_BOJE = {
    'Ca':    np.array([0.93, 0.87, 0.72]),
    'Ti':    np.array([0.96, 0.96, 0.94]),
    'Fe':    np.array([0.68, 0.20, 0.03]),
    'Cu':    np.array([0.09, 0.27, 0.70]),
    'Pb_La': np.array([0.94, 0.91, 0.80]),
    'Sn':    np.array([0.84, 0.68, 0.03]),
}

TEZINE = {'Ca': 0.45, 'Ti': 0.40, 'Fe': 1.50, 'Cu': 1.40, 'Pb_La': 0.75, 'Sn': 1.15}

def norm_excess(mapa, q_lo=8, q_hi=99):
    bg   = np.percentile(mapa, q_lo)
    peak = np.percentile(mapa, q_hi)
    return np.clip((mapa - bg) / (peak - bg + 1e-10), 0, 1)

def rekonstruisi(mape):
    norme_ex = {el: norm_excess(mape[el]) for el in ELEMENTI}
    zbir_boje   = np.zeros((ROWS, COLS, 3))
    zbir_tegovi = np.zeros((ROWS, COLS))
    for el in ELEMENTI:
        n  = norme_ex[el]
        w  = n * TEZINE[el]
        zbir_boje   += w[:, :, np.newaxis] * PIGMENTI_BOJE[el]
        zbir_tegovi += w
    nazivnik  = np.where(zbir_tegovi > 1e-6, zbir_tegovi, 1.0)
    raw = np.clip(zbir_boje / nazivnik[:, :, np.newaxis], 0, 1)
    blur = np.stack([gaussian_filter(raw[:, :, c], sigma=1.5) for c in range(3)], axis=2)
    aged = np.power(np.clip(blur, 0, 1), 0.87)
    rng = np.random.default_rng(42)
    return np.clip(aged + rng.normal(0, 0.012, aged.shape), 0, 1)

print("  Računam rekonstrukciju prova1...")
rek1 = rekonstruisi(mape1)
print("  Računam rekonstrukciju prova2...")
rek2 = rekonstruisi(mape2)

fig, axes = plt.subplots(1, 2, figsize=(26, 10))

axes[0].imshow(rek1, origin='upper', aspect='equal', interpolation='bicubic')
axes[0].set_title("Rekonstrukcija freske – prova1\nPonderisane pigmentne boje (γ=0.87, σ=1.5)",
                  fontsize=12, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(rek2, origin='upper', aspect='equal', interpolation='bicubic')
axes[1].set_title("Rekonstrukcija freske – prova2\nPonderisane pigmentne boje (γ=0.87, σ=1.5)",
                  fontsize=12, fontweight='bold')
axes[1].axis('off')

fig.suptitle("Poređenje rekonstrukcija freske: prova1 vs prova2\n"
             "(Isti algoritam, iste pigmentne boje – razlike odražavaju razlike u XRF signalu)",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IZLAZ, '4_rekonstrukcije.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  Sačuvano: 4_rekonstrukcije.png")


# ══════════════════════════════════════════════════════════════════════════════
#  SLIKA 5 – Statistika: srednji intenziteti i Pearsonova korelacija
# ══════════════════════════════════════════════════════════════════════════════

print("Crtam sliku 5: Statistička poređenja...")

srednje1 = {el: mape1[el].mean() for el in ELEMENTI}
srednje2 = {el: mape2[el].mean() for el in ELEMENTI}
max1     = {el: mape1[el].max()  for el in ELEMENTI}
max2     = {el: mape2[el].max()  for el in ELEMENTI}
korel    = {el: pearsonr(mape1[el].flatten(), mape2[el].flatten())[0] for el in ELEMENTI}

x     = np.arange(len(ELEMENTI))
width = 0.35

fig, axes = plt.subplots(1, 3, figsize=(24, 7))

# ─── Srednji intenzitet ───────────────────────────────────────────────────────
bars1 = axes[0].bar(x - width/2, [srednje1[el] for el in ELEMENTI],
                    width, label='prova1', color='steelblue', alpha=0.85)
bars2 = axes[0].bar(x + width/2, [srednje2[el] for el in ELEMENTI],
                    width, label='prova2', color='tomato',    alpha=0.85)
axes[0].set_title("Srednji intenzitet (cela mreža)", fontsize=11, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(ELEMENTI, fontsize=10)
axes[0].set_ylabel("Broj broja (XRF counts)")
axes[0].legend()
axes[0].bar_label(bars1, fmt='%.0f', fontsize=7, rotation=45)
axes[0].bar_label(bars2, fmt='%.0f', fontsize=7, rotation=45)

# ─── Maksimalni intenzitet ────────────────────────────────────────────────────
bars3 = axes[1].bar(x - width/2, [max1[el] for el in ELEMENTI],
                    width, label='prova1', color='steelblue', alpha=0.85)
bars4 = axes[1].bar(x + width/2, [max2[el] for el in ELEMENTI],
                    width, label='prova2', color='tomato',    alpha=0.85)
axes[1].set_title("Maksimalni intenzitet (peak signal)", fontsize=11, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(ELEMENTI, fontsize=10)
axes[1].set_ylabel("Broj broja (XRF counts)")
axes[1].legend()
axes[1].bar_label(bars3, fmt='%.0f', fontsize=7, rotation=45)
axes[1].bar_label(bars4, fmt='%.0f', fontsize=7, rotation=45)

# ─── Pearsonova korelacija (prostorna sličnost) ───────────────────────────────
r_vals  = [korel[el] for el in ELEMENTI]
boje_r  = ['green' if r > 0.9 else 'orange' if r > 0.7 else 'red' for r in r_vals]
bars5   = axes[2].bar(x, r_vals, color=boje_r, alpha=0.85, edgecolor='black', linewidth=0.5)
axes[2].set_title("Prostorna korelacija prova1↔prova2\n(Pearsonov r, stacked)", fontsize=11, fontweight='bold')
axes[2].set_xticks(x)
axes[2].set_xticklabels(ELEMENTI, fontsize=10)
axes[2].set_ylabel("Pearsonov r")
axes[2].set_ylim(0, 1.05)
axes[2].axhline(0.9, color='green',  linestyle='--', linewidth=1, alpha=0.6, label='r=0.9')
axes[2].axhline(0.7, color='orange', linestyle='--', linewidth=1, alpha=0.6, label='r=0.7')
axes[2].legend(fontsize=9)
axes[2].bar_label(bars5, fmt='%.3f', fontsize=9)

fig.suptitle("Statistička analiza: prova1 vs prova2", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IZLAZ, '5_statistika.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Sačuvano: 5_statistika.png")


# ══════════════════════════════════════════════════════════════════════════════
#  README.md
# ══════════════════════════════════════════════════════════════════════════════

print("Generišem README.md...")

zaključci = []
for el in ELEMENTI:
    r  = korel[el]
    s1 = srednje1[el]
    s2 = srednje2[el]
    rel_prom = (s2 - s1) / (s1 + 1e-10) * 100
    if abs(rel_prom) > 20:
        smer = "viši" if rel_prom > 0 else "niži"
        zaključci.append(f"- **{el}**: signal u prova2 je {smer} za {abs(rel_prom):.1f}% u poređenju sa prova1 (r={r:.3f})")
    else:
        zaključci.append(f"- **{el}**: sličan signal u oba skeniranja (Δ={rel_prom:+.1f}%, r={r:.3f})")

zakl_str = "\n".join(zaključci)

stat_str = "\n".join([
    f"| {el:6s} | {srednje1[el]:8.0f} | {srednje2[el]:8.0f} | "
    f"{max1[el]:8.0f} | {max2[el]:8.0f} | {korel[el]:.4f} |"
    for el in ELEMENTI
])

readme = f"""# Poređenje XRF skeniranja: prova1 vs prova2

## O poređenju

Ovaj folder sadrži analizu razlika između dva XRF skeniranja iste freske:
- **prova1**: `aurora-antico1-prova1` – prvo skeniranje
- **prova2**: `aurora-antico1-prova2` – drugo skeniranje

Oba skena imaju istu mrežu (60 × 120 = 7200 tačaka), dwell = 3 s/tački, i iste detektore (10264 + 19511).

---

## Statistika

| Element | μ prova1 | μ prova2 | max prova1 | max prova2 | Korelacija r |
|---------|---------|---------|-----------|-----------|-------------|
{stat_str}

---

## Ključni zaključci

{zakl_str}

### Interpretacija korelacije

- **r > 0.90** → gotovo identična prostorna distribucija (isti pigment, ista zona)
- **r 0.70–0.90** → slična distribucija, ali sa primetnim razlikama
- **r < 0.70** → značajne razlike u rasporedu pigmenta između skeniranja

---

## Sadržaj foldera

| Fajl | Opis |
|------|------|
| `1_mape_paralela.png` | Sve elementne mape prova1 ↔ prova2 (6 elemenata × 2 dataseta) |
| `2_razlike.png` | Mape razlika: prova2 − prova1 po svakom elementu (crveno = viši u prova2) |
| `3_kompoziti.png` | Aditivni RGB kompozit oba dataseta (iste boje – direktno uporedivo) |
| `4_rekonstrukcije.png` | Rekonstrukcija freske oba dataseta (isti algoritam, direktno uporedivo) |
| `5_statistika.png` | Srednji/maks intenziteti i Pearsonova korelacija po elementu |

---

*Generisano skriptom `poredj_prova1_prova2.py`  |  VincaInstitute XRF Analiza*
"""

with open(os.path.join(IZLAZ, 'README.md'), 'w', encoding='utf-8') as f:
    f.write(readme)
print("  Sačuvano: README.md")

print("\n" + "═" * 60)
print("  Poređenje prova1 vs prova2 završeno!")
print(f"  Rezultati: {os.path.abspath(IZLAZ)}/")
print("═" * 60)
