"""
restauracija.py
─────────────────────────────────────────────────────────────────────────────
Naučna rekreacija izgleda freske na osnovu XRF pigmentnih mapa.

NAUČNA OSNOVA:
  XRF (fluorescencija X-zraka) direktno meri hemijske elemente u pigmentima.
  Svaki izmereni element odgovara specifičnom istorijskom pigmentu:

    Ca → kreč / geso (CaCO₃)     → krem-bela osnova (intonaco)
    Ti → titanijum bela (TiO₂)   → čista bela (MODERNA restauracija, ~post-1920)
    Fe → okra / hematit (Fe₂O₃)  → crveno-smeđa (konture, senke, inkarnat)
    Cu → azurit (Cu₃(CO₃)₂(OH)₂) → duboka plava (nebo, draperija)
    Pb → olovna bela (PbCO₃)     → topla bela, sjajevi, inkarnat
    Sn → olovo-kalaj žuta (Pb₂SnO₄) → zlatno-žuta (detalji, draperija)

METODA (ponderisani prosek pigmentnih boja):
  Za svaki piksel (i, j) računamo:
    boja_piksela = Σ(w_el × normirani_signal_el × RGB_boja_el) / Σ(w_el × normirani_signal_el)
  gde su w_el relativne "važnosti" svakog elementa kao hromofora.

  Ovo je fizički opravdano jer se realni pigmenti na freski
  mešaju aditivno u tankim slojevima.

OGRANIČENJA:
  ✗ Prostorna rezolucija = korak skeniranja (~2-3 mm/piksel)
  ✗ Degradacija pigmenata nije modelovana (azurit može postati zeleni malahit)
  ✗ Precizni odnosi mešanja nisu poznati – boje su aproksimativne
  ✗ Ne rekonstruiše se 3D tekstura površine
  ✓ Distribucija pigmenata i figura je naučno utemeljena
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter

# ─── Izbor dataseta ───────────────────────────────────────────────────────────
DATASET_LABEL = sys.argv[1] if len(sys.argv) > 1 else 'prova1'
_DATASET_MAP  = {
    'prova1': 'aurora-antico1-prova1',
    'prova2': 'aurora-antico1-prova2',
}
DATASET_DIR = _DATASET_MAP.get(DATASET_LABEL, DATASET_LABEL)
print(f"Dataset: {DATASET_LABEL}  →  {DATASET_DIR}")

# ─── Dimenzije mreže skeniranja ───────────────────────────────────────────────
ROWS, COLS = 60, 120

# ─── Putanje ──────────────────────────────────────────────────────────────────
NPY_DIR  = os.path.join('rezultati', '_npy_cache', DATASET_LABEL)
IZLAZ    = os.path.join('rezultati', DATASET_LABEL, 'restauracija')
os.makedirs(NPY_DIR, exist_ok=True)
os.makedirs(IZLAZ, exist_ok=True)

# ─── Istorijski verifikovane boje pigmenata (linearni RGB, [0.0 – 1.0]) ───────
# Svaka boja je aproksimacija stvarne boje pigmenta kakav bi bio kada je nanesen,
# uzimajući u obzir i naturalno starenje (žućenje veziva, patina).
PIGMENTI_BOJE = {
    'Ca':    np.array([0.93, 0.87, 0.72]),   # Krem-bela (intonaco, geso)
    'Ti':    np.array([0.96, 0.96, 0.94]),   # Čista bela (TiO₂ – moderna restauracija)
    'Fe':    np.array([0.68, 0.20, 0.03]),   # Topla crvena okra (Fe₂O₃ / FeOOH mix)
    'Cu':    np.array([0.09, 0.27, 0.70]),   # Azurit – duboka kobaltno-plava
    'Pb_La': np.array([0.94, 0.91, 0.80]),   # Olovna bela – mlečno-topla bela
    'Sn':    np.array([0.84, 0.68, 0.03]),   # Olovo-kalaj žuta – zlatno-žuta
}

# ─── Relativni tegovi (hromoforna snaga svakog elementa) ──────────────────────
# Fe i Cu daju jaku karakterističnu boju → veće težine
# Ca je pozadinski supstrat → manja težina (ne bi trebalo da "prebojavas" sve u krem)
TEZINE = {
    'Ca':    0.45,
    'Ti':    0.40,
    'Fe':    1.50,   # Jak hromofor
    'Cu':    1.40,   # Jak hromofor
    'Pb_La': 0.75,
    'Sn':    1.15,
}

ELEMENTI = list(PIGMENTI_BOJE.keys())


# ══════════════════════════════════════════════════════════════════════════════
#  UČITAVANJE PODATAKA
# ══════════════════════════════════════════════════════════════════════════════

print("Učitavam XRF mape iz keša...")

# Kombinujemo oba detektora: prosek 10264 i 19511 = ekvivalentno stacked podacima
mape = {}
for el in ELEMENTI:
    m10 = np.load(os.path.join(NPY_DIR, f'10264_{el}.npy'))
    m19 = np.load(os.path.join(NPY_DIR, f'19511_{el}.npy'))
    mape[el] = (m10 + m19) / 2.0   # Prosek oba detektora za maksimalni SNR
    print(f"  {el:6s}  μ={mape[el].mean():.0f}  max={mape[el].max():.0f}")


# ══════════════════════════════════════════════════════════════════════════════
#  NORMALIZACIJA – naglašavamo VIŠAK iznad pozadine (= pigment, ne substrat)
# ══════════════════════════════════════════════════════════════════════════════

def norm_excess(mapa, q_lo=8, q_hi=99):
    """
    Oduzima procenjeni pozadinski nivo (q_lo-ti percentil = difuzni signal supstrata)
    i normalizuje vrh distribucije na 1.0.
    Ovo izdvaja pravi pigmentni signal od pozadinskog šuma.
    """
    bg   = np.percentile(mapa, q_lo)    # Nivo pozadine
    peak = np.percentile(mapa, q_hi)    # Vrh (najintenzivnije pigmentne zone)
    return np.clip((mapa - bg) / (peak - bg + 1e-10), 0, 1)

norm = {el: norm_excess(mape[el]) for el in ELEMENTI}
print("\nNormalizacija završena.")


# ══════════════════════════════════════════════════════════════════════════════
#  REKONSTRUKCIJA BOJA – ponderisani prosek pigmentnih boja
# ══════════════════════════════════════════════════════════════════════════════

print("Računam rekonstrukciju boja (ponderisani prosek)...")

zbir_boje   = np.zeros((ROWS, COLS, 3))
zbir_tegovi = np.zeros((ROWS, COLS))

for el in ELEMENTI:
    n    = norm[el]                     # Normirani signal (60×120)
    boja = PIGMENTI_BOJE[el]            # RGB boja pigmenta (3,)
    teg  = TEZINE[el]
    w    = n * teg                      # Efektivni teg po pikselu

    zbir_boje   += w[:, :, np.newaxis] * boja
    zbir_tegovi += w

# Ponderisani prosek – zaštita od deljenja nulom
nazivnik  = np.where(zbir_tegovi > 1e-6, zbir_tegovi, 1.0)
rekons_raw = np.clip(zbir_boje / nazivnik[:, :, np.newaxis], 0, 1)

# ─── Verzija 1: sirova rekonstrukcija (bez post-obrade) ───────────────────────
v1_sirova = rekons_raw.copy()

# ─── Verzija 2: Gausovo glajzovanje σ=1.5 ─────────────────────────────────────
# Realni pigmenti imaju meke, graduirane prelaze – ne oštre granice po pikselu
v2_blur = np.stack([
    gaussian_filter(rekons_raw[:, :, c], sigma=1.5)
    for c in range(3)
], axis=2)
v2_blur = np.clip(v2_blur, 0, 1)

# ─── Verzija 3: Blago starenje + finalna tekstura ─────────────────────────────
# Gamma < 1.0 tamni sliku neznatno (simulira patinu veziva i starenje boja)
gamma    = 0.87
v3_aged  = np.power(v2_blur, gamma)

# Veoma blagi sluДajni šum (±1%) simulira mikroteksturu freske
rng       = np.random.default_rng(42)
tekstura  = rng.normal(0, 0.012, v3_aged.shape)
v3_final  = np.clip(v3_aged + tekstura, 0, 1)

print("  Sve verzije rekonstrukcije izračunate.\n")


# ══════════════════════════════════════════════════════════════════════════════
#  SLIKA 1 – Tri verzije rekonstrukcije (panel)
# ══════════════════════════════════════════════════════════════════════════════

print("Crtam sliku 1: Tri verzije rekonstrukcije...")

fig, axes = plt.subplots(1, 3, figsize=(26, 8))
verzije   = [v1_sirova, v2_blur, v3_final]
naslovi   = [
    "v1 – Sirova rekonstrukcija\n(samo ponderisane boje)",
    "v2 – Sa Gausovim glajzovanjem\n(σ = 1.5 piksela)",
    "v3 – Finalna verzija\n(glajzovanje + γ=0.87 + tekstura)",
]

for ax, v, naslov in zip(axes, verzije, naslovi):
    ax.imshow(v, origin='upper', aspect='equal', interpolation='bicubic')
    ax.set_title(naslov, fontsize=11, fontweight='bold')
    ax.axis('off')

fig.suptitle("XRF Rekonstrukcija freske – evolucija post-obrade\n"
             "Osnova: prosek detektora 10264 + 19511  |  Mreža 60×120 piksela",
             fontsize=14, fontweight='bold')

# Legenda pigmenata
patches = [
    mpatches.Patch(color=PIGMENTI_BOJE[el],
                   label=f"{el}  –  {naziv}")
    for el, naziv in [
        ('Ca',    'Kalcijum – intonaco (krem-bela)'),
        ('Ti',    'Titanijum – moderna restauracija (bela)'),
        ('Fe',    'Gvožđe – okra/hematit (crveno-smeđa)'),
        ('Cu',    'Bakar – azurit (duboka plava)'),
        ('Pb_La', 'Olovo – olovna bela (topla bela)'),
        ('Sn',    'Kalaj – olovo-kalaj žuta (zlatna)'),
    ]
]
fig.legend(handles=patches, loc='lower center', ncol=6,
           fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.04),
           title="Dodeljene boje pigmenata", title_fontsize=10)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(os.path.join(IZLAZ, '1_rekonstrukcija_verzije.png'),
            dpi=180, bbox_inches='tight')
plt.close()
print("  Sačuvano: 1_rekonstrukcija_verzije.png")


# ══════════════════════════════════════════════════════════════════════════════
#  SLIKA 2 – Finalna rekonstrukcija uvećana (visoka rezolucija)
# ══════════════════════════════════════════════════════════════════════════════

print("Crtam sliku 2: Finalna rekonstrukcija (visoka rezolucija)...")

fig, ax = plt.subplots(figsize=(22, 11))
ax.imshow(v3_final, origin='upper', aspect='equal', interpolation='bicubic')
ax.set_title(
    f"XRF Rekonstrukcija freske  –  {DATASET_DIR}\n"
    "Ponderisano mešanje istorijskih pigmentnih boja na osnovu XRF signala",
    fontsize=15, fontweight='bold', pad=15
)
ax.axis('off')

# Ugaone anotacije
ax.text(0.01, 0.99,
        "NAUČNA APROKSIMACIJA\nNe zamenjuje fotografsku dokumentaciju",
        transform=ax.transAxes, fontsize=9, color='white', va='top',
        bbox=dict(boxstyle='round', facecolor='black', alpha=0.55))

ax.text(0.99, 0.01,
        f"Rezolucija: {COLS}×{ROWS} tačaka  |  Korak: ~2-3 mm/piksel\n"
        f"Detektori: 10264 + 19511  |  Vreme/tački: 3 s",
        transform=ax.transAxes, fontsize=8, color='white', va='bottom', ha='right',
        bbox=dict(boxstyle='round', facecolor='black', alpha=0.55))

plt.tight_layout()
plt.savefig(os.path.join(IZLAZ, '2_rekonstrukcija_finalna.png'),
            dpi=220, bbox_inches='tight')
plt.close()
print("  Sačuvano: 2_rekonstrukcija_finalna.png")


# ══════════════════════════════════════════════════════════════════════════════
#  SLIKA 3 – RGB naučni vs Rekonstrukcija (poređenje)
# ══════════════════════════════════════════════════════════════════════════════

print("Crtam sliku 3: RGB naučni vs Rekonstrukcija...")

# Naučni RGB: Fe=R, Cu=G, Pb=B (direktna vizualizacija merenih kanala)
rgb_naucni = np.stack([
    norm_excess(mape['Fe']),
    norm_excess(mape['Cu']),
    norm_excess(mape['Pb_La']),
], axis=2)
rgb_naucni = np.clip(rgb_naucni, 0, 1)

fig, axes = plt.subplots(1, 2, figsize=(26, 9))

axes[0].imshow(rgb_naucni, origin='upper', aspect='equal', interpolation='bicubic')
axes[0].set_title("XRF False-Color (naučni prikaz)\n"
                  "R = Fe (Gvožđe)  |  G = Cu (Bakar)  |  B = Pb (Olovo)",
                  fontsize=12, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(v3_final, origin='upper', aspect='equal', interpolation='bicubic')
axes[1].set_title("XRF Rekonstrukcija (ponderisane pigmentne boje)\n"
                  "Svi elementi  |  Gausovo glajzovanje  |  γ=0.87",
                  fontsize=12, fontweight='bold')
axes[1].axis('off')

# Strelice koje objašnjavaju razliku
fig.text(0.5, 0.02,
         "Levo: MERENI SIGNALI mapirani na R/G/B  →  naučna vizualizacija\n"
         "Desno: MERENI SIGNALI prevedeni u realne pigmentne boje  →  aproksimacija stvarnog izgleda",
         ha='center', fontsize=11, style='italic',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

fig.suptitle("Od naučnih merenja do rekonstrukcije izgleda freske",
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(os.path.join(IZLAZ, '3_naucni_vs_rekonstrukcija.png'),
            dpi=180, bbox_inches='tight')
plt.close()
print("  Sačuvano: 3_naucni_vs_rekonstrukcija.png")


# ══════════════════════════════════════════════════════════════════════════════
#  SLIKA 4 – Doprinos svakog pigmenta (6-panelni prikaz)
# ══════════════════════════════════════════════════════════════════════════════

print("Crtam sliku 4: Doprinos svakog pigmenta posebno...")

fig, axes = plt.subplots(2, 3, figsize=(22, 13))
axes_flat = axes.flatten()

naslovi_el = {
    'Ca':    'Kalcijum (Ca)\nIntonaco – podloga',
    'Ti':    'Titanijum (Ti)\nModerna restauracija',
    'Fe':    'Gvožđe (Fe)\nOkra, hematit – konture',
    'Cu':    'Bakar (Cu)\nAzurit – plava zona',
    'Pb_La': 'Olovo (Pb)\nOlovna bela – sjajevi',
    'Sn':    'Kalaj (Sn)\nOlovo-kalaj žuta',
}

for i, el in enumerate(ELEMENTI):
    # Svaki element prikazujemo u SVOJOJ boji na crnoj pozadini
    sloj = norm[el][:, :, np.newaxis] * PIGMENTI_BOJE[el]
    sloj = np.clip(sloj, 0, 1)
    sloj_blur = np.stack([
        gaussian_filter(sloj[:, :, c], sigma=1.5) for c in range(3)
    ], axis=2)

    axes_flat[i].imshow(sloj_blur, origin='upper', aspect='equal',
                        interpolation='bicubic')
    axes_flat[i].set_title(naslovi_el[el], fontsize=11, fontweight='bold')

    # Statistika u uglu
    axes_flat[i].text(0.02, 0.03,
                      f"max signal: {mape[el].max():.0f}\nμ: {mape[el].mean():.0f}",
                      transform=axes_flat[i].transAxes, fontsize=8,
                      color='white', va='bottom',
                      bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))
    axes_flat[i].axis('off')

fig.suptitle("Doprinos svakog pigmenta rekonstrukciji\n"
             "(svaki element prikazan u svojoj karakterističnoj boji)",
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IZLAZ, '4_doprinosi_pigmenata.png'),
            dpi=160, bbox_inches='tight')
plt.close()
print("  Sačuvano: 4_doprinosi_pigmenata.png")


# ══════════════════════════════════════════════════════════════════════════════
#  README.md
# ══════════════════════════════════════════════════════════════════════════════

print("Generišem README.md...")

readme = f"""# XRF Rekonstrukcija freske – {DATASET_DIR}

## Da li je rekonstrukcija naučno opravdana?

**Da – uz jasno definisana ograničenja.**

XRF spektroskopija direktno meri koji hemijski elementi (= pigmenti) se nalaze
na svakoj tački skeniranog objekta. Budući da svaki istorijski pigment ima
poznatu hemijsku formulu i karakterističnu boju, moguće je svaki izmereni
element prevesti u odgovarajuću pigmentnu boju i rekonstruisati aproksimativni
izgled slike.

Ova metoda se koristi u muzejima i konzervatorskim institutima (npr. Louvre,
National Gallery) za **neinvazivnu analizu podslojeva** i delimičnu
rekonstrukciju degradiranih ili prekrivenih dela.

---

## Metoda rekonstrukcije

### 1. Pigmentno mapiranje

| Element | XRF linija | Pigment | Istorijska boja | Mogući kontekst |
|---------|-----------|---------|----------------|----------------|
| **Ca** | Kα 3.69 keV | Kreč (CaCO₃), geso | Krem-bela | Intonaco (podloga fresKe) |
| **Ti** | Kα 4.51 keV | Titanijum bela (TiO₂) | Čista bela | MODERNA restauracija (post-1920) |
| **Fe** | Kα 6.40 keV | Crvena okra (Fe₂O₃), žuta okra (FeOOH) | Crveno-smeđa topla | Konture, senke, inkarnat |
| **Cu** | Kα 8.05 keV | Azurit (Cu₃(CO₃)₂(OH)₂) | Duboka plava | Nebo, draperija, plavi tonovi |
| **Pb** | Lα 10.55 keV | Olovna bela (2PbCO₃·Pb(OH)₂) | Topla bela | Sjajevi, inkarnat, preparacija |
| **Sn** | Kα 25.27 keV | Olovo-kalaj žuta (Pb₂SnO₄) | Zlatno-žuta | Detalji, draperija, zlatni tonovi |

### 2. Algoritam

Za svaki piksel (i, j):
```
boja[i,j] = Σ( w_el × norm_signal_el[i,j] × RGB_pigment_el )
             ──────────────────────────────────────────────────
             Σ( w_el × norm_signal_el[i,j] )
```
Ovo je **ponderisani prosek pigmentnih boja** gde je teg srazmeran
izmerenom XRF intenzitetu na datoj tački.

### 3. Post-obrada

| Korak | Parametar | Razlog |
|-------|----------|--------|
| Gausovo glajzovanje | σ = 1.5 px | Realni pigmenti imaju meke prelaze |
| Gamma korekcija | γ = 0.87 | Simulacija starenja boja i patine |
| Teksturni šum | ±1.2% | Mikrotekstura freske |

---

## Sadržaj foldera

| Fajl | Opis |
|------|------|
| `1_rekonstrukcija_verzije.png` | Tri verzije: sirova → glajzovana → finalna |
| `2_rekonstrukcija_finalna.png` | Finalna rekonstrukcija (visoka rezolucija) |
| `3_naucni_vs_rekonstrukcija.png` | Poređenje: RGB naučni prikaz ↔ rekonstrukcija |
| `4_doprinosi_pigmenata.png` | Doprinos svakog pigmenta posebno (6 panela) |

---

## Naučna ograničenja

### Šta rekonstrukcija MOŽE pokazati:
- ✅ **Prostornu distribuciju** svakog pigmenta na površini freske
- ✅ **Konture figura** (Fe – okra crtanje) i svetle zone (Pb – olovna bela)
- ✅ **Zone restauracije** (Ti – titanijum bela ukazuje na moderna popravke)
- ✅ **Grube tonove** – gde je slika bila crvena, plava, žuta, bela
- ✅ **Strukturne detalje** – linije, draperija, anatomske konture ruku

### Šta rekonstrukcija NE MOŽE pokazati:
- ❌ Tačne nijanse boja (ne znamo tačne odnose mešanja pigmenata)
- ❌ Degradaciju (azurit može potamneti/pozeleniti, olovna bela može počrniti)
- ❌ Detalje sitnije od koraka skeniranja (~2-3 mm po pikselu)
- ❌ 3D topografiju površine (reljef, pukotine)
- ❌ Organske materijale (ulje, jaje, veziva) koji nemaju XRF signal

### Ključna napomena o veličini piksela:
Svaki piksel u rekonstrukciji odgovara jednoj skeniranoj tački (~2-3 mm²).
Za 60×120 mreže, cela skenirana površina je ~18 cm × 36 cm.

---

## Interpretacija rekonstrukcije

Na osnovu raspodele pigmenata, rekonstrukcija prikazuje:

1. **Figurativni motiv** – jasno se vidi kompozicija sa figurama (ruke, prsti)
   koji su iscrtani okrom (Fe – crveno-smeđa) na svetloj (Pb – olovna bela) podlozi

2. **Plava zona** (Cu/azurit) – verovatno nebo ili plavetni pigment u pozadini

3. **Zlatni akcenti** (Sn) – olovo-kalaj žuta na specifičnim zonama;
   tamni pravougaoni region na Sn mapi ukazuje da je ta oblast BEZ originalnog
   zlatnog pigmenta – moguće zona restauracije ili drugačiji sloj

4. **Moderna intervencija** (Ti) – prisustvo titanijum-bele ukazuje da je
   deo freske restauriran savremenim materijalom

---
*Generisano skriptom `restauracija.py`  |  Naučna aproksimacija, nije fotografska rekonstrukcija*
"""

with open(os.path.join(IZLAZ, 'README.md'), 'w', encoding='utf-8') as f:
    f.write(readme)

print("  Sačuvano: README.md")

print("\n" + "═" * 60)
print("  Restauracija završena!")
print(f"  Rezultati: {os.path.abspath(IZLAZ)}/")
print("═" * 60)
