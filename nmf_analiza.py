"""
nmf_analiza.py
───────────────────────────────────────────────────────────────────────────────
Non-negative Matrix Factorization (NMF) spektralna dekompozicija XRF mapa.

PRINCIP:
  Matrica D (7200 piksela x 1023 kanala) se faktorise na:
      D ≈ W × H
  gde je:
      W = (7200 x K)  prostorne mape K komponenti ("abundance" mape)
      H = (K x 1023)  spektralni potpisi K komponenti ("endmembers")

  Svaka komponenta idealno odgovara jednom pigmentu ili materijalu.
  NMF automatski pronalazi:
    - Spektralne potpise (koji elementi/pikovi dominiraju)
    - Prostornu distribuciju svakog materijala
    - Skrivene materijale koji nisu eksplicitno trazeni
  Sve bez pretpostavljanja koji elementi postoje.

PREDNOSTI NAD RUCNOM ANALIZOM:
  - Nema subjektivnog biranja prozora oko pikova
  - Automatski razdvaja preklapajuce pikove
  - Otkriva korelacije izmedju elemenata (pigmentne recepture)
  - Objektivna segmentacija materijala
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import linregress
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize

# ─── Konfiguracija ───────────────────────────────────────────────────────────
DATASET_LABEL = sys.argv[1] if len(sys.argv) > 1 else 'prova1'
_DATASET_MAP  = {
    'prova1': ('aurora-antico1-prova1', 120, 60),
    'prova2': ('aurora-antico1-prova2', 120, 60),
}
DATASET_DIR, COLS, ROWS = _DATASET_MAP.get(DATASET_LABEL, (DATASET_LABEL, 120, 60))
DETEKTORI = ['10264', '19511']

IZLAZ = os.path.join('rezultati', DATASET_LABEL, 'nmf')
os.makedirs(IZLAZ, exist_ok=True)

# ─── Kalibracija: kanal -> keV ───────────────────────────────────────────────
_CAL = np.array([[219, 6.4], [278, 8.0], [363, 10.5], [436, 12.6], [869, 25.3]])
_SLOPE, _INTERCEPT, *_ = linregress(_CAL[:, 0], _CAL[:, 1])

# Poznati XRF pikovi za anotaciju
POZNATI_PIKOVI = {
    'K':     3.31,
    'Ca':    3.69,
    'Ti':    4.51,
    'Fe Ka': 6.40,
    'Fe Kb': 7.06,
    'Cu Ka': 8.05,
    'Zn Ka': 8.64,
    'Pb La': 10.55,
    'Pb Lb': 12.61,
    'Sr Ka': 14.16,
    'Sn Ka': 25.27,
}


# ══════════════════════════════════════════════════════════════════════════════
#  UCITAVANJE SIROVIH SPEKTARA
# ══════════════════════════════════════════════════════════════════════════════

def parse_mca_file(filepath):
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
                in_data = False
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


TOTAL = ROWS * COLS
CACHE_PATH = os.path.join(IZLAZ, f'spektri_matrica_{DATASET_LABEL}.npy')

if os.path.exists(CACHE_PATH):
    print(f"Ucitavam kesirane spektre iz {CACHE_PATH}...")
    D = np.load(CACHE_PATH)
    n_channels = D.shape[1]
else:
    print(f"Ucitavam {TOTAL} spektara iz oba detektora...")
    # Ucitaj prvi fajl da odredimo broj kanala
    test_path = os.path.join(DATASET_DIR, DETEKTORI[0], 'None_1.mca')
    test_data = parse_mca_file(test_path)
    n_channels = len(test_data['counts'])
    print(f"  Kanala po spektru: {n_channels}")

    D = np.zeros((TOTAL, n_channels), dtype=np.float64)

    for det in DETEKTORI:
        folder = os.path.join(DATASET_DIR, det)
        print(f"  Detektor {det}...", end=" ", flush=True)
        for i in range(1, TOTAL + 1):
            path = os.path.join(folder, f'None_{i}.mca')
            data = parse_mca_file(path)
            counts = data['counts']
            t = data['time']
            # Normalizacija na CPS (counts per second)
            cps = counts / max(t, 0.1)
            # Dodajemo (prosek dva detektora)
            D[i - 1] += cps / len(DETEKTORI)

            if i % 1000 == 0:
                print(f"{i}", end=" ", flush=True)
        print("OK")

    # Sacuvaj kes
    np.save(CACHE_PATH, D)
    print(f"  Kesirano: {CACHE_PATH}")

print(f"  Matrica spektara: {D.shape} ({D.shape[0]} piksela x {D.shape[1]} kanala)")

# Energijska osa
energy = np.arange(n_channels) * _SLOPE + _INTERCEPT


# ══════════════════════════════════════════════════════════════════════════════
#  PRETPROCESIRANJE
# ══════════════════════════════════════════════════════════════════════════════

print("Pretprocesiranje...")

# Ogranicimo na koristan opseg (1 - 30 keV) da izbacimo sum na krajevima
ch_lo = max(0, int((1.0 - _INTERCEPT) / _SLOPE))
ch_hi = min(n_channels, int((30.0 - _INTERCEPT) / _SLOPE))
D_trim = D[:, ch_lo:ch_hi]
energy_trim = energy[ch_lo:ch_hi]

# NMF zahteva ne-negativne vrednosti
D_trim = np.maximum(D_trim, 0)

print(f"  Opseg: {energy_trim[0]:.1f} - {energy_trim[-1]:.1f} keV ({D_trim.shape[1]} kanala)")
print(f"  Srednji spektar: min={D_trim.mean(axis=0).min():.1f}, max={D_trim.mean(axis=0).max():.1f} CPS")


# ══════════════════════════════════════════════════════════════════════════════
#  NMF DEKOMPOZICIJA
# ══════════════════════════════════════════════════════════════════════════════

# Testiramo razlicite K vrednosti i pratimo gresku rekonstrukcije
K_values = list(range(2, 11))
errors = []

print("\nOdredjivanje optimalnog broja komponenti...")
for k in K_values:
    model = NMF(n_components=k, init='nndsvda', max_iter=500, random_state=42)
    W = model.fit_transform(D_trim)
    err = model.reconstruction_err_
    errors.append(err)
    print(f"  K={k:2d}  greska={err:.0f}")

# ─── Slika 0: Greska rekonstrukcije vs K ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(K_values, errors, 'o-', linewidth=2, markersize=8)
ax.set_xlabel('Broj komponenti (K)', fontsize=12)
ax.set_ylabel('Greska rekonstrukcije (Frobenius norma)', fontsize=12)
ax.set_title('NMF: Odabir optimalnog broja komponenti', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Racunamo "koleno" (elbow) - tacku najveceg pada
diffs = np.diff(errors)
diffs2 = np.diff(diffs)
K_opt = K_values[np.argmax(diffs2) + 2]  # +2 za offset od duplih diff-ova
ax.axvline(K_opt, color='red', linestyle='--', alpha=0.7, label=f'Optimalno K={K_opt}')
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(IZLAZ, '0_optimalni_K.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\nOptimalni K={K_opt} (elbow metod)")


# ══════════════════════════════════════════════════════════════════════════════
#  FINALNI NMF SA OPTIMALNIM K
# ══════════════════════════════════════════════════════════════════════════════

print(f"\nFinalni NMF sa K={K_opt}...")
model_final = NMF(n_components=K_opt, init='nndsvda', max_iter=1000, random_state=42)
W_final = model_final.fit_transform(D_trim)   # (7200, K) - prostorne mape
H_final = model_final.components_              # (K, n_ch)  - spektralni potpisi

# Reshape W u prostorne mape
mape_nmf = W_final.reshape(ROWS, COLS, K_opt)  # (60, 120, K)

# Normalizacija svake komponente za vizualizaciju
for k in range(K_opt):
    peak = np.percentile(mape_nmf[:, :, k], 99)
    if peak > 0:
        mape_nmf[:, :, k] /= peak

print(f"  W: {W_final.shape} (prostorne mape)")
print(f"  H: {H_final.shape} (spektralni potpisi)")


# ══════════════════════════════════════════════════════════════════════════════
#  IDENTIFIKACIJA KOMPONENTI
# ══════════════════════════════════════════════════════════════════════════════

def identifikuj_komponentu(spektar, energy_axis):
    """
    Za dati NMF spektralni potpis, nadji dominantne pikove
    i uporedi sa poznatim XRF linijama.
    """
    # Nadji lokalne maksimume
    from scipy.signal import find_peaks
    peaks_idx, props = find_peaks(spektar, height=np.max(spektar) * 0.1,
                                  distance=5, prominence=np.max(spektar) * 0.05)
    peaks_kev = energy_axis[peaks_idx]
    peaks_h   = spektar[peaks_idx]

    # Mapiranje na poznate elemente
    identifikacije = []
    for kev, h in zip(peaks_kev, peaks_h):
        best_el, best_dist = None, 999
        for el, el_kev in POZNATI_PIKOVI.items():
            d = abs(kev - el_kev)
            if d < best_dist:
                best_dist = d
                best_el = el
        if best_dist < 0.4:  # tolerancija 0.4 keV
            identifikacije.append((best_el, kev, h))

    return identifikacije, peaks_idx


# ══════════════════════════════════════════════════════════════════════════════
#  SLIKA 1: Spektralni potpisi komponenti
# ══════════════════════════════════════════════════════════════════════════════

print("Crtam spektralne potpise komponenti...")

colors = plt.cm.Set1(np.linspace(0, 1, K_opt))
nazivi_komp = []

fig, axes = plt.subplots(K_opt, 1, figsize=(16, 3 * K_opt), sharex=True)
if K_opt == 1:
    axes = [axes]

for k in range(K_opt):
    ax = axes[k]
    spektar = H_final[k]
    ax.fill_between(energy_trim, spektar, alpha=0.3, color=colors[k])
    ax.plot(energy_trim, spektar, color=colors[k], linewidth=1.5)

    # Identifikuj pikove
    ident, _ = identifikuj_komponentu(spektar, energy_trim)
    elementi = [el for el, _, _ in ident]
    naziv = " + ".join(elementi) if elementi else f"Komponenta {k+1}"
    nazivi_komp.append(naziv)

    # Anotacija pikova
    for el, kev, h in ident:
        ax.annotate(el, xy=(kev, h), fontsize=9, fontweight='bold',
                    ha='center', va='bottom', color='black',
                    xytext=(0, 5), textcoords='offset points')

    ax.set_ylabel('Intenzitet', fontsize=10)
    ax.set_title(f'Komponenta {k+1}: {naziv}', fontsize=11,
                 fontweight='bold', color=colors[k])
    ax.set_xlim(energy_trim[0], energy_trim[-1])
    ax.grid(True, alpha=0.2)

axes[-1].set_xlabel('Energija (keV)', fontsize=12)

fig.suptitle(f'NMF spektralna dekompozicija (K={K_opt})\n'
             f'Automatski otkriveni spektralni potpisi materijala',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IZLAZ, '1_spektralni_potpisi.png'),
            dpi=180, bbox_inches='tight')
plt.close()
print("  Sacuvano: 1_spektralni_potpisi.png")


# ══════════════════════════════════════════════════════════════════════════════
#  SLIKA 2: Prostorne mape komponenti
# ══════════════════════════════════════════════════════════════════════════════

print("Crtam prostorne mape komponenti...")

n_cols_grid = min(K_opt, 4)
n_rows_grid = (K_opt + n_cols_grid - 1) // n_cols_grid
fig, axes = plt.subplots(n_rows_grid, n_cols_grid,
                         figsize=(6 * n_cols_grid, 5 * n_rows_grid))
if K_opt == 1:
    axes = np.array([axes])
axes_flat = axes.flatten()

for k in range(K_opt):
    ax = axes_flat[k]
    im = ax.imshow(mape_nmf[:, :, k], origin='upper', aspect='equal',
                   cmap='hot', interpolation='bicubic', vmin=0, vmax=1)
    ax.set_title(f'Komponenta {k+1}\n{nazivi_komp[k]}',
                 fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

for k in range(K_opt, len(axes_flat)):
    axes_flat[k].axis('off')

fig.suptitle(f'NMF prostorne mape (K={K_opt})\n'
             f'Distribucija automatski otkrivenih materijala',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IZLAZ, '2_prostorne_mape.png'),
            dpi=180, bbox_inches='tight')
plt.close()
print("  Sacuvano: 2_prostorne_mape.png")


# ══════════════════════════════════════════════════════════════════════════════
#  SLIKA 3: RGB kompozit od 3 najjace komponente
# ══════════════════════════════════════════════════════════════════════════════

print("Crtam RGB NMF kompozit...")

# Sortiraj komponente po ukupnom intenzitetu (najjace prve)
total_signal = [mape_nmf[:, :, k].sum() for k in range(K_opt)]
sorted_k = np.argsort(total_signal)[::-1]

fig, axes = plt.subplots(1, 2, figsize=(24, 9))

# RGB od top-3 komponenti
rgb_nmf = np.zeros((ROWS, COLS, 3))
for i, ch in enumerate(sorted_k[:3]):
    rgb_nmf[:, :, i] = np.clip(mape_nmf[:, :, ch], 0, 1)

axes[0].imshow(rgb_nmf, origin='upper', aspect='equal', interpolation='bicubic')
axes[0].set_title(
    f'NMF RGB kompozit\n'
    f'R={nazivi_komp[sorted_k[0]]}  |  G={nazivi_komp[sorted_k[1]]}  |  B={nazivi_komp[sorted_k[2]]}',
    fontsize=12, fontweight='bold'
)
axes[0].axis('off')

# Poredenje sa rucnim Fe/Cu/Pb RGB
from scipy.ndimage import gaussian_filter
npy_dir = os.path.join('rezultati', '_npy_cache', DATASET_LABEL)
try:
    fe_map = (np.load(os.path.join(npy_dir, '10264_Fe.npy')) +
              np.load(os.path.join(npy_dir, '19511_Fe.npy'))) / 2.0
    cu_map = (np.load(os.path.join(npy_dir, '10264_Cu.npy')) +
              np.load(os.path.join(npy_dir, '19511_Cu.npy'))) / 2.0
    pb_map = (np.load(os.path.join(npy_dir, '10264_Pb_La.npy')) +
              np.load(os.path.join(npy_dir, '19511_Pb_La.npy'))) / 2.0

    def norm_q(m, lo=5, hi=99):
        bg = np.percentile(m, lo)
        pk = np.percentile(m, hi)
        return np.clip((m - bg) / (pk - bg + 1e-10), 0, 1)

    rgb_manual = np.stack([norm_q(fe_map), norm_q(cu_map), norm_q(pb_map)], axis=2)
    axes[1].imshow(rgb_manual, origin='upper', aspect='equal', interpolation='bicubic')
    axes[1].set_title('Rucni RGB\nR=Fe | G=Cu | B=Pb',
                      fontsize=12, fontweight='bold')
except FileNotFoundError:
    axes[1].text(0.5, 0.5, 'Nema kesiranih\nelement mapa',
                 ha='center', va='center', fontsize=14, transform=axes[1].transAxes)
    axes[1].set_title('Rucni RGB (nedostupan)', fontsize=12, fontweight='bold')
axes[1].axis('off')

fig.suptitle('NMF automatska segmentacija vs rucna analiza',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IZLAZ, '3_rgb_nmf_vs_rucni.png'),
            dpi=180, bbox_inches='tight')
plt.close()
print("  Sacuvano: 3_rgb_nmf_vs_rucni.png")


# ══════════════════════════════════════════════════════════════════════════════
#  SLIKA 4: Korelaciona matrica komponenti
# ══════════════════════════════════════════════════════════════════════════════

print("Crtam korelacionu matricu...")

corr = np.corrcoef(W_final.T)  # (K x K) korelacija izmedju prostornih mapa

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
for i in range(K_opt):
    for j in range(K_opt):
        ax.text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center',
                fontsize=10, fontweight='bold',
                color='white' if abs(corr[i,j]) > 0.5 else 'black')

labels = [f'K{k+1}\n{nazivi_komp[k][:15]}' for k in range(K_opt)]
ax.set_xticks(range(K_opt))
ax.set_xticklabels(labels, fontsize=9, rotation=45, ha='right')
ax.set_yticks(range(K_opt))
ax.set_yticklabels(labels, fontsize=9)
plt.colorbar(im, ax=ax, label='Pearsonov koeficijent korelacije')
ax.set_title('Korelacija prostornih mapa NMF komponenti\n'
             'Pozitivna korelacija = isti materijal/pigment',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IZLAZ, '4_korelaciona_matrica.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Sacuvano: 4_korelaciona_matrica.png")


# ══════════════════════════════════════════════════════════════════════════════
#  SLIKA 5: Svi spektri + NMF rekonstrukcija (validacija)
# ══════════════════════════════════════════════════════════════════════════════

print("Crtam validaciju rekonstrukcije...")

# Prosecni spektar i njegova NMF rekonstrukcija
srednji_spektar = D_trim.mean(axis=0)
W_mean = W_final.mean(axis=0)  # (K,)
rekons_srednji = W_mean @ H_final  # (n_ch,)

fig, ax = plt.subplots(figsize=(16, 6))
ax.semilogy(energy_trim, srednji_spektar, 'k-', linewidth=1.5,
            label='Prosecni izmereni spektar', alpha=0.8)
ax.semilogy(energy_trim, rekons_srednji, 'r--', linewidth=1.5,
            label=f'NMF rekonstrukcija (K={K_opt})', alpha=0.8)

# Doprinos svake komponente
for k in range(K_opt):
    doprinos = W_mean[k] * H_final[k]
    ax.fill_between(energy_trim, doprinos, alpha=0.2, color=colors[k],
                    label=f'K{k+1}: {nazivi_komp[k]}')

# Anotacije pikova
for el, kev in POZNATI_PIKOVI.items():
    if energy_trim[0] <= kev <= energy_trim[-1]:
        ax.axvline(kev, color='gray', linestyle=':', alpha=0.4)
        ax.text(kev, ax.get_ylim()[1] * 0.7, el, fontsize=7,
                ha='center', rotation=90, alpha=0.6)

ax.set_xlabel('Energija (keV)', fontsize=12)
ax.set_ylabel('Intenzitet (CPS)', fontsize=12)
ax.set_title(f'Validacija NMF dekompozicije\n'
             f'Prosecni spektar = suma {K_opt} komponenti',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.2)
ax.set_xlim(energy_trim[0], energy_trim[-1])
plt.tight_layout()
plt.savefig(os.path.join(IZLAZ, '5_validacija_rekonstrukcije.png'),
            dpi=180, bbox_inches='tight')
plt.close()
print("  Sacuvano: 5_validacija_rekonstrukcije.png")


# ══════════════════════════════════════════════════════════════════════════════
#  SLIKA 6: Kombinovani pregled (spektar + mapa za svaku komponentu)
# ══════════════════════════════════════════════════════════════════════════════

print("Crtam kombinovani pregled...")

fig = plt.figure(figsize=(20, 4 * K_opt))
gs = GridSpec(K_opt, 2, width_ratios=[1.5, 1], hspace=0.35, wspace=0.2)

for k in range(K_opt):
    # Levo: spektar
    ax_sp = fig.add_subplot(gs[k, 0])
    ax_sp.fill_between(energy_trim, H_final[k], alpha=0.3, color=colors[k])
    ax_sp.plot(energy_trim, H_final[k], color=colors[k], linewidth=1.5)

    ident, _ = identifikuj_komponentu(H_final[k], energy_trim)
    for el, kev, h in ident:
        ax_sp.annotate(el, xy=(kev, h), fontsize=10, fontweight='bold',
                       ha='center', va='bottom',
                       xytext=(0, 8), textcoords='offset points',
                       arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

    ax_sp.set_xlim(1, 28)
    ax_sp.set_ylabel('Intenzitet')
    ax_sp.set_title(f'Komponenta {k+1}: {nazivi_komp[k]}',
                    fontsize=12, fontweight='bold', color=colors[k])
    ax_sp.grid(True, alpha=0.2)
    if k == K_opt - 1:
        ax_sp.set_xlabel('Energija (keV)')

    # Desno: prostorna mapa
    ax_map = fig.add_subplot(gs[k, 1])
    im = ax_map.imshow(mape_nmf[:, :, k], origin='upper', aspect='equal',
                       cmap='hot', interpolation='bicubic', vmin=0, vmax=1)
    ax_map.axis('off')
    plt.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)

fig.suptitle(f'NMF kompletna analiza: spektralni potpisi + prostorne mape\n'
             f'Dataset: {DATASET_DIR} | K={K_opt} komponenti | '
             f'{TOTAL} piksela x {n_channels} kanala',
             fontsize=14, fontweight='bold', y=1.01)
plt.savefig(os.path.join(IZLAZ, '6_kombinovani_pregled.png'),
            dpi=180, bbox_inches='tight')
plt.close()
print("  Sacuvano: 6_kombinovani_pregled.png")


# ══════════════════════════════════════════════════════════════════════════════
#  ZAVRSNI IZVESTAJ
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  NMF analiza zavrsena!")
print(f"  Rezultati: {os.path.abspath(IZLAZ)}/")
print(f"  Optimalni K: {K_opt}")
print(f"  Komponente:")
for k in range(K_opt):
    print(f"    {k+1}. {nazivi_komp[k]}")
print("=" * 60)
