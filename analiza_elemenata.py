"""
analiza_elemenata.py
────────────────────────────────────────────────────────────────────────────────
Kompleksna XRF analiza za sve detektore i sve elemente.

Pokretanje:
  python3 analiza_elemenata.py prova1   # aurora-antico1-prova1 (podrazumevano)
  python3 analiza_elemenata.py prova2   # aurora-antico1-prova2

Izlaz:
  rezultati/{dataset}/10264/   – mape, zbirni spektar, README.md
  rezultati/{dataset}/19511/   – mape, zbirni spektar, README.md
  rezultati/{dataset}/stacked/ – mape, zbirni spektar, README.md
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ─── Dataset selekcija ────────────────────────────────────────────────────────
# Pokretanje: python3 analiza_elemenata.py prova1  ili  prova2
DATASET_LABEL = sys.argv[1] if len(sys.argv) > 1 else 'prova1'
_DATASET_MAP  = {
    'prova1': 'aurora-antico1-prova1',
    'prova2': 'aurora-antico1-prova2',
}
DATASET_DIR = _DATASET_MAP.get(DATASET_LABEL, DATASET_LABEL)
print(f"Dataset: {DATASET_LABEL}  →  {DATASET_DIR}")

# ─── Konfiguracija mreže skeniranja (iz colonneXrighe.txt) ────────────────────
ROWS   = 60    # Broj redova skeniranja
COLS   = 120   # Broj kolona skeniranja
UKUPNO = ROWS * COLS  # 7200 tačaka ukupno

# ─── Kalibracione tačke (iste za oba detektora) ───────────────────────────────
# Format: [kanal, energija_keV]
# Svaka tačka odgovara poznatoj XRF liniji elementa koji je prisutan u uzorku
KALIB = np.array([
    [219,  6.400],   # Fe Kα  – gvožđe
    [278,  8.046],   # Cu Kα  – bakar
    [363, 10.551],   # Pb Lα  – olovo
    [436, 12.614],   # Pb Lβ  – olovo (druga linija)
    [869, 25.271],   # Sn Kα  – kalaj
])

# Linearna regresija kalibracije: E(keV) = a * kanal + b
# R² treba biti > 0.9999 jer je XRF kalibracija gotovo savršeno linearna
_a, _b, _r, _, _ = linregress(KALIB[:, 0], KALIB[:, 1])

def kanal_za_energiju(e_kev):
    """Pretvara energiju u keV u broj kanala koristeći kalibracionu pravu."""
    return int(round((e_kev - _b) / _a))

def energija_za_kanal(kanal):
    """Pretvara broj kanala u energiju u keV."""
    return _a * kanal + _b

print(f"Kalibracija: E = {_a:.5f} × kanal + {_b:.4f} keV  (R² = {_r**2:.6f})")

# ─── Definicija elemenata za mapiranje ────────────────────────────────────────
# Za svaki element definišemo:
#   naziv      – kratka oznaka za ime fajla
#   puni_naziv – pun naziv elementa na srpskom
#   linija     – XRF linija (Kα, Kβ, Lα, Lβ...)
#   energija   – tabelarna energija linije u keV
#   kanal      – izračunati centralni kanal (iz kalibracione prave)
#   prozor     – polu-širina prozora integracije (±prozor kanala)
#   pigmenti   – mogući pigmenti/minerali u kojima se element nalazi
#   paleta     – paleta boja za prikaz mape
ELEMENTI = [
    {
        "naziv":      "Ca",
        "puni_naziv": "Kalcijum",
        "linija":     "Kα",
        "energija":   3.692,
        "kanal":      kanal_za_energiju(3.692),
        "prozor":     15,
        "pigmenti":   "Kreč (CaCO₃), gips (CaSO₄·2H₂O) – osnova maltera i preparacije",
        "paleta":     "Blues",
    },
    {
        "naziv":      "Ti",
        "puni_naziv": "Titanijum",
        "linija":     "Kα",
        "energija":   4.510,
        "kanal":      kanal_za_energiju(4.510),
        "prozor":     10,
        "pigmenti":   "Titanijum bela (TiO₂) – moguće moderno restauratorsko premazivanje",
        "paleta":     "Purples",
    },
    {
        "naziv":      "Fe",
        "puni_naziv": "Gvozdje",
        "linija":     "Kα",
        "energija":   6.400,
        "kanal":      219,      # Tačna kalibraciona tačka – bez zaokruživanja
        "prozor":     10,
        "pigmenti":   "Crvena okra (Fe₂O₃, hematit), žuta okra (FeOOH, getit), umbra",
        "paleta":     "YlOrRd",
    },
    {
        "naziv":      "Cu",
        "puni_naziv": "Bakar",
        "linija":     "Kα",
        "energija":   8.046,
        "kanal":      278,      # Tačna kalibraciona tačka
        "prozor":     10,
        "pigmenti":   "Azurit (Cu₃(CO₃)₂(OH)₂), malahit (Cu₂CO₃(OH)₂), verdigris – plava/zelena",
        "paleta":     "Greens",
    },
    {
        "naziv":      "Pb_La",
        "puni_naziv": "Olovo La",
        "linija":     "Lα",
        "energija":   10.551,
        "kanal":      363,      # Tačna kalibraciona tačka
        "prozor":     10,
        "pigmenti":   "Olovna bela (2PbCO₃·Pb(OH)₂), minijum (Pb₃O₄) – bela/crvena boja",
        "paleta":     "hot",
    },
    {
        "naziv":      "Pb_Lb",
        "puni_naziv": "Olovo Lb",
        "linija":     "Lβ",
        "energija":   12.614,
        "kanal":      436,      # Tačna kalibraciona tačka
        "prozor":     10,
        "pigmenti":   "Druga spektralna linija olova – potvrda i provera Pb signala",
        "paleta":     "afmhot",
    },
    {
        "naziv":      "Sn",
        "puni_naziv": "Kalaj",
        "linija":     "Kα",
        "energija":   25.271,
        "kanal":      869,      # Tačna kalibraciona tačka
        "prozor":     15,
        "pigmenti":   "Olovo-kalaj žuta (Pb₂SnO₄) – zlatno-žuta boja, česta u starim freskama",
        "paleta":     "copper",
    },
]

# Ispisujemo izračunate kanale za proveru
print("\nIzračunati kanali po elementu:")
print(f"  {'Element':<12} {'Linija':<6} {'Energija':>10}  {'Kanal':>6}  {'Prozor':>8}")
print(f"  {'-'*50}")
for el in ELEMENTI:
    print(f"  {el['puni_naziv']:<12} {el['linija']:<6} {el['energija']:>8.3f} keV  "
          f"ch {el['kanal']:>4}  ±{el['prozor']} ch")

# ─── Putanje ──────────────────────────────────────────────────────────────────
BAZA_DIR    = DATASET_DIR                               # npr. aurora-antico1-prova1
DETEKTORI   = ['10264', '19511', 'stacked']
REZULTATI   = os.path.join('rezultati', DATASET_LABEL)  # rezultati/prova1/ ili prova2/


# ══════════════════════════════════════════════════════════════════════════════
#  POMOĆNE FUNKCIJE
# ══════════════════════════════════════════════════════════════════════════════

def parse_mca_counts(filepath):
    """
    Čita samo DATA sekciju .mca fajla i vraća niz odbroja po kanalu.
    Brza verzija – ne parsira metapodatke ni kalibraciju.
    """
    counts = []
    in_data = False
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line == "<<DATA>>":
                in_data = True   # Ušli smo u sekciju sa podacima
                continue
            if line == "<<END>>":
                break            # Kraj fajla – prekidamo čitanje
            if in_data and line:
                counts.append(int(line))
    return np.array(counts, dtype=np.int32)


def window_integral(counts, centar, polusirina):
    """
    Pravougaoni prozor integracije: sabira odbroje u opsegu
    [centar - polusirina, centar + polusirina].
    Handles granice niza – neće preći van 0 ili len(counts)-1.
    """
    lo = max(0, centar - polusirina)
    hi = min(len(counts) - 1, centar + polusirina)
    return int(counts[lo : hi + 1].sum())


# ══════════════════════════════════════════════════════════════════════════════
#  OBRADA JEDNOG DETEKTORA
# ══════════════════════════════════════════════════════════════════════════════

def obradi_detektor(ime_detektora):
    """
    Čita sve None_N.mca fajlove za zadati detektor.

    Vraća:
      mape          – rečnik {naziv_elementa: 2D numpy niz (ROWS × COLS)}
      zbirni_spektar – zbir svih 7200 spektara (1D numpy niz)
    """
    data_dir = os.path.join(BAZA_DIR, ime_detektora)

    # Inicijalizacija: prazna mapa za svaki element
    mape = {el["naziv"]: np.zeros((ROWS, COLS), dtype=np.float64) for el in ELEMENTI}
    zbirni_spektar = None  # Biće inicijalizovan na prvu veličinu fajla

    for n in range(1, UKUPNO + 1):
        filepath = os.path.join(data_dir, f"None_{n}.mca")

        # Računamo red i kolonu iz linearnog indeksa (skeniranje levo→desno, gore→dole)
        row = (n - 1) // COLS
        col = (n - 1) % COLS

        try:
            counts = parse_mca_counts(filepath)

            # Akumuliramo sve spektre u zbirni (za vizualizaciju ukupnog spektra)
            if zbirni_spektar is None:
                zbirni_spektar = np.zeros(len(counts), dtype=np.float64)
            zbirni_spektar += counts

            # Za svaki element izračunavamo integral pika i upisujemo u mapu
            for el in ELEMENTI:
                mape[el["naziv"]][row, col] = window_integral(
                    counts, el["kanal"], el["prozor"]
                )

        except FileNotFoundError:
            # Fajl ne postoji – ostavljamo 0 u mapi
            pass
        except Exception as e:
            print(f"  Greška u None_{n}.mca: {e}")

        # Napredak: ispisujemo svakih 1000 fajlova
        if n % 1000 == 0:
            print(f"  Obrađeno: {n}/{UKUPNO} fajlova ({n*100//UKUPNO}%)")

    return mape, zbirni_spektar


# ══════════════════════════════════════════════════════════════════════════════
#  FUNKCIJE ZA VIZUALIZACIJU
# ══════════════════════════════════════════════════════════════════════════════

def crtaj_mapu_elementa(mapa_2d, el, detektor_ime, putanja):
    """
    Crta distribucionu mapu jednog elementa i čuva je kao PNG.

    Parametri boje:
      - paleta definisana u ELEMENTI rečniku
      - bele/svetle oblasti = visoka koncentracija elementa
      - tamne oblasti = niska koncentracija
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(
        mapa_2d,
        origin='upper',           # Red 0 na vrhu – odgovara redosledu skeniranja
        aspect='equal',           # Pikseli kvadratni (1:1 aspekt)
        cmap=el["paleta"],
        interpolation='nearest'   # Bez interpolacije – originalne vrednosti
    )

    # Traka boja sa opisom
    cbar = fig.colorbar(im, ax=ax, shrink=0.75)
    cbar.set_label(
        f'{el["puni_naziv"]} ({el["naziv"]} {el["linija"]}) odbroji\n'
        f'Prozor: kan. {el["kanal"] - el["prozor"]}–{el["kanal"] + el["prozor"]}'
        f'  ({energija_za_kanal(el["kanal"] - el["prozor"]):.2f}–'
        f'{energija_za_kanal(el["kanal"] + el["prozor"]):.2f} keV)',
        fontsize=9
    )

    ax.set_title(
        f'Mapa distribucije: {el["puni_naziv"]} ({el["naziv"]} {el["linija"]}, '
        f'{el["energija"]:.3f} keV)\n'
        f'Detektor: {detektor_ime}  |  {el["pigmenti"]}',
        fontsize=10
    )
    ax.set_xlabel('Kolona (0–119)', fontsize=10)
    ax.set_ylabel('Red (0–59)', fontsize=10)

    # Statistike u uglu grafika
    srednja = mapa_2d.mean()
    std     = mapa_2d.std()
    cv      = std / srednja * 100 if srednja > 0 else 0
    tekst_stat = (
        f'min  = {mapa_2d.min():.0f}\n'
        f'max  = {mapa_2d.max():.0f}\n'
        f'μ    = {srednja:.1f}\n'
        f'σ    = {std:.1f}\n'
        f'CV   = {cv:.0f}%'
    )
    ax.text(
        0.01, 0.02, tekst_stat,
        transform=ax.transAxes,
        fontsize=8, verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8)
    )

    plt.tight_layout()
    plt.savefig(putanja, dpi=150, bbox_inches='tight')
    plt.close(fig)


def crtaj_pregled_svih(mape, detektor_ime, putanja):
    """
    Crta sve elementne mape u jednom velikom pregledu (grid 2 × 4).
    Koristan za brzo poređenje prostorne distribucije svih elemenata.
    """
    n_el  = len(ELEMENTI)
    n_col = 4
    n_row = (n_el + n_col - 1) // n_col   # Zaokruživanje gore

    fig, axes = plt.subplots(n_row, n_col, figsize=(22, n_row * 5.5))
    axes_flat = axes.flatten()

    for i, el in enumerate(ELEMENTI):
        mapa = mape[el["naziv"]]
        im = axes_flat[i].imshow(
            mapa, origin='upper', aspect='auto',
            cmap=el["paleta"], interpolation='nearest'
        )
        axes_flat[i].set_title(
            f'{el["puni_naziv"]}\n{el["naziv"]} {el["linija"]} – {el["energija"]:.2f} keV',
            fontsize=9
        )
        axes_flat[i].set_xlabel('Kolona', fontsize=7)
        axes_flat[i].set_ylabel('Red', fontsize=7)
        plt.colorbar(im, ax=axes_flat[i], shrink=0.85)

        # Kratke statistike ispod svakog grafika
        srednja = mapa.mean()
        axes_flat[i].set_xlabel(
            f'Kolona   [μ={srednja:.0f}, max={mapa.max():.0f}]', fontsize=7
        )

    # Sakrijemo prazne panele (ako broj elemenata nije deljiv sa n_col)
    for j in range(len(ELEMENTI), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f'Pregled distribucije svih {n_el} elemenata – Detektor {detektor_ime}\n'
        f'Mreža {ROWS}×{COLS}, {UKUPNO} tačaka',
        fontsize=14, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    plt.savefig(putanja, dpi=120, bbox_inches='tight')
    plt.close(fig)


def crtaj_zbirni_spektar(spektar, detektor_ime, putanja):
    """
    Crta zbir svih 7200 spektara sa logaritamskom Y-osom.
    Vertikalne linije označavaju XRF pikove svakog elementa koji analiziramo.
    Ovaj grafik potvrđuje koje elemente zaista možemo detektovati.
    """
    N        = len(spektar)
    kanali   = np.arange(N)
    energije = _a * kanali + _b   # Kalibracija kanala → keV

    # Zamenjujemo nule i negatives sa 0.1 da bi log skala radila
    spektar_plot = np.where(spektar > 0, spektar, 0.1)

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.semilogy(energije, spektar_plot, color='steelblue', lw=0.7, alpha=0.9,
                label='Zbirni spektar')

    # Paleta boja za vertikalne oznake elemenata
    BOJE_EL = ['royalblue', 'mediumpurple', 'crimson', 'forestgreen',
               'darkorange', 'peru', 'saddlebrown']

    y_vrh = spektar.max()

    for el, boja in zip(ELEMENTI, BOJE_EL):
        e = el["energija"]
        if e <= 30:
            # Vertikalna isprekidana linija na energiji elementa
            ax.axvline(e, color=boja, linestyle='--', alpha=0.75, lw=1.3)

            # Sivi pojas koji prikazuje prozor integracije
            e_lo = energija_za_kanal(el["kanal"] - el["prozor"])
            e_hi = energija_za_kanal(el["kanal"] + el["prozor"])
            ax.axvspan(e_lo, e_hi, alpha=0.08, color=boja)

            # Tekstualna oznaka elementa iznad linije
            ax.text(
                e + 0.15, y_vrh * 0.4,
                f'{el["naziv"]}\n{el["linija"]}\n{e:.2f} keV',
                fontsize=7, color=boja, va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6)
            )

    ax.set_xlim(0, 30)
    ax.set_ylim(1, y_vrh * 20)
    ax.set_xlabel('Energija (keV)', fontsize=12)
    ax.set_ylabel('Ukupni odbroji – logaritamska skala', fontsize=12)
    ax.set_title(
        f'Zbirni spektar svih {UKUPNO} tačaka – Detektor {detektor_ime}\n'
        f'(Sivi pojas = prozor integracije za svaki element)',
        fontsize=12
    )
    ax.grid(True, which='both', alpha=0.25, ls=':')

    plt.tight_layout()
    plt.savefig(putanja, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  GENERISANJE README.md ZA SVAKI DETEKTOR
# ══════════════════════════════════════════════════════════════════════════════

def generisi_readme(detektor_ime, mape, putanja):
    """
    Generiše README.md fajl sa:
    - listom sadržaja foldera
    - statističkim rezimeom svake elementne mape
    - automatskim zaključcima o distribuciji elemenata
    """
    redovi = []

    redovi.append(f"# Rezultati XRF analize – Detektor {detektor_ime}\n\n")
    redovi.append(f"**Dataset:** aurora-antico1-prova1  \n")
    redovi.append(f"**Mreža skeniranja:** {ROWS} redova × {COLS} kolona = **{UKUPNO} tačaka**  \n")
    redovi.append(f"**Vreme merenja po tački:** 3 sekunde  \n")
    redovi.append(f"**Kalibracija:** E(keV) = {_a:.5f} × kanal + {_b:.4f}  (R² = {_r**2:.6f})  \n\n")

    redovi.append("---\n\n")
    redovi.append("## Sadržaj foldera\n\n")
    redovi.append("| Fajl | Element | XRF linija | Energija | Pigmenti/Minerali |\n")
    redovi.append("|------|---------|-----------|----------|-------------------|\n")

    for el in ELEMENTI:
        ime_f = f"{el['naziv']}_{el['puni_naziv'].replace(' ', '_')}_mapa.png"
        redovi.append(
            f"| `{ime_f}` | **{el['puni_naziv']}** | {el['naziv']} {el['linija']} "
            f"| {el['energija']:.3f} keV | {el['pigmenti']} |\n"
        )

    redovi.append("| `zbirni_spektar.png` | – | – | 0–30 keV | Zbir svih spektara sa označenim pikovima |\n")
    redovi.append("| `sve_mape_pregled.png` | – | – | – | Svih 7 mapa u jednom pregledu (grid) |\n\n")

    redovi.append("---\n\n")
    redovi.append("## Statistike po elementu\n\n")
    redovi.append(
        "| Element | Linija | Kanal | Prozor | Min | Max | Srednja | Std | CV% | Ocena distribucije |\n"
    )
    redovi.append("|---------|--------|-------|--------|-----|-----|---------|-----|-----|-------------------|\n")

    zakljucci    = []
    uniformni    = []
    neujednaceni = []

    for el in ELEMENTI:
        mapa    = mape[el["naziv"]]
        mn      = mapa.min()
        mx      = mapa.max()
        srednja = mapa.mean()
        std     = mapa.std()
        cv      = std / srednja * 100 if srednja > 0 else 0

        # Automatska ocena distribucije na osnovu koeficijenta varijacije (CV)
        if cv > 40:
            ocena = "Jako neravnomerna – lokalni pigmenti"
            neujednaceni.append(el)
        elif cv > 20:
            ocena = "Umereno neravnomerna – mešovita"
            neujednaceni.append(el)
        else:
            ocena = "Uniformna – pozadinska komponenta"
            uniformni.append(el)

        redovi.append(
            f"| **{el['puni_naziv']}** | {el['linija']} | {el['kanal']} | "
            f"±{el['prozor']} | {mn:.0f} | {mx:.0f} | {srednja:.1f} | "
            f"{std:.1f} | {cv:.0f}% | {ocena} |\n"
        )

        # Posebna napomena za elemente sa izrazitim lokalnim pikovima
        if mx > srednja + 3 * std:
            zakljucci.append(
                f"- **{el['puni_naziv']} ({el['naziv']} {el['linija']})**: "
                f"max = {mx:.0f} (>{srednja + 3*std:.0f} = μ+3σ) – "
                f"jasne zone visokog intenziteta → {el['pigmenti']}"
            )

    redovi.append("\n---\n\n")
    redovi.append("## Zaključci\n\n")
    redovi.append("### Elementi sa neravnomernom distribucijom (pigmenti)\n\n")

    if neujednaceni:
        for el in neujednaceni:
            redovi.append(
                f"- **{el['puni_naziv']} ({el['naziv']} {el['linija']}, "
                f"{el['energija']:.2f} keV)**: {el['pigmenti']}\n"
            )
    else:
        redovi.append("- Nema elemenata sa izrazito neravnomernom distribucijom.\n")

    redovi.append("\n### Elementi sa uniformnom distribucijom (supstrat/pozadina)\n\n")
    if uniformni:
        for el in uniformni:
            redovi.append(
                f"- **{el['puni_naziv']} ({el['naziv']} {el['linija']}, "
                f"{el['energija']:.2f} keV)**: {el['pigmenti']}\n"
            )
    else:
        redovi.append("- Svi elementi pokazuju neravnomernu distribuciju.\n")

    if zakljucci:
        redovi.append("\n### Elementi sa izrazitim lokalnim pikovima (μ + 3σ kriterijum)\n\n")
        redovi.extend([z + "\n" for z in zakljucci])

    redovi.append("\n---\n\n")
    redovi.append("## Napomena o pigmentima u staroj fresko-tehnici\n\n")
    redovi.append(
        "| Element | Moguće poreklo |\n"
        "|---------|---------------|\n"
        "| **Ca (Kalcijum)** | Osnova malternog sloja (intonaco) od kreča (CaCO₃). "
        "Uniformna distribucija je normalna. |\n"
        "| **Ti (Titanijum)** | TiO₂ pigment je moderan (uveden ~1920). Prisustvo ukazuje "
        "na restauraciju ili moderni premaz. |\n"
        "| **Fe (Gvožđe)** | Najstariji pigmenti – okre i umbra. Crvena (Fe₂O₃) i žuta (FeOOH). |\n"
        "| **Cu (Bakar)** | Azurit i malahit – plava i zelena boja. Karakteristični za "
        "sredn­jovekovne freske. |\n"
        "| **Pb (Olovo)** | Olovna bela (PbCO₃·Pb(OH)₂) i minijum (Pb₃O₄). "
        "Lα i Lβ treba da koreliraju. |\n"
        "| **Sn (Kalaj)** | Olovo-kalaj žuta (Pb₂SnO₄) – zlatni tonovi, "
        "tipični za gotiku i renesansu. |\n\n"
    )

    redovi.append("---\n")
    redovi.append("*Generisano automatski skriptom `analiza_elemenata.py`*\n")

    with open(putanja, 'w', encoding='utf-8') as f:
        f.writelines(redovi)


# ══════════════════════════════════════════════════════════════════════════════
#  GLAVNI DEO PROGRAMA
# ══════════════════════════════════════════════════════════════════════════════

for detektor in DETEKTORI:
    print(f"\n{'═'*64}")
    print(f"  Obrada detektora: {detektor}")
    print(f"{'═'*64}")

    # Kreiramo izlazni folder ako ne postoji
    izlaz_dir = os.path.join(REZULTATI, detektor)
    os.makedirs(izlaz_dir, exist_ok=True)

    # ── Faza 1: Čitanje svih fajlova ──────────────────────────────────────────
    print(f"  Čitam {UKUPNO} .mca fajlova – može potrajati...")
    mape, zbirni = obradi_detektor(detektor)
    print(f"  Svi fajlovi obrađeni.")

    # ── Faza 2: Mapa za svaki element ─────────────────────────────────────────
    for el in ELEMENTI:
        ime_f   = f"{el['naziv']}_{el['puni_naziv'].replace(' ', '_')}_mapa.png"
        putanja = os.path.join(izlaz_dir, ime_f)
        crtaj_mapu_elementa(mape[el["naziv"]], el, detektor, putanja)
        srednja = mape[el["naziv"]].mean()
        mx      = mape[el["naziv"]].max()
        print(f"  [{el['naziv']:6s}] Sačuvano: {ime_f}  (μ={srednja:.0f}, max={mx:.0f})")

    # ── Faza 3: Pregled svih mapa u jednoj slici ──────────────────────────────
    p_pregled = os.path.join(izlaz_dir, 'sve_mape_pregled.png')
    crtaj_pregled_svih(mape, detektor, p_pregled)
    print(f"  Sačuvano: sve_mape_pregled.png")

    # ── Faza 4: Zbirni spektar ────────────────────────────────────────────────
    if zbirni is not None:
        p_spektar = os.path.join(izlaz_dir, 'zbirni_spektar.png')
        crtaj_zbirni_spektar(zbirni, detektor, p_spektar)
        print(f"  Sačuvano: zbirni_spektar.png")

    # ── Faza 5: README.md ─────────────────────────────────────────────────────
    p_readme = os.path.join(izlaz_dir, 'README.md')
    generisi_readme(detektor, mape, p_readme)
    print(f"  Sačuvano: README.md")

print(f"\n{'═'*64}")
print("  Analiza završena!")
print(f"  Rezultati su u folderu: {os.path.abspath(REZULTATI)}/")
print(f"{'═'*64}")
