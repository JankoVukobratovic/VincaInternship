# XRF Spektralna Analiza Freske

Analiza historijskog umetničkog dela metodom **XRF fluorescencije X-zraka**.
Detektor skenira fresku piksel po piksel i meri koji hemijski elementi
(= pigmenti) se nalaze u svakoj tački. Iz tih spektara rekonstruišemo
distribuciju pigmenata, identifikujemo modernu restauraciju i mapiramo
zone hemijskog rizika.

📄 **Naučnu analizu, validaciju i zaključke pogledati u [`IZVESTAJ.md`](IZVESTAJ.md).**

**Datasets:** `aurora-antico1-prova1`, `aurora-antico1-prova2` (120×60 = 7200 piksela)
i `aurora-antico1-ruotato` (80×45 = 3600 piksela) · **Detektori:** 10264, 19511 · **Dwell:** 3s/piksel

---

## Struktura projekta

```
VincaInternship/
├── README.md                       # ovaj fajl
├── IZVESTAJ.md                     # detaljni naučni izveštaj
├── requirements.txt
│
├── src/
│   ├── xrf_core.py                 # XRF analitički engine (run_scan API)
│   └── elements.json               # definicije elementarnih linija + boje
│
├── scripts/                        # pokretačke skripte (run from project root)
│   ├── 01_run_analysis.py          # element mape (poziva xrf_core.run_scan)
│   ├── 02_vulnerability.py         # NMF + Chemical Vulnerability Index
│   ├── 03_sam_segmentation.py      # SAM segmentacija + per-region rizik
│   ├── compare_Ti.py               # Ti uporedna mapa: prova1 vs ruotato
│   └── generate_signals.py         # po-piksel spektar plot
│
├── results/                        # izlazi za prova1/prova2 (PNG + NPY cache)
├── results_rotated/                # izlazi za ruotato skeniranje
│
├── Resources/                      # SIROVI MCA PODACI — gitignored, vidi dole
├── models/                         # SAM checkpoint — gitignored, vidi dole
│
└── xrf-denoise/                    # zaseban U-Net denoising potprojekat
```

> Sve skripte se pokreću iz korenog direktorijuma:
> `python scripts/01_run_analysis.py`

---

## Setup

### 1. Sirovi MCA podaci

Skripte očekuju sledeći raspored u `Resources/` (nije u repou — preuzeti
od Vinča instituta ili mentora):

```
Resources/
├── aurora-antico1-prova1/
│   ├── 10264/None_1.mca ... None_7200.mca
│   └── 19511/None_1.mca ... None_7200.mca
├── aurora-antico1-prova2/
│   └── (ista struktura)
└── aurora-antico1-ruotato/
    ├── 10264/None_1.mca ... None_3600.mca
    └── 19511/None_1.mca ... None_3600.mca
```

### 2. Python okruženje

```bash
pip install -r requirements.txt
```

### 3. SAM checkpoint (samo za `03_sam_segmentation.py`)

Skinuti `sam_vit_b_01ec64.pth` iz zvaničnog
[Meta Segment Anything repo-a](https://github.com/facebookresearch/segment-anything#model-checkpoints)
i staviti ga u `models/sam_vit_b_01ec64.pth`.

---

## Pokretanje

```bash
# 1) Element mape (popunjava results/_npy_cache/)
python scripts/01_run_analysis.py

# 2) Mapiranje hemijskog rizika (NMF + CVI)
python scripts/02_vulnerability.py

# 3) SAM segmentacija + per-region rizik
python scripts/03_sam_segmentation.py prova1

# Pomoćne skripte
python scripts/compare_Ti.py
python scripts/generate_signals.py
```

`02_vulnerability.py` i `03_sam_segmentation.py` koriste keš iz
`results/_npy_cache/` — pokrenuti prvo `01_run_analysis.py` da se keš popuni.

---

## xrf-denoise (potprojekat)

Zaseban eksperiment: **1D U-Net za denoising sirovih XRF spektara** pre
ekstrakcije element-mapa. Cilj je da naučni pipeline radi i sa kraćim
vremenom skeniranja (manje SNR-a). Ima svoj `src/`, `scripts/`, `experiments/`
i `requirements.txt` (PyTorch ekosistem). `scripts/05_full_pipeline.py` je
end-to-end varijanta: `denoise → element maps → NMF → CVI → SAM → risk report`.

Vidi `xrf-denoise/` za detalje.

---

## Detektovani elementi

Iz validacije u `IZVESTAJ.md` §4 i §9:

| Element | Linija | keV | Pouzdanost | Pigment / izvor |
|---------|--------|-----|-----------|-----------------|
| **Pb** Olovo | Lβ1 | 12.61 | Visoka | Olovna bela, minijum (sjajevi, bela, crvena) |
| **Ca** Kalcijum | Kα | 3.69 | Visoka | Kreč CaCO₃ (intonaco — malterni sloj) |
| **Fe** Gvožđe | Kα | 6.40 | Visoka | Okra, hematit (konture, senke, inkarnat) |
| **Cu** Bakar | Kα | 8.05 | Visoka | Azurit, malahit (plava, zelena) |
| **Ti** Titanijum | Kα | 4.51 | Visoka | TiO₂ bela — **MODERNA restauracija** |
| **Zn** Cink | Kα | 8.64 | Visoka | Lokalizovan signal (korigovan za Cu Kβ) |
| **S** Sumpor | Kα | 2.31 | Visoka (det 10264) | Gips, sulfati |
| **K** Kalijum | Kα | 3.31 | Marginalna | Slab signal, na granici detekcije |
| **Sn** Kalaj | Kα | 25.27 | Marginalna | Olovo-kalaj žuta (zlatni tonovi) |
| ~~As~~ | Kα | 10.54 | **Odsutan** | Sav signal na 10.54 keV je Pb Lα |

---

## Ključni zaključci

- **Figura** vidljiva na Fe mapi (okra konture); **pozadina** je Pb-bela + Cu-azurit.
- **Ti signal ukazuje na modernu restauraciju** — TiO₂ se ne koristi u originalnim freskama.
- **Reproducibilnost** prova1 ↔ prova2: r > 0.97 za sve jače elemente (Pb, Ca, Fe, Cu, Zn).
- **Konzistentnost** prova ↔ ruotato: isti hemijski sastav, različita geometrija skeniranja.
- **Detaljnu validaciju, kalibraciju i metodologiju vidi u [`IZVESTAJ.md`](IZVESTAJ.md).**
