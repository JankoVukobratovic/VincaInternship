# VincaInternship – XRF Spektralna Analiza Freske

## O projektu

Analiza historijskog umetničkog dela metodom **XRF fluorescencije X-zraka** (X-ray Fluorescence).
Detektor skenira površinu piksel po piksel i meri koji hemijski elementi (= pigmenti) se nalaze
na svakoj tački. Na osnovu 7200 izmerenih spektara rekonstruišemo distribuciju pigmenata i
aproksimativni izgled freske.

**Dataseti:** `aurora-antico1-prova1/` i `aurora-antico1-prova2/`
**Mreža skeniranja:** 60 redova × 120 kolona = 7200 tačaka
**Vreme po tački:** 3 sekunde | **Detektori:** 10264, 19511, stacked

---

## Struktura projekta

```
VincaInternship-master/
│
├── aurora-antico1-prova1/          # Sirovi podaci – skeniranje 1
│   ├── 10264/      None_1.mca … None_7200.mca
│   ├── 19511/      None_1.mca … None_7200.mca
│   └── stacked/    None_1.mca … None_7200.mca
│
├── aurora-antico1-prova2/          # Sirovi podaci – skeniranje 2
│   ├── 10264/      None_1.mca … None_7200.mca
│   ├── 19511/      None_1.mca … None_7200.mca
│   └── stacked/    None_1.mca … None_7200.mca
│
├── main.py                         # Parsiranje .mca + spektralna heatmap analiza
├── analiza_elemenata.py            # Elementarne mape (7 el. × 3 det.) – arg: prova1|prova2
├── poredj_i_spoji.py               # Kompozit + poređenje detektora – arg: prova1|prova2
├── restauracija.py                 # Naučna rekreacija izgleda freske – arg: prova1|prova2
├── poredj_prova1_prova2.py         # Poređenje oba skeniranja (cross-dataset analiza)
│
└── rezultati/
    ├── prova1/                     # Rezultati za aurora-antico1-prova1
    │   ├── stacked/                # Elementarne mape (highest SNR)
    │   │   ├── Ca_Kalcijum_mapa.png
    │   │   ├── Ti_Titanijum_mapa.png
    │   │   ├── Fe_Gvozdje_mapa.png
    │   │   ├── Cu_Bakar_mapa.png
    │   │   ├── Pb_La_Olovo_La_mapa.png
    │   │   ├── Sn_Kalaj_mapa.png
    │   │   ├── sve_mape_pregled.png
    │   │   ├── zbirni_spektar.png
    │   │   └── README.md
    │   ├── poredj_i_spoji/         # Kompoziti i poređenje detektora
    │   │   ├── 1_aditivni_kompozit.png
    │   │   ├── 2_rgb_Fe_Cu_Pb.png
    │   │   ├── 3_poredj_svi_elementi.png
    │   │   ├── 4_preklopljeni_RG.png
    │   │   ├── 5_korelacija_scatter.png
    │   │   └── README.md
    │   └── restauracija/           # Naučna rekreacija izgleda freske
    │       ├── 1_rekonstrukcija_verzije.png
    │       ├── 2_rekonstrukcija_finalna.png
    │       ├── 3_naucni_vs_rekonstrukcija.png
    │       ├── 4_doprinosi_pigmenata.png
    │       └── README.md
    │
    ├── prova2/                     # Rezultati za aurora-antico1-prova2 (ista struktura)
    │   ├── stacked/
    │   ├── poredj_i_spoji/
    │   └── restauracija/
    │
    ├── prova1_vs_prova2/           # Cross-dataset poređenje
    │   ├── 1_mape_paralela.png     # Sve elementne mape prova1 ↔ prova2
    │   ├── 2_razlike.png           # Mape razlika prova2 − prova1
    │   ├── 3_kompoziti.png         # Aditivni kompozit oba dataseta
    │   ├── 4_rekonstrukcije.png    # Rekonstrukcija freske oba dataseta
    │   ├── 5_statistika.png        # Intenziteti i Pearsonova korelacija
    │   └── README.md
    │
    └── _npy_cache/                 # Keš numpy nizova (brzo ponovljeno pokretanje)
        ├── prova1/   *.npy
        └── prova2/   *.npy
```

---

## Python skripte

| Fajl | Opis | Pokretanje |
|------|------|-----------|
| `main.py` | Parsiranje .mca fajlova, heatmap svih spektara | `python3 main.py` |
| `analiza_elemenata.py` | 7 elementnih mapa × 3 detektora, zbirni spektri, README | `python3 analiza_elemenata.py [prova1\|prova2]` |
| `poredj_i_spoji.py` | Aditivni kompozit, RGB, poređenje detektora, scatter | `python3 poredj_i_spoji.py [prova1\|prova2]` |
| `restauracija.py` | Naučna rekreacija izgleda freske iz pigmentnih mapa | `python3 restauracija.py [prova1\|prova2]` |
| `poredj_prova1_prova2.py` | Cross-dataset poređenje: mape razlika, rekonstrukcije, statistika | `python3 poredj_prova1_prova2.py` |

> **Napomena:** `poredj_i_spoji.py`, `restauracija.py` i `poredj_prova1_prova2.py` koriste keš iz `rezultati/_npy_cache/`.
> Pokrenite prvo `analiza_elemenata.py` za odgovarajući dataset ako keš ne postoji.
>
> Podrazumevani dataset (bez argumenta) je `prova1`.

---

## Detektovani elementi i pigmenti

| Element | XRF linija | Energija | Pigment | Fresko-tehnika |
|---------|-----------|----------|---------|----------------|
| **Ca** Kalcijum | Kα | 3.69 keV | Kreč CaCO₃ | Intonaco – malterni sloj |
| **Ti** Titanijum | Kα | 4.51 keV | TiO₂ bela | MODERNA restauracija |
| **Fe** Gvožđe | Kα | 6.40 keV | Okra, hematit | Konture, senke, inkarnat |
| **Cu** Bakar | Kα | 8.05 keV | Azurit, malahit | Plava, zelena |
| **Pb** Olovo | Lα | 10.55 keV | Olovna bela, minijum | Sjajevi, bela, crvena |
| **Sn** Kalaj | Kα | 25.27 keV | Olovo-kalaj žuta | Zlatni tonovi |

---

## Ključni zaključci

1. **Figura** – jasno vidljive ruke/prsti na Fe mapi (crvena okra = konture)
2. **Pozadina** – Pb (olovna bela) kao osnova, Cu (azurit) u pozadinskim tonovima
3. **Zlatna zona** – Sn (olovo-kalaj žuta) van centralnog pravougaonog regiona
4. **Restauracija** – Ti signal ukazuje na modernu intervenciju u delu slike
5. **Oba detektora** se savršeno slažu (Fe: r=0.98, Cu: r=0.97) – pouzdani podaci
6. **Prova1 vs Prova2** – pogledati `rezultati/prova1_vs_prova2/README.md` za detalje

---

*Vinca Institute – XRF Analiza | Python 3 + NumPy + SciPy + Matplotlib*
