# XRF Analiza — Aurora Antico 1: Kompletan Izvestaj

**Datum:** Mart 2026
**Datasets:** prova1, prova2, ruotato
**Detektori:** 10264, 19511
**Skripte:** `analiza_korigovana.py`, `analiza_ruotato.py`, `mapa_kalaj.py`

---

## 1. Struktura Projekta

```
VincaInternship/
├── aurora-antico1-prova1/      # Sirovi MCA podaci: 120×60 = 7200 piksela, dwell=3s
│   ├── 10264/None_1.mca ... None_7200.mca
│   └── 19511/None_1.mca ... None_7200.mca
├── aurora-antico1-prova2/      # Isti objekat, ponavljanje skeniranja
│   ├── 10264/ ...
│   └── 19511/ ...
├── aurora-antico1-ruotato/     # Zarotiran sken: 80×45 = 3600 piksela, dwell=3s
│   ├── 10264/ ...
│   └── 19511/ ...
├── stacked/                    # Stacked MCA fajlovi (REAL_TIME=6s), 3600 fajlova
│
├── analiza_korigovana.py       # Glavna analiza za prova1 i prova2
├── analiza_ruotato.py          # Analiza za ruotato dataset
├── mapa_kalaj.py               # Analiza Sn (kalaj, 25.27 keV)
├── generate_prova1_signals.py  # Generator spektarnih plota po pikselu
│
├── rezultati_korigovani/       # Korigovane mape elemenata za prova1/prova2
│   ├── prova1/elementi_10264.png, elementi_19511.png
│   ├── prova2/elementi_10264.png, elementi_19511.png
│   ├── razlike_prova1_vs_prova2/diff_10264.png, diff_19511.png
│   └── sn_mape/sn_prova1.png, sn_prova2.png, sn_diff.png
└── rezultati_ruotato/          # Mape za ruotato
    ├── elementi_10264.png
    ├── elementi_19511.png
    ├── elementi_suma_det.png
    ├── diff_detektora.png
    └── spektri/stacked_10264.png, stacked_19511.png
```

---

## 2. Podaci i Instrumentacija

### 2.1 MCA Fajlovi
Svaki piksel odgovara jednom `.mca` (Multi-Channel Analyzer) fajlu sa 1024 kanala. Format: tekst, podaci izmedju `<<DATA>>` i `<<END>>` markera. Svaki kanal sadrzi ceo broj impulsa.

### 2.2 Energetska Kalibracija
Linearna kalibracija dobivena iz 5 referentnih pikova:

| Kanal | Energija (keV) | Element |
|-------|----------------|---------|
| 219   | 6.4            | Mn Kα (cca) |
| 278   | 8.0            | Cu Kα   |
| 363   | 10.5           | As/Pb Lα |
| 436   | 12.6           | Pb Lβ   |
| 869   | 25.3           | Sn Kα   |

**Rezultat kalibracije:**
- Nagib (slope): **0.02917 keV/kanal**
- Odsecak (intercept): **−0.069 keV**
- R²: >0.9999 (odlicna linearnost)

Opseg: kanal 0 = −0.069 keV, kanal 1023 = 29.76 keV

### 2.3 Detektori — Razlike u Osetljivosti

| Element | Energija (keV) | det10264 / det19511 ratio |
|---------|----------------|---------------------------|
| S Kα    | 2.31           | **13.97×** (det10264 mnogo osetljiviji) |
| Ca Kα   | 3.69           | **6.14×** |
| Ti Kα   | 4.51           | **2.63×** |
| Fe Kα   | 6.40           | 1.22× |
| Cu Kα   | 8.04           | ~1.04× (gotovo isto) |
| Zn Kα   | 8.64           | 0.82× |
| Pb Lβ   | 12.61          | **0.73×** (det19511 osetljiviji) |

**Zakljucak:** Razlika je posledica razlicitih debljina Be prozora (berilium window) koji apsorbuje niskoeneregetska zracenja. Det 10264 ima tanji prozor → vece signale za S, Ca, Ti. Presek osetljivosti je oko 8 keV (Cu, Zn). Det 19511 ima deblji prozor → bolji SNR za Pb (>12 keV). Ova razlika je **intrinsicna geometrijska osobina** detektora, a ne varijacija uzorka — potvrdjeno konzistentnim ratiom u prova1, prova2 I ruotato.

---

## 3. Metodologija Ekstrakcije Signala

### 3.1 Osnovna Metoda: Linearna Subtrakcija Pozadine

Za svaki piksel i svaki element:

1. Pronadji kanal koji odgovara energiji pika (argmin |E - E_target|)
2. Definis prozor pika: ±`hw` keV oko centra
3. Definis dve boje pozadine: levi blok [lo−bg_hw, lo] i desni blok [hi, hi+bg_hw]
4. Izracunaj srednje vrednosti pozadine u oba bloka
5. Interpoliraj linearnu liniju pozadine ispod pika
6. Neto signal = Σ(counts_peak − bg_line), klipirano na 0

```
 counts
   |        /\   <- pik
   |  ___  /  \  ___
   | |BG |/    \/BG |
   |_|___|______|___|___→ kanal
       bg_lo   pik  bg_hi
```

**Prozori po elementu (prova1/prova2):**

| Element | Centar (keV) | hw (keV) | bg_hw (keV) |
|---------|-------------|----------|-------------|
| S Kα    | 2.31        | 0.20     | 0.25        |
| K Kα    | 3.3138      | specijalno (vidi §3.2) | — |
| Ca Kα   | 3.69        | 0.30     | 0.25        |
| Ti Kα   | 4.51        | 0.30     | 0.25        |
| Fe Kα   | 6.40        | 0.30     | 0.25        |
| Cu Kα   | 8.04        | 0.30     | 0.25        |
| Zn Kα   | 8.64        | 0.25     | 0.25        |
| As+PbLα | 10.54       | 0.30     | 0.25        |
| Pb Lβ1  | 12.61       | 0.30     | 0.25        |

### 3.2 Specijalna K Metoda (Tesni Sideband)

K Kα (3.3138 keV) je okruzen dvema interferencama:
- **Levo:** Ar Kα na 2.957 keV (argon iz vazduha u putu zraka) — rep ovog pika doseze do 3.2 keV. Vidljiv samo na det 10264 (tanji prozor).
- **Desno:** Ca Kα na 3.69 keV — nagib pocinje vec od 3.45 keV.

Standardni bg prozor [3.08–3.22] keV hvata rep Ar pika → precenjuje pozadinu → net K = 0.

**Resenje — tesni sideband blokovi u cistoj dolini:**
```
BG levo : [3.23, 3.28] keV  → posle Ar repa, pre K pika
K prozor: [3.296, 3.332] keV  (±2 kanala oko 3.3138 keV)
BG desno: [3.35, 3.42] keV  → posle K pika, pre Ca nagiba
```
Pozadina = prosek oba bloka po kanalu × broj kanala u piku.

### 3.3 Korekcija Zn — Doprinos Cu Kβ

Cu Kβ energija (8.903 keV) je izvan Zn prozora (8.64±0.25 keV), ali Cu ima relativno jacak Kβ. Koeficijent Cu Kβ/Kα ≈ 0.17 (standardna vrednost za Cu fluorescenciju).

**Korekcija:** `Zn_corr = max(0, Zn_raw − 0.17 × Cu_raw)`

### 3.4 Korekcija As — Doprinos Pb Lα

As Kα = 10.544 keV, Pb Lα = 10.551 keV — razdvajanje je samo **0.007 keV**, sto je daleko ispod rezolucije detektora (~0.25 keV FWHM na 10 keV). Pikovi su **apsolutno nerazdvojivi**.

**Metoda procene ratio-a Pb Lα/Lβ iz podataka:**
1. Uzmi piksele gde je As mali (donji kvartal — pretpostavljamo da tamo nema As)
2. Linearna regresija: As_raw ~ a × Pb_Lβ + b na tom skupu
3. Koeficijent `a` = Pb Lα/Lβ ratio iz podataka
4. Primeni: `As_corr = max(0, As_raw − a × Pb_Lβ)`

**Izmereni Pb Lα/Lβ ratioi:**

| Dataset    | Det 10264 | Det 19511 |
|------------|-----------|-----------|
| prova1     | 1.617     | 1.435     |
| prova2     | 1.619     | 1.440     |
| ruotato    | 1.639     | 1.461     |

Konzistentnost ratio-a kroz sve tri skeniranja (±1.5%) potvrdjuje validnost metode.

### 3.5 Korekcija S — Doprinos Pb Mα

Pb Mα = 2.346 keV, S Kα = 2.308 keV — razdvajanje 0.038 keV, takodjer ispod rezolucije.

**Isti pristup kao za As:** regresija na donji kvartal S vrednosti, estimacija Pb Mα/Lβ ratio-a, oduzimanje.

**Izmereni Pb Mα/Lβ ratioi:**

| Dataset    | Det 10264 | Det 19511 |
|------------|-----------|-----------|
| prova1     | 0.1235    | 0.0027    |
| prova2     | 0.1230    | 0.0025    |
| ruotato    | 0.1730    | 0.0019    |

**Kljucna razlika:** Pb Mα (2.35 keV) je jako niskoeneregetsko zracenje — deblji Be prozor det 19511 ga gotovo u potpunosti apsorbuje (ratio ≈ 0.002 vs 0.17 za det 10264).

---

## 4. Elementi — Validacija i Status

### 4.1 Prova1 / Prova2 (120×60 = 7200 piksela)

#### Ca Kα (3.69 keV) — VALIDAN
- Det 10264: mean=1539, max=6116, 100% piksela pozitivno
- Det 10264/19511 ratio = 6.1× (konzistentno)
- Reproducibilnost prova1/prova2: **r=0.991**, Δmean=1.6%
- Distribucija realna, mapa pokazuje jasnu prostornu strukturu

#### Ti Kα (4.51 keV) — VALIDAN
- Det 10264: mean=413, max=1903, 100% piksela
- Reproducibilnost: **r=0.959**, Δmean=1.6%
- Prostorni pattern korelisan sa Ca (r=0.236) — razlicita distribucija

#### Fe Kα (6.40 keV) — VALIDAN
- Det 10264: mean=269, max=2272, 100% piksela
- Reproducibilnost: **r=0.991**, Δmean=1.0%
- Odlicna korelacija prova1/prova2, najjaca reproducibilnost

#### Cu Kα (8.04 keV) — VALIDAN
- Det 10264: mean=213, max=1064, 70% piksela (ostalo = 0 = nema Cu)
- Reproducibilnost: **r=0.978**, Δmean=0.1%
- Jasno lokalizovan signal (ne posvuda)

#### Zn Kα (8.64 keV) — VALIDAN (korigovano)
- Pre korekcije: mean≈33, max≈3992; dopo korekcije: mean≈30, max≈3960
- Cu Kβ doprinos mali ali realan — korekcija ×0.17 smanjuje vrednosti malo
- Reproducibilnost: **r=0.985**, Δmean=0.9%
- Samo ~27% piksela pozitivno → jasno lokalizovano

#### Pb Lβ1 (12.61 keV) — VALIDAN
- Det 10264: mean=2635, max=6370, 100% piksela
- Det 19511: mean=3626, max=8829 (det 19511 osetljiviji na 12.6 keV!)
- Reproducibilnost: **r=0.987**, Δmean=0.1% — najkonzistentniji signal
- Koristi se i za korekciju As i S

#### S Kα (2.31 keV) — VALIDAN, razlicite vrednosti po detektoru
- Det 10264: mean=505 (post-corr), 100% piksela
- Det 19511: mean=50 (post-corr), 97% piksela
- Razlika 10× izmedju detektora posledica Be prozora
- Reproducibilnost: **r=0.916** (det10264), r=0.140 (det19511 — slab SNR)
- Det 19511 ima nizak SNR za S zbog absorpcije u Be prozoru

#### K Kα (3.3138 keV) — SLAB SIGNAL, PARCIJALNO VALIDAN
- Det 10264: mean=2.5, max=24.5, **48% piksela pozitivno**
- Det 19511: mean=3.3, max=36.8, **49% piksela pozitivno**
- Signal je meren ali je na granici detekcije (SNR ≈ 2-5)
- Reproducibilnost: **r≈0** — signal previse slab za piksel-to-piksel reprodukciju
- Globalne srednje vrednosti su konzistentne (Δ<4%)
- **Zakljucak:** K je prisutan u uzorku ali marginalni signal. Mapa pokazuje tendenciju ali ne individualne piksele pouzdano.

#### As Kα (10.54 keV, zajednicki prozor sa Pb Lα) — SUVISLO
- Pre korekcije: mean=3829 (det10264) — sve to je zapravo Pb Lα!
- **POSLE KOREKCIJE: mean=0.3, samo 45/7200 piksela pozitivno**
- Zakljucak: **U uzorku suvislo nema arsena.** Signal u ovom prozoru je skoro u potpunosti Pb Lα. Mapa oznacena "As+PbLα" u suštini prikazuje raspodelu olova.

#### K Kα — Beleska o Ar interferenci
Ar Kα (2.957 keV) je vidljiv u spektrima **iskljucivo na det 10264** (tanji Be prozor propusta Ar fluorescenciju iz vazduha u putu zraka). Na det 19511 nije vidljiv. Ova linija NIJE hemijski element u uzorku — to je artefakt instrumentacije.

### 4.2 Ruotato (80×45 = 3600 piksela) + Dodatne Pb Linije

Ruotato dataset sadrzi iste elemente kao prova1/prova2 — isti objekat, zarotiran ugao skeniranja.

**Dodatni elementi u ruotato analizi:**

#### Pb Ll (9.185 keV) — VALIDAN
- Det 10264: mean=115, max=391, 92% piksela
- Det 19511: mean=133, max=463, 92% piksela
- **Korelacija sa Pb Lβ: r=0.785–0.797** — potvrdjeno da je ista vrsta (Pb)
- Ova linija je realna Pb L-serija linija

#### Pb Lγ1 (14.77 keV) — VALIDAN
- Det 10264: mean=297, max=741, 100% piksela
- Det 19511: mean=468, max=1055, 100% piksela
- **Korelacija sa Pb Lβ: r=0.936–0.945** — odlicna, potvrdjeno Pb
- Det 19511 jaci signal (konzistentno sa vecom osetljivoscu na visokim energijama)

**Zasto su ove linije dodate samo u ruotato a ne i u prova1/prova2?**
Tehnoloskim razlogom — prova1/prova2 analiza je dizajnirana pre detaljnog pregleda spektara. Preporucuje se dodati PbLl i PbLg i u `analiza_korigovana.py`.

### 4.3 Elementi Koji SU PROVERAVANI I ISKLJUCENI

#### Cr Kα (5.42 keV), Mn Kα (5.90 keV), Co Kα (6.93 keV), Ni Kα (7.47 keV)
- **scipy.find_peaks() je inicijalno detektovao ove "pikove"** u sumiranom spektru
- **Proveravano na sva tri dataseta** (prova1, prova2, ruotato) sa oba detektora
- Sa pozadinskim oduzimanjem: **net signal = 0 za sve ove elemente u svim pikselima**
- **Zakljucak:** Lokalni maksimumi koje je find_peaks detektovao su talasanja Compton rasprsenosti kontinuuma, a ne pravi fluorescencioni pikovi. Ovi elementi NISU prisutni u uzorku.

#### Sn Kα (25.27 keV) — MARGINALANO
- Signal na 25.27 keV je na granici detekcije (~10-29 counts vs ~15 pozadina)
- Prostorna konzistentnost izmedju prova1 i prova2 sugeriše marginalnu realnost signala
- SNR je nedovoljan za pouzdanu kvantitativnu mapu
- **Mape generisane i sacuvane u `rezultati_korigovani/sn_mape/`**

---

## 5. Reproducibilnost: Prova1 vs Prova2

| Element | Det 10264 r | Det 19511 r | Status |
|---------|------------|------------|--------|
| Ca      | **0.991**  | **0.910**  | Odlicno |
| Fe      | **0.991**  | **0.988**  | Odlicno |
| Pb      | **0.987**  | **0.989**  | Odlicno |
| Cu      | **0.978**  | **0.973**  | Odlicno |
| Zn      | **0.985**  | **0.979**  | Odlicno |
| Ti      | **0.959**  | **0.783**  | Dobro  |
| S       | 0.916      | 0.140      | Det10264 OK, det19511 nizak SNR |
| K       | −0.014     | 0.005      | Signal previse slab za piksel korelaciju |
| As (kor.)| 0.062     | 0.023      | Suvislo nema As, suma su buka |

Srednje vrednosti signala su konzistentne izmedju prova1 i prova2 za sve jace elemente (Δ < 2%), sto potvrdjuje:
- Kalibracija je stabilna
- Uzorak se nije promenio izmedju skeniranja
- Metoda ekstrakcije je reproduktivna

---

## 6. Konzistentnost Ruotato vs Prova1/Prova2

| Element | Prova1 mean (det10264) | Ruotato mean (det10264) | Ratio |
|---------|----------------------|------------------------|-------|
| Ca      | 1539                 | 1610                   | 1.05  |
| Ti      | 413                  | 418                    | 1.01  |
| Fe      | 269                  | 318                    | 1.18  |
| Cu      | 213                  | 194                    | 0.91  |
| Pb      | 2635                 | 2551                   | 0.97  |

Svi elementi su konzistentni (±5-18%). Lagano veci Fe u ruotato moze biti posledica drugog ugla skeniranja koji prikazuje povrsinu sa vise zeleza. **Nema novih elemenata** u ruotato — isti hemijski sastav, razlicita geometrija skeniranja.

---

## 7. Razlike Mapa: Prova1 − Prova2 (Razlike Detektora)

### 7.1 Razlike Prova1 vs Prova2 (isti detektor)
Slike u `rezultati_korigovani/razlike_prova1_vs_prova2/`:
- **Crveno** = veci signal u prova1
- **Plavo** = veci signal u prova2
- **Belo/sivo** = jednak signal

Za Ca, Fe, Pb, Cu, Zn: razlike su statisticki zanemarive (r>0.98), mapa izgleda blizu uniformno sivom (beli sum). Ovo je **ocekivano** za dva skeniranja istog objekta.

### 7.2 Razlike Izmedju Detektora (ruotato)
Slika `rezultati_ruotato/diff_detektora.png`:
- Razlike su sistematske i odražavaju razlicitu osetljivost po energiji
- Det 10264 uvek visi za S, Ca, Ti; det 19511 uvek visi za Pb, PbLg
- Ovo je instrumentalni artefakt, ne prostorna varijacija uzorka

---

## 8. Kompletni Skup Generisanih Slika

### `rezultati_korigovani/`
| Slika | Sadrzaj |
|-------|---------|
| `prova1/elementi_10264.png` | 9 mapa elemenata, detektor 10264, prova1 skeniranje |
| `prova1/elementi_19511.png` | 9 mapa elemenata, detektor 19511, prova1 skeniranje |
| `prova2/elementi_10264.png` | 9 mapa elemenata, detektor 10264, prova2 skeniranje |
| `prova2/elementi_19511.png` | 9 mapa elemenata, detektor 19511, prova2 skeniranje |
| `razlike_prova1_vs_prova2/diff_10264.png` | prova1−prova2 razlika, det 10264 |
| `razlike_prova1_vs_prova2/diff_19511.png` | prova1−prova2 razlika, det 19511 |
| `sn_mape/sn_prova1.png` | Sn mapa (kalaj) za prova1, oba detektora |
| `sn_mape/sn_prova2.png` | Sn mapa (kalaj) za prova2, oba detektora |
| `sn_mape/sn_diff.png` | Sn razlika prova1−prova2 |

Skala boja: **crna = 0 (nema elementa), zasicena boja = max signal**. Svaka mapa normalizovana na 99. percentil da se eliminisu sporadicni outlieri.

### `rezultati_ruotato/`
| Slika | Sadrzaj |
|-------|---------|
| `elementi_10264.png` | 10 mapa elemenata (ukljucuje PbLl, PbLg), det 10264 |
| `elementi_19511.png` | 10 mapa elemenata, det 19511 |
| `elementi_suma_det.png` | Suma oba detektora (bolji SNR) |
| `diff_detektora.png` | Razlika det10264 − det19511 |
| `spektri/stacked_10264.png` | Sumovani spektar svih 3600 piksela, det 10264 |
| `spektri/stacked_19511.png` | Sumovani spektar svih 3600 piksela, det 19511 |

---

## 9. Identifikovani Elementi u Uzorku

Na osnovu svih analiza, u uzorku su potvrdjeni sledeci elementi:

| Element | Pouzdanost | Napomena |
|---------|-----------|---------|
| **Pb (olovo)** | Visoka | Lβ1, Ll, Lγ1 sve konzistentne; dominantan element |
| **Ca (kalcijum)** | Visoka | Jak signal, odlicna reprodukcija |
| **Fe (gvozdje)** | Visoka | Jak signal, odlicna reprodukcija |
| **S (sumpor)** | Visoka (det10264) | Realan signal, manja osetljivost det19511 |
| **Cu (bakar)** | Visoka | Jak, lokalizovan signal |
| **Ti (titanijum)** | Visoka | Jak signal |
| **Zn (cink)** | Visoka | Lokalizovan, korigovan za Cu Kβ |
| **K (kalijum)** | Marginalana | Slab signal, na granici detekcije |
| **Sn (kalaj)** | Marginalna | Verovatno trag, na granici detekcije |
| **As (arsen)** | Nema | Signal bio skoro iskljucivo Pb Lα; posle korekcije <50/7200 piksela |

**Karakterizacija uzorka:** Aurora Antico 1 je verovatno **olovom bogata bronzana smesa** (Pb, Cu, Zn), sa kalcijumskim i gvozdenim primesama, verovatno nanosene kao pigment ili prajmer. Visoki Ca i Fe uz S sugerisu prisustvo gipsa ili zemljanog pigmenta. Visoki Pb uz Cu/Zn ukazuje na olovnu bronzu ili minium (Pb3O4) kao pigment.

---

## 10. Tehnicke Napomene i Ogranicenja

1. **Energetska rezolucija:** FWHM detektora ~0.25 keV na 10 keV. Pikovi razmaknuti manje od 0.25 keV su nerazdvojivi (As/Pb Lα, S/Pb Mα).

2. **NPY cache sistem:** Sirovi integrali se cuvaju u `_npy_cache/*.npy`. Korekcije se primenjuju u memoriji pri svakom pokretanju. Cache ubrzava ponovni racun 100×.

3. **Negativni signali:** Posle pozadinskog oduzimanja i korekcija, sve vrednosti su klipirane na 0 (fizicki minimum).

4. **Vizualizacija:** `vmin=0, vmax=percentile(99)` — gornji 1% se klipira da bi slabiji signali bili vidljivi na fonu jakih.

5. **K pouzdanost:** K signal je slab (SNR 2-5). Tesni sideband metod je neophodan (standardna metoda daje net=0 zbog Ar i Ca interferencija), ali cak i sa ovim pristupom individualni pikseli nisu pouzdani. Suma/srednja vrednost po regijama je pouzdana.

---

## 11. Uraðeno — Hronologicni Pregled

1. **Inicijalna analiza** (`main.py`, `better_main.py`) — fiksni prozor bez bg oduzimanja, pogresno K target (3.00 keV umesto 3.31 keV). Rezultati u `rezultati_novi/`. *(Obrisano kao zastarelo)*

2. **Identifikacija problema:**
   - K target je bio pogresan — K Kα na 3.3138 keV, ne 3.0 keV
   - Standardni bg prozor za K je zahvatao Ar Kα rep i Ca Kα nagib → net K = 0 uvek
   - Ni bg oduzimanja nije bilo → mape su pokazivale sirovi integral (sa pozadinom)
   - Zn je sadrzao Cu Kβ doprinos
   - As prozor je sadrzao Pb Lα doprinos
   - S prozor je sadrzao Pb Mα doprinos

3. **Korigovana analiza** (`analiza_korigovana.py`) — pozadinsko oduzimanje za sve elemente, tesni sideband K metoda, Zn−CuKβ, As−PbLα, S−PbMα korekcije. Rezultati u `rezultati_korigovani/`.

4. **Sn mape** (`mapa_kalaj.py`) — Sn Kα na 25.27 keV, slab signal ali prostorno konzistentan.

5. **Prova1 signal plotovi** (`generate_prova1_signals.py`) — generisani spektri po pikselu sa anotiranim elementima (375MB, obrisano kao zastarelo).

6. **Ruotato analiza** (`analiza_ruotato.py`) — inicijalno ukljucivala Cr/Mn/Co/Ni, potom verifikovano da su Compton artefakti, uklonjeni. Dodate PbLl i PbLg linije. Cache izbrisan i ponovo generisan sa ispravnim elementima.

7. **Validacija cross-dataset:** Prova1 vs prova2 vs ruotato — konzistentni elementi, konzistentni ratio-i, konzistentna osetljivost detektora. Potvrdjeno: svi tri skeniranja su isti objekat.

8. **Reorganizacija projekta:** Ruotato data premestena u `aurora-antico1-ruotato/`. Obrisano: `main.py`, `better_main.py`, `razlika_po_detektoru.py`, `restauracija.py`, `analiza_novi.py`, `map.png`, `rezultati/`, `rezultati_novi/`, `prova1_signal/`.

---

*Izvestaj generisan automatski na osnovu numericke analize svih NPY cache fajlova i skripti.*
