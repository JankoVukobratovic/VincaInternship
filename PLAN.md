# Plan rada — Geometry-resolved Characterization of a Dual-detector MA-XRF Scanner

Radni plan za drugi rad (abstract: `abstract.tex` / `abstract2.tex`,
autori: `authors.md`). Posao je podeljen na dva paralelna koloseka
(Osoba A i Osoba B) sa samo dve tačke primopredaje, tako da niko nikog
ne čeka.

---

## 1. Ideja rada

MA-XRF skeneri sa dva detektora normalno sabiraju signale radi boljeg
SNR-a, a razlika među detektorima se ignoriše. Mi pokazujemo da ta
razlika, plus jedan dodatni sken istog platna nagnutog napred
(`antico1-prova4-ruotato`), razdvaja dve komponente:

- **detektorska svojstva** (odnos efikasnosti, Be prozor) — ne zavise
  od nagiba;
- **geometrijski efekti** (izlazni uglovi fotona ka detektorima) —
  menjaju se sa nagibom na predvidiv, energetski zavisan način.

Iz toga dobijamo: odnose efikasnosti po elementu sa nesigurnostima,
efektivne uglove gledanja oba detektora, mapu geometrijske
neuniformnosti, osetljivost elementnih mapa na pozicioniranje platna i
fuziju dva detektora koja nadmašuje prost zbir.

## 2. Potvrda izvodljivosti (izmereno na postojećim podacima)

Odnos detektora R = det10264 / det19511 po elementu, globalna suma po
pikselima, bootstrap nesigurnost (2000 uzoraka). Baseline = razlika
prova1↔prova2 (ista geometrija, 7 dana razmaka); tilt = pomeraj
ruotato skena u odnosu na frontalni prosek.

| Element | keV   | R frontal | R nagnut | baseline | tilt pomeraj | značajnost |
|---------|-------|-----------|----------|----------|--------------|------------|
| Ca      | 3.69  | 6.03      | 6.33     | 1.0%     | **+4.9%**    | 13σ        |
| Ti      | 4.51  | 2.54      | 2.61     | 0.9%     | **+2.9%**    | 5.6σ       |
| Fe      | 6.40  | 1.24      | 1.25     | 0.5%     | +0.9%        | 2.4σ       |
| Cu      | 8.04  | 1.08      | 1.04     | 1.2%     | **−3.7%**    | 8σ         |
| Pb Ll   | 9.19  | 0.860     | 0.862    | 0.5%     | +0.3%        | 0.4σ       |
| Pb Lα   | 10.54 | 0.800     | 0.781    | 0.4%     | **−2.3%**    | 13.5σ      |
| Pb Lβ   | 12.61 | 0.727     | 0.711    | 0.2%     | **−2.2%**    | 14σ        |
| Pb Lγ   | 14.77 | 0.650     | 0.634    | 1.4%     | **−2.5%**    | 8.5σ       |

(K je na nivou šuma; Zn je redak i zavisi od Cu-korekcije — ne ulaze u
fit.)

Zaključci:

1. Nagib pomera odnos detektora 3–14σ iznad baseline-a → geometrija i
   detektorska svojstva su razdvojivi. Centralna pretpostavka rada važi.
2. Signal je energetski struktuiran: pozitivan ispod ~6 keV, prolazak
   kroz nulu oko 6–9 keV, negativan iznad — kriva koju model treba da
   fituje.
3. Četiri Pb linije (isti element, isti pikseli, ista kompozicija) daju
   čistu energetsku zavisnost, imunu na prigovor da nagnuti sken pokriva
   drugi kadar (80×45 naspram 120×60).

Reprodukcija: `python scripts/06_efficiency_ratios.py`

## 3. Osoba A — model, karakterizacija, mali ML

### Faza 1 (odmah, bez zavisnosti)

1. **Tabela 1** — `scripts/06_efficiency_ratios.py`: per-element R sa
   bootstrap nesigurnostima, baseline vs. tilt (skripta postoji,
   dopuniti po potrebi).
2. **Ugao nagiba** — pitati Ridolfija koliki je bio nagib kod
   `prova4-ruotato` (poslati mejl prvog dana).
3. **Skica forward modela** R(E, nagib): odnos efikasnosti (ΔBe prozor +
   odnos prostornih uglova) × geometrijski član (efektivni uglovi θ₁,
   θ₂). Kod fita pripremiti tako da prima ugao kao parametar; testirati
   sa pretpostavljenim uglom (npr. 10°).

### Faza 2 (posle handoff-a 1)

4. **Procena ugla iz skraćenja** — iz faktora vertikalne kompresije
   (registracija, Osoba B); ukrstiti sa Ridolfijevim brojem.
5. **Fit geometrije** — `scripts/07_geometry_fit.py`: fit modela na 8
   upotrebljivih tačaka (Ca…Pb Lγ), nesigurnosti iz kovarijanse; Pb
   multi-linija kao interna kontrola. Uz to Gaussian process regresija
   iste krive kao neparametarska provera sa pojasom nesigurnosti →
   **centralna figura rada** (tilt pomeraj vs. energija + fit).
6. **ML: regresija lokalnog nagiba** — ulaz: log-ratio vektor po
   pikselu (~8 dimenzija); etikete: 0° (frontalni pikseli) / poznati
   ugao (nagnuti pikseli); ridge ili mali MLP → mapa topografije /
   iskrivljenosti platna iz frontalnog skena.

### Pisanje

Uvod, Metod (model + regresija), Rezultati karakterizacije.

## 4. Osoba B — registracija, mape, fuzija, glavni ML

### Faza 1 (odmah, bez zavisnosti)

1. **Registracija** — `scripts/08_registration.py`: afina registracija
   prova1 ↔ ruotato (proširenje pristupa iz `compare_Ti.py`); usput
   izračunati faktor vertikalnog skraćenja → **handoff 1 za Osobu A**.
2. **Log-ratio mape** — prerada `scripts/05_detector_diff.py`:
   log(D1/D2) po elementu za sva tri skena. Glatka prostorna
   komponenta = mapa geometrijske neuniformnosti; rezidual =
   topografija platna. (Diff mape prate abundancu elemenata; ratio
   mape prate geometriju.)
3. **Noise2Noise dataloader** — parovi spektara po pikselu (dva
   detektora = dve nezavisne Poisson realizacije istog signala;
   18 000 piksela × 2 smera ≈ 36 000 primera). Split: trening
   prova1+ruotato, evaluacija prova2 (prostorni/po-skenu split,
   nikako slučajan po pikselima). Probni trening može odmah, bez
   skaliranja targeta.

### Faza 2 (posle handoff-a 2)

4. **ML: trening fuzije** — 1D U-Net iz `xrf-denoise` (warm-start sa
   `experiments/A_scratch/checkpoints/best_model.pt`); target skaliran
   odnosom efikasnosti iz Tabele 1 da mreža ne uči trivijalno
   skaliranje. Laptop-skala.
5. **Benchmark fuzije** — `scripts/09_fusion.py`: na held-out skenu
   uporediti SNR po svih 8 elemenata za tri nivoa: prost zbir <
   inverse-variance ponderisanje < naučena fuzija → **druga ključna
   figura/tabela rada**.
6. **Osetljivost na pozicioniranje** — promena svake elementne mape
   frontal→nagnut posle registracije; kad ugao stigne od Osobe A,
   izraziti u %/stepenu ("practical estimate of the error caused by
   imperfect mounting" iz abstracta).

### Pisanje

Instrument/Data, fuzija (klasična + naučena), validacija, figure mapa.

## 5. Tačke primopredaje

| # | Smer | Sadržaj | Kada |
|---|------|---------|------|
| 1 | B → A | faktor vertikalnog skraćenja iz registracije | rano (dan-dva) |
| 2 | A → B | Tabela 1 (odnosi efikasnosti sa nesigurnostima) | sredina |

Svaki handoff je jedan broj odnosno jedna tabela (commit u repo), ne
kod. Redosled nije bitan — obe faze 1 su potpuno nezavisne i pokrivaju
~60% posla.

## 6. ML komponente — šta se trenira

| Komponenta | Tip | Podaci | Obim |
|------------|-----|--------|------|
| Noise2Noise fuzija (B) | 1D U-Net, self-supervised, od nule ili warm-start sopstvenog checkpointa | ~36 000 parova spektara | laptop, sati |
| Regresija nagiba (A) | ridge / mali MLP, supervizovano (etikete iz geometrije eksperimenta) | nekoliko hiljada piksela | sekunde |
| GP kriva R(E) (A) | Gaussian process regresija | 8 tačaka | trivijalno |

Nema spoljnih pre-trained modela — sve se trenira na sopstvenim
podacima koji se već dele preko GitHub Releases, pa je ceo ML deo
reproduktibilan.

## 7. Izmene abstracta

- "fully characterize the instrument" ublažiti u karakterizaciju
  *relativnog* odziva (odnosi efikasnosti + efektivni uglovi, ne
  apsolutna efikasnost).
- "flat-field correction map" preimenovati u mapu geometrijske
  neuniformnosti.
- Rečenicu o fuziji proširiti naučenom fuzijom (self-supervised,
  dva detektora kao nezavisne realizacije šuma).
