# Plan rada - Geometry-resolved Characterization of a Dual-detector MA-XRF Scanner

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

- **detektorska svojstva** (odnos efikasnosti, Be prozor) - ne zavise
  od nagiba;
- **geometrijski efekti** (izlazni uglovi fotona ka detektorima) - menjaju se sa nagibom na predvidiv, energetski zavisan način.

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

(K je na nivou šuma; Zn je redak i zavisi od Cu-korekcije - ne ulaze u
fit.)

Zaključci:

1. Nagib pomera odnos detektora 3–14σ iznad baseline-a → geometrija i
   detektorska svojstva su razdvojivi. Centralna pretpostavka rada važi.
2. Signal je energetski struktuiran: pozitivan ispod ~6 keV, prolazak
   kroz nulu oko 6–9 keV, negativan iznad - kriva koju model treba da
   fituje.
3. Četiri Pb linije (isti element, isti pikseli, ista kompozicija) daju
   čistu energetsku zavisnost, imunu na prigovor da nagnuti sken pokriva
   drugi kadar (80×45 naspram 120×60).

Reprodukcija: `python scripts/06_efficiency_ratios.py`

## 3. Osoba A - model, karakterizacija, mali ML

### Faza 1 (odmah, bez zavisnosti)

1. **Tabela 1** - `scripts/06_efficiency_ratios.py`: per-element R sa
   bootstrap nesigurnostima, baseline vs. tilt (skripta postoji,
   dopuniti po potrebi).
2. **Ugao nagiba** - pitati Ridolfija koliki je bio nagib kod
   `prova4-ruotato` (poslati mejl prvog dana).
3. **Skica forward modela** R(E, nagib): odnos efikasnosti (ΔBe prozor +
   odnos prostornih uglova) × geometrijski član (efektivni uglovi θ₁,
   θ₂). Kod fita pripremiti tako da prima ugao kao parametar; testirati
   sa pretpostavljenim uglom (npr. 10°).

### Faza 2 (posle handoff-a 1)

4. **Procena ugla iz skraćenja** - iz faktora vertikalne kompresije
   (registracija, Osoba B); ukrstiti sa Ridolfijevim brojem.
5. **Fit geometrije** - `scripts/07_geometry_fit.py`: fit modela na 8
   upotrebljivih tačaka (Ca…Pb Lγ), nesigurnosti iz kovarijanse; Pb
   multi-linija kao interna kontrola. Uz to Gaussian process regresija
   iste krive kao neparametarska provera sa pojasom nesigurnosti →
   **centralna figura rada** (tilt pomeraj vs. energija + fit).
6. **ML: regresija lokalnog nagiba** - ulaz: log-ratio vektor po
   pikselu (~8 dimenzija); etikete: 0° (frontalni pikseli) / poznati
   ugao (nagnuti pikseli); ridge ili mali MLP → mapa topografije /
   iskrivljenosti platna iz frontalnog skena.

### Pisanje

Uvod, Metod (model + regresija), Rezultati karakterizacije.

## 4. Osoba B - registracija, mape, fuzija, glavni ML

### Faza 1 (odmah, bez zavisnosti)

1. **Registracija** - `scripts/08_registration.py`: afina registracija
   prova1 ↔ ruotato (proširenje pristupa iz `compare_Ti.py`); usput
   izračunati faktor vertikalnog skraćenja → **handoff 1 za Osobu A**.
2. **Log-ratio mape** - prerada `scripts/05_detector_diff.py`:
   log(D1/D2) po elementu za sva tri skena. Glatka prostorna
   komponenta = mapa geometrijske neuniformnosti; rezidual =
   topografija platna. (Diff mape prate abundancu elemenata; ratio
   mape prate geometriju.)
3. **Noise2Noise dataloader** - parovi spektara po pikselu (dva
   detektora = dve nezavisne Poisson realizacije istog signala;
   18 000 piksela × 2 smera ≈ 36 000 primera). Split: trening
   prova1+ruotato, evaluacija prova2 (prostorni/po-skenu split,
   nikako slučajan po pikselima). Probni trening može odmah, bez
   skaliranja targeta.

### Faza 2 (posle handoff-a 2)

4. **ML: trening fuzije** - 1D U-Net iz `xrf-denoise` (warm-start sa
   `experiments/A_scratch/checkpoints/best_model.pt`); target skaliran
   odnosom efikasnosti iz Tabele 1 da mreža ne uči trivijalno
   skaliranje. Laptop-skala.
5. **Benchmark fuzije** - `scripts/09_fusion.py`: na held-out skenu
   uporediti SNR po svih 8 elemenata za tri nivoa: prost zbir <
   inverse-variance ponderisanje < naučena fuzija → **druga ključna
   figura/tabela rada**.
6. **Osetljivost na pozicioniranje** - promena svake elementne mape
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
kod. Redosled nije bitan - obe faze 1 su potpuno nezavisne i pokrivaju
~60% posla.

## 6. ML komponente - šta se trenira

| Komponenta | Tip | Podaci | Obim |
|------------|-----|--------|------|
| Noise2Noise fuzija (B) | 1D U-Net, self-supervised, od nule ili warm-start sopstvenog checkpointa | ~36 000 parova spektara | laptop, sati |
| Regresija nagiba (A) | ridge / mali MLP, supervizovano (etikete iz geometrije eksperimenta) | nekoliko hiljada piksela | sekunde |
| GP kriva R(E) (A) | Gaussian process regresija | 8 tačaka | trivijalno |

Nema spoljnih pre-trained modela - sve se trenira na sopstvenim
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

## 8. Amandmani posle fact-check-a (2026-08-09)

Fact-check plana protiv stvarnog izlaza `06_efficiency_ratios.py`
potvrdio je sve brojeve u Tabeli 1, energije linija, bootstrap i
aritmetiku piksela. Sledeći amandmani ništa ne brišu iz plana - raspored, podela A/B i tačke primopredaje ostaju identični; menjaju se
formulacije tvrdnji i po jedan detalj u tri zadatka (~1–2 dana ukupno).

### 8.1 Značajnost: dodati sistematsku grešku (A, dopuna 06)

Bootstrap σ je čisto statistički, a prova1↔prova2 se razlikuju ~2–4×
više nego što bootstrap predviđa → postoji dan-za-dan sistematika.
U `06_efficiency_ratios.py` dodati σ_sys ≈ |R₁−R₂|/√2 po elementu i
prijaviti kombinovano σ = √(σ_stat² + σ_sys²). U tekstu rada glavna
metrika je **pomeraj / ponovljivost = 2–12×**, ne "3–14σ". Fe (1.9×)
i Pb Lγ (1.85×) označiti kao marginalne; fit ih koristi sa naduvanim
greškama, ne izbacuje.

### 8.2 Cu i Pb Ll outlieri (A, dopuna 07)

Kriva tilt-pomeraja nije monotona: Cu (8.04 keV, −3.7%) je ispod
visokoenergetskog platoa (−2.3%), a Pb Ll (9.19 keV, +0.3%) iznad.
Tri poteza:

1. fit sa svih 8 tačaka + **leave-one-out** stabilnost parametara;
   reziduali Cu i Ll prijavljeni otvoreno;
2. proveriti prozor Ll linije (rep Cu Kβ ~8.9 keV / Zn linija);
3. Cu odstupanje uokviriti kao **signal dubine sloja** (element dublje
   u stratigrafiji → jači apsorpcioni efekat) - pasus u Diskusiji,
   veza sa ML regresijom lokalnog nagiba.

### 8.3 Forward model: eksplicitan apsorpcioni član (A, izmena skice u §3.1.3)

Prostorni uglovi su energetski nezavisni, a Be prozor se ne menja s
nagibom - energetsku zavisnost tilt signala nosi apsorpcija duž
izlaznog puta. Model:

    Δln R(E) = a + b · μ̄(E)

- `a` - konstantan geometrijski ofset (promena prostornih uglova);
  visokoenergetski plato ≈ −2.3% ga direktno čita;
- `b·μ̄(E)` - apsorpcioni član; μ̄(E) iz tabela (xraylib / NIST XCOM)
  za pretpostavljenu matricu (npr. olovna bela).

Obavezna **analiza osetljivosti na izbor matrice** (olovna bela vs.
kalcitna osnova). Iz `a` i `b` slede efektivni uglovi. GP ostaje kao
neparametarska kontrola.

### 8.4 N2N skaliranje i handoff 2 (B, dopuna §4.3–4.4)

Dva detektora nisu realizacije *istog* signala (R ide 0.65–6.06 preko
spektra) - jesu uslovno nezavisne Poisson realizacije *različitih*
odziva. Zato:

- **handoff 2 postaje: Tabela 1 + glatka R(E) kriva iz GP fita**
  (CSV po kanalima); dataloader skalira target kanal-po-kanal sa R(E),
  ne per-element skalarima;
- **loss maskirati na ~3.5–15.5 keV** (van toga je ekstrapolacija
  krive);
- u radu pošteno uokviriti: evaluacija na prova2 testira generalizaciju
  preko realizacija šuma na istoj slici, ne preko sadržaja.

### 8.5 Benchmark fuzije: bez unapred obećane hijerarhije (B, izmena §4.5)

Za čist Poisson sa proporcionalnim efikasnostima **prost zbir je
sufficient statistic** - nijedno linearno ponderisanje ga ne može
pobediti. Benchmark zato glasi: (1) zbir = teorijski optimalna
linearna baseline; (2) inverse-variance = provera ne-Poisson
komponente (≈ zbir je takođe nalaz: "detektori su Poisson-limitirani");
(3) naučena fuzija = jedini kandidat koji dodaje vrednost (spektralni/
prostorni prior). Figura ima poentu koji god rezultat izađe.

### 8.6 Jeftini dodaci

- Posle registracije (08) preračunati frontalni R **samo na
  preklapajućem regionu** → prigovor "drugi kadar" pada u potpunosti.
- K liniju prikazati na figuri kao otvoren marker (isključena iz
  fita): +20% na 3.3 keV kvalitativno pojačava niskoenergetski trend.
- N2N novitet formulisati kao "primena na MA-XRF" (analogije postoje u
  cryo-EM / CT even-odd split), ne "prvi put ikada".

### 8.7 Nalaz iz registracije (B1 urađeno, 2026-08-09) - VAŽNO ZA OSOBU A

`scripts/08_registration.py` (NCC 0.965): ruotato je **isti korak
skeniranja, samo pomeren kadar** (sx≈sy≈1.00, rotacija 1.4°, pomeraj
~18 px; pokriva 48% frontalnog kadra) - ne umanjena slika, kako je
pretpostavljao `compare_Ti.py` (ta figura poredi neregistrovan sadržaj
i ne treba je koristiti).

Posledica overlap poređenja (amandman 8.6): **prividni prolazak kroz
nulu u Tabeli 1 bio je artefakt kadra**. Na preklapajućem regionu tilt
pomeraj je monoton pozitivan pad sa energijom: Ca +9.5% (24σ),
Ti +7.9%, Fe +3.5%, Cu +2.5%, Pb Lα +2.2%, Pb Lβ +1.8%, Pb Lγ +0.6%;
K +21% nastavlja trend. Cu više NIJE outlier; Pb Ll (+4.1%) ostaje.
→ **Fit u 07 raditi na `results/registration/overlap_ratios.csv`**, ne
na `efficiency_ratios.csv`; u modelu iz 8.3 očekivati a ≈ 0 (mali
prostorno-ugaoni član) i dominantan apsorpcioni član b·μ̄(E).

Handoff 1: f = 0.9995 ± 0.0058 (kontrola prova2: 0.9963) → nagib je
mali, α ≲ 8°; skraćenje daje samo gornju granicu - Ridolfijev broj je
presudan (`results/registration/handoff1_foreshortening.md`).

### 8.8 Handoff 2 isporučen (2026-08-11)

`07_geometry_fit.py` sada eksportuje glatku krivu
`results/detector_diff/handoff2_ratio_curve.csv` (+ `.md` sa Tabelom 1):
R(E) = R_det(E) × exp(GP na log-rezidualima), 2–20 keV, kolone `R`
(frontalni skenovi), `R_sigma`, `R_model`, `R_tilt` (za ruotato, do
+10%). GP je regularizovan (belo šum fiksiran na 0.5%, dužina skale ≥
1.5 keV): slobodan fit degeneriše, a skale ispod ~1 keV bi fitovale
sistematiku prozora linija - rezidual je +6.1% na Cu i −5.2% na Pb Ll,
tj. 11% preko 1.1 keV, što potvrđuje sumnju iz 8.2 na kontaminaciju Ll
prozora. Zatvaranje na 8 izmerenih tačaka: max 0.31%.

Provizorna kriva iz `efficiency_ratios.csv` (full-frame) se više ne
koristi; N2N dataloader prima rečnik krivih po skenu.

### 8.9 Flat-field vs. akvizicioni artefakt (dopuna 10)

Kvantifikovano u `flatfield_map.txt`: scatter-rep okvir i flat-field su
ista prostorna struktura (r = 0.855; odnos +7.2% u vrućim pikselima
naspram −2.7% u ostatku; 47% vrućih piksela je u graničnom pojasu, pri
osnovi 28%). Hg pravougaonik je komplementaran region sa suprotnim
znakom (−3.9%), ali slabo korelisan (r = −0.236) i samo 6% u pojasu.
Formulacija za rad: jedna akviziciona geometrija viđena iz dva ugla - ne dve nezavisne potvrde iste mape.

### 8.10 Naučena fuzija - preliminarni brojevi (B4/B5, 2026-08-11)

`xrf-denoise/scripts/07_train_cross_detector.py` (1D U-Net, warm start
sa A_scratch, MSE maskiran na 3.5–15.5 keV, lr 3e-4, best epoch 2,
laptop/MPS ~5 min). Dve stvari su bile presudne i obe slede iz R(E):

1. **težinjenje gubitka** - skaliranje detektora B sa R množi varijansu
   sa R², pa nekompenzovan MSE dominira niskim energijama i mreža
   izgladi Ca mapu (cv odnos 0.45–0.67); težine 1/R (smer 0) i R
   (smer 1) to poništavaju;
2. **kombinovanje dva smera pri izlazu** - inverzno-varijansno R:1 po
   kanalu, ne prosta sredina (na Ca to je 85:15, isto što klasična
   fuzija nezavisno nalazi kao w = 0.89).

**Finalni rezultat** (`fusion_weighted.txt`, held-out pikseli): srednji
dobitak nad prostim zbirom **+17.7%**, naspram +0.9% za klasično
ponderisanje. Srednja vrednost je osetljiva na jednu liniju - medijana
je +11.9%, a bez Pb Ll srednja pada na +10.4%, pa se u radu navodi
raspon 10–18% zavisno od agregacije. Šest od osam linija pozitivno: Pb Ll +68.6%, Pb Lγ +33.3%,
Fe +28.5%, Cu +17.1%, Pb Lβ +6.7%, Pb Lα +0.6%; Ca −9.4% i Ti −3.7%
ostaju negativne. U jedinicama vremena: +17.7% SNR vredi 39% duže
akvizicije.

Presudan korak nije bila arhitektura nego **pune inverzno-varijansne
težine gubitka**: pored R² faktora treba računati i nivo odbroja, jer su
Pb linije dva reda veličine jače od Ca pa preuzimaju gradijent. Var =
R·E[a] (smer 0) i E[b]/R (smer 1), a očekivanja se procenjuju iz *ulaza*
koji je nezavisan od targeta, pa težine ne unose pristrasnost:
w = 1/(R·(a+1)) i R/(b+1). Prelaz sa 1/R težina na pune podigao je
rezultat sa +5.4% na +17.7% i popravio Ti sa −22.6% na −3.7%.

**Ablacija** (`fusion_ablation.txt`) - ista mreža, prekidači isključeni:

| težine gubitka | fuzija | srednji dobitak |
|---|---|---|
| nema | 1:1 | −0.1% |
| nema | R:1 | +7.1% |
| 1/R | 1:1 | +3.8% |
| 1/R | R:1 | +4.4% / +5.4% (lr 1e-4 / 3e-4) |
| puna | 1:1 | +14.4% |
| **puna** | **R:1** | **+17.7%** |

Mreža sama ne pobeđuje zbir (−0.1%). Pobeđuje ga tek sa varijansnom
strukturom koju propisuje izmereno R(E), a razlika između dve stope
učenja je manja od razlike između šema težinjenja - dakle nije artefakt
štimovanja.

Kontrolne kolone `cv_ratio` (srednje 0.967) i `r_vs_sum` (0.947 na Ca,
≥ 0.98 drugde) pokazuju da dobitak nije zamućenje.

**Nađeno u fact-checku 2026-08-12 - apsolutni nivo.** SNR, cv i r su svi
invarijantni na skalu, pa nijedan ne vidi grešku u samom intenzitetu.
Merena posebno (kolona `bias_learned_pct` u 09): −3 do −7% na Fe, Cu i
Pb linijama, ali **+33% na Ca i −30% na Ti**. Uzrok nije po-linijski
ofset nego izobličenje spektra na niskim energijama: fuzovani kontinuum
je 1.4× nivoa detektora A na 4.2–4.8 keV, 2.7× na 4.0–4.2 keV i 2.1×
ispod 3.5 keV, dok je Pb oblast ravna na 0.92×. B grana se množi sa
R(E) koje na Ca dostiže 6, pa se svaka pozitivna pristrasnost u
niskoodbrojnim kanalima 19511 tako i uvećava; kod Ca se na to dodaje i
to što donji fon (3.14–3.39 keV) leži ispod treniranog prozora.

Slika je bar konzistentna: gde je oblik spektra očuvan (Fe, Cu, Pb - nivo unutar 7%) fuzija dobija 17–69% SNR-a; gde nije (Ca, Ti) gubi.

**Dve popravke probane, obe pale** (2026-08-12):

1. *Prozor 3.0–15.5 keV* (da fonovi Ca budu unutra) - Ca se raspada:
   intenzitet −90%, cv 5.0. Dodati kanali nose R do 34, pa varijansa
   skaliranog targeta eksplodira baš tamo.
2. *Izbacivanje B grane ispod R = 2.5* (uklanja put kojim se pristrasnost
   detektora 19511 množi sa R) - Ti pada na −25.7% (bilo −3.7%), Ca bias
   raste na +41%.

Dakle greška je u samoj rekonstrukciji niskoenergetskog kontinuuma, ne u
skaliranju ni u kombinovanju. **Odluka za rad: naučena fuzija se navodi
za šest linija od Fe naviše** (nivo očuvan unutar 7%, dobitak
0.6–68.6%), Ca i Ti ostaju na prostom zbiru. Sledeći kandidat je
funkcija cilja koja čuva integrisani intenzitet linije umesto MSE po
kanalu.

Izbor modela: šema težinjenja po argumentu o varijansi, stopa učenja po
validacionom gubitku. Pošteno napomenuti: benchmark je preračunavan dok
su se konfiguracije razvijale, pa finalni broj jeste hold-out procena
ali ne i slepa - raspon ablacije (−0.1% do +17.7%) je ipak mnogo veći od
bilo kog uverljivog efekta selekcije. Držani pikseli takođe nisu potpuno
netaknuti: nemaju gradijent, ali je validacioni gubitak na njima birao
epohu zaustavljanja; prova2, druga polovina svakog SNR para, mreža nije
videla nikako. Usput nalaz vredan pomena u radu: validaciona MSE - i puna i
ograničena na prozore linija - **ne rangira modele isto kao SNR mapa**
(lr 1e-4 pobeđuje po oba validaciona kriterijuma, a gubi na mapama),
jer MSE nagrađuje skupljanje ka uslovnoj sredini. Zato je `cv_ratio`
obavezan uz svaki SNR broj.

### 8.11 B6 - osetljivost na pozicioniranje (2026-08-12)

`scripts/11_positioning_sensitivity.py`: frontalna mapa se registruje u
kadar nagnutog (afina transformacija iz 08) i poredi piksel-po-piksel,
pa se deli uglom. Rezultat je *diferencijalna* greška - zajednički mod
(−0.40 %, razlika ukupnog nivoa između skenova) je uklonjen jer je
degenerisan sa driftom sesije; ono što ostaje pogađa odnose elemenata,
tj. baš ono na čemu počiva identifikacija pigmenata.

Na sumiranoj mapi: **Ca +0.50 %/°, Ti +0.45 %/°**, Fe i Cu u šumu
(±0.04 %/°), **Pb Lβ −0.13 %/°**, Pb Lγ −0.08 %/°. Raspon između
ekstrema **0.63 procentna poena po stepenu**. Prag ponovljivosti
(prova1 vs prova2, ista geometrija) je 0.96 % RMS, tj. 0.12 %/° - Fe,
Cu i Pb Lγ ga ne prelaze i tako su označeni.

**Ugao je gornja granica, pa su ovo donje granice**: ako je platno bilo
nagnuto manje od 7.7°, ista promena mape se desila preko manje stepeni,
pa je greška po stepenu veća. To je obrnuto od uobičajenog čitanja
gornje granice i tako mora biti formulisano u radu.

Usput: implicirani pomeraj R iz ovih per-detektorskih brojeva slaže se
sa 08 po obliku i znaku, ali je sistematski niži (RMS 0.73 pp, max 1.2
pp na Ca). To je razlika između poređenja regiona (08) i poređenja
piksela (11) i treba je citirati kao sistematiku na figuri tilt
pomeraja.
