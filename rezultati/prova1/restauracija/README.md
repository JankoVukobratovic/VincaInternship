# XRF Rekonstrukcija freske – aurora-antico1-prova1

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
