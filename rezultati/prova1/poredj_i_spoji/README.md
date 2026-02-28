# Poređenje i kompozit – Detektori 10264 vs 19511

## Sadržaj foldera

| Fajl | Opis |
|------|------|
| `1_aditivni_kompozit.png` | Svih 6 elemenata preklopljeno bojama prema intenzitetu |
| `2_rgb_Fe_Cu_Pb.png` | Klasični RGB: Fe→Crvena, Cu→Zelena, Pb→Plava |
| `3_poredj_svi_elementi.png` | Normalizovane mape A i B + mapa razlike |
| `4_preklopljeni_RG.png` | 10264=Crvena, 19511=Zelena, saglasnost=Žuta |
| `5_korelacija_scatter.png` | Scatter korelacija 7200 tačaka: A vs B |

## Statistike korelacije između detektora

| Element | Korelacija r | Nagib regresije | Zaključak |
|---------|-------------|-----------------|----------|
| **Kalcijum (Ca)** | r = 0.8154 | y = 0.153×x | Umerena saglasnost – strukturni obrasci slični |
| **Titanijum (Ti)** | r = 0.8501 | y = 0.322×x | Dobra saglasnost – manje razlike u efikasnosti detektora |
| **Gvozdje (Fe)** | r = 0.9837 | y = 0.837×x | Odlična saglasnost – oba detektora vide istu strukturu |
| **Bakar (Cu)** | r = 0.9662 | y = 0.969×x | Odlična saglasnost – oba detektora vide istu strukturu |
| **Olovo La (Pb_La)** | r = 0.9543 | y = 1.203×x | Odlična saglasnost – oba detektora vide istu strukturu |
| **Kalaj (Sn)** | r = 0.9645 | y = 1.526×x | Odlična saglasnost – oba detektora vide istu strukturu |

## Opis slika

### 1. Aditivni kompozit
Svaki element dobija svoju boju i normalizuje se na [0,1].
Boje se aditivno mešaju – onde gde ima više elemenata boje se kombinuju.
Osnova interpretacije: oblasti sličnih boja = slični hemijski sastav.

**Dodeljene boje:**
- `#00A5FF` → **Kalcijum (Ca)**
- `#A500FF` → **Titanijum (Ti)**
- `#FF3300` → **Gvozdje (Fe)**
- `#00E526` → **Bakar (Cu)**
- `#FFE500` → **Olovo La (Pb_La)**
- `#FF00CC` → **Kalaj (Sn)**

### 2. RGB kompozit (Fe / Cu / Pb)
Klasični XRF false-color prikaz sa 3 kanala:
- **Crvene** oblasti = gvožđe (okre) dominira
- **Zelene** oblasti = bakar (azurit, malahit) dominira
- **Plave** oblasti = olovo (olovna bela, minijum) dominira
- **Žuta** = gvožđe + bakar zajedno
- **Bela** = svi elementi prisutni u sličnoj meri

### 3. Mape razlike
Normalizovane mape (p1–p99) oduzimaju se: **10264 − 19511**.
- **Crvene** zone → detektor 10264 meri više signala
- **Plave** zone → detektor 19511 meri više signala
- **Bele** zone → detektori se savršeno slažu

Razlike mogu nastati zbog: ugla detektora, senčenja uzorka, različite energetske efikasnosti.

### 4. Preklopljeni R/G prikaz
Najintuitivniji prikaz saglasnosti:
- **Žuta** → oba detektora registruju visok signal (saglasnost)
- **Crvena** → samo 10264 vidi signal (lokalna razlika)
- **Zelena** → samo 19511 vidi signal
- **Crna** → oba detektora imaju nizak signal

### 5. Scatter korelacija
Svaka od 7200 skeniranih tačaka prikazana kao tačka u dijagramu.
Nagib regresione prave < 1 → detektor 19511 generalno meri više.
Nagib > 1 → detektor 10264 meri više.

---
*Generisano skriptom `poredj_i_spoji.py`*
