# Rezultati XRF analize – Detektor stacked

**Dataset:** aurora-antico1-prova1  
**Mreža skeniranja:** 60 redova × 120 kolona = **7200 tačaka**  
**Vreme merenja po tački:** 3 sekunde  
**Kalibracija:** E(keV) = 0.02908 × kanal + -0.0145  (R² = 0.999975)  

---

## Sadržaj foldera

| Fajl | Element | XRF linija | Energija | Pigmenti/Minerali |
|------|---------|-----------|----------|-------------------|
| `Ca_Kalcijum_mapa.png` | **Kalcijum** | Ca Kα | 3.692 keV | Kreč (CaCO₃), gips (CaSO₄·2H₂O) – osnova maltera i preparacije |
| `Ti_Titanijum_mapa.png` | **Titanijum** | Ti Kα | 4.510 keV | Titanijum bela (TiO₂) – moguće moderno restauratorsko premazivanje |
| `Fe_Gvozdje_mapa.png` | **Gvozdje** | Fe Kα | 6.400 keV | Crvena okra (Fe₂O₃, hematit), žuta okra (FeOOH, getit), umbra |
| `Cu_Bakar_mapa.png` | **Bakar** | Cu Kα | 8.046 keV | Azurit (Cu₃(CO₃)₂(OH)₂), malahit (Cu₂CO₃(OH)₂), verdigris – plava/zelena |
| `Pb_La_Olovo_La_mapa.png` | **Olovo La** | Pb_La Lα | 10.551 keV | Olovna bela (2PbCO₃·Pb(OH)₂), minijum (Pb₃O₄) – bela/crvena boja |
| `Pb_Lb_Olovo_Lb_mapa.png` | **Olovo Lb** | Pb_Lb Lβ | 12.614 keV | Druga spektralna linija olova – potvrda i provera Pb signala |
| `Sn_Kalaj_mapa.png` | **Kalaj** | Sn Kα | 25.271 keV | Olovo-kalaj žuta (Pb₂SnO₄) – zlatno-žuta boja, česta u starim freskama |
| `zbirni_spektar.png` | – | – | 0–30 keV | Zbir svih spektara sa označenim pikovima |
| `sve_mape_pregled.png` | – | – | – | Svih 7 mapa u jednom pregledu (grid) |

---

## Statistike po elementu

| Element | Linija | Kanal | Prozor | Min | Max | Srednja | Std | CV% | Ocena distribucije |
|---------|--------|-------|--------|-----|-----|---------|-----|-----|-------------------|
| **Kalcijum** | Kα | 127 | ±15 | 1109 | 9003 | 3128.7 | 1076.6 | 34% | Umereno neravnomerna – mešovita |
| **Titanijum** | Kα | 156 | ±10 | 694 | 3854 | 1493.7 | 412.6 | 28% | Umereno neravnomerna – mešovita |
| **Gvozdje** | Kα | 219 | ±10 | 401 | 4466 | 873.4 | 650.1 | 74% | Jako neravnomerna – lokalni pigmenti |
| **Bakar** | Kα | 278 | ±10 | 379 | 2760 | 989.5 | 405.1 | 41% | Jako neravnomerna – lokalni pigmenti |
| **Olovo La** | Lα | 363 | ±10 | 3004 | 21025 | 9722.4 | 2745.0 | 28% | Umereno neravnomerna – mešovita |
| **Olovo Lb** | Lβ | 436 | ±10 | 2692 | 17586 | 7986.4 | 2175.6 | 27% | Umereno neravnomerna – mešovita |
| **Kalaj** | Kα | 869 | ±15 | 476 | 1540 | 1005.3 | 288.4 | 29% | Umereno neravnomerna – mešovita |

---

## Zaključci

### Elementi sa neravnomernom distribucijom (pigmenti)

- **Kalcijum (Ca Kα, 3.69 keV)**: Kreč (CaCO₃), gips (CaSO₄·2H₂O) – osnova maltera i preparacije
- **Titanijum (Ti Kα, 4.51 keV)**: Titanijum bela (TiO₂) – moguće moderno restauratorsko premazivanje
- **Gvozdje (Fe Kα, 6.40 keV)**: Crvena okra (Fe₂O₃, hematit), žuta okra (FeOOH, getit), umbra
- **Bakar (Cu Kα, 8.05 keV)**: Azurit (Cu₃(CO₃)₂(OH)₂), malahit (Cu₂CO₃(OH)₂), verdigris – plava/zelena
- **Olovo La (Pb_La Lα, 10.55 keV)**: Olovna bela (2PbCO₃·Pb(OH)₂), minijum (Pb₃O₄) – bela/crvena boja
- **Olovo Lb (Pb_Lb Lβ, 12.61 keV)**: Druga spektralna linija olova – potvrda i provera Pb signala
- **Kalaj (Sn Kα, 25.27 keV)**: Olovo-kalaj žuta (Pb₂SnO₄) – zlatno-žuta boja, česta u starim freskama

### Elementi sa uniformnom distribucijom (supstrat/pozadina)

- Svi elementi pokazuju neravnomernu distribuciju.

### Elementi sa izrazitim lokalnim pikovima (μ + 3σ kriterijum)

- **Kalcijum (Ca Kα)**: max = 9003 (>6359 = μ+3σ) – jasne zone visokog intenziteta → Kreč (CaCO₃), gips (CaSO₄·2H₂O) – osnova maltera i preparacije
- **Titanijum (Ti Kα)**: max = 3854 (>2731 = μ+3σ) – jasne zone visokog intenziteta → Titanijum bela (TiO₂) – moguće moderno restauratorsko premazivanje
- **Gvozdje (Fe Kα)**: max = 4466 (>2824 = μ+3σ) – jasne zone visokog intenziteta → Crvena okra (Fe₂O₃, hematit), žuta okra (FeOOH, getit), umbra
- **Bakar (Cu Kα)**: max = 2760 (>2205 = μ+3σ) – jasne zone visokog intenziteta → Azurit (Cu₃(CO₃)₂(OH)₂), malahit (Cu₂CO₃(OH)₂), verdigris – plava/zelena
- **Olovo La (Pb_La Lα)**: max = 21025 (>17957 = μ+3σ) – jasne zone visokog intenziteta → Olovna bela (2PbCO₃·Pb(OH)₂), minijum (Pb₃O₄) – bela/crvena boja
- **Olovo Lb (Pb_Lb Lβ)**: max = 17586 (>14513 = μ+3σ) – jasne zone visokog intenziteta → Druga spektralna linija olova – potvrda i provera Pb signala

---

## Napomena o pigmentima u staroj fresko-tehnici

| Element | Moguće poreklo |
|---------|---------------|
| **Ca (Kalcijum)** | Osnova malternog sloja (intonaco) od kreča (CaCO₃). Uniformna distribucija je normalna. |
| **Ti (Titanijum)** | TiO₂ pigment je moderan (uveden ~1920). Prisustvo ukazuje na restauraciju ili moderni premaz. |
| **Fe (Gvožđe)** | Najstariji pigmenti – okre i umbra. Crvena (Fe₂O₃) i žuta (FeOOH). |
| **Cu (Bakar)** | Azurit i malahit – plava i zelena boja. Karakteristični za sredn­jovekovne freske. |
| **Pb (Olovo)** | Olovna bela (PbCO₃·Pb(OH)₂) i minijum (Pb₃O₄). Lα i Lβ treba da koreliraju. |
| **Sn (Kalaj)** | Olovo-kalaj žuta (Pb₂SnO₄) – zlatni tonovi, tipični za gotiku i renesansu. |

---
*Generisano automatski skriptom `analiza_elemenata.py`*
