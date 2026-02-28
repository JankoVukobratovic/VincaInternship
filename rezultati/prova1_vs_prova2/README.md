# Poređenje XRF skeniranja: prova1 vs prova2

## O poređenju

Ovaj folder sadrži analizu razlika između dva XRF skeniranja iste freske:
- **prova1**: `aurora-antico1-prova1` – prvo skeniranje
- **prova2**: `aurora-antico1-prova2` – drugo skeniranje

Oba skena imaju istu mrežu (60 × 120 = 7200 tačaka), dwell = 3 s/tački, i iste detektore (10264 + 19511).

---

## Statistika

| Element | μ prova1 | μ prova2 | max prova1 | max prova2 | Korelacija r |
|---------|---------|---------|-----------|-----------|-------------|
| Ca     |     1564 |     1550 |     4502 |     4429 | 0.9914 |
| Ti     |      747 |      741 |     1927 |     1892 | 0.9837 |
| Fe     |      437 |      434 |     2233 |     2205 | 0.9949 |
| Cu     |      495 |      493 |     1380 |     1396 | 0.9872 |
| Pb_La  |     4861 |     4847 |    10512 |    10599 | 0.9919 |
| Sn     |      503 |      502 |      770 |      760 | 0.9859 |

---

## Ključni zaključci

- **Ca**: sličan signal u oba skeniranja (Δ=-0.9%, r=0.991)
- **Ti**: sličan signal u oba skeniranja (Δ=-0.8%, r=0.984)
- **Fe**: sličan signal u oba skeniranja (Δ=-0.6%, r=0.995)
- **Cu**: sličan signal u oba skeniranja (Δ=-0.3%, r=0.987)
- **Pb_La**: sličan signal u oba skeniranja (Δ=-0.3%, r=0.992)
- **Sn**: sličan signal u oba skeniranja (Δ=-0.1%, r=0.986)

### Interpretacija korelacije

- **r > 0.90** → gotovo identična prostorna distribucija (isti pigment, ista zona)
- **r 0.70–0.90** → slična distribucija, ali sa primetnim razlikama
- **r < 0.70** → značajne razlike u rasporedu pigmenta između skeniranja

---

## Sadržaj foldera

| Fajl | Opis |
|------|------|
| `1_mape_paralela.png` | Sve elementne mape prova1 ↔ prova2 (6 elemenata × 2 dataseta) |
| `2_razlike.png` | Mape razlika: prova2 − prova1 po svakom elementu (crveno = viši u prova2) |
| `3_kompoziti.png` | Aditivni RGB kompozit oba dataseta (iste boje – direktno uporedivo) |
| `4_rekonstrukcije.png` | Rekonstrukcija freske oba dataseta (isti algoritam, direktno uporedivo) |
| `5_statistika.png` | Srednji/maks intenziteti i Pearsonova korelacija po elementu |

---

*Generisano skriptom `poredj_prova1_prova2.py`  |  VincaInstitute XRF Analiza*
