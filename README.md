# Automated Chemical Vulnerability Assessment of Paintings from XRF Spectral Imaging

Companion code for:

> D. Pešić, J. Vukobratović, A. Stojanović, G. Ristori, S. Ridolfi,
> M. Gajić-Kvaščev, G. Kvaščev, **"Automated Chemical Vulnerability
> Assessment of Canvas Paintings from XRF Spectral Imaging Using Deep
> Learning and Foundation Models"**, ICETRAN 2026.
> (draft: [`ICETRAN.pdf`](ICETRAN.pdf), source: [`main.tex`](main.tex))

A macro-XRF scanner rasters a painting pixel by pixel and records a full
X-ray fluorescence spectrum at every point. This repository turns those
raw spectra into a conservator-facing risk assessment, fully
automatically and with no expert-labeled training data:

1. **Self-supervised denoising** - a 1D U-Net trained with Poisson
   splitting suppresses photon-counting noise.
2. **Elemental extraction** - per-pixel net peak intensities
   (background-subtracted, overlap-corrected) become 2-D element maps
   (Ca, Ti, Fe, Cu, Pb).
3. **NMF corroboration** - blind NMF (K = 5, CPS-normalised spectra,
   1-15 keV, acquisition-artifact channels excluded) independently
   recovers the same material layers, validating the extraction.
4. **Chemical Vulnerability Index (CVI)** - five literature-grounded
   degradation rules aggregate into a per-pixel risk score with four
   zones (low / moderate / elevated / critical).
5. **SAM segmentation** - Meta's Segment Anything Model partitions the
   painting into coherent regions; per-region mean CVI and dominant
   mechanism produce an actionable conservation report.

**Datasets:** `aurora-antico1-prova1`, `aurora-antico1-prova2`
(120x60 = 7200 px, repeated after 7 days) and the rotated scan
`antico1-prova4-ruotato` (80x45 px) - detector 10264 (19511 also
recorded), dwell 3 s/px, 1024 channels, ~0.0292 keV/channel.

Detailed lab notes in Serbian: [`IZVESTAJ.md`](IZVESTAJ.md)

---

## Setup

```bash
pip install -r requirements.txt
```

**Raw data** (not in git, ~400 MB unpacked) - download from the
repository's **Releases** page and unpack into `Resources/`:

```bash
curl -L -o xrf-data.zip \
  https://github.com/JankoVukobratovic/VincaInternship/releases/download/data-v1/aurora-antico1-xrf-data.zip
mkdir -p Resources
tar -xf xrf-data.zip -C Resources
```

**SAM checkpoint** (only for the SAM scripts, ~375 MB):

```bash
curl -L -o models/sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

**U-Net checkpoint** ships with the repo
(`xrf-denoise/experiments/A_scratch/checkpoints/best_model.pt`) - no
action needed.

---

## Running

Run everything from the project root. Element maps are cached as `.npy`
after the first run, so reruns are fast. Scripts skip phases whose
inputs are missing instead of failing.

| Command | Output | What it does |
|---------|--------|--------------|
| `python scripts/01_run_analysis.py` | `results/10264/` | element maps, prova1 vs prova2 |
| `python scripts/02_vulnerability.py` | `results/vulnerability_mapping/` | NMF + CVI + cross-scan validation |
| `python scripts/03_sam_segmentation.py prova1` | `results/sam_segmentation/` | SAM regions + conservation report |
| `python scripts/04_run_ruotato.py` | `results_rotated/` | rotated scan, per-detector maps |
| `python scripts/ablation_denoising.py` | `results/vulnerability_mapping/ablation_denoising.json` | denoising ablation (Table VII) |
| `python scripts/make_submission_figures.py` | `submission/` | camera-ready paper figures + captions |
| `cd xrf-denoise && python scripts/05_full_pipeline.py` | `xrf-denoise/experiments/full_pipeline/` | full denoise-to-report pipeline |

`figures/` holds the figure set referenced by `main.tex`.

---

## Method in one paragraph

Per pixel: `CVI = max_k { w_k * sqrt(A_k * B_k) }` over five degradation
rules on 8th/99th-percentile normalised element maps (single-element
rules use `w_k * A_k`), smoothed with a sigma = 1 px Gaussian; zones cut
at 0.25 / 0.50 / 0.75.

| ID | Mechanism | Elements | w | Source |
|----|-----------|----------|-----|--------|
| R1 | TiO2/CaCO3 thermal mismatch | Ti/Ca | 0.40 | Mora et al. 1984 |
| R2 | Cu-based green pigment degradation | Cu | 1.00 | Scott 2002 |
| R3 | Lead white darkening (PbS) | Pb | 1.00 | Cotte et al. 2006 |
| R4 | Trapped moisture under TiO2 | Ti/Cu | 0.60 | Schiessl 1998 |
| R5 | Fe-catalyzed Pb oxidation | Fe/Pb | 0.80 | Gonzalez et al. 2017 |

---

## Key results (detector 10264, prova1 vs prova2)

- Element-map reproducibility: **r = 0.96-0.99** per element
  (Ca 0.991, Fe 0.990, Pb 0.990, Cu 0.977, Ti 0.960).
- Composite CVI stability: **W1 = 0.0029, pixel-wise r = 0.993,
  SSIM = 0.978** (all per-rule W1 < 0.01).
- Risk zones (prova1): 16 % low, 50 % moderate, 28 % elevated,
  **6 % critical** - dominated by R2 (Cu green) and R3 (lead white).
- NMF: K = 5 components matching the pigment set (Pb, Ca+Fe, Cu, Ti +
  residual), relative reconstruction error 7.8 %.
- SAM: **9 coherent regions**; Cu- and Pb-rich regions concentrate the
  elevated/critical risk.
- Denoising ablation: the U-Net improves structural agreement
  (SSIM 0.978 -> 0.980, per-element r up for Ti and Cu) while both
  configurations stay far below the W1 = 0.05 stability threshold.
- End-to-end runtime: **< 70 s cold, < 10 s warm** for 7200 spectra
  (Intel Core i5-12450H, CPU only).

## Citation

If you use this code or data, please cite the ICETRAN 2026 paper above.
