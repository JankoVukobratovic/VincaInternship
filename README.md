# Automated Chemical Vulnerability Assessment of Paintings from XRF Spectral Imaging

Companion code for:

> D. Pešić, J. Vukobratović, A. Stojanović, G. Ristori, S. Ridolfi,
> M. Gajić-Kvaščev, G. Kvaščev, **"Automated Chemical Vulnerability
> Assessment of Canvas Paintings from XRF Spectral Imaging Using Deep
> Learning and Foundation Models"**, ICETRAN 2026.
> (draft: [`ICETRAN.pdf`](ICETRAN.pdf))

A macro-XRF scanner rasters a painting pixel by pixel and records a full
X-ray fluorescence spectrum at every point. This repository turns those
raw spectra into a conservator-facing risk assessment, fully
automatically and with no expert-labeled training data:

1. **Self-supervised denoising** — a 1D U-Net trained with Poisson
   splitting (no clean targets needed) suppresses photon-counting noise.
2. **Elemental extraction** — per-pixel net peak intensities
   (background-subtracted, spectral-overlap-corrected) become 2-D
   element maps (Ca, Ti, Fe, Cu, Pb, …).
3. **NMF corroboration** — blind non-negative matrix factorization
   independently recovers the same material layers, validating the
   physics-based extraction.
4. **Chemical Vulnerability Index (CVI)** — a literature-grounded
   composite index aggregates five published degradation mechanisms over
   the element maps and classifies every pixel into four risk zones.
5. **SAM segmentation** — Meta's Segment Anything Model partitions the
   painting into coherent regions; per-region mean CVI and dominant
   degradation mechanism produce an actionable conservation report.

Cross-scan reproducibility of the element maps reaches **r > 0.98**, the
composite CVI is stable across independent scans (**W₁ ≈ 0.003–0.005,
pixel-wise r ≈ 0.99, SSIM ≈ 0.98**), and the full 7200-spectrum pipeline
runs in about **70 s** on a laptop CPU.

**Datasets:** `aurora-antico1-prova1`, `aurora-antico1-prova2`
(120×60 = 7200 px, repeated after 7 days) and the rotated scan
`antico1-prova4-ruotato` (80×45 = 3600 px) · **Detectors:** 10264, 19511 ·
**Dwell:** 3 s/px · 1024 channels, ~0.0292 keV/channel

📄 Detailed lab notes, calibration and validation (in Serbian):
[`IZVESTAJ.md`](IZVESTAJ.md)

---

## Project structure

```
VincaInternship/
├── README.md                       # this file
├── ICETRAN.pdf                     # conference paper (context for everything)
├── IZVESTAJ.md                     # detailed lab report (Serbian)
├── requirements.txt
│
├── src/
│   ├── xrf_core.py                 # XRF analysis engine (run_scan API)
│   └── elements.json               # element line definitions + colors
│
├── scripts/                        # entry points (run from project root)
│   ├── 01_run_analysis.py          # element maps, prova1 vs prova2 (det 10264)
│   ├── 02_vulnerability.py         # NMF + CVI + cross-scan validation
│   ├── 03_sam_segmentation.py      # SAM segmentation + per-region risk
│   ├── 04_run_ruotato.py           # rotated scan: per-detector maps, sum, diff
│   ├── compare_Ti.py               # Ti map: prova1 vs rotated scan
│   └── generate_signals.py         # one annotated spectrum plot per pixel
│
├── results/                        # outputs for prova1/prova2 (PNG + npy cache)
├── results_rotated/                # outputs for the rotated scan
│
├── Resources/                      # RAW MCA DATA — gitignored, see Setup
├── models/                         # SAM checkpoint — gitignored, see Setup
│
└── xrf-denoise/                    # U-Net denoising subproject
    └── scripts/05_full_pipeline.py # denoise → maps → NMF → CVI → SAM → report
```

---

## Setup

### 1. Python environment

```bash
pip install -r requirements.txt
```

### 2. Raw MCA data

The scripts look for the scans first under `Resources/`, then in the
repository root (both layouts work):

```
Resources/
├── aurora-antico1-prova1/
│   ├── 10264/None_1.mca … None_7200.mca
│   ├── 19511/None_1.mca … None_7200.mca
│   └── stacked/ · colonneXrighe.txt · map.png
├── aurora-antico1-prova2/          # same structure (repeat scan)
└── antico1-prova4-ruotato/         # rotated scan, 80×45 px
    ├── 10264/None_1.mca … None_3600.mca
    └── 19511/None_1.mca … None_3600.mca
```

The raw data are not tracked in git (≈54 000 files, ≈400 MB) — download
them from the repository's **Releases** page and unpack into `Resources/`:

```bash
curl -L -o xrf-data.zip \
  https://github.com/JankoVukobratovic/VincaInternship/releases/download/data-v1/aurora-antico1-xrf-data.zip
mkdir -p Resources
tar -xf xrf-data.zip -C Resources     # or: unzip xrf-data.zip -d Resources
```

Element maps are cached as `.npy` after the first run, so every rerun is
incremental.

### 3. SAM checkpoint (only for `03_sam_segmentation.py` / SAM stage of the full pipeline)

Download `sam_vit_b_01ec64.pth` (~375 MB) from the official
[Segment Anything repository](https://github.com/facebookresearch/segment-anything#model-checkpoints)
and place it at `models/sam_vit_b_01ec64.pth`:

```bash
curl -L -o models/sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### 4. U-Net checkpoint (only for `xrf-denoise/scripts/05_full_pipeline.py`)

The trained denoiser ships with the repository
(`xrf-denoise/experiments/A_scratch/checkpoints/best_model.pt`, 6.4 MB) —
no action needed. Retrain with the `xrf-denoise` scripts if desired.

---

## Running

```bash
# 1) Element maps (fills the npy cache; prova2 processed when available)
python scripts/01_run_analysis.py

# 2) Chemical risk mapping: NMF + CVI + prova1-vs-prova2 validation
python scripts/02_vulnerability.py

# 3) SAM segmentation + per-region risk report (needs SAM checkpoint)
python scripts/03_sam_segmentation.py prova1

# 4) Rotated scan: per-detector maps, detector sum and difference
python scripts/04_run_ruotato.py

# Helpers
python scripts/compare_Ti.py          # Ti: prova1 vs rotated scan
python scripts/generate_signals.py    # 14400 per-pixel spectrum PNGs (slow)

# Full denoising pipeline (from xrf-denoise/)
cd xrf-denoise
python scripts/05_full_pipeline.py [--no-sam]
```

Scripts degrade gracefully: phases whose inputs are missing (raw
spectra, prova2, checkpoints) are skipped with a clear note instead of
failing. `02` and `03` consume the cache produced by `01`.

### Where the results land

| Script | Output folder | Main artifacts |
|--------|---------------|----------------|
| `01_run_analysis.py` | `results/10264/` | `prova1/element_maps.png`, `prova2/element_maps.png`, `compare/diff_…png`, `compare/strips_…png` |
| `02_vulnerability.py` | `results/vulnerability_mapping/` | `1_nmf_components.png` … `7_rule_stability.png` |
| `03_sam_segmentation.py` | `results/sam_segmentation/` | 6 figures + `conservation_report.txt` |
| `04_run_ruotato.py` | `results_rotated/` | `elements_det*.png`, `elements_sum_detectors.png`, `diff_detectors.png`, `individual/`, `spectra/` |
| `compare_Ti.py` | `results_rotated/` | `Ti_prova1_vs_ruotato.png` |
| `05_full_pipeline.py` | `xrf-denoise/experiments/full_pipeline/figures/` | publication figure, denoise comparison, CVI maps, `risk_report.txt`, `pipeline_summary.json` |

---

## Method summary (paper Tables I–II)

**Degradation rules** (chemistry-first severity weights):

| ID | Mechanism | Elements | w | Source |
|----|-----------|----------|-----|--------|
| R1 | TiO₂/CaCO₃ thermal mismatch | Ti/Ca | 0.40 | Mora et al. 1984 |
| R2 | Cu-based green pigment degradation | Cu | 1.00 | Scott 2002 |
| R3 | Lead white darkening (PbS) | Pb | 1.00 | Cotte et al. 2006 |
| R4 | Trapped moisture under TiO₂ | Ti/Cu | 0.60 | Schiessl 1998 |
| R5 | Fe-catalyzed Pb oxidation | Fe/Pb | 0.80 | Gonzalez et al. 2017 |

Per pixel: `CVI = max_k { w_k · sqrt(A_k · B_k) }` on 8th/99th-percentile
normalised maps (single-element rules use `w_k · A_k`), smoothed with a
σ=1 px Gaussian. Zones at quartile cut-offs: low < 0.25 ≤ moderate <
0.50 ≤ elevated < 0.75 ≤ critical.

**Detected elements** (validation in `IZVESTAJ.md` §4, §9):

| Element | Line | keV | Confidence | Pigment / source |
|---------|------|-----|------------|------------------|
| **Pb** | Lα/Lβ/Ll/Lγ | 10.55/12.61/… | High | Lead white |
| **Ca** | Kα | 3.69 | High | Lime/chalk (CaCO₃) |
| **Fe** | Kα | 6.40 | High | Red and yellow ochre (contours, figure) |
| **Cu** | Kα | 8.05 | High | Cu-based green pigment |
| **Ti** | Kα | 4.51 | High | Titanium white (commercial ground layer) |
| **Zn** | Kα | 8.64 | High | Localised (corrected for Cu Kβ) |
| **K** | Kα | 3.31 | Marginal | Weak, near detection limit |
| ~~Sn~~ | Kα | 25.27 | Excluded | Acquisition artifact |
| ~~As~~ | Kα | 10.54 | Absent | Signal at 10.54 keV is Pb Lα |

---

## Key results (single detector 10264, per the paper)

- Element-map reproducibility prova1 ↔ prova2: **r > 0.98** for all five
  CVI-relevant elements (Ca, Ti, Fe, Cu, Pb).
- Composite CVI stability across scans: **W₁ ≈ 0.003–0.005**,
  pixel-wise **r ≈ 0.99**, **SSIM ≈ 0.98**.
- Zone coverage (prova1): ~14–16 % low, ~50 % moderate, ~28–30 %
  elevated, **~6 % critical** — dominated by R2 (Cu green pigment) and
  R3 (lead white darkening).
- SAM identifies **13 coherent regions**; Cu- and Pb-rich regions
  concentrate the elevated/critical risk.
- End-to-end runtime: **< 70 s cold, < 10 s warm** for 7200 spectra
  (Intel Core i5-12450H, CPU only).

## Citation

If you use this code or data, please cite the ICETRAN 2026 paper above.
