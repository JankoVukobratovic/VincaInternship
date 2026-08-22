# Slide Content Suggestions
## "Automated Chemical Vulnerability Assessment of Canvas Paintings from XRF Spectral Imaging Using Deep Learning and Foundation Models"

**10-minute talk · 7 content slides + title = ~8 slides**

---

## Slide 1 — Title
- **Title** (as in paper)
- Authors + affiliations (ETF Belgrade, Ars Mensurae, IDArtScience, VINARH/Vinča)
- Conference name, date
- *Visual: photograph of the mockup canvas painting (fig_mop.png)*

---

## Slide 2 — Motivation
**Headline:** *Why automate conservation risk assessment?*

**Bullets:**
- Historical canvas paintings degrade silently — damage often invisible until irreversible
- Traditional assessment: expert-driven, time-consuming, qualitative
- XRF scanning gives us per-pixel elemental data across the entire surface — but interpreting 7,200 spectra by hand is impractical
- **Goal:** Build an automated, label-free pipeline that turns raw XRF spectra into a conservation-prioritization map — under 70 seconds, no expert annotation required

**Visual idea:** Side-by-side: conservator manually examining a painting vs. a heatmap of chemical risk zones

---

## Slide 3 — Pipeline Overview
**Headline:** *Five stages, one automated workflow*

**Diagram (linear flow with 5 boxes):**
```
Raw XRF Spectra
      ↓
[1] Self-Supervised Denoising (1D U-Net)
      ↓
[2] Physics-Based Elemental Extraction (calibrated ROI + background subtraction)
      ↓
[3] NMF Decomposition (corroboration only)
      ↓
[4] Chemical Vulnerability Index (CVI) — literature-grounded rules
      ↓
[5] SAM Region Segmentation → per-region risk summary
```

**Key point to highlight:**
- No labeled training data at any stage
- Runs in <70 s on a standard laptop (Intel i5-12450H, 16 GB RAM)

---

## Slide 4 — Spectral Analysis
**Headline:** *From noisy counts to clean element maps*

**Two sub-points:**

**4a. Self-Supervised Denoising**
- XRF spectra follow Poisson statistics → classical denoisers introduce bias
- Noise2Self approach: split each spectrum into two Poisson half-draws → train 1D U-Net to predict one half from the other
- No clean reference spectrum needed

**4b. Elemental Extraction**
- 5-point linear energy calibration (R² > 0.9999)
- ROI integration with linear background subtraction
- Spectral overlap corrections: Cu Kβ/Kα ratio for Zn, Pb/As regression (ΔE = 0.007 keV — below detector FWHM)
- Output: 2D spatial maps for Ca, Ti, Fe, Cu, Pb Lα, Pb Lβ

**Visual:** fig_element_maps.pdf — the 5 element spatial maps

**Bonus bullet:** NMF (K=5) independently recovers the same 5 material groups → cross-validates physics-based extraction (reconstruction error < 5%)

---

## Slide 5 — How Do We Measure Risk? (CVI)
**Headline:** *Literature-grounded Chemical Vulnerability Index*

**Formula (display equation):**
```
CVI(i,j) = max_k { w_k · √(A_k(i,j) · B_k(i,j)) }
```

**Rules table (compact — all 5 rows):**

| ID | Mechanism | Elements | w |
|----|-----------|----------|---|
| R1 | TiO₂/CaCO₃ thermal mismatch | Ti / Ca | 0.40 |
| R2 | Cu-based green pigment degradation | Cu | 1.00 |
| R3 | Lead white darkening (PbS) | Pb | 1.00 |
| R4 | Trapped moisture under TiO₂ | Ti / Cu | 0.60 |
| R5 | Fe-catalyzed Pb oxidation | Fe / Pb | 0.80 |

**Key design choices to say out loud:**
- Geometric mean → high score ONLY when BOTH incompatible elements co-occur
- Weights encode conservation priority (irreversibility, propagation speed, visual impact) — not fitted, not arbitrary
- Four zones: Low / Moderate / Elevated / Critical (< 0.25 / 0.25–0.50 / 0.50–0.75 / ≥ 0.75) — aligns with conservator action taxonomy
- Sensitivity: ±15% weight perturbation → <8% change in mean CVI, top-decile ranking preserved in >92% of pixels

---

## Slide 6 — Results
**Headline:** *Stable, reproducible, actionable risk maps*

**Three columns or three sub-bullets:**

**① Reproducibility (r > 0.98)**
- Two independent scans, 7 days apart
- All 5 CVI-relevant elements: r > 0.98 (Pearson)
- Composite CVI: W₁ = 0.0054, r = 0.994, SSIM = 0.982
- Cross-scan agreement comes primarily from physics-based extraction, not denoising

**② CVI Zone Distribution**
| Zone | Coverage |
|------|----------|
| Low | 13.9% |
| Moderate | 49.9% |
| Elevated | 30.3% |
| Critical | 5.9% |
- R2 (Cu degradation) elevated on ~22% of pixels; R3 (Pb darkening) on ~19%
- R5 (Fe/Pb oxidation) traces figural contours

**③ SAM Segmentation — 13 regions**
- Pb-dominated segment: max CVI 0.93, mean 0.54 → urgent (R3)
- Cu-dominated segment: max CVI 0.97, mean 0.45 → monitor (R2)
- Ti/Cu overlap segment: mean CVI ~0.45 → moisture entrapment (R4)

**Visual:** fig_cvi_map.png (left: continuous map, right: 4-zone classification)
OR fig_sam_segmentation.pdf side by side with CVI map

---

## Slide 7 — Further Work
**Headline:** *What comes next*

**Bullets:**
- Validate CVI against expert conservator assessments on real paintings with known degradation outcomes (current validation is internal consistency on one mockup)
- Extend to real paintings with documented conservation histories — may require adapted rule weights
- Integrate NMF abundance maps as CVI inputs to capture material mixtures invisible to single-element thresholds
- Resolve As Kα / Pb Lα overlap (ΔE = 0.007 keV) with higher-resolution detectors or deconvolution
- Evaluate on multi-detector setups and larger-format scans to test linear runtime scaling
- Develop GUI or REST API for direct use by conservators (no Python expertise required)

---

## Timing Guide (10 minutes)
| Slide | Topic | Time |
|-------|-------|------|
| 1 | Title | 0:20 |
| 2 | Motivation | 1:30 |
| 3 | Pipeline Overview | 1:30 |
| 4 | Spectral Analysis | 2:00 |
| 5 | CVI | 2:00 |
| 6 | Results | 2:00 |
| 7 | Further Work | 0:40 |
| — | Buffer / Q&A transition | 0:00 |
| **Total** | | **10:00** |
