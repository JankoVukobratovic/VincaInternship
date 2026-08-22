# Presentation Script — 10-Minute Conference Talk
## "Automated Chemical Vulnerability Assessment of Canvas Paintings from XRF Spectral Imaging Using Deep Learning and Foundation Models"

**Instructions:** Each section has a target time. Speak at a comfortable pace (~130 words/min). Cues in [brackets] are stage directions for you.

---

## SLIDE 1 — Title (0:00–0:20)

"Good [morning/afternoon]. My name is [name], and today I'll be presenting our work on automated chemical vulnerability assessment of canvas paintings using XRF spectral imaging. This is a collaboration between the School of Electrical Engineering at the University of Belgrade, Ars Mensurae and IDArtScience in Rome, and the Vinča Institute of Nuclear Sciences."

[Advance slide]

---

## SLIDE 2 — Motivation (0:20–1:50)

"Let me start with the problem we're trying to solve."

"Historical paintings degrade — and the dangerous part is that most degradation is invisible until it becomes irreversible. Pigments interact chemically: copper-based greens corrode, lead whites turn black, iron compounds catalyze oxidation in neighboring layers. By the time you can see the damage, significant material has already been lost."

"The standard approach is expert analysis — a conservator examines the painting, potentially taking micro-samples, and makes qualitative judgments. This is skilled, expensive, and doesn't scale."

"Macro-XRF scanning offers something much better: you get a full per-pixel elemental map of the entire painting surface. For our dataset, that's 7,200 spectra — one per spatial position. The catch is that interpreting 7,200 spectra manually is not practical."

"So our goal was to build a fully automated pipeline that takes raw XRF spectra as input and outputs a conservation-prioritization map — without requiring any expert-labeled training data. The whole thing runs in under 70 seconds on a standard laptop."

[Advance slide]

---

## SLIDE 3 — Pipeline Overview (1:50–3:20)

"Here's the high-level architecture — five stages chained together."

"We start with raw XRF spectra. The first stage is self-supervised denoising using a 1D U-Net. Then physics-based elemental extraction gives us 2D spatial maps for each element of interest. We run NMF decomposition as an independent corroboration step. Then comes the Chemical Vulnerability Index, which is the core risk assessment. And finally, SAM — the Segment Anything Model — partitions the canvas into coherent regions so we can give conservators per-region summaries."

"Two things worth emphasizing. First: no stage requires labeled data. The denoising is self-supervised, the extraction is physics-based, the CVI uses published conservation chemistry, and SAM is a foundation model used zero-shot. Second: end-to-end runtime is under 70 seconds cold — U-Net inference and SAM together account for 80% of that time, the rest is negligible."

[Advance slide]

---

## SLIDE 4 — Spectral Analysis (3:20–5:20)

"Let me go a bit deeper into the first two stages."

**Denoising:**
"XRF photon counts follow Poisson statistics — which means you can't use standard Gaussian denoisers without introducing systematic bias. We use a Noise2Self approach: for each spectrum, we randomly split the observed counts into two statistically independent halves using a Binomial split. We train a 1D U-Net to predict one half from the other. This works because the two halves share the same underlying signal but have independent noise — so the network learns to recover signal without ever seeing clean data."

"The architecture is a 4-level U-Net with 1.66 million parameters, trained with Poisson negative log-likelihood loss."

**Elemental Extraction:**
"After denoising, we extract element intensities using a 5-point linear energy calibration — R² exceeds 0.9999. For each element, we integrate counts in a calibrated window around the fluorescence peak, subtract a linearly interpolated background from adjacent sidebands, and apply spectral overlap corrections where needed."

"The trickiest case is arsenic Kα and lead Lα — their peaks are only 0.007 keV apart, which is below the detector's energy resolution. We handle this using a regression-estimated Pb Lα/Lβ ratio to separate the two contributions."

"The output is a set of 2D spatial maps — you can see them on this slide — for Ca, Ti, Fe, Cu, and Pb."

"As a sanity check, we run NMF on the raw spectra. With K=5 components it independently recovers the same five material groups: a Ca/Fe ochre layer, lead white, localized Cu-based pigment, titanium white, and a Compton scattering background. Reconstruction error is under 5%. This agreement between a supervised physics-based method and an unsupervised factorization gives us confidence that we're extracting real materials, not spectral artifacts."

[Advance slide]

---

## SLIDE 5 — How Do We Measure Risk? (5:20–7:20)

"Now for the core contribution: the Chemical Vulnerability Index."

"The CVI is defined as the per-pixel maximum over five degradation rules drawn from the conservation chemistry literature. For rules involving two incompatible elements, we use the geometric mean of their normalized intensities — this ensures a high score only when both elements are genuinely co-present. Single-element rules use just that element's intensity directly."

"You can see the five rules on this slide. R2 and R3 — copper-based pigment degradation and lead white darkening — carry the highest weight of 1.0, because they directly destroy the visible image and are essentially irreversible. R5, iron-catalyzed lead oxidation, gets 0.8 — in this painting it propagates along figural contours. R4, moisture trapped under titanium white, gets 0.6 — it's a compound mechanism. R1, thermal mismatch between TiO₂ and CaCO₃, gets 0.4 — slowest and most localized."

"The CVI lives in [0,1] by construction, so we define four zones at fixed thresholds: Low, Moderate, Elevated, and Critical. These align directly with the conservator action taxonomy: no action, monitor, intervene, urgent."

"One important property: the weights are interpretable and easy to adjust. You can adapt the regime — say, prioritizing mechanical risk in a thermally stressed object — with a one-line edit. A sensitivity analysis shows that ±15% perturbation in any weight changes the mean CVI by less than 8% and preserves the top-decile pixel ranking in over 92% of locations."

[Advance slide]

---

## SLIDE 6 — Results (7:20–9:20)

"Let me walk through the three main results."

**Reproducibility:**
"We scanned the same mockup painting twice, seven days apart, under identical conditions. For all five CVI-relevant elements, the Pearson correlation between the two scans exceeds 0.98 — the highest being Ca and Fe at 0.992 and 0.993. On the composite CVI, the Wasserstein distance between the two scan distributions is 0.0054, pixel-wise correlation is 0.994, and SSIM is 0.982. This tells us the pipeline is stable — the variability you see is photon-counting noise, not systematic drift."

"We also ran an ablation removing the U-Net denoising stage. Without denoising, the Wasserstein distance increases by 65% and SSIM drops by 0.018 — so denoising does contribute, but the dominant source of stability is the physics-based extraction itself."

**CVI Distribution:**
"Under our chemistry-first weight regime, 5.9% of pixels fall in the critical zone and 30.3% in the elevated zone. The dominant mechanisms are R2 and R3 — copper degradation and lead darkening — each exceeding the elevated threshold on over 20% of pixels. R5 appears as narrow linear features tracing the figural contours."

**SAM Regions:**
"SAM automatically identifies 13 coherent regions. Three representative ones illustrate the output: a lead-dominated region with peak CVI of 0.93, flagged for urgent assessment; a copper-dominated region with peak CVI of 0.97, flagged for monitoring; and a titanium/copper overlap region signaling moisture entrapment under what is likely a modern restoration layer."

"Each region comes with its dominant material, mean and max CVI, and dominant degradation mechanism — directly actionable for a conservator, generated entirely from raw spectra."

[Advance slide]

---

## SLIDE 7 — Further Work (9:20–10:00)

"I'll close with the main directions forward."

"The validation in this paper is internal consistency — two scans of one mockup. The critical next step is comparing CVI predictions against expert conservator assessments on real paintings with documented degradation outcomes. That's the ground truth we need."

"Methodologically, we want to integrate NMF abundance maps into the CVI inputs, which would capture material mixtures invisible to single-element thresholds. We also want to extend to paintings with more complex histories, where the rule weights may need adaptation."

"And on the practical side — building a tool conservators can actually use, without requiring Python expertise."

"Thank you. I'm happy to take questions."

---

## Key Numbers to Know Cold (without looking at slides)

| Fact | Value |
|------|-------|
| Scan grid | 120 × 60 = 7,200 pixels, 3 s/pixel |
| Two scans, interval | 7 days apart |
| Reproducibility (all elements) | r > 0.98 |
| Composite CVI cross-scan | W₁ = 0.0054, r = 0.994, SSIM = 0.982 |
| Critical zone coverage | 5.9% |
| Elevated zone coverage | 30.3% |
| SAM regions | 13 |
| End-to-end runtime (cold) | < 70 seconds |
| End-to-end runtime (warm cache) | < 10 seconds |
| U-Net parameters | ~1.66 million |
| Calibration R² | > 0.9999 |
| NMF components (K) | 5 |
| NMF reconstruction error | < 5% |
| As/Pb Lα energy separation | 0.007 keV (below detector FWHM) |
| Denoising ablation W₁ change | +65% without denoising |
| Weight sensitivity | ±15% → <8% change in mean CVI; >92% top-decile ranking preserved |
| Hardware | Intel Core i5-12450H, 16 GB RAM |

---

## Likely Questions and Answers

**Q: How do you choose the CVI weights? Are they empirically validated?**
A: "The weights encode conservation priority — how irreversible the damage is, how fast it propagates, how visually disruptive the outcome is — based on the referenced conservation literature. They are not empirically fitted to degradation outcome data. We show they're robust to ±15% perturbation, but true validation requires comparison against conservator assessments on paintings with known degradation histories. That's explicitly our next step."

**Q: Why SAM and not k-means or superpixels?**
A: "We use SAM as a zero-shot, no-tuning region generator. Classical alternatives like SLIC or k-means could substitute without changing the upstream chemistry. We chose SAM because it produces coherent, conservator-aggregable regions out of the box on low-contrast XRF input without per-image parameter tuning. The contribution isn't the segmentation algorithm — it's the per-region CVI aggregation it enables."

**Q: How does the pipeline handle the Pb/As overlap?**
A: "Arsenic Kα and Pb Lα are 0.007 keV apart — below the detector's energy resolution, so they can't be resolved directly. We use the known Pb Lα/Lβ intensity ratio, estimated by regression from Pb-rich pixels, to subtract the Pb contribution from the overlapping window and recover the As signal."

**Q: Does this generalize beyond the mockup canvas?**
A: "The physics-based extraction generalizes naturally to any XRF dataset with similar detectors. The CVI rules are drawn from the conservation literature and describe real degradation mechanisms, so they should apply broadly. However, paintings with different materials — for example, different historical pigments or unusual restoration interventions — may require adapted rule sets or weight regimes. We're careful to position this as a workflow contribution, not a universally calibrated tool."

**Q: What if the scan has a different number of pixels?**
A: "Runtime scales approximately linearly with pixel count, and the element map caching makes reruns with changed weights very fast — under 10 seconds warm. The calibration and extraction are parameterized by energy windows, not pixel count, so they adapt automatically."
