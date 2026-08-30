# Poster Plan - Heidelberg Conference

**Work:** *Geometry-resolved Characterization of a Dual-detector MA-XRF Scanner*
(the second paper - `PLAN.md`, `DRAFT.md`, `abstract.tex`, authors in `authors.md`)

> **TODO before designing anything:** check the conference's poster
> specification - required size (usually A0), portrait vs. landscape,
> whether a template/logo is mandatory, and whether they print it or
> you bring it. Everything below assumes **A0 portrait (841 × 1189 mm)**
> until confirmed.

---

## 1. The one-sentence takeaway (the whole poster serves this)

> **The difference between a scanner's two detectors - normally summed
> away - plus one tilted scan of the same painting, characterizes the
> instrument for free: detector response, acquisition geometry, canvas
> topography, and a learned fusion that beats summing by 10–18 % SNR.**

Every block on the poster must either set up or prove this sentence.
If a figure or paragraph doesn't, it goes in the backup slides of the
talk, not on the poster.

## 2. Most important parts of the work (what earns a place)

Ranked; the top four are the poster, the rest is optional filler.

1. **The idea / trick** - two detectors see the same spot from two
   angles. Tilting the canvas changes photon exit paths but not the
   detectors. Comparing frontal vs. tilted scans therefore *splits*
   the channel ratio R(E) into a detector part (tilt-invariant) and a
   geometric part (tilt-dependent, energy-structured). No calibration
   standards, no extra hardware - two routine scans.
   *Figure:* `presentation/figs/geometry_schematic.png`

2. **Two-stage decomposition result** (the hero figure):
   - R = det10264/det19511 falls from **≈5.8 at Ca Kα to 0.63 at
     Pb Lγ** - the two channels are up to sixfold apart, i.e. far from
     interchangeable.
   - A **3-parameter detector model** reproduces the frontal curve over
     two orders of magnitude; fitted absorber **973 ± 2 µm
     Be-equivalent** → cannot be an entrance window; corresponds to
     ~15–20 cm extra air path / collimation in front of det 19511.
   - Tilt shifts R **monotonically: +9.5 % at Ca (24σ) → +0.6 % at
     Pb Lγ**; a thick-sample fluorescence model with antisymmetric
     take-off angles reproduces the shape (lever arm s = 0.53 ± 0.10,
     tilt ≲ 8°).
   *Figure:* `results/detector_diff/geometry_fit.png` (both panels)

3. **Learned dual-detector fusion (Noise2Noise)** - the two channels
   are conditionally independent Poisson realizations; a 1D U-Net with
   *full inverse-variance loss weights* (prescribed by the measured
   R(E)) fuses them:
   - **+17.7 % mean SNR over simple summing** (median +11.9 %; quote
     10–18 % depending on aggregation) on held-out pixels; classical
     inverse-variance weighting gives only +0.9 %.
   - Six of eight lines positive: Pb Lℓ +69 %, Pb Lγ +33 %, Fe +28 %,
     Cu +17 %; Ca and Ti stay on the plain sum (low-energy level bias).
   - Ablation: same network with unweighted loss = **−0.1 %** - the
     gain comes from the measured variance structure, not from the
     architecture. In acquisition time, +17.7 % SNR ≈ 39 % longer dwell.
   *Figures:* `results/detector_diff/fusion_showcase.png` (visual
   before/after - great crowd magnet) and/or `fusion_benchmark.png`.

4. **Canvas topography from detector disagreement** - inverting
   per-pixel ratio residuals through the measured tilt response turns
   the eight lines into repeated measurements of local surface slope:
   a **relief map of the canvas from a single scan** (cross-scan
   r = 0.73, RMS ≈ 12°, χ²/dof = 0.85, 88 % of pixels consistent with
   pure geometry).
   *Figure:* `results/detector_diff/canvas_topography.png` (combined
   panel only).

5. *(Optional, small)* **Positioning error budget** - element maps move
   by up to **0.63 percentage points per degree** of mounting error
   (Ca +0.50 %/°, Ti +0.45 %/° vs. Pb Lβ −0.13 %/°), in the same
   energy order the model predicts; these are lower bounds.
   *Figure or just a 3-row mini-table from
   `results/registration/positioning_sensitivity.png`.*

6. *(Optional, one sentence in Methods)* Full-frame ratios mix in a
   field-of-view artifact big enough to flip signs - all ratios are
   computed on the **registered overlap** (affine registration,
   NCC 0.965). This is an honest-methods point reviewers/visitors like.

**Leave off the poster:** NMF corroboration details, CVI/first-paper
material, ablation sub-tables, GP hyperparameters, the flat-field/Hg
three-way correlation story (one sentence at most), all TODO numbers
awaiting the instrument builder.

## 3. Poster structure (A0 portrait, 3-column grid)

```
┌──────────────────────────────────────────────────────────┐
│  TITLE (short!)  ·  authors  ·  affiliations  ·  logos   │  ~12 %
│  [QR → GitHub repo]              [QR → paper/abstract]   │
├──────────────────┬──────────────────┬────────────────────┤
│ ① THE IDEA       │ ③ HERO RESULT    │ ④ LEARNED FUSION   │
│ schematic +      │ geometry_fit.png │ fusion_showcase    │
│ 4 sentences      │ (spans full      │ (crop: Pb Lℓ rows) │
│                  │  column width,   │ + 3 bullets        │
│ ② INSTRUMENT &   │  large!)         ├────────────────────┤
│ DATA             │                  │ ⑤ TOPOGRAPHY       │
│ mockup photo,    │ 5 bullets with   │ combined map +     │
│ 2 detectors,     │ the numbers      │ 2 bullets          │
│ 3 scans, 7200 px │ from §2.2        │                    │
├──────────────────┴──────────────────┴────────────────────┤
│ ⑥ TAKE-HOME BOX (large type, 3 lines max) · acknowledg-  │  ~10 %
│   ments · 3–5 references in small print                  │
└──────────────────────────────────────────────────────────┘
```

Reading order = numbered blocks, top-to-bottom within each column,
columns left-to-right. Number the blocks visibly (①②③…) so nobody
has to guess.

**Assertion headlines.** Name each block with its *finding*, not its
category:
- ① "One tilted scan separates detector from geometry" (not "Introduction")
- ③ "The two channels differ sixfold - and the model explains it" (not "Results")
- ④ "Learned fusion beats summing by 10–18 % SNR" (not "Deep learning")
- ⑤ "The discarded difference is also a relief map" (not "Topography")

A visitor who reads *only the headlines* should get the whole story.

## 4. Rules of thumb (first poster - what actually matters)

- **Design for three depths of engagement:** 3 seconds (title + hero
  figure), 30 seconds (headlines + take-home box), 3 minutes (bullets
  and numbers). Most visitors never reach depth three - that's normal;
  depth three is *you* talking them through it.
- **Word budget: ≤ 600 words total.** The paper is the archive; the
  poster is an advertisement for a conversation. No paragraphs longer
  than 3 lines. Bullets everywhere except the idea block.
- **Font sizes (A0):** title 90–110 pt, authors 48 pt, block headlines
  40–48 pt, body 28–32 pt, captions/references ≥ 20 pt. Test: print
  one block on A4 - if you can read it at arm's length, the A0 poster
  reads from 1.5 m.
- **Area budget:** ~45 % figures, ~25 % text, ~30 % white space. White
  space is not wasted space; a cramped poster reads as noise.
- **Figures:** every figure gets a one-line bold caption stating the
  conclusion ("Tilt shift is monotonic in energy - geometry and
  detector separate"). Axis labels must survive shrinking - regenerate
  at poster font sizes if needed (matplotlib: `plt.rcParams['font.size']`
  up to ~18–20 for these figure widths, export PDF/300-dpi PNG).
- **Color:** the existing figures already use a blue/orange palette - make that the poster accent palette and touch nothing else. One
  accent color for headlines and the take-home box.
- **QR codes:** one to the GitHub repo (data is already shared via
  Releases - reproducibility is a selling point, say so next to the QR),
  one to the paper/abstract PDF. Test both from 1 m away.
- **Take-home box:** the §1 sentence, verbatim, big. Bottom-center or
  bottom-right (where the eye exits).

## 5. Production checklist

- [ ] Confirm size/orientation + submission deadline from the
      conference (some want the PDF uploaded in advance).
- [ ] Build in a vector tool at true A0 (PowerPoint page 84.1 × 118.9 cm,
      LaTeX `beamerposter`/`baposter`/`tikzposter`, or Figma/Inkscape).
      Export **vector PDF**, embed fonts.
- [ ] Regenerate the 3 poster figures with poster-size fonts; crop
      `fusion_showcase.png` to the two Pb Lℓ rows (full 4×3 grid is
      too dense for a poster).
- [ ] Proof at 100 % zoom on screen + one A4 test print of the densest
      block.
- [ ] Print: matte paper (no glare under hall lighting) or fabric if
      flying with hand luggage; allow 2–3 working days.
- [ ] Prepare the **30-second pitch** (≈ the take-home box) and the
      **2-minute walkthrough** (blocks ① → ③ → ④, in that order).
      Practice both out loud once.
- [ ] Bring: a few A4 printouts of the poster or the abstract as
      handouts; the QR links double as a handout substitute.

## 6. Division of labor suggestion

Same spirit as `PLAN.md`: content blocks (§2 numbers, figure regeneration)
can be split per person along the existing A/B split; one person owns
layout/typography end-to-end so the poster has a single visual voice.
