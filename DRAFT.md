# Draft — Results (dual-detector paper)

Working text for the Results section, in paper order. Numbers come
from `results/detector_diff/` (scripts 07, 07b) computed on the
registered-overlap ratios of `results/registration/overlap_ratios.csv`
(script 08); the full-frame ratios of script 06 are kept only for the
frame-artifact comparison. Placeholders marked [TODO] wait for Person
B's remaining results or for confirmation from the instrument builder.

## Per-element efficiency ratios

The per-element count ratio R = det10264 / det19511 is strongly
energy-structured: it falls monotonically from about 5.8 at the Ca Ka
line (3.7 keV) to 0.63 at Pb Lg (14.8 keV), crossing unity near 8 keV
(Table 1). All ratios are computed on the registered overlap region
of the compared scans: a crop test showed that full-frame ratios mix
in a field-of-view artifact (different frame coverage of the two
scans), which in the tilted comparison is large enough to flip the
sign of the shift for several lines. The two frontal scans give
ratios consistent to within their bootstrap errors for all reliable
lines; K and Zn are excluded (overlapping lines / low counts). The
two channels of the scanner are therefore far from interchangeable:
the same pigment produces up to a sixfold different signal depending
on which detector is read out.

## Detector model

A three-parameter detector model — an energy-independent factor, a
differential low-energy absorber (Be-equivalent), and the ratio of
active Si thicknesses — reproduces the frontal ratio curve across two
orders of magnitude (Fig. X, left). The fitted absorber is
974 +- 1 um Be-equivalent, i.e. 40-100x a typical SDD entrance
window. The low-energy imbalance therefore cannot be a window
difference: it corresponds to roughly 15-20 cm of extra air path (or
equivalent collimation) in front of detector 19511. [TODO: confirm
nominal head geometry with the instrument builder.]

## Tilt-induced shift and geometric model

Tilting the canvas forward increases the ratio at every reliable
line, with a monotonic energy structure: +9.5 % at Ca Ka decaying to
+0.6 % at Pb Lg (Fig. X, right), significant at up to 24 sigma
against the frontal repeatability. (On full-frame ratios the shift
appears to change sign near 6.5 keV; the overlap crop test shows this
zero crossing is a frame-coverage artifact, resolving the Cu anomaly
noted in an earlier draft.) A thick-sample fluorescence model in
which the tilt moves the two effective take-off angles
antisymmetrically reproduces this shape. With a single tilt the
individual take-off angles are not identifiable; the identifiable
parameters are the lever arm s = 0.53 +- 0.10 (degrees of take-off
change per degree of canvas tilt, i.e. an effective take-off shift of
4.1 +- 0.8 degrees at the nominal tilt), the matrix-attenuation
energy scale Ec = 3.55 +- 0.48 keV, and a solid-angle offset of
+1.22 +- 0.19 %. The fit quality improves markedly over the
full-frame version (chi2/dof 19.1/5 against 48.3/5). A nonparametric
Gaussian-process regression of the same data reproduces the shape
independently, and the parametric curve stays within 1.1 sigma of the
GP mean over the whole energy range — the shape is a property of the
data, not of the model choice.

## Tilt angle from foreshortening

The mounting angle of the tilted scan was not recorded; we recover it
from the data. A forward tilt compresses the scanned image vertically
by cos(theta), so registering the tilted scan onto a frontal scan
with independent x/y scales measures the tilt through the scale ratio
sy/sx = 1/cos(theta), with the unknown scan step sizes cancelling.
Joint registration over five element maps gives theta = 7.7 +- 1.0
(element spread) +- 1.8 (registration floor) degrees. Foreshortening
at this angle sits near the resolution limit of the method: a no-tilt
control pair (prova2 onto prova1) returns an apparent "5.3 degrees",
so we quote the result as an upper bound (theta <~ 8 degrees) until
the instrument builder confirms the mounting angle; the stage-2
geometric fit uses the nominal 7.7 degrees as a conditional input.
The registration floor is itself a finding: two nominally identical
frontal scans differ in vertical scale by 0.43 %, a direct measure of
the positioning/calibration drift between scanning sessions. [TODO:
compare with the builder-confirmed angle if it arrives.]

## Still to merge (Person B)

- Registered detector-difference maps and the geometric
  non-uniformity map (script 08+). Numbers ready in
  `results/detector_diff/flatfield_map.txt`, including the overlap with
  the scatter-tail artifact (r = 0.86) that ties the flat-field, the
  masked artifact bands and the acquisition geometry into one story.
- Learned fusion: benchmark table ready in
  `results/detector_diff/fusion_weighted.txt` (summing vs
  inverse-variance vs N2N; +5.4% mean SNR over summing on unseen
  pixels, with the Ti and Ca caveats of PLAN 8.10). Prose still to
  write.
- Final figure numbering and Table 1 formatting. Table 1 rows are
  emitted by script 07 into
  `results/detector_diff/handoff2_ratio_curve.md`.
