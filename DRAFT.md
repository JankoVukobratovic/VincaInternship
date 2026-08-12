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

## Instrument and data

The scanner carries two silicon drift detectors, 10264 and 19511, that
read the same excited pixel simultaneously. Three scans of the same
canvas are used here: two frontal scans seven days apart (prova1,
prova2; 120 x 60 pixels) and one scan with the canvas tilted forward
(ruotato; 80 x 45 pixels), each pixel a 1024-channel spectrum at
0.0292 keV per channel and 3.0 s dwell. The frontal pair fixes the
repeatability of everything reported below; the tilted scan supplies the
geometry contrast.

Eight emission lines carry the analysis: Ca Ka, Ti Ka, Fe Ka, Cu Ka and
the four Pb L lines. K and Zn are excluded throughout (overlapping lines
and low counts). Net intensities use a fixed half-window with linear
sideband subtraction, the same integrator as the rest of the pipeline.
All cross-scan comparisons are made after the affine registration of
[Sec. registration], on the common footprint only.

## Detector fusion

Summing the two channels is what the instrument does by default. Two
alternatives are compared against it on the same metric: inverse-variance
weighting with per-element weights, and a learned fusion. The metric
needs no ground truth: prova1 and prova2 are two noise realizations of
the same painting, so for each element map SNR = mean / (std of the
difference / sqrt 2), with the Pearson correlation of the two scans as a
consistency check. Weights are estimated on one checkerboard of pixels
and all numbers are evaluated on the other.

Inverse-variance weighting gains 0.9 % in mean SNR over plain summing,
and nothing at all on the four Pb lines (Table X). That null result is
informative rather than disappointing: for Poisson counting with
proportional efficiencies the plain sum is a sufficient statistic, so no
linear reweighting can beat it, and measuring the expected nothing shows
the two channels are Poisson-limited over the whole range.

The learned fusion is a 1D U-Net trained Noise2Noise across the two
detectors: the two channels observe the same pixel at the same instant,
so each is a noisy realization of the other's expected spectrum once the
response ratio R(E) is divided out. There is never a clean target, and
none is needed. Training uses prova1 and the tilted scan with spatial
blocks held out for validation; prova2 is never seen. The loss is
restricted to 3.5-15.5 keV, outside which R(E) is extrapolation.

On the pixels the network never saw, the learned fusion gains 17.7 % in
mean SNR over summing, against the 0.9 % of the classical weighting
(Table X, Fig. Y). Six of the eight lines improve, Pb Ll by 68.6 %, Pb
Lg by 33.3 % and Fe by 28.5 %; Ca loses 9.4 % and Ti 3.7 %. In counting
terms a 17.7 % SNR gain is worth 39 % more acquisition time on the same
canvas.

Two decisions carry that result, and both are read off the measured
R(E) rather than tuned. First, the loss weights each channel by the
inverse variance of its target: rescaling detector B by R multiplies its
variance by R^2 while the mean grows only by R, and the Pb lines are two
orders of magnitude brighter than Ca, so an unweighted MSE is dominated
twice over and leaves the light lines under-fitted. Second, the two
prediction directions are combined per channel inverse-variance, R : 1,
rather than averaged -- at the Ca line that is 85:15, which is what the
classical weighting independently finds (w = 0.89). The ablation is
unambiguous: with both corrections off, the same network gains -0.1 %,
i.e. it does not beat summing at all (Table Z). The gap between the two
learning rates tried is smaller than the gap between weighting schemes.

Because a network that merely blurs a map also raises this SNR, every
row carries two guards: the ratio of spatial coefficients of variation
against the summed map (0.97 on average, so contrast is preserved) and
the correlation with it (0.95 or better everywhere). The two lines that
lose are exactly the two where the guards flag structure: Ca sits at
cv 0.69, the residue of shrinkage toward the mean spectrum where
detector B contributes almost no counts, and Ti at cv 1.16, where the
network adds variance instead of removing it.

## Validation

Three things are held out rather than assumed. Spatially, the network
never sees the validation blocks, and all fusion numbers above are
quoted on those pixels. Across scans, prova2 is a test scan for the
network in every configuration. Across models, the choice between
learning rates was made on validation loss before any benchmark number
was computed; the choice between weighting schemes follows the variance
argument above, not the benchmark.

What this does not establish should be stated plainly: prova1 and prova2
differ in noise realization, not in content, so the evaluation tests
generalization across noise on one painting, not across paintings. The
repeatability floor of the whole pipeline is set by the two frontal
scans, 0.96 % RMS on the element maps and 0.4 % on the vertical scale of
the registration.

## Map figures

Fig. M shows what the two channels see differently and where. The ratio
log(d10264/d19511) removes composition to first order, so its spatial
structure is geometry: the same inner rectangle appears at Ca Ka and at
Pb La, at opposite ends of the response curve. Taking the median over
the eight lines after removing each line's mean leaves the geometric
non-uniformity of the ratio, 8.9 % RMS with a +-16 % span, reproducible
between the two frontal scans at r = 0.70.

That map is not a detector quirk. It coincides with the acquisition
artifact already known from the scatter tail above 12.95 keV, which is
masked out of the NMF input for exactly this reason: the two agree pixel
by pixel at r = 0.86, and the ratio runs +7.2 % where the scatter band
is strong against -2.7 % elsewhere. The Hg-line rectangle is the
complementary region, carrying the opposite sign at r = -0.24. The
non-uniformity of the channel ratio, the scatter frame and the Hg
rectangle are one acquisition geometry seen three ways, and the
flat-field map divides all three out.

## Positioning sensitivity

The mounting angle is not a nuisance parameter here but a measurable
error budget. Registering the frontal scan into the tilted scan's frame
and comparing every element map pixel by pixel gives the change each map
suffers from a known change of mounting. The overall level difference
between the two scans (-0.4 %) is degenerate with session drift and is
removed as a common mode; what remains is the differential, element to
element, which is what pigment identification actually depends on.

Per degree of tilt, the summed maps move by +0.50 % at Ca Ka and
+0.45 % at Ti Ka, against -0.13 % at Pb Lb and -0.08 % at Pb Lg; Fe, Cu
and Pb Lg do not clear the 0.12 %/deg repeatability floor set by the
frontal pair. Between the extremes that is 0.63 percentage points per
degree of mounting error, in the same monotonic energy order as the
detector-ratio shift, and with the two channels moving in opposite
directions as the geometric model predicts.

These are lower bounds. The angle comes from foreshortening and is
itself an upper bound (theta <~ 8 deg), so if the canvas was tilted less
than assumed, the same map change accrued over fewer degrees and the
per-degree error is larger. [TODO: replace with the builder-confirmed
angle when it arrives; the numbers scale as 1/theta.]

The tilt shift of R implied by these per-detector changes agrees with
the region-level estimate of [Sec. tilt] in shape and in sign on every
line, but reads systematically lower, by 0.73 pp RMS and at most 1.2 pp
at Ca. That difference between region-matched and pixel-matched
comparison is the systematic of the method and is quoted as such on
Fig. [tilt].

## Still to merge (Person B)

- Final figure numbering and Table 1 formatting. Table 1 rows are
  emitted by script 07 into
  `results/detector_diff/handoff2_ratio_curve.md`.
- Ca (cv 0.69) and Ti (cv 1.16) in the learned fusion: the two lines
  that still lose. Both are shrinkage/variance effects at low counts
  rather than pipeline errors, and a map-level validation criterion
  (Poisson thinning inside the training scans) is the natural next step
  -- validation MSE demonstrably does not rank models the way map SNR
  does.
