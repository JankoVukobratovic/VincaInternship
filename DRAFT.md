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
canvas are used here: two frontal scans acquired seven days apart
(prova1, prova2; 120 x 60 pixels) and one scan with the canvas tilted
forward (ruotato; 80 x 45 pixels), each pixel a 1024-channel spectrum at
0.0292 keV per channel and a dwell of 3.004, 3.003 and 3.013 s
respectively. The seven-day interval is the operator's record (private
communication); the MCA headers carry a placeholder acquisition
timestamp and cannot confirm it. That interval matters only as the
timescale over which the repeatability floor below was accumulated. The frontal pair fixes the
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

On pixels that carried no gradient during training, the learned fusion
gains 17.7 % in mean SNR over summing, against the 0.9 % of the
classical weighting (Table X, Fig. Y). Six of the eight lines improve,
Pb Ll by 68.6 %, Pb Lg by 33.3 % and Fe by 28.5 %; Ca loses 9.4 % and
Ti 3.7 %. The mean is carried disproportionately by the weakest line:
the median gain is 11.9 %, and dropping Pb Ll alone takes the mean to
10.4 %, so the honest summary is a gain of roughly 10-18 % depending on
how the lines are aggregated. In counting terms even the lower figure
is worth about 22 % more acquisition time, the upper one 39 %.

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
the correlation with it (0.947 at Ca, 0.98 or better elsewhere). The two lines that
lose are exactly the two where the guards flag structure: Ca sits at
cv 0.69, the residue of shrinkage toward the mean spectrum where
detector B contributes almost no counts, and Ti at cv 1.16, where the
network adds variance instead of removing it.

## Validation

The fusion numbers are quoted on the validation blocks of the frontal
training scan, paired with the same pixels of prova2. Those pixels carry
no gradient, and prova2 is never seen by the network in any
configuration; but the validation loss on the same blocks selected the
stopping epoch, so they are held out from fitting, not from every use.

Three limitations follow and are better stated than argued away. First,
prova1 and prova2 differ in noise realization, not in content: the
evaluation tests generalization across noise on one painting, not across
paintings. Second, the weighting schemes were chosen on the variance
argument above and the learning rate on validation loss, but the
benchmark was recomputed as those configurations were developed, so the
final figure is a hold-out estimate that development has touched, not a
blind one; the ablation spread (-0.1 % to +17.7 %) is far larger than
any plausible selection effect, but the single number should be read
with that in mind. Third, the repeatability floor of the pipeline is set
by the two frontal scans, 0.96 % RMS on the element maps and 0.43 % on
the vertical scale of the registration.

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
+0.45 % at Ti Ka, against -0.13 % at Pb Lb and -0.08 % at Pb Lg. Each
line is judged against its own repeatability rather than a common band,
because the frontal pair is far from uniform across lines: Ca and Ti
clear their own floors (0.22 and 0.19 %/deg) by a factor 2.2-2.5 only,
the sharp Pb lines clear theirs (0.015-0.018 %/deg) by 5-9, and Fe, Cu
and Pb Lg do not clear theirs at all. Between the extremes that is 0.63 percentage points per
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
- Ca and Ti in the learned fusion: reported on the summed maps for now.
  Two remedies were tried and failed (see above); the next candidates
  are an intensity-preserving training objective and a map-level
  validation criterion, since validation MSE demonstrably does not rank
  models the way map SNR does.
