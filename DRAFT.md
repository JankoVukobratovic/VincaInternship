# Draft — Results (dual-detector paper)

Working text for the Results section, in paper order. Numbers come
from `results/detector_diff/` (scripts 06, 07, 07b). Placeholders
marked [TODO] wait for Person B's registration/non-uniformity results
or for confirmation from the instrument builder.

## Per-element efficiency ratios

The per-element count ratio R = det10264 / det19511 is strongly
energy-structured: it falls monotonically from about 6 at the Ca Ka
line (3.7 keV) to 0.65 at Pb Lg (14.8 keV), crossing unity near 8 keV
(Table 1). The two frontal scans give ratios consistent to within
their bootstrap errors for all reliable lines; K and Zn are excluded
(overlapping lines / low counts). The two channels of the scanner are
therefore far from interchangeable: the same pigment produces up to a
sixfold different signal depending on which detector is read out.

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

Tilting the canvas forward changes the ratio by up to a few percent —
small, but 3-14 sigma significant thanks to the full-map statistics —
with a characteristic energy structure: positive below 6 keV, a zero
crossing near 6.5 keV, and a negative plateau above 10 keV (Fig. X,
right). A thick-sample fluorescence model in which the tilt moves the
two effective take-off angles antisymmetrically reproduces this
shape. With a single tilt the individual take-off angles are not
identifiable; the identifiable parameters are the lever arm
s = 0.475 +- 0.074 (degrees of take-off change per degree of canvas
tilt, i.e. an effective take-off shift of 3.7 +- 0.6 degrees at the
measured tilt), the matrix-attenuation energy scale
Ec = 3.75 +- 0.46 keV, and a solid-angle offset of -2.80 +- 0.17 %.
A nonparametric Gaussian-process regression of the same data
reproduces the shape independently, and the parametric curve stays
within 0.9 sigma of the GP mean over the whole energy range — the
shape is a property of the data, not of the model choice. Cu deviates
from the smooth trend (-3.7 % against a predicted -1 %); a
frame-coverage effect is the leading suspect. [TODO: crop test.]

## Tilt angle from foreshortening

The mounting angle of the tilted scan was not recorded; we recover it
from the data. A forward tilt compresses the scanned image vertically
by cos(theta), so registering the tilted scan onto a frontal scan
with independent x/y scales measures the tilt through the scale ratio
sy/sx = 1/cos(theta), with the unknown scan step sizes cancelling.
Joint registration over five element maps gives theta = 7.7 +- 1.0
(element spread) +- 1.8 (registration floor) degrees, and the
three-scan consistency triangle closes to within the errors. The
registration floor is itself a finding: two nominally identical
frontal scans differ in vertical scale by 0.43 %, a direct measure of
the positioning/calibration drift between scanning sessions. [TODO:
compare with the builder-confirmed angle if it arrives.]

## Still to merge (Person B)

- Registered detector-difference maps and the geometric
  non-uniformity map (script 08+).
- Learned fusion / ML results.
- Final figure numbering and Table 1 formatting.
