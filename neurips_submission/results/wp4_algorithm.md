# The audit-refine loop as a procedure

The closed loop of Section 5 is not a one-off fix; it is an explicit,
repeatable procedure, run for four rounds against the single real scan.

**Round 0 (nominal).** Whitened distance to the real scan: 2.72 for the
cubic simulator, 8.92 for the historical bilinear one. Both exceed the
null floor of 0.95 +/- 0.24 (the distance between two nominal-vs-nominal
pairs), so the simulator is measurably wrong before any fitting.

**Round 1.** Prior = the measured calibration jitter plus the
bilinear-vs-cubic hypothesis at 50/50. Posterior rejects bilinear
completely (0.000) and reaches d_min = 2.16. The posterior predictive
check (PPC) rejects all 12 posterior draws, uniformly with verdict
"noise" (12/12).

**Round 2.** The round-1 PPC's per-line residual (variance ratios 0.9 to
4.6 across lines) motivates a per-line noise knob. d_min improves to
2.02. The PPC still rejects every draw; the noise_k verdict drops from
12/12 to 7/12 (3 gain_like, 1 warp_shift, 1 blur), sampling noise at
N=12, not a qualitative change.

**Round 3.** forward_model.py's own docstring names the per-pixel
flat-field as unmodelled; a one-parameter radial vignetting knob (5%
sd, a round-number placeholder, not a measured quantity) is added.
d_min improves only to 1.98, and the flat-field posterior
(-0.022 +/- 0.053) is statistically indistinguishable from its prior
(0.002 +/- 0.050): the knob absorbs no signal. The PPC verdict returns
to 10/12 noise_k, unchanged in character.

**Stopping rule.** A round is added only if it is motivated by a
specific pattern in the previous round's PPC decomposition and
corresponds to a documented, physically named mechanism, never an
arbitrary knob. The loop is judged converged when a round no longer
moves d_min beyond null-floor noise and does not change the PPC verdict
distribution. Round 3 meets both conditions: the procedure has
converged, and the remaining residual is attributed, by elimination
across three physically motivated hypotheses, to structure the audit
battery cannot resolve with this knob set (scatter background,
intra-scan dwell drift).
