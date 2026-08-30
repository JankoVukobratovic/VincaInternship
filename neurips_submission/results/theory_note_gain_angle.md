# Theory note: why gain and angle are degenerate at one tilt (ready for the appendix)

Source numbers: `results/registration/positioning_sensitivity.csv`, column `tilt_pct_sum`
(measured percent level response at the 7.7 deg reference mounting).

## Derivation

The forward model's per-line gain at believed angle `a = angle_deg + angle_bias_deg`,
with `gain_scale = 1 + delta_s`, is (ignoring the independent per-line offset knob,
which is not part of this degeneracy):

    g_el(a, delta_s) = 1 + (1 + delta_s) * (p_el / 100) * (a / 7.7)

where `p_el` is the measured `tilt_pct_sum` of line `el`. Write `a = a0 + delta_a`
for the true mounting angle `a0` (7.7 deg for the real scan T) and a small angle
bias `delta_a`. Linearising in the two small perturbations `delta_s, delta_a`:

    g_el - g_el^nom  =  (g_el^nom - 1) * ( delta_s + delta_a / a0 )  + O(delta^2)

where `g_el^nom - 1 = (p_el/100) * (a0/7.7)` is the nominal gain deviation of
line `el` at the scan's own angle `a0` (for `a0 = 7.7`, this is exactly `p_el/100`).

**The consequence.** The perturbation of every line's gain factors into (i) a
fixed, measured, per-line "sensitivity profile" `v_el = g_el^nom - 1`, and (ii)
a single scalar `kappa = delta_s + delta_a / a0` that depends only on the pair
`(gain_scale, angle_bias)` and the scan's own angle `a0`. For one fixed scan,
the map `(delta_s, delta_a) -> kappa` is many-to-one (any line `delta_a = a0 *
(kappa - delta_s)` gives the same kappa), so **the set of gain vectors reachable
by varying gain_scale and angle_bias at fixed a0 is one-dimensional**, spanned
by the fixed profile `v_el`, no matter how the two knobs are combined. Any
statistic that reads only that profile (a per-line level ratio, or any linear
functional of it) can estimate kappa, but cannot separate `delta_s` from
`delta_a`: infinitely many pairs produce the identical first-order signal on
one scan.

**Breaking it needs a second angle.** A second measurement at a different angle
`a1 != a0` gives an independent combination `kappa' = delta_s + delta_a / a1`;
the 2x2 system in `(delta_s, delta_a)` has determinant proportional to
`1/a0 - 1/a1 != 0`, so the two knobs are identifiable in principle from two
angles, though the achievable precision still depends on the calibration
noise floor relative to `1/a0 - 1/a1`.

**Why this explains three separate empirical results already in the paper,
without new experiments:**

1. WP2's pre-registered rule groups gain and angle-belief as one family and
   this derivation is why: they are not merely correlated, they are exactly
   degenerate to first order at one angle.
2. WP2's post-hoc `gain_proj` statistic is the projection of the per-line
   level deviation onto `v_el` — this derivation shows that projection is
   (under isotropic noise) the maximum-likelihood estimator of `kappa` along
   the single observable direction. That is exactly why it recovers the
   extreme rungs (`gain x0`, `gain x2`, large `|kappa|`) but not `x0.5`,
   `x1.5`, or `angle +-2/+-5 deg` (small `|kappa|`, inside the noise floor of
   the kappa estimate) — the statistic was, without anyone deriving it this
   way at the time, already the right one for the identifiable direction.
3. WP4's ABC posterior marginals for `gain_scale` and `angle_bias` match the
   prior (Fig. 3 omits them for exactly this reason): the real scan is one
   measurement at one angle (7.7 deg), so the ABC likelihood surface is flat
   along the `delta_a = a0*(kappa - delta_s)` line for any fixed kappa the
   data does constrain — the posterior is not weak by accident, it is flat in
   a direction the experiment cannot see.

Numeric illustration (the measured profile `v_el = p_el/100`, `p_el` from
`positioning_sensitivity.csv`): Ca +0.0342, Ti +0.0309, Fe -0.0007, Cu -0.0073,
PbLl +0.0024, PbLa -0.0111, PbLb -0.0140, PbLg -0.0101. Any `(gain_scale,
angle_bias)` pair that is degenerate at 7.7 deg reproduces this exact ratio
across all 8 lines; only its overall scale (kappa) is observable.

## Suggested placement

- One sentence in the WP2 main-text paragraph, right after "gain vs angle is
  degenerate at a single tilt": *"a short linearisation (Appendix~C) shows
  this is an exact rank-one degeneracy of the gain vector at one angle, not
  merely a correlation, and identifies gain\_proj as its maximum-likelihood
  direction."*
- A short new Appendix C ("Why gain and angle are degenerate"), roughly the
  derivation above compressed to 10-12 lines plus the one-sentence WP4
  callback ("this is also why the ABC posterior of Fig. 3 leaves gain_scale
  and angle_bias at the prior: one scan at one angle cannot see the
  orthogonal direction").
- No new figure needed; the existing `positioning_sensitivity.csv` numbers
  are enough, already quoted in Section 2 of the paper.

## LaTeX-ready block (paste into a new `\subsection` or `\paragraph` in the appendix)

```latex
\paragraph{Why gain and angle are degenerate at one tilt.}
Writing the believed angle as $a=a_0+\delta_a$ and $\mathrm{gain\_scale}=1+\delta_s$,
the per-line gain $g_{\mathrm{el}}=1+(1+\delta_s)\frac{p_{\mathrm{el}}}{100}\frac{a}{7.7}$
linearises, for the fixed angle $a_0$ of one scan, to
$$g_{\mathrm{el}}-g_{\mathrm{el}}^{\mathrm{nom}} \approx (g_{\mathrm{el}}^{\mathrm{nom}}-1)\,\Big(\delta_s+\tfrac{\delta_a}{a_0}\Big),$$
a product of a fixed, measured per-line profile and a single scalar
$\kappa=\delta_s+\delta_a/a_0$. At fixed $a_0$ the reachable gain vectors form
a one-dimensional subspace regardless of how $\delta_s,\delta_a$ are combined,
so no statistic built from one scan can separate the two knobs, only estimate
$\kappa$; a second scan at a different angle would (determinant
$\propto 1/a_0-1/a_1\neq0$). This is why the pre-registered rule scores
gain and angle-belief as one family (Section~4), why the post-hoc $\mathrm{gain\_proj}$
statistic (the projection onto the measured profile, i.e.\ the maximum-likelihood
estimator of $\kappa$) recovers only the rungs with large $|\kappa|$, and why the
ABC posterior of Fig.~3 leaves gain\_scale and angle\_bias at the prior: one
scan at 7.7$^\circ$ cannot see the orthogonal direction.
```
