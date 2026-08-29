> NOTE (2026-08-28): working draft. The submitted text is paper/main.tex; where wording differs (e.g. the coverage claim is stated as under-coverage with heavy tails, not near-calibration), the paper is authoritative. Numbers here match the summary files.

# WP3 results text for the paper (numbers from results/wp3_regime_summary.txt, run of 2026-08-28)

Grid: 4 angles (8, 14, 20, 25 deg) x 5 holes (none, 6x8, 10x14, 14x20, 18x26 px in the tilted frame) x 3 doses (1, 0.5, 0.25) x 3 noise seeds = 180 cases, source prova2 (never trained on), validated emulator. Candidates: physics inverse (det), the nominal single net (MVP), the WP1 jitter-ensemble mean, four classical fills applied in the tilted frame before the same physics inverse (nearest, biharmonic, OpenCV Telea, OpenCV Navier-Stokes), and one hybrid (biharmonic fill, then the net with the hole declared valid). Scores are r against prova2, mean over seeds and the 8 lines, inside the hole and on the footprint.

## Method sentence

To find out where the learned prior pays at all, we sweep tilt angle, hole size and dose and compare the learned restorations against classical inpainting controls that never learned anything; without this comparison the paper could not claim that the prior is needed.

## Results paragraph: the regime map

**On the measured region the learned prior always pays; inside a hole it never does.** On the footprint every learned candidate beats every classical one at every cell (no hole: net 0.985 vs physics 0.977; 14x20 hole: ensemble 0.952, net 0.948, best classical 0.937, physics 0.781), and the gain grows with hole size because the learned models also repair the warp blur around the hole. Inside the hole the ranking reverses: the biharmonic fill beats the nominal net in 16 of 16 cells at dose 1 (20 deg, 14x20: r 0.467 vs 0.415; 6x8: 0.529 vs 0.359), the jitter-ensemble mean wins only the 10x14 column and by 0.005 to 0.015, and the gap widens as the dose drops: at dose 0.25 in the 14x20 hole the net falls to r 0.03 and the ensemble to -0.01 while the biharmonic fill stays at 0.455 (classical fills are dose-insensitive by construction, the learned fills are not). The only learned candidate that wins every hole cell is the hybrid, a classical fill followed by the net, and its margin over the fill alone is 0.005 to 0.03: the net adds a sharpening on top of a smooth fill, it does not add a better fill. The regime map (Fig. wp3_regime_map) is therefore almost entirely blue in the hole columns and its rows are flat: the hole-region r is angle-independent by construction, because the measured warp does not scale with the angle (only the per-line gains do, and r is gain-invariant), so the axis that matters is hole size and dose, not tilt.

**Consequences for the paper's story.** The harsh-regime demo of the MVP (hole r -0.07 to 0.57 on Ca) was real but was measured against physics alone; against a classical fill it is a loss. What the learned prior buys on this instrument is contrast and noise on the measured region (cv_ratio 0.98 to 1.00 vs 0.93 to 0.95), the restoration of the neighbourhood of a hole, and calibrated uncertainty (WP1); what it does not buy is hole content, where a smooth PDE fill of the tilted frame is both better and dose-proof. The honest recommendation is the hybrid: fill classically, then let the net do what it is good at.

**Acquisition-blur sensitivity.** Re-simulating the harsh case (20 deg, 14x20, dose 1) with the sharp (cubic) acquisition instead of the validated bilinear emulator moves footprint r by at most 0.007 and hole r by at most 0.014 and changes no ranking; it moves the net's contrast ratio from 0.944 to 0.970, consistent with the v1/v2 lesson (the bilinear emulator over-blurs the test input by the same amount it over-blurred the training input).

**Real anchor.** On the measured tilted scan the ensemble mean matches the physics inverse (r 0.9527 vs 0.9526, cv 0.979 vs 0.947) and the single nominal net loses on all 8 lines (0.9485), reproducing the WP1 finding in the WP3 pipeline.

## The three sentences

The learned prior beats both physics and four classical inpainting controls everywhere on the measured region, but loses to a biharmonic fill inside every hole of a 180-case grid (0.467 vs 0.415 at 20 deg, 14x20) and collapses at low dose where the classical fill does not (0.03 vs 0.455 at a quarter dose). A classical fill followed by the net is the best candidate in every cell, by 0.005 to 0.03 over the fill alone. Hole size and dose, not tilt angle, are the axes of the regime map, because the measured warp does not scale with the angle.

## Figure caption

- **wp3_regime_map**: Left three panels: r of the nominal net minus the best non-learned candidate (physics or the best classical fill), evaluated inside the hole (footprint for the no-hole column), mean over three seeds and the headline lines Ca, Cu, PbLa, one panel per dose; blue = the non-learned candidate wins. Right: r against hole area at 20 deg and dose 1 for all candidates, inside the hole and on the footprint (physics orange, learned navy, classical grey, hybrid navy triangles).

## Honest caveats

1. The classical controls fill in the tilted frame and then pass through the same nominal inverse as everything else; the comparison isolates fill quality, not pipeline differences.
2. The net was trained with zero-filled dropout blocks of 8 to 17 px; the 18x26 hole is outside its training distribution and the 6x8 hole is at its lower edge, which is where the biharmonic margin is largest.
3. The hole-region r is computed on 48 to 468 frontal pixels of a single source map; the seed spread is small (three seeds agree to 0.01), the painting-content dependence is untested (one painting).
4. The hybrid declares the filled hole valid to the net; a net trained on classically filled holes might do better and was not trained (time).
