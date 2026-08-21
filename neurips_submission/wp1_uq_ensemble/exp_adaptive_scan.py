"""WP1 / experiment 2 (STRETCH) - uncertainty-guided adaptive scanning.
OWNER: Dimitrije.  Attempt ONLY after exp_ensemble_uq is done; if time
runs short this becomes one outlook paragraph in the paper.

CLAIM UNDER TEST
    Measuring only a fraction of pixels and letting the physics+prior
    fill the rest, with the NEXT pixels chosen where the ensemble is
    most uncertain, reaches a target map quality with fewer measured
    pixels than raster/random acquisition - i.e. less irradiation of
    the painting and shorter scan time.

SKETCH (all pieces exist)
    1. Start from a sparse validity mask (e.g. every 4th pixel of the
       tilted frame measured, the rest zeroed).
    2. Restore with the WP1 jitter ensemble -> mean + std maps.
    3. Acquisition step: "measure" (reveal from the full simulated
       tilted scan) the B pixels with the largest predictive std,
       update the validity mask.
    4. Repeat; log quality-vs-measured-fraction after each step for
       adaptive / random / raster orderings.

OUTPUT   results/wp1_adaptive_scan.csv:
         strategy, step, measured_frac, element, region, r, bias_pct
FIGURE   figures/wp1_adaptive.png: r vs measured fraction, three
         strategies, headline lines (config.FIG_LINES).

DEFINITION OF DONE: the three-curve figure on >= 2 simulated cases, or
an explicit decision to drop to outlook status (tell the team).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run(quick: bool = False):
    raise NotImplementedError(
        "TODO(WP1, stretch): implement per the module docstring")


if __name__ == "__main__":
    run()
