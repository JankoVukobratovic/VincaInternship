"""WP3 - degradation grid, classical controls, regime map.
OWNER: ______ (upisi ime)

CLAIM UNDER TEST
    Where exactly does the learned prior pay?  We sweep degradation
    severity (angle x hole size x dose, config.GRID), restore every
    case with deterministic physics, physics+U-Net, and the classical
    inpainting controls, and condense the sweep into ONE regime-map
    figure: the frontier beyond which learning beats everything that
    does not learn.  Without the classical controls the paper cannot
    claim the prior is NEEDED - this comparison is the paper's spine.

METHOD (inference-only sweep - no training; uses the MVP checkpoint)
    1. Cases from prova2 (never in training) via common.restore.degrade
       with the validated instrument emulator.                  [done]
    2. Candidates: det / net / classical_* (common.classical).  [done]
    3. Frozen scoring, footprint + hole regions.                [done]
    4. TODO(WP3): OpenCV inpainting control (classical.opencv_fill) -
       one more column, reviewers will expect it.
    5. TODO(WP3): the regime-map figure (make_figures below).
    6. TODO(WP3): real-scan anchor row + repeat the harsh case with
       sim='sharp' as the acquisition-blur sensitivity note.

OUTPUTS
    results/wp3_degradation_grid.csv   angle, hole_px, dose, seed,
        element, candidate, region, r, ssim, bias_pct, cv_ratio, n_px
    figures/wp3_regime_map.png

DEFINITION OF DONE
    Full grid ran (all seeds); regime map with the learned-vs-best-
    classical frontier marked; three-sentence reading of the map for
    the paper (mild regime -> physics suffices; holes -> only the
    learned prior recovers structure; the crossover location).

Run from the repo root:
    python neurips_submission/wp3_degradation_benchmark/exp_degradation_grid.py --quick
"""

import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from common import classical, io_utils, restore


def run(quick: bool = False):
    net = restore.load_mvp_net()
    if net is None:
        raise SystemExit(
            "MVP checkpoint missing (neurips-restore/experiments/"
            "checkpoint.pt) - run neurips-restore/scripts/"
            "03_learned_restoration.py first or fetch the checkpoint")

    grid = config.GRID_QUICK if quick else config.GRID
    cases = list(itertools.product(grid["angles"], grid["holes"],
                                   grid["doses"], grid["seeds"]))
    rows = []
    for i, (angle, (h, w), dose, seed) in enumerate(cases):
        print(f"[{i + 1}/{len(cases)}] angle={angle:g} hole={h}x{w} "
              f"dose={dose:g} seed={seed}")
        case = restore.degrade(source="prova2", angle=angle,
                               block=restore.centered_block(h, w),
                               dose=dose, seed=seed)
        det, learned = restore.apply_network(net, case["tilted"],
                                             case["angle"],
                                             validity=case["validity"])
        cands = {"det": det, "net": learned}
        # classical fills operate in the TILTED frame where data was lost
        cands.update(classical.classical_restorations(
            case["tilted"], case["v_tilt"], case["angle"]))
        scored = restore.score_candidates(
            cands, case["truth"],
            {"footprint": case["fp"], "hole": case["hole"]})
        rows += [{"angle": angle, "hole_px": h * w, "hole": f"{h}x{w}",
                  "dose": dose, "seed": seed, **r} for r in scored]
        if (i + 1) % 10 == 0:
            io_utils.write_rows("wp3_degradation_grid", rows)
    path = io_utils.write_rows("wp3_degradation_grid", rows)
    print(f"saved: {path}  ({len(rows)} rows)")


def make_figures():
    """TODO(WP3): the regime map from results/wp3_degradation_grid.csv.

    Suggested form (discuss with the team before finalizing):
      - x = hole area (px), y = angle, one panel per dose;
      - cell color = mean over seeds/headline lines of
        r(net) - max(r(det), r(best classical)) in the HOLE region
        (footprint region for the no-hole column);
      - contour/frontier where the difference crosses zero, annotated
        "learning pays right of this line";
      - one small side panel: r vs hole area at the harsh angle, all
        candidates as separate lines (the classical controls must be
        visibly separate from the net - that is the money panel).
    Save to io_utils.fig_path('wp3_regime_map.png')."""
    raise NotImplementedError("TODO(WP3): implement make_figures")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    if args.figures:
        make_figures()
    else:
        run(quick=args.quick)
