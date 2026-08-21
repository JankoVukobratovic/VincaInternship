"""WP2 / experiment 1 - defect-tolerance curves.
OWNER: ______ (upisi ime)

CLAIM UNDER TEST
    "How imperfect may a training simulator be before the learned
    restoration stops helping - or starts hurting?"  We train one net
    per DELIBERATELY BROKEN simulator (config.DEFECT_LADDERS: noise
    constant, tilt gains, angle belief, resampling blur, registration
    shift) and measure the restoration quality it delivers on (a)
    nominal simulated test cases and (b) the REAL ruotato scan.  The
    blur rung reproduces our organic v1 mistake as a controlled point
    on the curve - the paper's motivating case study becomes one rung
    of a systematic ladder.

METHOD (glue below is written; loop = train per rung + evaluate)
    1. Control: net trained on the NOMINAL simulator.           [done]
    2. Per rung: training.make_batch_fn(knobs=defect_knobs).    [done]
    3. Eval on shared test cases + real scan, frozen metrics.   [done]
    4. TODO(WP2): confirm/extend the ladders in config.py - rung
       spacing IS the experiment design; verify each family reaches
       both "still fine" and "clearly hurting".
    5. TODO(WP2): tolerance-curve figure + interpretation: per family,
       x = defect magnitude, y = delta(r) and delta(cv_ratio) vs the
       nominal-trained net, real-scan points overlaid.

OUTPUTS
    results/wp2_defect_tolerance.csv  rows: defect_family, defect,
        element, candidate(det|net), region, testbed(sim|real), metrics
    figures/wp2_tolerance_curves.png

DEFINITION OF DONE
    Every family in DEFECT_LADDERS has a curve; each curve states where
    the net drops below the deterministic baseline; three-sentence
    takeaway per family for the paper.

Run from the repo root:
    python neurips_submission/wp2_simulator_audit/exp_defect_tolerance.py --quick
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from common import core, io_utils, perturb, restore, training


def eval_net(net, label, family, quick):
    """Shared testbed: simulated cases from prova2 + the real scan."""
    rows = []
    grid = config.GRID_QUICK if quick else config.GRID
    for angle in grid["angles"]:
        case = restore.degrade(source="prova2", angle=angle,
                               block=restore.centered_block(14, 20), seed=0)
        det, learned = restore.apply_network(net, case["tilted"],
                                             case["angle"],
                                             validity=case["validity"])
        r = restore.score_candidates(
            {"det": det, "net": learned}, case["truth"],
            {"footprint": case["fp"], "hole": case["hole"]})
        rows += [{"defect_family": family, "defect": label,
                  "testbed": f"sim_{angle:g}deg", **x} for x in r]

    ruo = core.fm.load_summed_maps("ruotato")
    truth2 = core.fm.load_summed_maps("prova2")
    det, learned = restore.apply_network(net, ruo, core.fm.REF_ANGLE_DEG)
    r = restore.score_candidates({"det": det, "net": learned}, truth2,
                                 {"footprint": core.dg.footprint()})
    rows += [{"defect_family": family, "defect": label,
              "testbed": "REAL_ruotato", **x} for x in r]
    return rows


def run(quick: bool = False):
    tcfg = config.QUICK_TRAIN if quick else config.TRAIN
    ladder = [("nominal", "control", perturb.NOMINAL)]
    for family, rungs in config.DEFECT_LADDERS.items():
        for label, kw in rungs:
            ladder.append((family, label,
                           perturb.SimKnobs(label=label, **kw)))
    if quick:
        ladder = ladder[:3]
        print("QUICK: first 3 rungs only")

    rows = []
    for i, (family, label, knobs) in enumerate(ladder):
        print(f"[{i + 1}/{len(ladder)}] {family}/{label}")
        net, hist = training.train_net(
            training.make_batch_fn(knobs=knobs), tcfg,
            seed=config.BASE_SEED + 7 * i)
        rung_rows = eval_net(net, label, family, quick)
        for r in rung_rows:
            r["train_val_l1"] = hist["best_val_l1"]
        rows += rung_rows
        io_utils.write_rows("wp2_defect_tolerance", rows)  # checkpoint often
    print(f"saved: {io_utils._path('wp2_defect_tolerance')} "
          f"({len(rows)} rows)")


def make_figures():
    """TODO(WP2): tolerance curves from results/wp2_defect_tolerance.csv.

    Per defect family one panel: x = rung (defect magnitude), y =
    net-minus-det delta of r (and a second row for cv_ratio), one line
    per element in config.FIG_LINES, horizontal zero line = "the net
    stops helping", real-scan testbed as marker overlay.  Save to
    io_utils.fig_path('wp2_tolerance_curves.png')."""
    raise NotImplementedError("TODO(WP2): implement make_figures")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    if args.figures:
        make_figures()
    else:
        run(quick=args.quick)
