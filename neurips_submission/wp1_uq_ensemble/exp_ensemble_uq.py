"""WP1 / experiment 1 - simulator-uncertainty ensemble UQ.
OWNER: Dimitrije

CLAIM UNDER TEST
    An ensemble trained across simulators drawn WITHIN calibration
    uncertainty (config.JITTER) yields per-pixel uncertainty that (a) is
    calibrated on held-out simulated cases, and (b) separates data noise
    from simulator uncertainty - the fixed-simulator control ensemble
    (same N, same seeds, nominal knobs) captures noise+init variance
    only, so the SPREAD DIFFERENCE is attributable to the imperfect
    simulator.  This is the "UQ that knows the model is imperfect"
    headline of the paper.

METHOD
    1. Train N = config.ENSEMBLE_N nets, each on perturb.jittered knobs
       (member i has its own knobs AND its own seed).           [done]
    2. Train N control nets with NOMINAL knobs, same seeds.     [done]
    3. Predict on held-out test cases (simulated from prova2 via
       common.restore.degrade) and on the REAL ruotato.         [done]
    4. Coverage: fraction of pixels with |mean - truth| <= z*std for
       z in config.COVERAGE_Z, per line, per ensemble.          [TODO]
    5. Analysis: calibration curve figure, jitter-vs-control spread
       maps, the sigma story for the paper.                     [TODO]

OUTPUTS
    results/wp1_ensemble_members.csv   per-member training summary
    results/wp1_uq_coverage.csv        coverage rows: ensemble, element,
                                       case, region, z, coverage, n_px
    results/wp1_ensemble/*.pt          member checkpoints (gitignored)
    figures/wp1_calibration.png        z vs empirical coverage, both
                                       ensembles (perfect = diagonal)

DEFINITION OF DONE
    Coverage numbers for both ensembles on >= 6 simulated cases + the
    real scan; a one-figure calibration plot; three sentences of
    interpretation in the paper's results section.

Run from the repo root:
    python neurips_submission/wp1_uq_ensemble/exp_ensemble_uq.py --quick
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import config
from common import core, io_utils, perturb, restore, training


def member_dir():
    d = os.path.join(core.RESULTS_DIR, "wp1_ensemble")
    os.makedirs(d, exist_ok=True)
    return d


def train_ensemble(kind: str, n: int, train_cfg: dict) -> list:
    """Train (or load cached) members; kind in {'jitter', 'control'}."""
    members = []
    rows = []
    for i in range(n):
        seed = config.BASE_SEED + 100 * i + (0 if kind == "jitter" else 1)
        ckpt = os.path.join(member_dir(), f"{kind}_{i:02d}.pt")
        net = core.RestorationUNet()
        if os.path.exists(ckpt):
            net.load_state_dict(torch.load(ckpt, weights_only=True))
            net.eval()
            print(f"[{kind} {i}] cached")
        else:
            rng = np.random.default_rng(seed)
            knobs = (perturb.jittered(rng, config.JITTER, f"jitter_{i}")
                     if kind == "jitter" else perturb.NOMINAL)
            print(f"[{kind} {i}] training  knobs={knobs.to_meta()}")
            net, hist = training.train_net(
                training.make_batch_fn(knobs=knobs), train_cfg, seed=seed)
            torch.save(net.state_dict(), ckpt)
            rows.append({"ensemble": kind, "member": i, **knobs.to_meta(),
                         **hist})
        members.append(net)
    if rows:
        io_utils.write_rows("wp1_ensemble_members", rows, append=True)
    return members


def ensemble_predict(members: list, tilted, angle, validity=None):
    """(mean maps, std maps) over member restorations, physical units."""
    preds = {el: [] for el in core.ELEMENTS}
    for net in members:
        _, learned = restore.apply_network(net, tilted, angle,
                                           validity=validity)
        for el in core.ELEMENTS:
            preds[el].append(learned[el])
    mean = {el: np.nanmean(np.stack(v), axis=0) for el, v in preds.items()}
    std = {el: np.nanstd(np.stack(v), axis=0, ddof=1)
           for el, v in preds.items()}
    return mean, std


def test_cases(quick: bool) -> list:
    """Held-out cases: simulated from prova2 (never in training)."""
    grid = config.GRID_QUICK if quick else config.GRID
    cases = []
    for angle in grid["angles"]:
        for (h, w) in grid["holes"][:2]:
            cases.append(restore.degrade(
                source="prova2", angle=angle,
                block=restore.centered_block(h, w), seed=0))
    return cases


# ---------------------------------------------------------------------------
# TODO(WP1): the scientific core
# ---------------------------------------------------------------------------

def coverage_rows(mean, std, truth, regions, meta) -> list:
    """TODO(WP1): per-line empirical coverage of |mean-truth| <= z*std.

    For each element, region (footprint / hole) and z in
    config.COVERAGE_Z: coverage = fraction of region pixels inside the
    band.  Guard std == 0 (zero-spread pixels count as covered only if
    the error is 0).  Return rows carrying `meta` + element, region, z,
    coverage, n_px.  Decide and DOCUMENT whether std needs the
    finite-ensemble inflation factor.
    """
    raise NotImplementedError("TODO(WP1): implement coverage_rows")


def make_figures():
    """TODO(WP1): calibration curve (z vs coverage, jitter vs control,
    diagonal = perfect) from results/wp1_uq_coverage.csv, plus one
    spread-map panel (jitter std vs control std, same color scale) on a
    harsh case.  Save via io_utils.fig_path."""
    raise NotImplementedError("TODO(WP1): implement make_figures")


def run(quick: bool = False):
    n = config.ENSEMBLE_N_QUICK if quick else config.ENSEMBLE_N
    tcfg = config.QUICK_TRAIN if quick else config.TRAIN
    jitter = train_ensemble("jitter", n, tcfg)
    control = train_ensemble("control", n, tcfg)

    rows = []
    for case in test_cases(quick):
        regions = {"footprint": case["fp"], "hole": case["hole"]}
        meta = {"case_angle": case["angle"], "case_dose": case["dose"],
                "case_block": str(case["block"]), "source": case["source"]}
        for kind, members in (("jitter", jitter), ("control", control)):
            mean, std = ensemble_predict(members, case["tilted"],
                                         case["angle"],
                                         validity=case["validity"])
            rows += [{"ensemble": kind, **r} for r in coverage_rows(
                mean, std, case["truth"], regions, meta)]
    # real-scan anchor (no truth-free coverage here: score vs prova2)
    ruo = core.fm.load_summed_maps("ruotato")
    truth2 = core.fm.load_summed_maps("prova2")
    for kind, members in (("jitter", jitter), ("control", control)):
        mean, std = ensemble_predict(members, ruo, core.fm.REF_ANGLE_DEG)
        rows += [{"ensemble": kind, **r} for r in coverage_rows(
            mean, std, truth2, {"footprint": core.dg.footprint()},
            {"case_angle": core.fm.REF_ANGLE_DEG, "source": "REAL_ruotato"})]

    path = io_utils.write_rows("wp1_uq_coverage", rows)
    print(f"saved: {path}  ({len(rows)} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    if args.figures:
        make_figures()
    else:
        run(quick=args.quick)
