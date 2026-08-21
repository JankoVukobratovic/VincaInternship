"""WP2 / experiment 2 - blind simulator diagnostics.
OWNER: ______ (upisi ime)

CLAIM UNDER TEST
    A small battery of summary statistics, compared between REAL
    measurements and simulations, can not only DETECT that a simulator
    is wrong but IDENTIFY WHICH component is broken - turning our
    hand-made blur-mismatch discovery into a reusable procedure.

DESIGN
    Statistics (per element line, tilted frame):
      hf_ratio        high-frequency power real/sim - catches resampling
                      blur (this is the statistic that caught v1) [done]
      var_slope       slope of local variance vs local mean - catches a
                      wrong noise constant k                      [TODO]
      level_ratio     mean level real/sim - catches gain errors    [TODO]
      cv_ratio        contrast ratio - catches blur AND gains
                      (degeneracy: document it)                    [TODO]
      edge_shift      cross-correlation peak offset real vs sim -
                      catches warp/registration errors             [TODO]

    Blind test protocol (the punchline):
      for each rung in config.DEFECT_LADDERS: simulate a "real"
      measurement with the DEFECTIVE simulator, simulate the reference
      with the NOMINAL one, run the battery, and let identify() name
      the broken component WITHOUT being told.  Report the confusion
      matrix defect-family x diagnosis.  Then run the battery on the
      actual REAL ruotato vs the nominal simulator and report what it
      flags (expected: blur statistic fires for the bilinear v1
      simulator variant, stays quiet for the v2 cubic one).

OUTPUTS
    results/wp2_diagnostics.csv       statistic values per case
    results/wp2_diag_confusion.csv    truth family x diagnosed family
    figures/wp2_diag_confusion.png    the confusion matrix

DEFINITION OF DONE
    Battery of >= 4 statistics; confusion matrix on all ladder rungs;
    the real-scan verdict paragraph for the paper.

Run from the repo root:
    python neurips_submission/wp2_simulator_audit/exp_diagnostics.py --quick
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config
from common import core, io_utils, perturb


def hf_ratio(real: np.ndarray, sim: np.ndarray, frac: float = 0.25) -> float:
    """High-frequency power ratio real/sim (the statistic that caught
    the v1 blur mismatch).  frac = fraction of the spectrum counted as
    'high' along each axis."""
    def hf(img):
        F = np.abs(np.fft.rfft2(img - img.mean())) ** 2
        h, w = F.shape
        return float(F[int(h * (1 - frac)) :, :].sum()
                     + F[:, int(w * (1 - frac)) :].sum())
    return hf(real) / max(hf(sim), 1e-12)


# ---------------------------------------------------------------------------
# TODO(WP2): the rest of the battery + the identifier
# ---------------------------------------------------------------------------

def battery(real_maps: dict, sim_maps: dict) -> dict:
    """TODO(WP2): {statistic_name: {element: value}} for the full
    battery in the module docstring.  hf_ratio above is the template:
    each statistic is a scalar per element line comparing real vs sim."""
    raise NotImplementedError("TODO(WP2): implement the battery")


def identify(stats: dict) -> str:
    """TODO(WP2): map a battery result to a diagnosis in
    {'noise_k', 'gain', 'angle_bias', 'blur', 'warp', 'ok'}.  Start
    with interpretable z-score thresholds against the nominal-vs-
    nominal spread (estimate it from repeated nominal draws); a learned
    classifier is optional polish, not the core."""
    raise NotImplementedError("TODO(WP2): implement identify")


def run(quick: bool = False):
    rng = np.random.default_rng(config.BASE_SEED)
    p1 = core.fm.load_summed_maps("prova1")
    angle = core.fm.REF_ANGLE_DEG

    rows, confusion = [], []
    ladders = list(config.DEFECT_LADDERS.items())
    if quick:
        ladders = ladders[:2]
    for family, rungs in ladders:
        for label, kw in rungs:
            defective = perturb.SimKnobs(label=label, **kw)
            pseudo_real = perturb.forward_perturbed(p1, angle, rng, defective)
            reference = perturb.forward_perturbed(p1, angle, rng,
                                                  perturb.NOMINAL)
            stats = battery(pseudo_real, reference)
            verdict = identify(stats)
            confusion.append({"truth": family, "defect": label,
                              "diagnosis": verdict,
                              "correct": verdict == family})
            for stat, per_el in stats.items():
                for el, v in per_el.items():
                    rows.append({"defect_family": family, "defect": label,
                                 "statistic": stat, "element": el,
                                 "value": v})
    io_utils.write_rows("wp2_diagnostics", rows)
    path = io_utils.write_rows("wp2_diag_confusion", confusion)
    n_ok = sum(c["correct"] for c in confusion)
    print(f"saved: {path}  blind identification {n_ok}/{len(confusion)}")

    # the real-scan verdict: REAL ruotato vs its nominal simulation
    ruo = core.fm.load_summed_maps("ruotato")
    sim = perturb.forward_perturbed(p1, angle, rng, perturb.NOMINAL)
    # NOTE: ruo lives on the tilted grid already; sim is built from
    # prova1 - same grid, comparable.  TODO(WP2): run battery(ruo, sim),
    # append to the CSV with defect_family='REAL', and write the
    # verdict paragraph.


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    run(quick=args.quick)
