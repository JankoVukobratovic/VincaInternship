"""WP3 addendum - a parameter-free adaptive blend of the learned prior and
a classical fill, using the WP1 jitter ensemble's OWN per-pixel spread.

CLAIM UNDER TEST
    exp_degradation_grid.py's fixed hybrid ("classical_biharmonic+net":
    fill classically, THEN always trust the net) is a hand-picked
    combination.  Can a parameter-free rule that trusts the net exactly
    where its own ensemble says it is confident, and falls back on the
    classical fill where it is not, do at least as well - without ever
    looking at the test grid to tune anything?

METHOD (precision-weighted blend, no free hyperparameter)
    Per pixel, per element line:
        w_net = var_ref[el] / (var_ref[el] + std_jitter[el]^2)
        blend[el] = w_net * mean_jitter[el] + (1 - w_net) * biharmonic[el]

    var_ref[el] is fixed BEFORE seeing any test case: the squared P99
    normalisation scale of the training source (datagen.norm_scales()),
    i.e. a nominal "how big is a typical value of this line" reference,
    with a global constant c = 1 (var_ref = norm_scale^2 exactly - no
    grid search, no per-case fit).  This is the textbook inverse-
    variance ("precision") weighting of two independent estimates of
    the same quantity, with var_ref standing in for the (unknown, not
    estimated) variance of the classical fill: it is deliberately a
    ROUGH, fixed prior, not a fitted one, and the point of the
    experiment is to see whether even this crude rule is competitive.

    mean_jitter, std_jitter come from wp1_uq_ensemble.exp_ensemble_uq.
    ensemble_predict() on the SAME 12 jitter members used everywhere
    else in this package; biharmonic is common.classical.biharmonic_fill
    through the nominal physics inverse, exactly as in
    exp_degradation_grid.candidates().

    Where std_jitter is large (e.g. inside a dropout hole, see WP1),
    w_net -> 0 and the candidate reduces to the classical fill; where
    std_jitter is small (measured pixels), w_net -> 1 and it reduces to
    the ensemble mean.  No case-by-case decision, no threshold.

CANDIDATES SCORED (same grid as exp_degradation_grid.py, footprint+hole)
    det, net (MVP single), ens_jitter (plain mean, reference),
    classical_biharmonic (reference), classical_biharmonic+net (the
    existing fixed hybrid, reference), adaptive_blend (this addendum).

OUTPUTS (new files only - the original WP3 CSV/summary are untouched)
    results/wp3_adaptive_blend.csv
    results/wp3_adaptive_blend_summary.txt

Run from the repo root:
    python neurips_submission/wp3_degradation_benchmark/exp_adaptive_blend.py --quick
    python neurips_submission/wp3_degradation_benchmark/exp_adaptive_blend.py
    python neurips_submission/wp3_degradation_benchmark/exp_adaptive_blend.py --summary
"""

import argparse
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import config
from common import classical, core, io_utils, restore
from wp3_degradation_benchmark.exp_degradation_grid import (
    HARSH, load_jitter_members, case_key,
)
from wp1_uq_ensemble import exp_ensemble_uq as uq

ELEMENTS = core.ELEMENTS
CSV = "wp3_adaptive_blend"


def var_ref() -> dict:
    """Fixed, a priori per-line reference variance (P99-of-training-
    source scale, squared, c=1) - computed once, never touched by a
    test case."""
    scales = core.dg.norm_scales()
    return {el: float(scales[i]) ** 2 for i, el in enumerate(ELEMENTS)}


_VAR_REF = None


def get_var_ref() -> dict:
    global _VAR_REF
    if _VAR_REF is None:
        _VAR_REF = var_ref()
    return _VAR_REF


def adaptive_blend(mean_jitter: dict, std_jitter: dict,
                   biharmonic: dict) -> dict:
    vr = get_var_ref()
    out = {}
    for el in ELEMENTS:
        w = vr[el] / (vr[el] + std_jitter[el] ** 2)
        out[el] = w * mean_jitter[el] + (1.0 - w) * biharmonic[el]
    return out


def candidates(net, members, tilted, v_tilt, angle, validity):
    """Same candidate set as exp_degradation_grid.candidates(), plus
    adaptive_blend; v_tilt is required here (holes are the point)."""
    det, learned = restore.apply_network(net, tilted, angle,
                                         validity=validity)
    cands = {"det": det, "net": learned}
    mean_j = std_j = bih = None
    if members:
        mean_j, std_j, _, _ = uq.ensemble_predict(members, tilted, angle,
                                                  validity=validity)
        cands["ens_jitter"] = mean_j
    if v_tilt is not None:
        bih = classical.biharmonic_fill(tilted, v_tilt)
        cands["classical_biharmonic"] = {
            el: restore.fm.inverse(bih, angle_deg=angle)[el]
            for el in ELEMENTS}
        _, hybrid = restore.apply_network(net, bih, angle, validity=None)
        cands["classical_biharmonic+net"] = hybrid
        if mean_j is not None:
            cands["adaptive_blend"] = adaptive_blend(
                mean_j, std_j, cands["classical_biharmonic"])
    return cands


def done_keys(rows):
    return {case_key(r["angle"], r["hole"], r["dose"], r["seed"], r["sim"])
            for r in rows}


def run_case(net, members, angle, h, w, dose, seed, sim):
    case = restore.degrade(source="prova2", angle=angle,
                           block=restore.centered_block(h, w), dose=dose,
                           seed=seed, sim=sim)
    cands = candidates(net, members, case["tilted"], case["v_tilt"],
                       case["angle"], case["validity"])
    scored = restore.score_candidates(
        cands, case["truth"], {"footprint": case["fp"], "hole": case["hole"]})
    return [{"angle": angle, "hole_px": h * w, "hole": f"{h}x{w}",
             "dose": dose, "seed": seed, "sim": sim, **r} for r in scored]


def run(quick: bool = False):
    torch.set_num_threads(2)
    net = restore.load_mvp_net()
    if net is None:
        raise SystemExit("MVP checkpoint missing - see exp_degradation_grid")
    members = load_jitter_members()
    if not members:
        raise SystemExit("WP1 jitter ensemble missing/incomplete - "
                         "adaptive_blend needs it")

    grid = config.GRID_QUICK if quick else config.GRID
    plan = [(a, hw, d, s, "validated") for a, hw, d, s in itertools.product(
        grid["angles"], grid["holes"], grid["doses"], grid["seeds"])]

    rows = io_utils.read_rows(CSV)
    done = done_keys(rows)
    t0 = time.time()
    n_new = 0
    for i, (angle, (h, w), dose, seed, sim) in enumerate(plan):
        key = case_key(angle, f"{h}x{w}", dose, seed, sim)
        if key in done:
            continue
        print(f"[{i + 1}/{len(plan)}] angle={angle:g} hole={h}x{w} "
              f"dose={dose:g} seed={seed}  [{time.time() - t0:.0f} s]",
              flush=True)
        rows += run_case(net, members, angle, h, w, dose, seed, sim)
        done.add(key)
        n_new += 1
        if n_new % 10 == 0:
            io_utils.write_rows(CSV, rows)

    # real-scan anchor
    if case_key(core.fm.REF_ANGLE_DEG, "0x0", 1.0, -1, "real") not in done:
        ruo = core.fm.load_summed_maps("ruotato")
        truth2 = core.fm.load_summed_maps("prova2")
        cands = candidates(net, members, ruo, np.ones(core.TILTED_SHAPE),
                           core.fm.REF_ANGLE_DEG, None)
        scored = restore.score_candidates(cands, truth2,
                                          {"footprint": core.dg.footprint()})
        rows += [{"angle": core.fm.REF_ANGLE_DEG, "hole_px": 0, "hole": "0x0",
                  "dose": 1.0, "seed": -1, "sim": "real", **r}
                 for r in scored]
        print("real anchor done")
    path = io_utils.write_rows(CSV, rows)
    print(f"saved: {path}  ({len(rows)} rows, {n_new} new cases,"
          f" {time.time() - t0:.0f} s)")
    summarize()


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def _sel(rows, **cond):
    out = []
    for r in rows:
        ok = True
        for k, v in cond.items():
            rv = r.get(k, "")
            if callable(v):
                ok &= bool(v(rv))
            else:
                try:
                    ok &= float(rv) == float(v)
                except (TypeError, ValueError):
                    ok &= str(rv) == str(v)
        if ok:
            out.append(r)
    return out


def _mean(rows, key="r"):
    v = []
    for r in rows:
        try:
            x = float(r[key])
        except (TypeError, ValueError):
            continue
        if np.isfinite(x):
            v.append(x)
    return float(np.mean(v)) if v else float("nan")


CANDS = ("det", "net", "ens_jitter", "classical_biharmonic",
         "classical_biharmonic+net", "adaptive_blend")


def summarize():
    rows = io_utils.read_rows(CSV)
    if not rows:
        print("no rows yet - run without --summary first")
        return ""
    sim = _sel(rows, sim="validated")
    holes = [f"{h}x{w}" for (h, w) in config.GRID["holes"]]
    angles = list(config.GRID["angles"])
    present = [c for c in CANDS if _sel(sim, candidate=c)]

    L = []
    P = L.append
    P("WP3 addendum - parameter-free adaptive blend (precision-weighted:"
      " ensemble mean where jitter spread is low, biharmonic fill where"
      " it is high; var_ref fixed a priori = norm_scales()^2, no fit)")
    P(f"rows {len(rows)}; candidates: {', '.join(present)}")
    P("")

    def table(region, dose):
        P(f"[{region}, dose {dose:g}, mean r over seeds and all 8 lines]")
        P("    " + f"{'angle':>6s} {'hole':>6s}" + "".join(
            f"{c[:24]:>26s}" for c in present) + f"{'best':>26s}")
        for a in angles:
            for hole in holes:
                if region == "hole" and hole == "0x0":
                    continue
                vals = {c: _mean(_sel(sim, angle=a, hole=hole, dose=dose,
                                      candidate=c, region=region))
                        for c in present}
                best = max(vals, key=lambda c: (vals[c] if np.isfinite(
                    vals[c]) else -9))
                P("    " + f"{a:6g} {hole:>6s}" + "".join(
                    f"{vals[c]:26.4f}" for c in present) + f"{best:>26s}")
        P("")

    table("hole", 1.0)
    table("footprint", 1.0)

    # the honest head-to-head verdicts
    P("[verdict: adaptive_blend vs the fixed hybrid classical_biharmonic+net]")
    for region in ("hole", "footprint"):
        holes_r = holes[1:] if region == "hole" else holes
        d_ab, d_hy = [], []
        for a in angles:
            for hole in holes_r:
                v_ab = _mean(_sel(sim, angle=a, hole=hole, dose=1.0,
                                  candidate="adaptive_blend", region=region))
                v_hy = _mean(_sel(sim, angle=a, hole=hole, dose=1.0,
                                  candidate="classical_biharmonic+net",
                                  region=region))
                if np.isfinite(v_ab) and np.isfinite(v_hy):
                    d_ab.append(v_ab)
                    d_hy.append(v_hy)
        d_ab, d_hy = np.array(d_ab), np.array(d_hy)
        wins = int(np.sum(d_ab > d_hy))
        P(f"  {region}: adaptive_blend beats the fixed hybrid in"
          f" {wins}/{len(d_ab)} cells (dose 1); mean r"
          f" adaptive_blend {d_ab.mean():.4f} vs hybrid {d_hy.mean():.4f}"
          f" (delta {d_ab.mean() - d_hy.mean():+.4f})")

    P("")
    P("[verdict: adaptive_blend vs plain ens_jitter and vs classical_"
      "biharmonic alone, hole region, dose 1]")
    for other in ("ens_jitter", "classical_biharmonic"):
        d_ab, d_o = [], []
        for a in angles:
            for hole in holes[1:]:
                v_ab = _mean(_sel(sim, angle=a, hole=hole, dose=1.0,
                                  candidate="adaptive_blend", region="hole"))
                v_o = _mean(_sel(sim, angle=a, hole=hole, dose=1.0,
                                 candidate=other, region="hole"))
                if np.isfinite(v_ab) and np.isfinite(v_o):
                    d_ab.append(v_ab)
                    d_o.append(v_o)
        d_ab, d_o = np.array(d_ab), np.array(d_o)
        wins = int(np.sum(d_ab > d_o))
        P(f"  vs {other}: adaptive_blend wins {wins}/{len(d_ab)} cells;"
          f" mean r adaptive_blend {d_ab.mean():.4f} vs {other}"
          f" {d_o.mean():.4f} (delta {d_ab.mean() - d_o.mean():+.4f})")

    # dose effect
    P("")
    P("[dose effect: mean r over angles, seeds, 8 lines; hole 14x20 /"
      " footprint 0x0]")
    for d in config.GRID["doses"]:
        parts = []
        for c in present:
            v_h = _mean(_sel(sim, hole="14x20", dose=d, candidate=c,
                             region="hole"))
            v_f = _mean(_sel(sim, hole="0x0", dose=d, candidate=c,
                             region="footprint"))
            parts.append(f"{c} {v_h:.3f}/{v_f:.3f}")
        P(f"  dose {d:g}: " + "  ".join(parts))

    # real anchor
    real = _sel(rows, sim="real")
    if real:
        P("")
        P("[REAL anchor: ruotato vs prova2, footprint, mean r (cv_ratio)]")
        rc = [c for c in present if _sel(real, candidate=c)]
        for c in rc:
            P(f"    {c:26s} {_mean(_sel(real, candidate=c)):.4f}"
              f" ({_mean(_sel(real, candidate=c), 'cv_ratio'):.3f})")

    text = "\n".join(L)
    print(text)
    path = os.path.join(core.RESULTS_DIR, "wp3_adaptive_blend_summary.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summarize()
    else:
        run(quick=args.quick)
