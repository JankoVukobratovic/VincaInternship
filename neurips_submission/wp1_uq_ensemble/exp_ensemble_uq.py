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

METHOD (all implemented)
    1. Train N = config.ENSEMBLE_N nets, each on perturb.jittered knobs
       (member i has its own knobs AND its own seed).
    2. Train N control nets with NOMINAL knobs, same seeds.
    3. Predict on held-out cases (config.WP1_CASES: simulated from
       prova2 - never in training - over angle x hole x test-simulator
       x dose) and on the REAL ruotato scan.
    4. Coverage: fraction of pixels with |mean - truth| <= z*sigma for
       z in config.COVERAGE_Z, per line, region and ensemble, for two
       uncertainty bands:
         band "ens"   : sigma = ensemble spread only (epistemic:
                        init + noise-of-training + simulator draw)
         band "total" : sigma^2 = spread^2 + sigma_alea^2, where
                        sigma_alea is the spread of the ensemble MEAN
                        over config.WP1_NOISE_REPS noise replicates of
                        the same case (the simulator used as an
                        aleatoric-uncertainty provider).  On the REAL
                        scan the 7.7 deg simulated twin supplies
                        sigma_alea and the calibrated reference-map
                        noise k*truth is added, because the reference
                        (prova2) is itself a noisy measurement.
    5. Variance decomposition per region: var_jitter - var_control is
       the variance attributable to the imperfect simulator; error-
       ranking diagnostics: Spearman(sigma, |err|) and the AUSE of the
       sparsification curve, jitter vs control.
    6. Accuracy of the ensemble means vs the deterministic inverse, a
       single member and the MVP net (does jitter training cost r?).

FINITE-ENSEMBLE NOTE (decision documented as required by the TODO)
    sigma is the ddof=1 sample std over N members; NO inflation factor
    is applied.  The c4 bias of the sample std is 2.2 % at N = 12
    (0.978) and a Student-t band with 11 dof widens z = 2 by ~10 %;
    both are far below the calibration effects of interest (factors of
    2-3) and are stated in the paper, not corrected.

OUTPUTS
    results/wp1_ensemble_members.csv   per-member knobs + training summary
    results/wp1_uq_coverage.csv        ensemble, band, element, case,
                                       region, z, coverage, expected, n_px
    results/wp1_uq_diagnostics.csv     variance decomposition + ranking
    results/wp1_uq_accuracy.csv        r/ssim/bias of the candidates
    results/wp1_summary.txt            aggregated numbers for the paper
    results/wp1_ensemble/*.pt, *.json  member checkpoints (gitignored)
    results/wp1_ensemble/maps_*.npz    per-case maps for figures
    figures/wp1_calibration.png        z vs empirical coverage
    figures/wp1_spread_maps.png        jitter vs control spread, harsh case
    figures/wp1_error_vs_sigma.png     binned reliability diagram

DEFINITION OF DONE
    Coverage numbers for both ensembles on >= 6 simulated cases + the
    real scan; a one-figure calibration plot; three sentences of
    interpretation in the paper's results section.

Run from the repo root:
    python neurips_submission/wp1_uq_ensemble/exp_ensemble_uq.py --quick
    # parallel training on a many-core box (4 shells, then the full run):
    python .../exp_ensemble_uq.py --train-only --kind jitter  --members 0-5  --threads 3
    python .../exp_ensemble_uq.py --train-only --kind jitter  --members 6-11 --threads 3
    python .../exp_ensemble_uq.py --train-only --kind control --members 0-5  --threads 3
    python .../exp_ensemble_uq.py --train-only --kind control --members 6-11 --threads 3
    python .../exp_ensemble_uq.py            # picks up the cached members
"""

import argparse
import json
import math
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy.stats import spearmanr

import config
from common import core, io_utils, perturb, restore, training

ELEMENTS = core.ELEMENTS
KINDS = ("jitter", "control")
_trapz = getattr(np, "trapezoid", None) or np.trapz


def gauss_cov(z: float) -> float:
    """Coverage of a |N(0,1)| <= z band (the perfect-calibration target)."""
    return math.erf(z / math.sqrt(2.0))


# ---------------------------------------------------------------------------
# members: seeds, knobs, training, caching
# ---------------------------------------------------------------------------

_QUICK = False   # quick runs keep their (under-trained) members apart


def set_quick(flag: bool):
    global _QUICK
    _QUICK = bool(flag)


def member_dir():
    d = os.path.join(core.RESULTS_DIR,
                     "wp1_ensemble" + ("_quick" if _QUICK else ""))
    os.makedirs(d, exist_ok=True)
    return d


def member_seed(kind: str, i: int) -> int:
    return config.BASE_SEED + 100 * i + (0 if kind == "jitter" else 1)


def member_knobs(kind: str, i: int) -> perturb.SimKnobs:
    """Member i's simulator belief: a jitter draw or the nominal knobs."""
    if kind == "control":
        return perturb.NOMINAL
    rng = np.random.default_rng(member_seed(kind, i))
    return perturb.jittered(rng, config.JITTER, f"jitter_{i}")


def _member_paths(kind: str, i: int):
    base = os.path.join(member_dir(), f"{kind}_{i:02d}")
    return base + ".pt", base + ".json"


def train_member(kind: str, i: int, train_cfg: dict, verbose: bool = False):
    """Train one member or load it from the cache; returns the net."""
    ckpt, meta = _member_paths(kind, i)
    net = core.RestorationUNet()
    if os.path.exists(ckpt):
        net.load_state_dict(torch.load(ckpt, weights_only=True))
        net.eval()
        print(f"[{kind} {i:02d}] cached")
        return net
    knobs = member_knobs(kind, i)
    print(f"[{kind} {i:02d}] training  knobs={knobs.to_meta()}", flush=True)
    net, hist = training.train_net(training.make_batch_fn(knobs=knobs),
                                   train_cfg, seed=member_seed(kind, i),
                                   verbose=verbose)
    torch.save(net.state_dict(), ckpt)
    with open(meta, "w", encoding="utf-8") as fh:
        json.dump({"ensemble": kind, "member": i, **knobs.to_meta(), **hist,
                   "train_steps": train_cfg.get("steps"),
                   "train_batch": train_cfg.get("batch")}, fh, indent=1)
    print(f"[{kind} {i:02d}] done  best val L1 {hist['best_val_l1']:.5f}"
          f" at step {hist['best_step']}  ({hist['wall_s']:.0f} s)",
          flush=True)
    return net


def train_ensemble(kind: str, n: int, train_cfg: dict, members=None,
                   verbose: bool = False) -> list:
    idx = range(n) if members is None else members
    return [train_member(kind, i, train_cfg, verbose) for i in idx]


def write_members_csv():
    """Rebuild results/wp1_ensemble_members.csv from the member json files
    (safe under parallel training - no append races)."""
    rows = []
    for f in sorted(os.listdir(member_dir())):
        if f.endswith(".json") and (f.startswith("jitter_")
                                    or f.startswith("control_")):
            with open(os.path.join(member_dir(), f), encoding="utf-8") as fh:
                rows.append(json.load(fh))
    if rows:
        io_utils.write_rows("wp1_ensemble_members", rows)
    return rows


# ---------------------------------------------------------------------------
# prediction
# ---------------------------------------------------------------------------

def ensemble_predict(members: list, tilted, angle, validity=None):
    """(mean, std, det, stack) over member restorations, physical units.

    std is the ddof=1 sample std (see the finite-ensemble note); det is
    the deterministic physics inverse (identical for every member).
    """
    preds = {el: [] for el in ELEMENTS}
    det = None
    for net in members:
        det, learned = restore.apply_network(net, tilted, angle,
                                             validity=validity)
        for el in ELEMENTS:
            preds[el].append(learned[el])
    stack = {el: np.stack(v) for el, v in preds.items()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = {el: np.nanmean(s, axis=0) for el, s in stack.items()}
        if len(members) > 1:
            std = {el: np.nanstd(s, axis=0, ddof=1) for el, s in stack.items()}
        else:
            std = {el: np.zeros_like(mean[el]) for el in ELEMENTS}
    return mean, std, det, stack


def aleatoric_sigma(members: list, case: dict, reps: int):
    """Propagated measurement noise: spread of the ensemble mean (and of
    the deterministic inverse) over `reps` noise replicates of the case.
    Returns (sigma_net, sigma_det) dicts."""
    means = {el: [] for el in ELEMENTS}
    dets = {el: [] for el in ELEMENTS}
    for j in range(reps):
        c = restore.degrade(source=case["source"], angle=case["angle"],
                            block=case["block"], dose=case["dose"],
                            seed=10_000 + j, sim=case["sim"])
        m, _, d, _ = ensemble_predict(members, c["tilted"], c["angle"],
                                      validity=c["validity"])
        for el in ELEMENTS:
            means[el].append(m[el])
            dets[el].append(d[el])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        s_net = {el: np.nanstd(np.stack(v), axis=0, ddof=1)
                 for el, v in means.items()}
        s_det = {el: np.nanstd(np.stack(v), axis=0, ddof=1)
                 for el, v in dets.items()}
    return s_net, s_det


# ---------------------------------------------------------------------------
# test cases
# ---------------------------------------------------------------------------

def case_tag(angle, hole, sim, dose) -> str:
    return f"a{angle:g}_h{hole[0]}x{hole[1]}_{sim}_d{dose:g}"


def case_meta(case: dict) -> dict:
    h, w = (case["block"][2], case["block"][3]) if case["block"] else (0, 0)
    return {"case": case_tag(case["angle"], (h, w), case["sim"],
                             case["dose"]),
            "case_angle": case["angle"], "case_hole": f"{h}x{w}",
            "case_sim": case["sim"], "case_dose": case["dose"],
            "source": case["source"]}


def test_cases(quick: bool) -> list:
    """Held-out cases: simulated from prova2 (never in training)."""
    spec = config.WP1_CASES_QUICK if quick else config.WP1_CASES
    cases = []
    for sim in spec["sims"]:
        for dose in spec["doses"]:
            for angle in spec["angles"]:
                for (h, w) in spec["holes"]:
                    cases.append(restore.degrade(
                        source="prova2", angle=angle,
                        block=restore.centered_block(h, w), dose=dose,
                        seed=spec["seed"], sim=sim))
    return cases


# ---------------------------------------------------------------------------
# the scientific core: coverage, decomposition, ranking, accuracy
# ---------------------------------------------------------------------------

def coverage_rows(mean, std, truth, regions, meta, extra_var=None,
                  band="ens") -> list:
    """Per-line empirical coverage of |mean - truth| <= z * sigma.

    sigma = std (band "ens") or sqrt(std^2 + extra_var) (band "total").
    Zero-spread pixels count as covered only if the error is exactly 0
    (the <= comparison does that by itself).  No finite-ensemble
    inflation is applied (see the module docstring).
    """
    rows = []
    for el in ELEMENTS:
        err = np.abs(mean[el] - truth[el])
        sig = std[el] if extra_var is None else np.sqrt(
            std[el] ** 2 + extra_var[el])
        for reg, mask in regions.items():
            if mask is None:
                continue
            m = mask & np.isfinite(err) & np.isfinite(sig)
            n = int(m.sum())
            if n < 9:
                continue
            e, s = err[m], sig[m]
            for z in config.COVERAGE_Z:
                rows.append({**meta, "band": band, "element": el,
                             "region": reg, "z": z,
                             "coverage": float(np.mean(e <= z * s)),
                             "expected": gauss_cov(z), "n_px": n,
                             "sigma_rms": float(np.sqrt(np.mean(s ** 2))),
                             "err_rms": float(np.sqrt(np.mean(e ** 2)))})
    return rows


def sparsification_ause(err: np.ndarray, sig: np.ndarray,
                        nbins: int = 20) -> float:
    """Area between the sigma-ordered and the oracle (error-ordered)
    sparsification curves, normalised by the full RMSE (0 = perfect
    ranking of the errors by the predicted uncertainty)."""
    n = len(err)
    if n < 20 or not np.isfinite(err).all():
        return float("nan")
    fr = np.linspace(0.0, 0.9, nbins)
    e2 = err ** 2

    def curve(order):
        e = e2[order]
        return np.array([np.sqrt(e[int(f * n):].mean()) for f in fr])

    cs = curve(np.argsort(-sig, kind="stable"))
    co = curve(np.argsort(-err, kind="stable"))
    return float(_trapz(cs - co, fr) / cs[0]) if cs[0] > 0 else float("nan")


def _spearman(a, b) -> float:
    if len(a) < 10 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def diagnostic_rows(pred: dict, truth, regions, meta) -> list:
    """Variance decomposition and error-ranking quality per line/region.

    pred = {"jitter": (mean, std), "control": (mean, std),
            "alea": sigma_alea (jitter ensemble), "alea_det": sigma_alea
            of the deterministic inverse}
    """
    mean_j, std_j = pred["jitter"]
    mean_c, std_c = pred["control"]
    alea, alea_det = pred["alea"], pred["alea_det"]
    rows = []
    for el in ELEMENTS:
        for reg, mask in regions.items():
            if mask is None:
                continue
            m = (mask & np.isfinite(mean_j[el]) & np.isfinite(mean_c[el])
                 & np.isfinite(truth[el]))
            n = int(m.sum())
            if n < 9:
                continue
            ej = np.abs(mean_j[el] - truth[el])[m]
            ec = np.abs(mean_c[el] - truth[el])[m]
            sj, sc, sa = std_j[el][m], std_c[el][m], alea[el][m]
            vj, vc, va = float(np.mean(sj ** 2)), float(np.mean(sc ** 2)), \
                float(np.mean(sa ** 2))
            s_sim = np.sqrt(np.clip(sj ** 2 - sc ** 2, 0.0, None))
            rows.append({
                **meta, "element": el, "region": reg, "n_px": n,
                "var_control": vc, "var_jitter": vj,
                "var_sim": max(vj - vc, 0.0), "var_alea": va,
                "var_alea_det": float(np.mean(alea_det[el][m] ** 2)),
                "frac_sim": (vj - vc) / vj if vj > 0 else float("nan"),
                "spread_ratio": math.sqrt(vj / vc) if vc > 0 else float("nan"),
                "mse_jitter": float(np.mean(ej ** 2)),
                "mse_control": float(np.mean(ec ** 2)),
                "truth_var": float(np.var(truth[el][m])),
                "spearman_jitter": _spearman(sj, ej),
                "spearman_control": _spearman(sc, ec),
                "spearman_sim": _spearman(s_sim, ej),
                "spearman_total_jitter": _spearman(np.sqrt(sj ** 2 + sa ** 2),
                                                   ej),
                "spearman_total_control": _spearman(
                    np.sqrt(sc ** 2 + sa ** 2), ec),
                "ause_jitter": sparsification_ause(ej, sj),
                "ause_control": sparsification_ause(ec, sc),
                "ause_total_jitter": sparsification_ause(
                    ej, np.sqrt(sj ** 2 + sa ** 2)),
                "ause_total_control": sparsification_ause(
                    ec, np.sqrt(sc ** 2 + sa ** 2))})
    return rows


def accuracy_rows(cands: dict, truth, regions, meta) -> list:
    return [{**meta, **r} for r in
            restore.score_candidates(cands, truth, regions)]


# ---------------------------------------------------------------------------
# one case end-to-end
# ---------------------------------------------------------------------------

def evaluate_case(ensembles: dict, case: dict, meta: dict, regions: dict,
                  truth: dict, mvp_net, reps: int, real: bool = False,
                  twin: dict | None = None):
    """Coverage / diagnostics / accuracy rows for one case (sim or real).

    For the real scan `case["tilted"]` is the measured ruotato,
    `twin` the 7.7 deg simulated case that provides sigma_alea, and the
    reference-map noise k*truth enters the total band.
    """
    cov, diag, acc = [], [], []
    pred = {}
    stacks = {}
    for kind in KINDS:
        mean, std, det, stack = ensemble_predict(
            ensembles[kind], case["tilted"], case["angle"],
            validity=case.get("validity"))
        pred[kind] = (mean, std)
        stacks[kind] = stack
        pred["det"] = det
    alea = {}
    for kind in KINDS:
        alea[kind], alea_det = aleatoric_sigma(
            ensembles[kind], twin if twin is not None else case, reps)
    pred["alea"], pred["alea_det"] = alea["jitter"], alea_det

    ref_var = None
    if real:
        ks = core.fm.calibrate_noise()
        ref_var = {el: ks[el] * np.clip(truth[el], 0.0, None)
                   for el in ELEMENTS}

    for kind in KINDS:
        mean, std = pred[kind]
        base = {"ensemble": kind, **meta}
        cov += coverage_rows(mean, std, truth, regions, base, band="ens")
        extra = {el: alea[kind][el] ** 2 + (ref_var[el] if ref_var else 0.0)
                 for el in ELEMENTS}
        cov += coverage_rows(mean, std, truth, regions, base,
                             extra_var=extra, band="total")
        if real:
            extra2 = {el: alea[kind][el] ** 2 for el in ELEMENTS}
            cov += coverage_rows(mean, std, truth, regions, base,
                                 extra_var=extra2, band="total_noref")
        else:
            # the simulated cases are ALSO scored against a measured
            # (noisy) reference whose noise partly passes through the
            # forward warp into the input; adding the full k*truth is an
            # UPPER bound on that contribution, "total" a lower bound
            ks = core.fm.calibrate_noise()
            extra3 = {el: alea[kind][el] ** 2
                      + ks[el] * np.clip(truth[el], 0.0, None)
                      for el in ELEMENTS}
            cov += coverage_rows(mean, std, truth, regions, base,
                                 extra_var=extra3, band="total_ref")
    diag += diagnostic_rows(pred, truth, regions, meta)

    cands = {"deterministic": pred["det"],
             "control_mean": pred["control"][0],
             "jitter_mean": pred["jitter"][0],
             "control_single": {el: stacks["control"][el][0]
                                for el in ELEMENTS},
             "jitter_single": {el: stacks["jitter"][el][0]
                               for el in ELEMENTS}}
    if mvp_net is not None:
        _, mvp = restore.apply_network(mvp_net, case["tilted"],
                                       case["angle"],
                                       validity=case.get("validity"))
        cands["mvp_single"] = mvp
    acc += accuracy_rows(cands, truth, regions, meta)

    # maps for the figures (gitignored, rebuilt by any rerun)
    np.savez_compressed(
        os.path.join(member_dir(), f"maps_{meta['case']}.npz"),
        elements=np.array(ELEMENTS),
        truth=np.stack([truth[el] for el in ELEMENTS]).astype(np.float32),
        det=np.stack([pred["det"][el] for el in ELEMENTS]).astype(np.float32),
        mean_jitter=np.stack([pred["jitter"][0][el] for el in ELEMENTS]
                             ).astype(np.float32),
        std_jitter=np.stack([pred["jitter"][1][el] for el in ELEMENTS]
                            ).astype(np.float32),
        mean_control=np.stack([pred["control"][0][el] for el in ELEMENTS]
                              ).astype(np.float32),
        std_control=np.stack([pred["control"][1][el] for el in ELEMENTS]
                             ).astype(np.float32),
        alea_jitter=np.stack([alea["jitter"][el] for el in ELEMENTS]
                             ).astype(np.float32),
        alea_control=np.stack([alea["control"][el] for el in ELEMENTS]
                              ).astype(np.float32),
        alea_det=np.stack([alea_det[el] for el in ELEMENTS]).astype(np.float32),
        ref_var=(np.stack([ref_var[el] for el in ELEMENTS]).astype(np.float32)
                 if ref_var else np.zeros((len(ELEMENTS),) + core.FRONTAL_SHAPE,
                                          np.float32)),
        footprint=regions["footprint"],
        hole=(regions.get("hole") if regions.get("hole") is not None
              else np.zeros(core.FRONTAL_SHAPE, bool)),
        meta=json.dumps(meta))
    return cov, diag, acc


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run(quick: bool = False, verbose: bool = False):
    set_quick(quick)
    n = config.ENSEMBLE_N_QUICK if quick else config.ENSEMBLE_N
    tcfg = config.QUICK_TRAIN if quick else config.TRAIN
    reps = 3 if quick else config.WP1_NOISE_REPS
    t0 = time.time()
    ensembles = {"jitter": train_ensemble("jitter", n, tcfg, verbose=verbose),
                 "control": train_ensemble("control", n, tcfg,
                                           verbose=verbose)}
    write_members_csv()
    mvp_net = restore.load_mvp_net()
    print(f"members ready ({time.time() - t0:.0f} s); MVP net "
          f"{'found' if mvp_net else 'missing'}")

    cov, diag, acc = [], [], []
    cases = test_cases(quick)
    for k, case in enumerate(cases):
        meta = case_meta(case)
        regions = {"footprint": case["fp"], "hole": case["hole"]
                   if case["block"] is not None else None}
        c, d, a = evaluate_case(ensembles, case, meta, regions,
                                case["truth"], mvp_net, reps)
        cov += c
        diag += d
        acc += a
        print(f"  case {k + 1}/{len(cases)} {meta['case']:32s}"
              f" [{time.time() - t0:.0f} s]", flush=True)

    # real-scan anchor: REAL ruotato restored, scored against prova2
    ruo = core.fm.load_summed_maps("ruotato")
    truth2 = core.fm.load_summed_maps("prova2")
    twin = restore.degrade(source="prova2", angle=core.fm.REF_ANGLE_DEG,
                           block=None, dose=1.0, seed=0, sim="validated")
    real_case = {"tilted": ruo, "angle": core.fm.REF_ANGLE_DEG,
                 "validity": None}
    meta = {"case": "REAL_ruotato", "case_angle": core.fm.REF_ANGLE_DEG,
            "case_hole": "0x0", "case_sim": "real", "case_dose": 1.0,
            "source": "REAL_ruotato"}
    c, d, a = evaluate_case(ensembles, real_case, meta,
                            {"footprint": core.dg.footprint(), "hole": None},
                            truth2, mvp_net, reps, real=True, twin=twin)
    cov += c
    diag += d
    acc += a
    print(f"  real anchor done [{time.time() - t0:.0f} s]")

    print("saved:", io_utils.write_rows("wp1_uq_coverage", cov),
          f"({len(cov)} rows)")
    print("saved:", io_utils.write_rows("wp1_uq_diagnostics", diag),
          f"({len(diag)} rows)")
    print("saved:", io_utils.write_rows("wp1_uq_accuracy", acc),
          f"({len(acc)} rows)")
    summarize()


# ---------------------------------------------------------------------------
# summary for the paper
# ---------------------------------------------------------------------------

def _f(rows, **cond):
    out = []
    for r in rows:
        ok = True
        for k, v in cond.items():
            rv = r.get(k, "")
            if callable(v):
                ok &= bool(v(rv))
            else:
                ok &= (str(rv) == str(v))
        if ok:
            out.append(r)
    return out


def _mean(rows, key):
    v = [float(r[key]) for r in rows if r.get(key, "") not in ("", "nan")]
    v = [x for x in v if np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def summarize(path: str | None = None) -> str:
    cov = io_utils.read_rows("wp1_uq_coverage")
    diag = io_utils.read_rows("wp1_uq_diagnostics")
    acc = io_utils.read_rows("wp1_uq_accuracy")
    mem = io_utils.read_rows("wp1_ensemble_members")
    if not cov:
        print("no coverage rows yet")
        return ""
    L = []
    P = L.append
    is_sim = lambda s: s != "REAL_ruotato"  # noqa: E731
    is_real = lambda s: s == "REAL_ruotato"  # noqa: E731
    n_sim = len({r["case"] for r in cov if is_sim(r["source"])})
    nj = len(_f(mem, ensemble="jitter"))
    nc = len(_f(mem, ensemble="control"))
    P("WP1 - simulator-uncertainty ensemble UQ: summary")
    P(f"members: {nj} jitter + {nc} control; held-out simulated cases:"
      f" {n_sim}; real anchor: ruotato vs prova2")
    if mem:
        P("training: mean best val L1 jitter "
          f"{_mean(_f(mem, ensemble='jitter'), 'best_val_l1'):.5f}, control "
          f"{_mean(_f(mem, ensemble='control'), 'best_val_l1'):.5f}; mean "
          f"wall {_mean(mem, 'wall_s'):.0f} s / member")
    P("")
    P("[1] coverage of |mean - truth| <= z*sigma (mean over lines and"
      " cases; Gaussian target in brackets)")
    for src_name, src in (("simulated", is_sim), ("REAL", is_real)):
        regs = ("footprint", "hole") if src_name == "simulated" else \
            ("footprint",)
        bands = ("ens", "total", "total_ref") if src_name == "simulated" \
            else ("ens", "total_noref", "total")
        for reg in regs:
            P(f"  {src_name} / {reg}")
            head = "    " + f"{'band':12s}{'ensemble':9s}" + "".join(
                f"{'z=' + str(z):>12s}" for z in config.COVERAGE_Z)
            P(head)
            for band in bands:
                for kind in KINDS:
                    vals = []
                    for z in config.COVERAGE_Z:
                        rr = _f(cov, source=src, region=reg, band=band,
                                ensemble=kind, z=z)
                        vals.append(_mean(rr, "coverage"))
                    P("    " + f"{band:12s}{kind:9s}" + "".join(
                        f"{v:8.3f}[{gauss_cov(z):.2f}]"
                        for v, z in zip(vals, config.COVERAGE_Z)))
    P("")
    P("[2] spread decomposition (mean over lines and cases; sigma in"
      " counts, ratios dimensionless)")
    for src_name, src in (("simulated", is_sim), ("REAL", is_real)):
        regs = ("footprint", "hole") if src_name == "simulated" else \
            ("footprint",)
        for reg in regs:
            rr = _f(diag, source=src, region=reg)
            if not rr:
                continue
            vc, vj, va = _mean(rr, "var_control"), _mean(rr, "var_jitter"), \
                _mean(rr, "var_alea")
            P(f"  {src_name} / {reg}: sigma_control {math.sqrt(vc):.1f}"
              f"  sigma_jitter {math.sqrt(vj):.1f}"
              f"  sigma_sim(=sqrt(vj-vc)) {math.sqrt(max(vj - vc, 0)):.1f}"
              f"  sigma_alea {math.sqrt(va):.1f}"
              f"  rms_err(jitter mean) {math.sqrt(_mean(rr, 'mse_jitter')):.1f}"
              f"  rms_err(control mean) {math.sqrt(_mean(rr, 'mse_control')):.1f}")
            P(f"    spread ratio jitter/control {_mean(rr, 'spread_ratio'):.2f}"
              f"  frac of jitter variance from the simulator"
              f" {_mean(rr, 'frac_sim'):.2f}")
            P(f"    error ranking  Spearman(sigma,|err|): jitter"
              f" {_mean(rr, 'spearman_jitter'):.3f}  control"
              f" {_mean(rr, 'spearman_control'):.3f}  sim-only"
              f" {_mean(rr, 'spearman_sim'):.3f} | total band: jitter"
              f" {_mean(rr, 'spearman_total_jitter'):.3f}  control"
              f" {_mean(rr, 'spearman_total_control'):.3f}")
            P(f"    AUSE (0 = oracle): jitter {_mean(rr, 'ause_jitter'):.3f}"
              f"  control {_mean(rr, 'ause_control'):.3f} | total band:"
              f" jitter {_mean(rr, 'ause_total_jitter'):.3f}  control"
              f" {_mean(rr, 'ause_total_control'):.3f}")
    # per-angle spread ratio (does simulator uncertainty grow with the
    # extrapolation distance from the 7.7 deg calibration?)
    angles = sorted({float(r["case_angle"]) for r in diag
                     if is_sim(r["source"])})
    if angles:
        P("  spread ratio jitter/control by angle (sim, footprint): " + "  ".join(
            f"{a:g}deg {_mean(_f(diag, source=is_sim, region='footprint', case_angle=a), 'spread_ratio'):.2f}"
            for a in angles))
        P("  sigma_sim by angle (sim, footprint, counts): " + "  ".join(
            f"{a:g}deg {math.sqrt(max(_mean(_f(diag, source=is_sim, region='footprint', case_angle=a), 'var_sim'), 0)):.1f}"
            for a in angles))
    sims = sorted({r["case_sim"] for r in diag if is_sim(r["source"])})
    if len(sims) > 1:
        P("  by test simulator (footprint): " + "  ".join(
            f"{s}: rms_err(jitter) {math.sqrt(_mean(_f(diag, source=is_sim, region='footprint', case_sim=s), 'mse_jitter')):.1f}"
            f" rms_err(control) {math.sqrt(_mean(_f(diag, source=is_sim, region='footprint', case_sim=s), 'mse_control')):.1f}"
            for s in sims))
    P("")
    P("[3] accuracy of the candidates (mean r over lines and cases)")
    cands = []
    for r in acc:
        if r["candidate"] not in cands:
            cands.append(r["candidate"])
    for src_name, src in (("simulated", is_sim), ("REAL", is_real)):
        regs = ("footprint", "hole") if src_name == "simulated" else \
            ("footprint",)
        for reg in regs:
            P(f"  {src_name} / {reg}: " + "  ".join(
                f"{c} {_mean(_f(acc, source=src, region=reg, candidate=c), 'r'):.4f}"
                for c in cands))
        if src_name == "REAL":
            P("  REAL / footprint |bias %|: " + "  ".join(
                f"{c} {np.mean([abs(float(x['bias_pct'])) for x in _f(acc, source=src, region='footprint', candidate=c)]):.2f}"
                for c in cands))
            P("  REAL / footprint cv_ratio: " + "  ".join(
                f"{c} {_mean(_f(acc, source=src, region='footprint', candidate=c), 'cv_ratio'):.3f}"
                for c in cands))
    # per-line real anchor: does the jitter ensemble mean keep the r of the
    # deterministic inverse (the MVP single net lost r on 8/8 lines)?
    real_rows = _f(acc, source=is_real, region="footprint")
    if real_rows:
        P("")
        P("[4] REAL anchor per line, r (SSIM) on the footprint vs prova2")
        show = [c for c in ("deterministic", "mvp_single", "control_mean",
                            "jitter_mean") if c in cands]
        P("    " + f"{'line':6s}" + "".join(f"{c:>22s}" for c in show))
        wins = {c: 0 for c in show}
        for el in ELEMENTS:
            vals = {}
            for c in show:
                rr = _f(real_rows, element=el, candidate=c)
                vals[c] = (float(rr[0]["r"]), float(rr[0]["ssim"])) if rr \
                    else (float("nan"), float("nan"))
            det_r = vals.get("deterministic", (float("nan"),))[0]
            for c in show:
                if c != "deterministic" and vals[c][0] >= det_r - 1e-4:
                    wins[c] += 1
            P("    " + f"{el:6s}" + "".join(
                f"{vals[c][0]:12.4f} ({vals[c][1]:.3f})" for c in show))
        P("    lines with r >= deterministic: " + ", ".join(
            f"{c} {wins[c]}/{len(ELEMENTS)}" for c in show
            if c != "deterministic"))
    text = "\n".join(L)
    print(text)
    path = path or os.path.join(core.RESULTS_DIR, "wp1_summary.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return text


# ---------------------------------------------------------------------------
# figures (from CSV + the per-case npz only)
# ---------------------------------------------------------------------------

NAVY = "#1f2a44"
GREY = "#8c8c8c"
ORANGE = "#c8641e"


def _load_maps(tag: str):
    p = os.path.join(member_dir(), f"maps_{tag}.npz")
    return np.load(p, allow_pickle=False) if os.path.exists(p) else None


def make_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cov = io_utils.read_rows("wp1_uq_coverage")
    if not cov:
        raise NotImplementedError("run the experiment first (no coverage csv)")
    is_sim = lambda s: s != "REAL_ruotato"  # noqa: E731
    is_real = lambda s: s == "REAL_ruotato"  # noqa: E731
    zs = list(config.COVERAGE_Z)
    xg = [gauss_cov(z) for z in zs]

    # --- figure 1: calibration curves -------------------------------------
    panels = [("simulated, footprint", is_sim, "footprint"),
              ("simulated, hole", is_sim, "hole"),
              ("real scan, footprint", is_real, "footprint")]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.15), sharey=True)
    style = {("jitter", "ens"): dict(color=NAVY, ls=":", marker="o"),
             ("jitter", "total"): dict(color=NAVY, ls="-", marker="o"),
             ("jitter", "total_ref"): dict(color=NAVY, ls="--", marker="o"),
             ("control", "ens"): dict(color=GREY, ls=":", marker="s"),
             ("control", "total"): dict(color=GREY, ls="-", marker="s"),
             ("control", "total_ref"): dict(color=GREY, ls="--", marker="s")}
    band_label = {"ens": "spread only", "total": "+ noise",
                  "total_ref": "+ noise + ref. noise"}
    for ax, (title, src, reg) in zip(axes, panels):
        ax.plot([0, 1], [0, 1], color="k", lw=0.8, alpha=0.5)
        for (kind, band), st in style.items():
            # on the real panel the reference noise lives in "total"
            # (dashed) and the noise-only band is "total_noref" (solid)
            qband = band
            if src is is_real:
                qband = {"ens": "ens", "total": "total_noref",
                         "total_ref": "total"}[band]
            ys, es = [], []
            for z in zs:
                rr = _f(cov, source=src, region=reg, band=qband,
                        ensemble=kind, z=z)
                v = [float(r["coverage"]) for r in rr]
                ys.append(np.mean(v) if v else np.nan)
                es.append(np.std(v) if v else np.nan)
            if np.all(np.isnan(ys)):
                continue
            lab = f"{kind}, {band_label[band]}"
            ax.errorbar(xg, ys, yerr=es, label=lab, ms=4, lw=1.4,
                        capsize=2, **st)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("nominal Gaussian coverage", fontsize=9)
        ax.set_xlim(0.3, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("empirical coverage", fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    axes[1].legend(h, l, fontsize=8, loc="upper left", frameon=False)
    fig.tight_layout()
    out = io_utils.fig_path("wp1_calibration.png")
    fig.savefig(out, dpi=200)
    fig.savefig(out.replace(".png", ".pdf"))
    plt.close(fig)
    print("saved:", out)

    # --- figure 2: spread maps on the harsh case ---------------------------
    hc = config.WP1_HARSH_CASE
    tag = case_tag(hc["angle"], hc["hole"], hc["sim"], hc["dose"])
    d = _load_maps(tag)
    if d is None:
        # fall back to any available case with a hole
        for f in sorted(os.listdir(member_dir())):
            if f.startswith("maps_") and "_h14x20" in f:
                d = np.load(os.path.join(member_dir(), f))
                tag = f[5:-4]
                break
    if d is not None:
        els = list(d["elements"])
        lines = [el for el in config.FIG_LINES if el in els]
        fp = d["footprint"]
        hole = d["hole"]
        rows, cols = np.where(fp)
        r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
        crop = (slice(r0, r1), slice(c0, c1))
        cols_t = ["truth", "ensemble mean", "|error|", "spread, control",
                  "spread, jitter", "simulator part"]
        fig, axes = plt.subplots(len(lines), len(cols_t),
                                 figsize=(1.55 * len(cols_t), 0.95 * len(lines) + 0.45))
        axes = np.atleast_2d(axes)
        for i, el in enumerate(lines):
            k = els.index(el)
            truth = d["truth"][k]
            mean = d["mean_jitter"][k]
            err = np.abs(mean - truth)
            sc, sj = d["std_control"][k], d["std_jitter"][k]
            ssim_ = np.sqrt(np.clip(sj ** 2 - sc ** 2, 0, None))
            vmax = float(np.nanpercentile(truth[fp], 99))
            smax = float(np.nanpercentile(sj[fp], 99))
            panels_ = [(truth, 0, vmax, "viridis"), (mean, 0, vmax, "viridis"),
                       (err, 0, smax * 2, "magma"), (sc, 0, smax, "magma"),
                       (sj, 0, smax, "magma"), (ssim_, 0, smax, "magma")]
            for j, (img, lo, hi, cmap) in enumerate(panels_):
                ax = axes[i, j]
                im = np.where(fp, img, np.nan)[crop]
                ax.imshow(im, vmin=lo, vmax=hi, cmap=cmap,
                          interpolation="nearest")
                if hole.any():
                    ax.contour(hole[crop].astype(float), levels=[0.5],
                               colors=["w"], linewidths=0.7)
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    ax.set_title(cols_t[j], fontsize=9)
                if j == 0:
                    ax.set_ylabel(el, fontsize=10)
        fig.suptitle(f"{tag.replace('_', ' ')}: white outline = dropout hole;"
                     " spread panels share one colour scale per line",
                     fontsize=9)
        fig.tight_layout()
        out = io_utils.fig_path("wp1_spread_maps.png")
        fig.savefig(out, dpi=200)
        fig.savefig(out.replace(".png", ".pdf"))
        plt.close(fig)
        print("saved:", out)

    # --- figure 3: binned reliability (RMSE vs predicted sigma) ------------
    pooled = {("jitter", "ens"): ([], []), ("jitter", "total"): ([], []),
              ("control", "ens"): ([], []), ("control", "total"): ([], [])}
    n_cases = 0
    for f in sorted(os.listdir(member_dir())):
        if not (f.startswith("maps_") and f.endswith(".npz")) or "REAL" in f:
            continue
        d = np.load(os.path.join(member_dir(), f))
        n_cases += 1
        fp = d["footprint"]
        for k in range(len(d["elements"])):
            truth = d["truth"][k]
            scale = float(np.nanstd(truth[fp])) or 1.0   # per-line units
            for kind in KINDS:
                err = np.abs(d[f"mean_{kind}"][k] - truth)[fp] / scale
                s_ens = d[f"std_{kind}"][k][fp] / scale
                s_tot = np.sqrt(d[f"std_{kind}"][k] ** 2
                                + d[f"alea_{kind}"][k] ** 2)[fp] / scale
                pooled[(kind, "ens")][0].append(s_ens)
                pooled[(kind, "ens")][1].append(err)
                pooled[(kind, "total")][0].append(s_tot)
                pooled[(kind, "total")][1].append(err)
    if n_cases:
        fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.8), sharey=True)
        for ax, band in zip(axes, ("ens", "total")):
            lim = 0.0
            for kind in KINDS:
                s = np.concatenate(pooled[(kind, band)][0])
                e = np.concatenate(pooled[(kind, band)][1])
                ok = np.isfinite(s) & np.isfinite(e)
                s, e = s[ok], e[ok]
                edges = np.quantile(s, np.linspace(0, 1, 13))
                xs, ys = [], []
                for a, b in zip(edges[:-1], edges[1:]):
                    m = (s >= a) & (s <= b)
                    if m.sum() < 30:
                        continue
                    xs.append(np.sqrt(np.mean(s[m] ** 2)))
                    ys.append(np.sqrt(np.mean(e[m] ** 2)))
                ax.plot(xs, ys, marker="o", ms=4, lw=1.4,
                        color=NAVY if kind == "jitter" else GREY, label=kind)
                lim = max(lim, max(xs + ys))
            ax.plot([0, lim], [0, lim], color="k", lw=0.8, alpha=0.5)
            ax.set_title("spread only" if band == "ens" else "spread + noise",
                         fontsize=10)
            ax.set_xlabel("predicted sigma (units of the line's std)",
                          fontsize=9)
            ax.tick_params(labelsize=8)
            ax.grid(alpha=0.25)
        axes[0].set_ylabel("RMS error in the bin (down = better)",
                           fontsize=9)
        axes[0].legend(frameon=False, fontsize=9)
        fig.tight_layout()
        out = io_utils.fig_path("wp1_error_vs_sigma.png")
        fig.savefig(out, dpi=200)
        fig.savefig(out.replace(".png", ".pdf"))
        plt.close(fig)
        print("saved:", out)


# ---------------------------------------------------------------------------

def _parse_members(spec: str | None):
    if not spec:
        return None
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--train-only", action="store_true",
                    help="train the selected members and exit")
    ap.add_argument("--kind", choices=("jitter", "control", "both"),
                    default="both")
    ap.add_argument("--members", default=None, help="e.g. 0-5 or 0,2,4")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    set_quick(args.quick)
    if args.figures:
        make_figures()
    elif args.summary:
        summarize()
    elif args.train_only:
        n = config.ENSEMBLE_N_QUICK if args.quick else config.ENSEMBLE_N
        tcfg = config.QUICK_TRAIN if args.quick else config.TRAIN
        kinds = KINDS if args.kind == "both" else (args.kind,)
        for kind in kinds:
            train_ensemble(kind, n, tcfg, members=_parse_members(args.members),
                           verbose=args.verbose)
        write_members_csv()
    else:
        run(quick=args.quick, verbose=args.verbose)
