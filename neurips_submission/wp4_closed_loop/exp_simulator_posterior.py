"""WP4 - closing the loop: audit -> posterior over the simulator from ONE
real scan -> posterior ensemble.
OWNER: Dimitrije (agreed 2026-08-28)

CLAIM UNDER TEST
    The WP2 diagnostic battery is a set of summary statistics.  Used in
    rejection ABC against the single real tilted scan, it yields a
    posterior over the simulator's knobs (the same SimKnobs the WP1
    prior jitters) WITHOUT ground truth.  An ensemble trained on
    posterior draws should then be NARROWER than the prior (jitter)
    ensemble on the real scan while staying at least as well covered
    and without losing r: the model audits its own simulator, updates
    it from one measurement, and shrinks its uncertainty honestly.
    The claim can be lost; whatever comes out is reported.

PIECES
    1. ABC.  Prior = config.JITTER draws (perturb.jittered) plus
       blur_mode in {cubic, bilinear} with p = 0.5 (the v1 mistake is in
       the prior so the loop has to reject it by itself).  For each draw
       theta: S = forward_perturbed(prova1, 7.7 deg, theta); summary =
       exp_diagnostics.battery(ruotato, S) (9 statistics x 8 lines);
       whitening by the WP2 real-scan null (jittered sim of prova2 vs
       nominal sim of prova1, contains the session difference);
       distance = RMS z; accept the closest ACCEPT fraction (sensitivity
       at 2 / 5 / 10 %).
    2. Posterior predictive check: the WP2 rule's verdict on 12 posterior
       draws vs 12 prior draws vs nominal ('ok' = indistinguishable from
       the real scan within calibration uncertainty).
    3. Posterior ensemble: 12 nets on 12 accepted draws (sampled without
       replacement, seeded), WP1 jitter seeds, config.TRAIN, cached.
    4. Evaluation of control (WP1) / prior (WP1 jitter) / posterior on
       the REAL scan and on 8 validated dose-1 simulated cases: spread,
       coverage (WP1 bands), r / cv / bias, Spearman(sigma, |err|).

OUTPUTS
    results/wp4_abc_draws.csv, wp4_abc_marginals.csv, wp4_ppc.csv
    results/wp4_posterior/members.json, post_XX.pt (gitignored)
    results/wp4_posterior_coverage.csv, wp4_posterior_accuracy.csv,
    wp4_posterior_spread.csv, wp4_summary.txt
    figures/wp4_prior_posterior.png

Run from the repo root:
    python neurips_submission/wp4_closed_loop/exp_simulator_posterior.py --abc
    python .../exp_simulator_posterior.py --train-only --members 0-2 --threads 3   (x4 shells)
    python .../exp_simulator_posterior.py --eval
    python .../exp_simulator_posterior.py --figures
"""

import argparse
import dataclasses
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
from wp1_uq_ensemble import exp_ensemble_uq as uq
from wp2_simulator_audit import exp_diagnostics as dg

ELEMENTS = core.ELEMENTS
ANGLE = core.fm.REF_ANGLE_DEG
N_DRAWS = 3000
N_DRAWS_QUICK = 200
ACCEPT = 0.05
ACCEPT_SENS = (0.02, 0.05, 0.10)
N_MEMBERS = config.ENSEMBLE_N
KNOB_COLS = ("noise_k_scale", "gain_scale", "angle_bias_deg", "warp_rot_deg",
             "warp_dy", "warp_dx", "blur_bilinear", "flatfield_strength")
_QUICK = False


def set_quick(flag):
    global _QUICK
    _QUICK = bool(flag)
    uq.set_quick(flag)


def post_dir():
    d = os.path.join(core.RESULTS_DIR,
                     "wp4_posterior" + ("_quick" if _QUICK else ""))
    os.makedirs(d, exist_ok=True)
    return d


def _tag(name):
    return name + ("_quick" if _QUICK else "")


# ---------------------------------------------------------------------------
# knobs <-> flat row
# ---------------------------------------------------------------------------

def knobs_to_row(k: perturb.SimKnobs) -> dict:
    r = {"noise_k_scale": k.noise_k_scale, "gain_scale": k.gain_scale,
         "angle_bias_deg": k.angle_bias_deg, "warp_rot_deg": k.warp_rot_deg,
         "warp_dy": k.warp_shift_px[0], "warp_dx": k.warp_shift_px[1],
         "blur_bilinear": 1 if k.blur_mode == "bilinear" else 0,
         "flatfield_strength": k.flatfield_strength}
    for el, v in zip(ELEMENTS, k.gain_pct_offset):
        r[f"gain_off_{el}"] = v
    for el, v in zip(ELEMENTS, k.noise_k_line_scale):
        r[f"nkline_{el}"] = v
        r[f"nkeff_{el}"] = v * k.noise_k_scale
    return r


def row_to_knobs(r: dict, label="posterior") -> perturb.SimKnobs:
    return perturb.SimKnobs(
        noise_k_scale=float(r["noise_k_scale"]),
        gain_scale=float(r["gain_scale"]),
        angle_bias_deg=float(r["angle_bias_deg"]),
        blur_mode="bilinear" if int(float(r["blur_bilinear"])) else "cubic",
        warp_shift_px=(float(r["warp_dy"]), float(r["warp_dx"])),
        warp_rot_deg=float(r["warp_rot_deg"]),
        gain_pct_offset=tuple(float(r[f"gain_off_{el}"]) for el in ELEMENTS),
        noise_k_line_scale=tuple(float(r.get(f"nkline_{el}", 1.0))
                                 for el in ELEMENTS),
        flatfield_strength=float(r.get("flatfield_strength", 0.0)),
        label=label)


def prior_draw(rng, i, spec=None) -> perturb.SimKnobs:
    k = perturb.jittered(rng, spec or config.JITTER, f"abc_{i}")
    blur = "bilinear" if rng.random() < 0.5 else "cubic"
    return dataclasses.replace(k, blur_mode=blur)


# round-2 prior: round 1's PPC left a uniformly "noise" verdict whose
# per-line pattern the global k knob cannot express; round 2 adds the
# per-line noise multipliers with log-sd 0.6 (wide enough to cover the
# measured per-line variance ratios 0.9 to 4.6, MVP-2 check [3]).
ROUND2_SPEC = dict(config.JITTER, noise_k_line_log_sd=0.6)

# round-3 prior: round 2 still rejects every draw; the one component
# forward_model.py's own docstring names as unmodelled and not yet
# tried is the per-pixel flat-field ("What the simulator does NOT
# model: the flat-field of the detector ratio"). 5% radial vignetting
# sd is a ROUND-NUMBER PLACEHOLDER, not a measured quantity like the
# round-1 sigmas: no independent flat-field calibration exists for
# this instrument, so this is a plausible-magnitude guess only.
ROUND3_SPEC = dict(ROUND2_SPEC, flatfield_strength_sd=0.05)


# ---------------------------------------------------------------------------
# 1. ABC
# ---------------------------------------------------------------------------

def real_null(n_null):
    p1 = core.fm.load_summed_maps("prova1")
    p2 = core.fm.load_summed_maps("prova2")
    return dg.Null([dg.null_pair(p2, p1, ANGLE, 9000 + 2 * j)
                    for j in range(n_null)], ANGLE)


def run_abc(quick=False, spec=None, suffix="", update_members=True,
           ppc_label=None):
    n = N_DRAWS_QUICK if quick else N_DRAWS
    n_null = 8 if quick else dg.N_NULL
    p1 = core.fm.load_summed_maps("prova1")
    ruo = core.fm.load_summed_maps("ruotato")
    print(f"whitening null ({n_null} pairs) ...")
    null = real_null(n_null)
    t0 = time.time()
    rows = []
    rng = np.random.default_rng(config.BASE_SEED + 4242)

    def distance(knobs, seed):
        S = perturb.forward_perturbed(p1, ANGLE, np.random.default_rng(seed),
                                      knobs)
        b = dg.battery(ruo, S, ANGLE)
        z = null.z(b)
        zz = np.array([[z[s][el] for el in ELEMENTS] for s in dg.STATS])
        return float(np.sqrt(np.mean(zz ** 2))), \
            {f"A_{s}": float(np.mean(np.abs(zz[i]))) for i, s in enumerate(dg.STATS)}

    for i in range(n):
        knobs = prior_draw(rng, i, spec)
        d, A = distance(knobs, 100_000 + i)
        rows.append({"draw": i, **knobs_to_row(knobs), "d": d, **A})
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{n} draws [{time.time() - t0:.0f} s]", flush=True)
    # reference points
    for tag, kn in (("nominal_cubic", perturb.NOMINAL),
                    ("nominal_bilinear", dataclasses.replace(
                        perturb.NOMINAL, blur_mode="bilinear"))):
        d, A = distance(kn, 99)
        rows.append({"draw": -1, "ref": tag, **knobs_to_row(kn), "d": d, **A})
    draws = [r for r in rows if r["draw"] >= 0]
    order = np.argsort([r["d"] for r in draws])
    for frac in ACCEPT_SENS:
        k = max(int(round(frac * n)), 8)
        acc = set(int(order[j]) for j in range(k))
        for r in draws:
            r[f"acc_{int(frac * 100)}"] = int(r["draw"] in acc)
    io_utils.write_rows(_tag("wp4_abc_draws" + suffix), rows)

    # marginals
    marg = []
    for key in KNOB_COLS + tuple(f"gain_off_{el}" for el in ELEMENTS) \
            + tuple(f"nkline_{el}" for el in ELEMENTS) \
            + tuple(f"nkeff_{el}" for el in ELEMENTS):
        pri = np.array([float(r[key]) for r in draws])
        m = {"knob": key, "prior_mean": pri.mean(), "prior_sd": pri.std()}
        for frac in ACCEPT_SENS:
            a = np.array([float(r[key]) for r in draws
                          if r[f"acc_{int(frac * 100)}"]])
            m[f"post{int(frac * 100)}_mean"] = a.mean()
            m[f"post{int(frac * 100)}_sd"] = a.std()
            m[f"post{int(frac * 100)}_q05"] = float(np.quantile(a, 0.05))
            m[f"post{int(frac * 100)}_q95"] = float(np.quantile(a, 0.95))
        marg.append(m)
    io_utils.write_rows(_tag("wp4_abc_marginals" + suffix), marg)
    d_all = np.array([r["d"] for r in draws])
    print(f"ABC done: {n} draws in {time.time() - t0:.0f} s; d min {d_all.min():.2f}"
          f" median {np.median(d_all):.2f}; nominal cubic d ="
          f" {[r['d'] for r in rows if r.get('ref') == 'nominal_cubic'][0]:.2f},"
          f" nominal bilinear d ="
          f" {[r['d'] for r in rows if r.get('ref') == 'nominal_bilinear'][0]:.2f}")
    for m in marg:
        print(f"  {m['knob']:16s} prior {m['prior_mean']:+.3f} +- {m['prior_sd']:.3f}"
              f"   post5% {m['post5_mean']:+.3f} +- {m['post5_sd']:.3f}"
              f"   post2% {m['post2_mean']:+.3f} +- {m['post2_sd']:.3f}"
              f"   post10% {m['post10_mean']:+.3f} +- {m['post10_sd']:.3f}")
    if update_members:
        posterior_knobs(force=True)
        ppc(null)
    else:
        ppc_extra(rows, null, spec, suffix, ppc_label or suffix.lstrip("_"))


def ppc_extra(rows, null, spec, suffix, label):
    """PPC for a re-run ABC that must NOT touch the trained ensemble's
    members.json: posterior draws sampled from the fresh accepted set.
    `label` names this round (e.g. "r2", "r3") in the set names and the
    output CSV suffix."""
    p1 = core.fm.load_summed_maps("prova1")
    ruo = core.fm.load_summed_maps("ruotato")
    acc = [r for r in rows if r.get("draw", -1) >= 0
           and r[f"acc_{int(ACCEPT * 100)}"]]
    rng = np.random.default_rng(config.BASE_SEED + 78)
    idx = rng.choice(len(acc), min(12, len(acc)), replace=False)
    sets = {f"posterior_{label}": [row_to_knobs(acc[i], f"post_{label}_{i}")
                                   for i in idx]}
    rng2 = np.random.default_rng(config.BASE_SEED + 4242)
    sets[f"prior_{label}"] = [prior_draw(rng2, i, spec) for i in range(12)]
    sets["nominal"] = [perturb.NOMINAL]
    out = []
    for kind, ks in sets.items():
        for i, kn in enumerate(ks):
            S = perturb.forward_perturbed(p1, ANGLE,
                                          np.random.default_rng(500 + i), kn)
            b = dg.battery(ruo, S, ANGLE)
            d, flags, A, zs = dg.identify(b, null, posthoc=True)
            out.append({"set": kind, "member": i, "verdict": d,
                        **{f"A_{s}": A[s] for s in dg.STATS}})
    io_utils.write_rows(_tag("wp4_ppc" + suffix), out)
    for kind in sets:
        vs = [r["verdict"] for r in out if r["set"] == kind]
        print(f"  PPC {kind:13s}: ok {sum(v == 'ok' for v in vs)}/{len(vs)}"
              f"  verdicts: {', '.join(vs)}")


def posterior_knobs(force=False) -> list:
    """The 12 posterior member knob sets (sampled once, saved to json)."""
    path = os.path.join(post_dir(), "members.json")
    if os.path.exists(path) and not force:
        with open(path, encoding="utf-8") as fh:
            return [row_to_knobs(r, f"post_{i}") for i, r in enumerate(json.load(fh))]
    draws = [r for r in io_utils.read_rows(_tag("wp4_abc_draws"))
             if r.get("draw", "-1") not in ("-1", "") and int(r["draw"]) >= 0]
    acc = [r for r in draws if int(r[f"acc_{int(ACCEPT * 100)}"])]
    rng = np.random.default_rng(config.BASE_SEED + 77)
    n = config.ENSEMBLE_N_QUICK if _QUICK else N_MEMBERS
    idx = rng.choice(len(acc), n, replace=False)
    chosen = [acc[i] for i in idx]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([{k: float(c[k]) for k in KNOB_COLS
                    + tuple(f"gain_off_{el}" for el in ELEMENTS)}
                   for c in chosen], fh, indent=1)
    return [row_to_knobs(r, f"post_{i}") for i, r in enumerate(chosen)]


def ppc(null=None):
    """Posterior predictive check through the WP2 rule."""
    if null is None:
        null = real_null(8 if _QUICK else dg.N_NULL)
    p1 = core.fm.load_summed_maps("prova1")
    ruo = core.fm.load_summed_maps("ruotato")
    rows = []
    sets = {"posterior": posterior_knobs()}
    rng = np.random.default_rng(config.BASE_SEED + 4242)
    sets["prior"] = [prior_draw(rng, i) for i in range(len(sets["posterior"]))]
    sets["nominal"] = [perturb.NOMINAL]
    for kind, ks in sets.items():
        for i, kn in enumerate(ks):
            S = perturb.forward_perturbed(p1, ANGLE, np.random.default_rng(500 + i), kn)
            b = dg.battery(ruo, S, ANGLE)
            d, flags, A, zs = dg.identify(b, null, posthoc=True)
            rows.append({"set": kind, "member": i, "verdict": d,
                         **{f"A_{s}": A[s] for s in dg.STATS}})
    io_utils.write_rows(_tag("wp4_ppc"), rows)
    for kind in sets:
        vs = [r["verdict"] for r in rows if r["set"] == kind]
        print(f"  PPC {kind:10s}: ok {sum(v == 'ok' for v in vs)}/{len(vs)}"
              f"  verdicts: {', '.join(vs)}")


# ---------------------------------------------------------------------------
# 3. posterior ensemble
# ---------------------------------------------------------------------------

def train_member(i, tcfg, verbose=False):
    ckpt = os.path.join(post_dir(), f"post_{i:02d}.pt")
    meta = ckpt[:-3] + ".json"
    net = core.RestorationUNet()
    if os.path.exists(ckpt):
        net.load_state_dict(torch.load(ckpt, weights_only=True))
        net.eval()
        print(f"[posterior {i:02d}] cached")
        return net
    knobs = posterior_knobs()[i]
    seed = config.BASE_SEED + 100 * i          # == WP1 jitter member i
    print(f"[posterior {i:02d}] training  knobs={knobs.to_meta()}", flush=True)
    net, hist = training.train_net(training.make_batch_fn(knobs=knobs), tcfg,
                                   seed=seed, verbose=verbose)
    torch.save(net.state_dict(), ckpt)
    with open(meta, "w", encoding="utf-8") as fh:
        json.dump({"ensemble": "posterior", "member": i, **knobs.to_meta(),
                   **hist}, fh, indent=1)
    print(f"[posterior {i:02d}] done  best val L1 {hist['best_val_l1']:.5f}"
          f" at step {hist['best_step']}  ({hist['wall_s']:.0f} s)", flush=True)
    return net


def load_wp1(kind, n):
    out = []
    for i in range(n):
        ckpt, _ = uq._member_paths(kind, i)
        net = core.RestorationUNet()
        net.load_state_dict(torch.load(ckpt, weights_only=True))
        net.eval()
        out.append(net)
    return out


# ---------------------------------------------------------------------------
# 4. evaluation
# ---------------------------------------------------------------------------

def evaluate(quick=False):
    n = config.ENSEMBLE_N_QUICK if quick else N_MEMBERS
    tcfg = config.QUICK_TRAIN if quick else config.TRAIN
    reps = 3 if quick else config.WP1_NOISE_REPS
    ens = {"control": load_wp1("control", n), "prior": load_wp1("jitter", n),
           "posterior": [train_member(i, tcfg) for i in range(n)]}
    mvp = restore.load_mvp_net()
    ks = core.fm.calibrate_noise()

    cases = []
    ruo = core.fm.load_summed_maps("ruotato")
    truth2 = core.fm.load_summed_maps("prova2")
    twin = restore.degrade(source="prova2", angle=ANGLE, block=None, dose=1.0,
                           seed=0, sim="validated")
    cases.append(("REAL_ruotato", {"tilted": ruo, "angle": ANGLE,
                                   "validity": None}, truth2,
                  {"footprint": core.dg.footprint(), "hole": None}, twin, True))
    angles = (20.0,) if quick else config.WP1_CASES["angles"]
    for angle in angles:
        for (h, w) in config.WP1_CASES["holes"]:
            c = restore.degrade(source="prova2", angle=angle,
                                block=restore.centered_block(h, w), dose=1.0,
                                seed=0, sim="validated")
            cases.append((uq.case_tag(angle, (h, w), "validated", 1.0), c,
                          c["truth"], {"footprint": c["fp"],
                                       "hole": c["hole"] if c["block"] else None},
                          None, False))
    cov, acc, spr = [], [], []
    t0 = time.time()
    for tag, case, truth, regions, twin_, real in cases:
        meta = {"case": tag, "case_angle": case["angle"],
                "source": "REAL_ruotato" if real else "prova2"}
        cands = {}
        for kind, members in ens.items():
            mean, std, det, stack = uq.ensemble_predict(
                members, case["tilted"], case["angle"],
                validity=case.get("validity"))
            alea, _ = uq.aleatoric_sigma(members, twin_ if twin_ else case, reps)
            base = {"ensemble": kind, **meta}
            cov += uq.coverage_rows(mean, std, truth, regions, base, band="ens")
            extra = {el: alea[el] ** 2 for el in ELEMENTS}
            cov += uq.coverage_rows(mean, std, truth, regions, base,
                                    extra_var=extra, band="total_noref")
            if real:
                extra2 = {el: alea[el] ** 2 + ks[el] * np.clip(truth[el], 0, None)
                          for el in ELEMENTS}
                cov += uq.coverage_rows(mean, std, truth, regions, base,
                                        extra_var=extra2, band="total")
            cands["deterministic"] = det
            cands[f"{kind}_mean"] = mean
            for el in ELEMENTS:
                for reg, mask in regions.items():
                    if mask is None:
                        continue
                    m = mask & np.isfinite(mean[el]) & np.isfinite(truth[el])
                    if m.sum() < 9:
                        continue
                    e = np.abs(mean[el] - truth[el])[m]
                    s = std[el][m]
                    st = np.sqrt(s ** 2 + alea[el][m] ** 2)
                    spr.append({**base, "element": el, "region": reg,
                                "n_px": int(m.sum()),
                                "spread_rms": float(np.sqrt(np.mean(s ** 2))),
                                "alea_rms": float(np.sqrt(np.mean(alea[el][m] ** 2))),
                                "err_rms": float(np.sqrt(np.mean(e ** 2))),
                                "spearman_ens": float(spearmanr(s, e).statistic)
                                if np.std(s) > 0 else float("nan"),
                                "spearman_total": float(spearmanr(st, e).statistic)
                                if np.std(st) > 0 else float("nan")})
        if mvp is not None:
            _, cands["mvp_single"] = restore.apply_network(
                mvp, case["tilted"], case["angle"], validity=case.get("validity"))
        acc += uq.accuracy_rows(cands, truth, regions, meta)
        print(f"  {tag:32s} [{time.time() - t0:.0f} s]", flush=True)
    io_utils.write_rows(_tag("wp4_posterior_coverage"), cov)
    io_utils.write_rows(_tag("wp4_posterior_accuracy"), acc)
    io_utils.write_rows(_tag("wp4_posterior_spread"), spr)
    summarize()


# ---------------------------------------------------------------------------
# summary + figure
# ---------------------------------------------------------------------------

def _f(rows, **cond):
    out = []
    for r in rows:
        ok = True
        for k, v in cond.items():
            rv = r.get(k, "")
            ok &= bool(v(rv)) if callable(v) else (str(rv) == str(v))
        if ok:
            out.append(r)
    return out


def _mean(rows, key):
    v = [float(r[key]) for r in rows if r.get(key, "") not in ("", "nan")]
    v = [x for x in v if np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def summarize():
    marg = io_utils.read_rows(_tag("wp4_abc_marginals"))
    draws = io_utils.read_rows(_tag("wp4_abc_draws"))
    ppc_rows = io_utils.read_rows(_tag("wp4_ppc"))
    cov = io_utils.read_rows(_tag("wp4_posterior_coverage"))
    acc = io_utils.read_rows(_tag("wp4_posterior_accuracy"))
    spr = io_utils.read_rows(_tag("wp4_posterior_spread"))
    L = []
    P = L.append
    P("WP4 - closing the loop: ABC posterior over the simulator from one real scan")
    if marg:
        n = len([r for r in draws if r.get("draw", "-1") not in ("-1", "")
                 and int(r["draw"]) >= 0])
        refs = {r.get("ref"): float(r["d"]) for r in draws if r.get("ref")}
        P(f"[1] ABC: {n} prior draws; whitened RMS-z distance to the real scan:"
          f" nominal cubic {refs.get('nominal_cubic', float('nan')):.2f},"
          f" nominal bilinear {refs.get('nominal_bilinear', float('nan')):.2f}")
        P(f"    {'knob':16s}{'prior':>18s}{'post 2%':>18s}{'post 5%':>18s}{'post 10%':>18s}")
        for m in marg:
            P(f"    {m['knob']:16s}"
              f"{float(m['prior_mean']):+9.3f} +-{float(m['prior_sd']):6.3f}"
              f"{float(m['post2_mean']):+9.3f} +-{float(m['post2_sd']):6.3f}"
              f"{float(m['post5_mean']):+9.3f} +-{float(m['post5_sd']):6.3f}"
              f"{float(m['post10_mean']):+9.3f} +-{float(m['post10_sd']):6.3f}")
        P("    (blur_bilinear mean = posterior probability of the v1 bilinear "
          "sampling; prior 0.5)")
    if ppc_rows:
        P("")
        P("[2] posterior predictive check (WP2 rule, post-hoc, verdict 'ok' ="
          " indistinguishable from the real scan within calibration uncertainty)")
        for kind in ("nominal", "prior", "posterior"):
            vs = [r["verdict"] for r in ppc_rows if r["set"] == kind]
            if vs:
                P(f"    {kind:10s} ok {sum(v == 'ok' for v in vs)}/{len(vs)}:"
                  f" {', '.join(vs)}")
    if cov:
        is_real = lambda s: s == "REAL_ruotato"     # noqa: E731
        is_sim = lambda s: s != "REAL_ruotato"      # noqa: E731
        P("")
        P("[3] three ensembles on the REAL scan (footprint, mean over lines)")
        P(f"    {'ensemble':10s}{'r':>8s}{'cv':>7s}{'|bias|':>8s}{'spread':>8s}"
          f"{'alea':>7s}{'err':>7s}{'cov z=1':>9s}{'cov z=2':>9s}{'cov z=3':>9s}"
          f"{'Spear':>7s}")
        for kind in ("control", "prior", "posterior"):
            a = _f(acc, source=is_real, region="footprint", candidate=f"{kind}_mean")
            s = _f(spr, source=is_real, region="footprint", ensemble=kind)
            c = [_mean(_f(cov, source=is_real, region="footprint", band="total",
                          ensemble=kind, z=z), "coverage") for z in (1.0, 2.0, 3.0)]
            P(f"    {kind:10s}{_mean(a, 'r'):8.4f}{_mean(a, 'cv_ratio'):7.3f}"
              f"{np.mean([abs(float(x['bias_pct'])) for x in a]) if a else float('nan'):8.2f}"
              f"{_mean(s, 'spread_rms'):8.1f}{_mean(s, 'alea_rms'):7.1f}"
              f"{_mean(s, 'err_rms'):7.1f}{c[0]:9.3f}{c[1]:9.3f}{c[2]:9.3f}"
              f"{_mean(s, 'spearman_total'):7.3f}")
        d = _f(acc, source=is_real, region="footprint", candidate="deterministic")
        m = _f(acc, source=is_real, region="footprint", candidate="mvp_single")
        P(f"    {'det':10s}{_mean(d, 'r'):8.4f}{_mean(d, 'cv_ratio'):7.3f}"
          f"    {'mvp_single':10s} r {_mean(m, 'r'):.4f}")
        P("    per line r (det / control / prior / posterior):")
        for el in ELEMENTS:
            vals = [_mean(_f(acc, source=is_real, region="footprint",
                             candidate=c_, element=el), "r")
                    for c_ in ("deterministic", "control_mean", "prior_mean",
                               "posterior_mean")]
            P(f"      {el:5s} " + "  ".join(f"{v:.4f}" for v in vals))
        P("")
        P("[4] simulated held-out cases (validated emulator, dose 1; mean over"
          " lines and cases)")
        for reg in ("footprint", "hole"):
            P(f"    {reg}")
            for kind in ("control", "prior", "posterior"):
                a = _f(acc, source=is_sim, region=reg, candidate=f"{kind}_mean")
                s = _f(spr, source=is_sim, region=reg, ensemble=kind)
                c = [_mean(_f(cov, source=is_sim, region=reg, band="total_noref",
                              ensemble=kind, z=z), "coverage") for z in (1.0, 2.0)]
                P(f"      {kind:10s} r {_mean(a, 'r'):.4f}  spread {_mean(s, 'spread_rms'):7.1f}"
                  f"  err {_mean(s, 'err_rms'):7.1f}  cov z=1/2 {c[0]:.3f}/{c[1]:.3f}"
                  f"  Spearman {_mean(s, 'spearman_total'):.3f}")
    text = "\n".join(L)
    print(text)
    with open(os.path.join(core.RESULTS_DIR, _tag("wp4_summary") + ".txt"), "w",
              encoding="utf-8") as fh:
        fh.write(text + "\n")
    return text


def make_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    NAVY, GREY, ORANGE, LIGHT = "#1f2a44", "#8c8c8c", "#c8641e", "#b9c0d4"
    draws = [r for r in io_utils.read_rows(_tag("wp4_abc_draws"))
             if r.get("draw", "-1") not in ("-1", "") and int(r["draw"]) >= 0]
    if not draws:
        raise NotImplementedError("run --abc first")
    cov = io_utils.read_rows(_tag("wp4_posterior_coverage"))
    acc = io_utils.read_rows(_tag("wp4_posterior_accuracy"))
    spr = io_utils.read_rows(_tag("wp4_posterior_spread"))
    # gain_scale and angle_bias_deg marginals are omitted (posterior is
    # indistinguishable from the prior there; numbers in the summary)
    knobs = [("noise_k_scale", "noise constant (x)"),
             ("warp_rot_deg", "reg. rotation (deg)"),
             ("warp_dy", "reg. shift dy (px)"),
             ("blur_bilinear", "P(bilinear sampling)")]
    accepted = [r for r in draws if int(r[f"acc_{int(ACCEPT * 100)}"])]
    have_eval = bool(cov)
    fig, axes = plt.subplots(2 if have_eval else 1, 4,
                             figsize=(9.4, 3.15 if have_eval else 2.0))
    axes = np.atleast_2d(axes)
    for j, (key, title) in enumerate(knobs):
        ax = axes[0, j]
        pri = np.array([float(r[key]) for r in draws])
        pos = np.array([float(r[key]) for r in accepted])
        if key == "blur_bilinear":
            bar_h = [pri.mean(), pos.mean()]
            ax.bar([0, 1], bar_h, color=[GREY, NAVY], width=0.6)
            for xb, hb in zip([0, 1], bar_h):
                ax.text(xb, max(hb, 0.03) + 0.03, f"{hb:.3f}", ha="center",
                        fontsize=7.5)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["prior", "posterior"])
            ax.set_ylim(0, 1.05)
        else:
            bins = np.linspace(min(pri.min(), pos.min()), max(pri.max(), pos.max()), 26)
            ax.hist(pri, bins=bins, density=True, color=GREY, alpha=0.55)
            ax.hist(pos, bins=bins, density=True, histtype="stepfilled",
                    color=NAVY, alpha=0.6, edgecolor=NAVY, linewidth=1.6)
            ax.set_yticks([])
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=8)
    prior_patch = plt.Rectangle((0, 0), 1, 1, color=GREY, alpha=0.55,
                                label="prior")
    post_patch = plt.Rectangle((0, 0), 1, 1, color=NAVY, alpha=0.6,
                               label="posterior")
    fig.legend(handles=[prior_patch, post_patch], loc="upper left",
              bbox_to_anchor=(0.005, 1.06 if have_eval else 1.16), ncol=2,
              fontsize=9, frameon=False, handlelength=1.4)
    if have_eval:
        is_real = lambda s: s == "REAL_ruotato"     # noqa: E731
        kinds = ("control", "prior", "posterior")
        cols = {"control": GREY, "prior": "#5b76b3", "posterior": NAVY}
        mks = {"control": "s", "prior": "^", "posterior": "D"}
        # (a) calibration on the real scan
        ax = axes[1, 0]
        zs = list(config.COVERAGE_Z)
        xg = [uq.gauss_cov(z) for z in zs]
        ax.plot([0, 1], [0, 1], color="k", lw=0.8, alpha=0.5)
        for kind in kinds:
            ys = [_mean(_f(cov, source=is_real, region="footprint", band="total",
                           ensemble=kind, z=z), "coverage") for z in zs]
            ax.plot(xg, ys, marker=mks[kind], ms=5, lw=1.5, color=cols[kind])
        ax.set_xlabel("nominal coverage")
        ax.set_ylabel("empirical, real scan")
        ax.grid(alpha=0.25)
        # (b) spread per line
        ax = axes[1, 1]
        x = np.arange(len(ELEMENTS))
        for k_, kind in enumerate(kinds):
            vals = [_mean(_f(spr, source=is_real, region="footprint",
                             ensemble=kind, element=el), "spread_rms")
                    for el in ELEMENTS]
            ax.bar(x + (k_ - 1) * 0.27, vals, width=0.27, color=cols[kind])
        ax.set_xticks(x)
        ax.set_xticklabels(ELEMENTS, fontsize=7, rotation=45)
        ax.set_title("spread (counts)", fontsize=9)
        # (c) r per line
        ax = axes[1, 2]
        for cand, col, mk, lw in (
                ("deterministic", ORANGE, "o", 2.4),
                ("control_mean", GREY, mks["control"], 1.1),
                ("prior_mean", cols["prior"], mks["prior"], 1.1),
                ("posterior_mean", NAVY, mks["posterior"], 1.1)):
            vals = [_mean(_f(acc, source=is_real, region="footprint",
                             candidate=cand, element=el), "r") for el in ELEMENTS]
            ax.plot(x, vals, marker=mk, ms=5, lw=lw, color=col,
                    zorder=5 if cand == "deterministic" else 3)
        ax.set_xticks(x)
        ax.set_xticklabels(ELEMENTS, fontsize=7, rotation=45)
        ax.set_title("r vs F2 (up = better)", fontsize=9)
        ax.grid(alpha=0.25)
        # (d) spread vs coverage summary
        ax = axes[1, 3]
        for kind in kinds:
            s = _mean(_f(spr, source=is_real, region="footprint", ensemble=kind), "spread_rms")
            c = _mean(_f(cov, source=is_real, region="footprint", band="total",
                         ensemble=kind, z=2.0), "coverage")
            ax.scatter([s], [c], s=60, color=cols[kind], marker=mks[kind])
        ax.axhline(uq.gauss_cov(2.0), color="k", lw=0.8, alpha=0.5)
        ax.set_xlabel("spread rms (counts)")
        ax.set_ylabel("coverage at z = 2")
        ax.grid(alpha=0.25)

        row2_handles = [
            plt.Line2D([0], [0], color=ORANGE, marker="o", lw=2.4,
                      label="physics"),
            plt.Line2D([0], [0], color=GREY, marker=mks["control"], lw=1.5,
                      label="control"),
            plt.Line2D([0], [0], color=cols["prior"], marker=mks["prior"],
                      lw=1.5, label="prior ensemble"),
            plt.Line2D([0], [0], color=NAVY, marker=mks["posterior"], lw=1.5,
                      label="posterior ensemble"),
        ]
        fig.legend(handles=row2_handles, loc="lower center",
                  bbox_to_anchor=(0.5, -0.04), ncol=4, fontsize=8.5,
                  frameon=False, handlelength=1.6, columnspacing=1.4)
    fig.subplots_adjust(left=0.06, right=0.99,
                        top=0.86 if have_eval else 0.78,
                        bottom=0.2 if have_eval else 0.14,
                        hspace=0.5, wspace=0.35)
    out = io_utils.fig_path("wp4_prior_posterior.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved:", out)


def _parse(spec):
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


def run(quick=False):
    set_quick(quick)
    if not io_utils.read_rows(_tag("wp4_abc_draws")):
        run_abc(quick)
    evaluate(quick)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--abc", action="store_true")
    ap.add_argument("--abc2", action="store_true",
                    help="round 2: prior extended with per-line noise "
                         "multipliers; writes *_r2 CSVs, does not touch "
                         "the trained ensemble")
    ap.add_argument("--abc3", action="store_true",
                    help="round 3: prior extended with a flat-field knob "
                         "on top of round 2; writes *_r3 CSVs, does not "
                         "touch the trained ensemble")
    ap.add_argument("--ppc", action="store_true")
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--members", default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    set_quick(args.quick)
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.abc:
        run_abc(args.quick)
    elif args.abc2:
        run_abc(args.quick, spec=ROUND2_SPEC, suffix="_r2",
                update_members=False)
    elif args.abc3:
        run_abc(args.quick, spec=ROUND3_SPEC, suffix="_r3",
                update_members=False)
    elif args.ppc:
        ppc()
    elif args.train_only:
        tcfg = config.QUICK_TRAIN if args.quick else config.TRAIN
        n = config.ENSEMBLE_N_QUICK if args.quick else N_MEMBERS
        for i in (_parse(args.members) or range(n)):
            train_member(i, tcfg, args.verbose)
    elif args.eval:
        evaluate(args.quick)
    elif args.figures:
        make_figures()
    elif args.summary:
        summarize()
    else:
        run(args.quick)
