"""WP2 / experiment 1 - defect-tolerance curves.
OWNER: Dimitrije (took over the whole WP on 2026-08-28)

CLAIM UNDER TEST
    "How imperfect may a training simulator be before the learned
    restoration stops helping - or starts hurting?"  We train one net
    per DELIBERATELY BROKEN simulator (config.DEFECT_LADDERS: noise
    constant, tilt gains, angle belief, resampling blur, registration
    shift and rotation) and measure the restoration quality it delivers
    on (a) held-out simulated test cases from the NOMINAL validated
    emulator and (b) the REAL ruotato scan.  The blur rung reproduces
    our organic v1 mistake as a controlled point on the curve.

DESIGN DECISIONS
    - Every rung trains ONE net with the SAME seed (that of WP1's
      control_00), so rung-to-rung differences are the simulator, not
      the initialisation.  The initialisation/training-noise band is
      taken from WP1's 12 nominal control members (family
      "nominal_seeds"), which are evaluated on the same testbed; a rung
      "hurts" only if it leaves that band.
    - Nominal rung = WP1 control_00 itself (same seed, same simulator),
      loaded from results/wp1_ensemble if present.
    - Rungs are cached (results/wp2_rungs/<family>__<label>.pt + json)
      so training can run in parallel shells and the evaluation is
      restartable; the CSV is rebuilt from scratch at evaluation time.
    - Early stopping validates on the DEFECTIVE simulator (what a
      practitioner with a wrong simulator would do); train_val_l1 is
      therefore not comparable across rungs and is logged only.

OUTPUTS
    results/wp2_defect_tolerance.csv  defect_family, defect, x (numeric
        defect magnitude), element, candidate (det|net), region,
        testbed (sim_<angle>deg_<hole> | REAL_ruotato), metrics
    results/wp2_tolerance_summary.txt
    figures/wp2_tolerance_curves.png

DEFINITION OF DONE
    Every family in DEFECT_LADDERS has a curve; each curve states where
    the net drops below the deterministic baseline / out of the seed
    band; three-sentence takeaway per family for the paper.

Run from the repo root:
    python neurips_submission/wp2_simulator_audit/exp_defect_tolerance.py --quick
    # parallel training (4 shells), then the evaluation:
    python .../exp_defect_tolerance.py --train-only --rungs 0-4   --threads 3
    python .../exp_defect_tolerance.py --train-only --rungs 5-9   --threads 3
    python .../exp_defect_tolerance.py --train-only --rungs 10-13 --threads 3
    python .../exp_defect_tolerance.py --train-only --rungs 14-17 --threads 3
    python .../exp_defect_tolerance.py
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import config
from common import core, io_utils, perturb, restore, training

ELEMENTS = core.ELEMENTS
_QUICK = False
RUNG_SEED = config.BASE_SEED + 1          # == WP1 control_00
# borderline rungs re-trained with extra seeds (reviewer guard: is the
# small real-scan dip just a bad seed?)
EXTRA_SEED_RUNGS = (("warp_shift", "shift_0.5px"), ("blur", "bilinear_v1"))
N_EXTRA_SEEDS = 3


def set_quick(flag):
    global _QUICK
    _QUICK = bool(flag)


def rung_dir():
    d = os.path.join(core.RESULTS_DIR,
                     "wp2_rungs" + ("_quick" if _QUICK else ""))
    os.makedirs(d, exist_ok=True)
    return d


def ladder() -> list:
    """[(family, label, SimKnobs)] with the nominal control first."""
    out = [("nominal", "nominal", perturb.NOMINAL)]
    for family, rungs in config.DEFECT_LADDERS.items():
        for label, kw in rungs:
            out.append((family, label, perturb.SimKnobs(label=label, **kw)))
    return out


def defect_x(family: str, knobs: perturb.SimKnobs):
    """Numeric defect magnitude for the curves (nominal -> nominal value)."""
    key, nominal = config.DEFECT_X.get(family, (None, 0.0))
    if key is None:
        return 0.0
    v = getattr(knobs, key)
    if key == "warp_shift_px":
        return float(v[0])
    if key == "blur_mode":
        return 0.0 if v == "cubic" else 1.0
    return float(v)


def _paths(family, label):
    base = os.path.join(rung_dir(), f"{family}__{label}")
    return base + ".pt", base + ".json"


def _wp1_control(i=0):
    d = os.path.join(core.RESULTS_DIR,
                     "wp1_ensemble" + ("_quick" if _QUICK else ""))
    p = os.path.join(d, f"control_{i:02d}.pt")
    return p if os.path.exists(p) else None


def train_rung(family, label, knobs, tcfg, verbose=False):
    ckpt, meta = _paths(family, label)
    net = core.RestorationUNet()
    if family == "nominal" and not os.path.exists(ckpt):
        src = _wp1_control(0)
        if src is not None:
            net.load_state_dict(torch.load(src, weights_only=True))
            net.eval()
            print(f"[{family}/{label}] = WP1 control_00 (same seed, same "
                  f"simulator)")
            return net
    if os.path.exists(ckpt):
        net.load_state_dict(torch.load(ckpt, weights_only=True))
        net.eval()
        print(f"[{family}/{label}] cached")
        return net
    print(f"[{family}/{label}] training  knobs={knobs.to_meta()}", flush=True)
    net, hist = training.train_net(training.make_batch_fn(knobs=knobs),
                                   tcfg, seed=RUNG_SEED, verbose=verbose)
    torch.save(net.state_dict(), ckpt)
    with open(meta, "w", encoding="utf-8") as fh:
        json.dump({"defect_family": family, "defect": label,
                   **knobs.to_meta(), **hist}, fh, indent=1)
    print(f"[{family}/{label}] done  best val L1 {hist['best_val_l1']:.5f}"
          f" at step {hist['best_step']}  ({hist['wall_s']:.0f} s)",
          flush=True)
    return net


def rung_hist(family, label) -> dict:
    _, meta = _paths(family, label)
    if os.path.exists(meta):
        with open(meta, encoding="utf-8") as fh:
            h = json.load(fh)
        return {"train_val_l1": h.get("best_val_l1"),
                "train_best_step": h.get("best_step")}
    return {}


def train_extra_seeds(which=None, tcfg=None, verbose=False):
    """Train the extra-seed variants of the borderline rungs (cached)."""
    tcfg = tcfg or config.TRAIN
    lookup = {(f, l): k for f, l, k in ladder()}
    sel = EXTRA_SEED_RUNGS if which in (None, "both") else tuple(
        r for r in EXTRA_SEED_RUNGS
        if (which == "shift") == (r[0] == "warp_shift"))
    for (family, label) in sel:
        knobs = lookup[(family, label)]
        for k in range(1, N_EXTRA_SEEDS + 1):
            base = os.path.join(rung_dir(), f"{family}__{label}__s{k}")
            if os.path.exists(base + ".pt"):
                print(f"[{family}/{label} s{k}] cached")
                continue
            print(f"[{family}/{label} s{k}] training seed"
                  f" {RUNG_SEED + 1000 * k}", flush=True)
            net, hist = training.train_net(
                training.make_batch_fn(knobs=knobs), tcfg,
                seed=RUNG_SEED + 1000 * k, verbose=verbose)
            torch.save(net.state_dict(), base + ".pt")
            with open(base + ".json", "w", encoding="utf-8") as fh:
                json.dump({"defect_family": family, "defect": label,
                           "seed_variant": k, **knobs.to_meta(), **hist},
                          fh, indent=1)
            print(f"[{family}/{label} s{k}] done  best val L1"
                  f" {hist['best_val_l1']:.5f} ({hist['wall_s']:.0f} s)",
                  flush=True)


def eval_extra_seeds(quick) -> list:
    """Rows for whichever extra-seed variants exist on disk."""
    rows = []
    lookup = {(f, l): k for f, l, k in ladder()}
    for (family, label) in EXTRA_SEED_RUNGS:
        knobs = lookup[(family, label)]
        x = defect_x(family, knobs)
        for k in range(1, N_EXTRA_SEEDS + 1):
            ckpt = os.path.join(rung_dir(), f"{family}__{label}__s{k}.pt")
            if not os.path.exists(ckpt):
                continue
            net = core.RestorationUNet()
            net.load_state_dict(torch.load(ckpt, weights_only=True))
            net.eval()
            rows += eval_net(net, family, f"{label}#s{k}", x, quick)
    return rows


# ---------------------------------------------------------------------------
# testbed
# ---------------------------------------------------------------------------

_CASES = None


def test_cases(quick):
    global _CASES
    if _CASES is None:
        spec = config.WP2_TEST_QUICK if quick else config.WP2_TEST
        _CASES = []
        for angle in spec["angles"]:
            for (h, w) in spec["holes"]:
                c = restore.degrade(source="prova2", angle=angle,
                                    block=restore.centered_block(h, w),
                                    seed=spec["seed"])
                c["testbed"] = f"sim_{angle:g}deg_h{h}x{w}"
                _CASES.append(c)
    return _CASES


def eval_net(net, family, label, x, quick) -> list:
    rows = []
    for case in test_cases(quick):
        det, learned = restore.apply_network(net, case["tilted"],
                                             case["angle"],
                                             validity=case["validity"])
        regions = {"footprint": case["fp"],
                   "hole": case["hole"] if case["block"] else None}
        for r in restore.score_candidates({"det": det, "net": learned},
                                          case["truth"], regions):
            rows.append({"defect_family": family, "defect": label, "x": x,
                         "testbed": case["testbed"],
                         "case_angle": case["angle"], **r})
    ruo = core.fm.load_summed_maps("ruotato")
    truth2 = core.fm.load_summed_maps("prova2")
    det, learned = restore.apply_network(net, ruo, core.fm.REF_ANGLE_DEG)
    for r in restore.score_candidates({"det": det, "net": learned}, truth2,
                                      {"footprint": core.dg.footprint()}):
        rows.append({"defect_family": family, "defect": label, "x": x,
                     "testbed": "REAL_ruotato",
                     "case_angle": core.fm.REF_ANGLE_DEG, **r})
    return rows


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run(quick: bool = False, verbose: bool = False):
    set_quick(quick)
    tcfg = config.QUICK_TRAIN if quick else config.TRAIN
    rungs = ladder()
    if quick:
        rungs = rungs[:3]
        print("QUICK: first 3 rungs only")
    t0 = time.time()
    rows = []
    for i, (family, label, knobs) in enumerate(rungs):
        net = train_rung(family, label, knobs, tcfg, verbose)
        x = defect_x(family, knobs)
        rr = eval_net(net, family, label, x, quick)
        hist = rung_hist(family, label)
        for r in rr:
            r.update(hist)
        rows += rr
        io_utils.write_rows("wp2_defect_tolerance", rows)   # checkpoint
        print(f"  [{i + 1}/{len(rungs)}] {family}/{label} evaluated"
              f" [{time.time() - t0:.0f} s]", flush=True)
    # the seed band: WP1's nominal control members on the same testbed
    n = config.ENSEMBLE_N_QUICK if quick else config.ENSEMBLE_N
    for k in range(1, n):
        p = _wp1_control(k)
        if p is None:
            continue
        net = core.RestorationUNet()
        net.load_state_dict(torch.load(p, weights_only=True))
        net.eval()
        rows += eval_net(net, "nominal_seeds", f"seed_{k:02d}", 0.0, quick)
    rows += eval_extra_seeds(quick)
    path = io_utils.write_rows("wp2_defect_tolerance", rows)
    print(f"saved: {path}  ({len(rows)} rows)")
    summarize()


# ---------------------------------------------------------------------------
# summary + figure
# ---------------------------------------------------------------------------

def _rows(rows, **cond):
    out = []
    for r in rows:
        if all(str(r.get(k, "")) == str(v) for k, v in cond.items()):
            out.append(r)
    return out


def _delta_r(rows, family, label, testbed_sel, region, lines=ELEMENTS):
    """mean over lines and matching testbeds of r(net) - r(det)."""
    d = []
    sel = [r for r in rows if r["defect_family"] == family
           and r["defect"] == label and r["region"] == region
           and testbed_sel(r["testbed"]) and r["element"] in lines]
    by = {}
    for r in sel:
        by.setdefault((r["testbed"], r["element"]), {})[r["candidate"]] = r
    for k, v in by.items():
        if "det" in v and "net" in v:
            d.append(float(v["net"]["r"]) - float(v["det"]["r"]))
    return float(np.mean(d)) if d else float("nan")


def _metric(rows, family, label, testbed_sel, region, key, cand="net",
            lines=ELEMENTS):
    v = [float(r[key]) for r in rows if r["defect_family"] == family
         and r["defect"] == label and r["region"] == region
         and r["candidate"] == cand and testbed_sel(r["testbed"])
         and r["element"] in lines and r.get(key, "") not in ("", "nan")]
    return float(np.mean(v)) if v else float("nan")


def summarize():
    rows = io_utils.read_rows("wp2_defect_tolerance")
    if not rows:
        print("no rows")
        return
    is_sim = lambda t: t.startswith("sim_")            # noqa: E731
    is_real = lambda t: t == "REAL_ruotato"            # noqa: E731
    L = []
    P = L.append
    P("WP2 - defect tolerance: r(net) - r(det), mean over lines "
      "(sim: mean over 4 angles x 2 holes from prova2; REAL: ruotato vs prova2)")
    seeds = sorted({r["defect"] for r in rows
                    if r["defect_family"] == "nominal_seeds"})
    band = {}
    for reg, sel, name in (("footprint", is_sim, "sim_fp"),
                           ("hole", is_sim, "sim_hole"),
                           ("footprint", is_real, "real_fp")):
        vals = [_delta_r(rows, "nominal", "nominal", sel, reg)] + [
            _delta_r(rows, "nominal_seeds", s, sel, reg) for s in seeds]
        vals = [v for v in vals if np.isfinite(v)]
        band[name] = (min(vals), max(vals), float(np.mean(vals))) if vals \
            else (np.nan, np.nan, np.nan)
    P(f"nominal seed band (n={len(seeds) + 1}) of delta r: "
      + "  ".join(f"{k} [{v[0]:+.4f}, {v[1]:+.4f}] mean {v[2]:+.4f}"
                  for k, v in band.items()))
    P("")
    P(f"{'family':12s}{'rung':16s}{'x':>7s}{'sim fp':>10s}{'sim hole':>10s}"
      f"{'REAL fp':>10s}{'cv REAL':>9s}{'bias REAL':>11s}{'val L1':>9s}  flags")
    for family, rungs in [("nominal", [("nominal", None)])] + [
            (f, [(l, k) for l, k in r]) for f, r in config.DEFECT_LADDERS.items()]:
        for label, _ in rungs:
            sel = _rows(rows, defect_family=family, defect=label)
            if not sel:
                continue
            x = float(sel[0]["x"])
            d_fp = _delta_r(rows, family, label, is_sim, "footprint")
            d_h = _delta_r(rows, family, label, is_sim, "hole")
            d_re = _delta_r(rows, family, label, is_real, "footprint")
            cv = _metric(rows, family, label, is_real, "footprint", "cv_ratio")
            bias = np.mean([abs(float(r["bias_pct"])) for r in sel
                            if r["candidate"] == "net" and is_real(r["testbed"])
                            and r["region"] == "footprint"])
            vl = sel[0].get("train_val_l1", "")
            flags = []
            if np.isfinite(d_fp) and d_fp < band["sim_fp"][0]:
                flags.append("sim-fp<band")
            if np.isfinite(d_h) and d_h < band["sim_hole"][0]:
                flags.append("hole<band")
            if np.isfinite(d_re) and d_re < band["real_fp"][0]:
                flags.append("REAL<band")
            if np.isfinite(d_re) and d_re < 0 and band["real_fp"][2] >= 0:
                flags.append("REAL<det")
            if np.isfinite(d_fp) and d_fp < 0:
                flags.append("sim-fp<det")
            P(f"{family:12s}{label:16s}{x:7.2f}{d_fp:+10.4f}{d_h:+10.4f}"
              f"{d_re:+10.4f}{cv:9.3f}{bias:11.2f}"
              f"{(float(vl) if vl not in ('', None) else float('nan')):9.4f}"
              f"  {' '.join(flags)}")
    # seed spread on the borderline rungs (base seed + extra variants)
    extra = [(f, l) for (f, l) in EXTRA_SEED_RUNGS
             if _rows(rows, defect_family=f, defect=l + "#s1")]
    if extra:
        P("")
        P("seed check on the borderline rungs (delta r over the base seed"
          " and the extra-seed variants):")
        for (family, label) in extra:
            variants = [label] + [f"{label}#s{k}"
                                  for k in range(1, N_EXTRA_SEEDS + 1)
                                  if _rows(rows, defect_family=family,
                                           defect=f"{label}#s{k}")]
            for reg, sel, name in (("footprint", is_sim, "sim fp"),
                                   ("footprint", is_real, "REAL fp")):
                vals = [_delta_r(rows, family, v, sel, reg)
                        for v in variants]
                vals = [v for v in vals if np.isfinite(v)]
                P(f"  {family}/{label} {name}: n={len(vals)}  mean"
                  f" {np.mean(vals):+.4f}  range [{min(vals):+.4f},"
                  f" {max(vals):+.4f}]")
    P("")
    P("reading: 'band' = the range of the 12 nominal seeds; a rung that stays"
      " inside it is indistinguishable from a re-initialisation; '<det' = "
      "the net is worse than the physics inverse.")
    text = "\n".join(L)
    print(text)
    with open(os.path.join(core.RESULTS_DIR, "wp2_tolerance_summary.txt"),
              "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return text


def crosstab():
    """Visibility (WP2 diagnostics, pre-registered rule) x damage (WP2
    tolerance) per rung - the paper's audit payoff table.

    Visible  = majority rule diagnosis != 'ok' over the 5 seeds at the
               7.7 deg calibration angle (20 deg shown as well).
    Harmful  = delta r leaves the 12-seed nominal band on the simulated
               footprint or on the real scan (the hole region is
               excluded: its band is seed-dominated, see the summary).
    """
    rows = io_utils.read_rows("wp2_defect_tolerance")
    conf = io_utils.read_rows("wp2_diag_confusion")
    if not rows or not conf:
        print("need both wp2_defect_tolerance.csv and wp2_diag_confusion.csv")
        return
    is_sim = lambda t: t.startswith("sim_")            # noqa: E731
    is_real = lambda t: t == "REAL_ruotato"            # noqa: E731
    seeds = sorted({r["defect"] for r in rows
                    if r["defect_family"] == "nominal_seeds"})
    band = {}
    for reg, sel, name in (("footprint", is_sim, "sim_fp"),
                           ("footprint", is_real, "real_fp")):
        vals = [_delta_r(rows, "nominal", "nominal", sel, reg)] + [
            _delta_r(rows, "nominal_seeds", s_, sel, reg) for s_ in seeds]
        vals = [v for v in vals if np.isfinite(v)]
        band[name] = (min(vals), max(vals))

    def majority(defect, angle):
        ds = [c["diagnosis"] for c in conf
              if c["defect"] == defect and c.get("angle", "") == str(angle)]
        if not ds:
            return "n/a"
        return max(set(ds), key=ds.count)

    L = []
    P = L.append
    P("")
    P("VISIBILITY x DAMAGE crosstab (diagnosis = pre-registered rule,"
      " majority over 5 seeds; harmful = outside the nominal seed band)")
    P(f"{'family':12s}{'rung':16s}{'diag 7.7deg':>14s}{'diag 20deg':>13s}"
      f"{'d_r sim fp':>12s}{'d_r real':>10s}{'harmful':>9s}  quadrant")
    quad_count = {}
    for family, rungs in config.DEFECT_LADDERS.items():
        for label, _ in rungs:
            d_fp = _delta_r(rows, family, label, is_sim, "footprint")
            d_re = _delta_r(rows, family, label, is_real, "footprint")
            harmful = (np.isfinite(d_fp) and d_fp < band["sim_fp"][0]) or                       (np.isfinite(d_re) and d_re < band["real_fp"][0])
            dg1, dg2 = majority(label, 7.7), majority(label, 20.0)
            visible = dg1 != "ok" or dg2 != "ok"
            quad = ("visible+harmful" if visible and harmful else
                    "visible+harmless" if visible else
                    "INVISIBLE+HARMFUL" if harmful else
                    "invisible+harmless")
            quad_count[quad] = quad_count.get(quad, 0) + 1
            P(f"{family:12s}{label:16s}{dg1:>14s}{dg2:>13s}"
              f"{d_fp:+12.4f}{d_re:+10.4f}{str(harmful):>9s}  {quad}")
    P("")
    P("quadrants: " + "  ".join(f"{k} {v}" for k, v in
                                sorted(quad_count.items())))
    P("the dangerous quadrant is INVISIBLE+HARMFUL"
      + (" - EMPTY: every defect that hurts is caught by the audit"
         if quad_count.get("INVISIBLE+HARMFUL", 0) == 0 else
         f" - {quad_count['INVISIBLE+HARMFUL']} rung(s) fall in it"))
    text = "\n".join(L)
    print(text)
    with open(os.path.join(core.RESULTS_DIR, "wp2_tolerance_summary.txt"),
              "a", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return text


def make_figures():
    rows = io_utils.read_rows("wp2_defect_tolerance")
    if not rows:
        raise NotImplementedError("run exp_defect_tolerance first")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    NAVY, GREY, ORANGE = "#1f2a44", "#8c8c8c", "#c8641e"
    is_sim = lambda t: t.startswith("sim_")            # noqa: E731
    is_real = lambda t: t == "REAL_ruotato"            # noqa: E731
    families = [f for f in config.DEFECT_LADDERS if _rows(rows, defect_family=f)]
    seeds = sorted({r["defect"] for r in rows
                    if r["defect_family"] == "nominal_seeds"})
    # the sim-hole row is omitted: its 12-seed band spans [-0.03, +0.27]
    # (initialisation-dominated), so it carries no tolerance signal; the
    # numbers stay in the summary table
    panels = [("sim footprint", is_sim, "footprint",
               "r(net) - r(physics)\n(up = better)"),
              ("real scan", is_real, "footprint",
               "r(net) - r(physics)\n(up = better)")]
    fig, axes = plt.subplots(len(panels), len(families),
                             figsize=(1.45 * len(families),
                                      1.25 * len(panels)),
                             sharey="row")
    axes = np.atleast_2d(axes)
    for j, fam in enumerate(families):
        key, nominal = config.DEFECT_X[fam]
        rungs = [(l, k) for l, k in config.DEFECT_LADDERS[fam]]
        for i, (title, sel, reg, ylab) in enumerate(panels):
            ax = axes[i, j]
            # seed band at the nominal x
            b = [_delta_r(rows, "nominal", "nominal", sel, reg)] + [
                _delta_r(rows, "nominal_seeds", s, sel, reg) for s in seeds]
            b = [v for v in b if np.isfinite(v)]
            xs, ys = [], []
            x0 = 0.0 if key == "blur_mode" else float(nominal)
            xs.append(x0)
            ys.append(b[0] if b else np.nan)
            for label, _ in rungs:
                sel_rows = _rows(rows, defect_family=fam, defect=label)
                if not sel_rows:
                    continue
                xs.append(float(sel_rows[0]["x"]))
                ys.append(_delta_r(rows, fam, label, sel, reg))
            order = np.argsort(xs)
            xs, ys = np.array(xs)[order], np.array(ys)[order]
            if b:
                ax.axhspan(min(b), max(b), color=GREY, alpha=0.25, lw=0)
            ax.axhline(0.0, color="k", lw=0.7, alpha=0.6)
            ax.plot(xs, ys, marker="o", ms=4, lw=1.5, color=NAVY)
            ax.plot([x0], [ys[list(xs).index(x0)] if x0 in list(xs) else np.nan],
                    marker="s", ms=6, color=ORANGE, ls="none")
            if key in ("noise_k_scale", "gain_scale") and fam == "noise_k":
                from matplotlib.ticker import NullFormatter, ScalarFormatter
                ax.set_xscale("log")
                ax.set_xticks([0.25, 1.0, 4.0])
                ax.xaxis.set_major_formatter(ScalarFormatter())
                ax.xaxis.set_minor_formatter(NullFormatter())
            if key == "blur_mode":
                ax.set_xticks([0, 1])
                ax.set_xticklabels(["cubic", "bilinear"])
            if i == 0:
                ax.set_title({"noise_k": "noise const. k (x)",
                              "gain": "gain slope (x)",
                              "angle_bias": "angle belief (deg)",
                              "blur": "resampling blur",
                              "warp_shift": "reg. shift (px)",
                              "warp_rot": "reg. rotation (deg)"
                              }.get(fam, fam), fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{title}\n{ylab}", fontsize=8.5)
            ax.grid(alpha=0.25)
            ax.tick_params(labelsize=7.5)
    fig.suptitle("One net per broken training simulator; grey band = "
                 "12 nominal seeds, orange = nominal", fontsize=9)
    fig.tight_layout()
    out = io_utils.fig_path("wp2_tolerance_curves.png")
    fig.savefig(out, dpi=200)
    fig.savefig(out.replace(".png", ".pdf"))
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--crosstab", action="store_true")
    ap.add_argument("--extra-seeds", action="store_true")
    ap.add_argument("--which", choices=("shift", "blur", "both"),
                    default="both")
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--rungs", default=None,
                    help="indices into the ladder (0 = nominal), e.g. 1-5")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    set_quick(args.quick)
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.list:
        for i, (f, l, k) in enumerate(ladder()):
            print(i, f, l)
    elif args.figures:
        make_figures()
    elif args.summary:
        summarize()
    elif args.crosstab:
        crosstab()
    elif args.train_only:
        tcfg = config.QUICK_TRAIN if args.quick else config.TRAIN
        if args.extra_seeds:
            train_extra_seeds(args.which, tcfg, args.verbose)
        else:
            rungs = ladder()
            idx = _parse(args.rungs) or range(len(rungs))
            for i in idx:
                f, l, k = rungs[i]
                train_rung(f, l, k, tcfg, args.verbose)
    else:
        run(quick=args.quick, verbose=args.verbose)
