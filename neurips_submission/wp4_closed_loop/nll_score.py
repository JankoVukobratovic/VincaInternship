"""WP4 / addendum - Gaussian NLL as ONE proper score for the ensembles.

Coverage + Spearman + AUSE describe calibration and ranking separately;
the negative log likelihood under a per-pixel Gaussian N(mean, sigma^2)
is a single PROPER score that rewards a band that is both narrow and
calibrated, and settles the posterior-vs-prior tradeoff (27 % narrower
at 1-2 points less coverage) with one number.

    NLL = mean over pixels of [ 0.5 ln(2 pi v) + err^2 / (2 v) ],

with v the band variance.  Bands, exactly as in WP1/WP4:
    real scan : v = spread^2 + alea^2 + k*truth   ("total")
                v = spread^2 + alea^2             ("total_noref")
    simulated : v = spread^2 + alea^2             ("total")
                v = spread^2 + alea^2 + k*truth   ("total_ref", upper
                bound on the reference-noise contribution)

Ensembles: control (WP1 nominal), prior (WP1 jitter), posterior (WP4).
Inference only; nothing is trained here.

OUTPUT  results/wp4_nll.csv, results/wp4_nll_summary.txt

Run from the repo root:
    python neurips_submission/wp4_closed_loop/nll_score.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import config
from common import core, io_utils, restore
from wp1_uq_ensemble import exp_ensemble_uq as uq

ELEMENTS = core.ELEMENTS
EPS = 1e-6


def load_ensemble(kind):
    if kind == "posterior":
        d = os.path.join(core.RESULTS_DIR, "wp4_posterior")
        paths = [os.path.join(d, f"post_{i:02d}.pt") for i in range(12)]
    else:
        fname = "jitter" if kind == "prior" else kind
        d = os.path.join(core.RESULTS_DIR, "wp1_ensemble")
        paths = [os.path.join(d, f"{fname}_{i:02d}.pt") for i in range(12)]
    nets = []
    for p in paths:
        net = core.RestorationUNet()
        net.load_state_dict(torch.load(p, weights_only=True))
        net.eval()
        nets.append(net)
    return nets


def nll_rows(mean, var, truth, regions, meta):
    rows = []
    for el in ELEMENTS:
        err2 = (mean[el] - truth[el]) ** 2
        v = var[el] + EPS
        nll = 0.5 * np.log(2.0 * math.pi * v) + err2 / (2.0 * v)
        for reg, mask in regions.items():
            if mask is None:
                continue
            m = mask & np.isfinite(nll)
            if int(m.sum()) < 9:
                continue
            rows.append({**meta, "element": el, "region": reg,
                         "nll": float(np.mean(nll[m])),
                         "n_px": int(m.sum())})
    return rows


def main():
    torch.set_num_threads(2)
    ks = core.fm.calibrate_noise()
    ens = {k: load_ensemble(k) for k in ("control", "prior", "posterior")}
    print("ensembles loaded (12 members each)")
    rows = []

    # real anchor
    ruo = core.fm.load_summed_maps("ruotato")
    truth2 = core.fm.load_summed_maps("prova2")
    twin = restore.degrade(source="prova2", angle=core.fm.REF_ANGLE_DEG,
                           block=None, dose=1.0, seed=0, sim="validated")
    fp = core.dg.footprint()
    ref = {el: ks[el] * np.clip(truth2[el], 0.0, None) for el in ELEMENTS}
    for kind, members in ens.items():
        mean, std, det, _ = uq.ensemble_predict(members, ruo,
                                                core.fm.REF_ANGLE_DEG)
        alea, _ = uq.aleatoric_sigma(members, twin, config.WP1_NOISE_REPS)
        meta = {"source": "REAL_ruotato", "case": "REAL", "ensemble": kind}
        v_noref = {el: std[el] ** 2 + alea[el] ** 2 for el in ELEMENTS}
        v_tot = {el: v_noref[el] + ref[el] for el in ELEMENTS}
        rows += nll_rows(mean, v_tot, truth2, {"footprint": fp},
                         {**meta, "band": "total"})
        rows += nll_rows(mean, v_noref, truth2, {"footprint": fp},
                         {**meta, "band": "total_noref"})
        print(f"  real: {kind}")

    # simulated held-out cases (validated, dose 1)
    spec = config.WP1_CASES
    for angle in spec["angles"]:
        for (h, w) in spec["holes"]:
            case = restore.degrade(source="prova2", angle=angle,
                                   block=restore.centered_block(h, w),
                                   dose=1.0, seed=spec["seed"],
                                   sim="validated")
            regions = {"footprint": case["fp"],
                       "hole": case["hole"] if case["block"] else None}
            ref_c = {el: ks[el] * np.clip(case["truth"][el], 0.0, None)
                     for el in ELEMENTS}
            for kind, members in ens.items():
                mean, std, det, _ = uq.ensemble_predict(
                    members, case["tilted"], case["angle"],
                    validity=case["validity"])
                alea, _ = uq.aleatoric_sigma(members, case,
                                             config.WP1_NOISE_REPS)
                meta = {"source": "sim", "case": f"a{angle:g}_h{h}x{w}",
                        "ensemble": kind}
                v = {el: std[el] ** 2 + alea[el] ** 2 for el in ELEMENTS}
                v_ref = {el: v[el] + ref_c[el] for el in ELEMENTS}
                rows += nll_rows(mean, v, case["truth"], regions,
                                 {**meta, "band": "total"})
                rows += nll_rows(mean, v_ref, case["truth"], regions,
                                 {**meta, "band": "total_ref"})
            print(f"  sim a{angle:g} h{h}x{w} done", flush=True)

    path = io_utils.write_rows("wp4_nll", rows)
    print("saved:", path)
    summarize()


def summarize():
    rows = io_utils.read_rows("wp4_nll")
    L = []
    P = L.append
    P("WP4 addendum - Gaussian NLL per pixel (nats; lower = better;"
      " a proper score: narrow AND calibrated wins)")

    def mean_nll(**cond):
        v = [float(r["nll"]) for r in rows
             if all(str(r.get(k)) == str(x) for k, x in cond.items())]
        return float(np.mean(v)) if v else float("nan")

    P("")
    P(f"{'setting':34s}{'control':>10s}{'prior':>10s}{'posterior':>10s}")
    for src, reg, band in (("REAL_ruotato", "footprint", "total"),
                           ("REAL_ruotato", "footprint", "total_noref"),
                           ("sim", "footprint", "total"),
                           ("sim", "footprint", "total_ref"),
                           ("sim", "hole", "total")):
        vals = [mean_nll(source=src, region=reg, band=band, ensemble=k)
                for k in ("control", "prior", "posterior")]
        P(f"{src + ' / ' + reg + ' / ' + band:34s}"
          + "".join(f"{v:10.3f}" for v in vals))
    P("")
    P("REAL / footprint / total, per line:")
    P("    " + f"{'line':6s}{'control':>10s}{'prior':>10s}{'posterior':>10s}")
    for el in ELEMENTS:
        vals = [mean_nll(source="REAL_ruotato", region="footprint",
                         band="total", ensemble=k, element=el)
                for k in ("control", "prior", "posterior")]
        P("    " + f"{el:6s}" + "".join(f"{v:10.3f}" for v in vals))
    text = "\n".join(L)
    print(text)
    with open(os.path.join(core.RESULTS_DIR, "wp4_nll_summary.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
