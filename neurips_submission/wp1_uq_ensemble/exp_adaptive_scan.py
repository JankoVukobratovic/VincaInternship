"""WP1 / experiment 2 (STRETCH) - uncertainty-guided adaptive scanning.
OWNER: Dimitrije.  Runs after exp_ensemble_uq (needs its jitter
ensemble); if the result is a null it becomes one outlook paragraph.

CLAIM UNDER TEST
    Measuring only a fraction of the tilted frame and letting the
    physics + prior fill the rest, with the NEXT tiles chosen where the
    ensemble is most uncertain, reaches a target map quality with fewer
    measured pixels than raster or random acquisition, i.e. less
    irradiation of the painting and shorter scan time.

PROTOCOL
    1. The tilted frame (45 x 80) is divided into TILE x TILE tiles
       (config-free constants below); a scan starts with START_FRAC of
       the tiles measured (fixed random subset, same for every strategy)
       and the rest zeroed = dropout blocks the net was trained with.
    2. Restore with the WP1 jitter ensemble (inverse + net, validity
       channel) -> mean and spread maps in the frontal frame.
    3. Acquisition step: reveal B tiles chosen by the strategy
         adaptive : largest ensemble spread (summed over the headline
                    lines, each normalised by the line's spread scale,
                    warped to the tilted frame and averaged per tile)
         random   : uniform random tiles
         raster   : row-major order (the scanner's default)
         oracle   : largest actual error (upper bound, not realisable)
    4. Repeat until everything is measured; log r / bias vs measured
       fraction per line after every step.

OUTPUT   results/wp1_adaptive_scan.csv:
         case..., strategy, step, measured_frac, element, region, r,
         bias_pct (+ ssim, cv_ratio on the footprint)
FIGURE   figures/wp1_adaptive.png: r vs measured fraction, the
         strategies, headline lines (config.FIG_LINES).

DEFINITION OF DONE: the curves on >= 2 simulated cases, or an explicit
decision to drop to outlook status (tell the team).

Run from the repo root:
    python neurips_submission/wp1_uq_ensemble/exp_adaptive_scan.py [--quick]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config
from common import core, io_utils, restore
from wp1_uq_ensemble import exp_ensemble_uq as uq

ELEMENTS = core.ELEMENTS
TILE = 5                      # tile side in tilted-frame pixels
START_FRAC = 0.25             # fraction of tiles measured at the start
TILES_PER_STEP = 8            # B: tiles revealed per acquisition step
STRATEGIES = ("adaptive", "random", "raster", "oracle")
CASES = ((20.0, 1.0), (14.0, 1.0), (25.0, 0.5))      # (angle, dose)
CASES_QUICK = ((20.0, 1.0),)


def tile_grid():
    th, tw = core.TILTED_SHAPE
    ids = np.zeros(core.TILTED_SHAPE, dtype=int)
    k = 0
    tiles = []
    for r0 in range(0, th, TILE):
        for c0 in range(0, tw, TILE):
            ids[r0:r0 + TILE, c0:c0 + TILE] = k
            tiles.append((r0, c0))
            k += 1
    return ids, tiles


def measure(full_tilted: dict, measured: np.ndarray):
    """The scan as acquired so far: unmeasured pixels are zero (as in
    the training dropout blocks); validity mask in both frames."""
    v_tilt = measured.astype(float)
    tilted = {el: np.where(measured, m, 0.0) for el, m in full_tilted.items()}
    validity = np.nan_to_num(core.fm.warp_tilted_to_frontal(v_tilt),
                             nan=0.0).astype(np.float32)
    return tilted, validity


def tile_scores(kind: str, mean, std, truth, ids, n_tiles, rng):
    """Per-tile acquisition priority for one strategy (higher = first)."""
    if kind == "random":
        return rng.random(n_tiles)
    if kind == "raster":
        return -np.arange(n_tiles, dtype=float)
    fp = core.dg.footprint()
    acc = np.zeros(core.FRONTAL_SHAPE)
    for el in config.FIG_LINES:
        if kind == "adaptive":
            m = np.nan_to_num(std[el], nan=0.0)
        else:                                    # oracle
            m = np.nan_to_num(np.abs(mean[el] - truth[el]), nan=0.0)
        scale = np.nanstd(truth[el][fp]) or 1.0
        acc += m / scale
    # frontal priority -> tilted frame -> per-tile mean
    t = np.nan_to_num(core.fm.warp_frontal_to_tilted(acc), nan=0.0)
    return np.array([t[ids == k].mean() for k in range(n_tiles)])


def run_case(members, angle, dose, strategies, rng_seed=0):
    case = restore.degrade(source="prova2", angle=angle, block=None,
                           dose=dose, seed=0, sim="validated")
    ids, tiles = tile_grid()
    n_tiles = len(tiles)
    rng0 = np.random.default_rng(rng_seed)
    start = np.zeros(n_tiles, dtype=bool)
    start[rng0.choice(n_tiles, int(round(START_FRAC * n_tiles)),
                      replace=False)] = True
    regions = {"footprint": case["fp"]}
    meta = {"case": f"a{angle:g}_d{dose:g}", "case_angle": angle,
            "case_dose": dose, "source": "prova2"}
    rows = []
    for strat in strategies:
        rng = np.random.default_rng(rng_seed + 1)
        got = start.copy()
        step = 0
        while True:
            measured = got[ids]
            tilted, validity = measure(case["tilted"], measured)
            mean, std, det, _ = uq.ensemble_predict(members, tilted, angle,
                                                    validity=validity)
            frac = float(measured.mean())
            unmeasured = (case["fp"] & (validity < 0.5))
            regs = dict(regions)
            regs["unmeasured"] = unmeasured if unmeasured.sum() >= 9 else None
            for r in restore.score_candidates(
                    {"ensemble": mean, "deterministic": det}, case["truth"],
                    regs):
                rows.append({**meta, "strategy": strat, "step": step,
                             "measured_frac": frac, **r})
            if got.all():
                break
            score = tile_scores(strat, mean, std, case["truth"], ids,
                                n_tiles, rng)
            score = np.where(got, -np.inf, score)
            order = np.argsort(-score, kind="stable")
            got[order[:TILES_PER_STEP]] = True
            step += 1
        print(f"  {meta['case']} {strat:9s} {step} steps, final frac"
              f" {float(got[ids].mean()):.2f}", flush=True)
    return rows


def run(quick: bool = False):
    uq.set_quick(quick)
    n = config.ENSEMBLE_N_QUICK if quick else config.ENSEMBLE_N
    members = []
    for i in range(n):
        ckpt, _ = uq._member_paths("jitter", i)
        if not os.path.exists(ckpt):
            raise NotImplementedError(
                "WP1 jitter ensemble not trained yet - run exp_ensemble_uq "
                "first")
        members.append(uq.train_member("jitter", i, {}))
    rows = []
    for angle, dose in (CASES_QUICK if quick else CASES):
        rows += run_case(members, angle, dose, STRATEGIES)
    path = io_utils.write_rows("wp1_adaptive_scan", rows)
    print(f"saved: {path}  ({len(rows)} rows)")
    summarize()


def summarize():
    rows = io_utils.read_rows("wp1_adaptive_scan")
    if not rows:
        return
    lines = []
    P = lines.append
    P("WP1 stretch - adaptive scanning: r (ensemble, footprint) vs "
      "measured fraction, mean over cases and headline lines")
    fracs = sorted({round(float(r["measured_frac"]), 3) for r in rows})
    P("frac    " + "".join(f"{s:>10s}" for s in STRATEGIES))
    for f in fracs:
        vals = []
        for s in STRATEGIES:
            v = [float(r["r"]) for r in rows
                 if r["strategy"] == s and r["candidate"] == "ensemble"
                 and r["region"] == "footprint" and r["element"] in config.FIG_LINES
                 and round(float(r["measured_frac"]), 3) == f]
            vals.append(np.mean(v) if v else np.nan)
        P(f"{f:5.2f}   " + "".join(f"{v:10.4f}" for v in vals))
    # pixels needed to reach 0.95 of the full-scan r, per strategy
    full = {}
    for s in STRATEGIES:
        v = [float(r["r"]) for r in rows
             if r["strategy"] == s and r["candidate"] == "ensemble"
             and r["region"] == "footprint" and r["element"] in config.FIG_LINES
             and round(float(r["measured_frac"]), 3) == fracs[-1]]
        full[s] = np.mean(v)
    P("")
    for target in (0.95, 0.98, 0.99):
        out = []
        for s in STRATEGIES:
            need = None
            for f in fracs:
                v = [float(r["r"]) for r in rows
                     if r["strategy"] == s and r["candidate"] == "ensemble"
                     and r["region"] == "footprint"
                     and r["element"] in config.FIG_LINES
                     and round(float(r["measured_frac"]), 3) == f]
                if v and np.mean(v) >= target * full[s]:
                    need = f
                    break
            out.append(f"{s} {need if need is not None else 'n/a'}")
        P(f"measured fraction to reach {target:.2f} x full-scan r: "
          + ", ".join(out))
    # measured fraction needed for an ABSOLUTE r target, linearly
    # interpolated between acquisition steps (the step grid is 5.6 %)
    curves = {}
    for s in STRATEGIES:
        ys = []
        for f in fracs:
            v = [float(r["r"]) for r in rows
                 if r["strategy"] == s and r["candidate"] == "ensemble"
                 and r["region"] == "footprint" and r["element"] in config.FIG_LINES
                 and round(float(r["measured_frac"]), 3) == f]
            ys.append(np.mean(v) if v else np.nan)
        curves[s] = np.array(ys)
    P("")
    P("measured fraction needed for an absolute r target (interpolated):")
    for target in (0.80, 0.90, 0.95):
        out = []
        for s in STRATEGIES:
            y = curves[s]
            need = "n/a"
            for i in range(1, len(fracs)):
                if y[i] >= target > y[i - 1]:
                    need = fracs[i - 1] + (fracs[i] - fracs[i - 1]) * \
                        (target - y[i - 1]) / (y[i] - y[i - 1])
                    need = f"{need:.3f}"
                    break
            out.append(f"{s} {need}")
        P(f"  r >= {target:.2f}: " + ", ".join(out))
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(core.RESULTS_DIR, "wp1_adaptive_summary.txt"),
              "w", encoding="utf-8") as fh:
        fh.write(text + "\n")


def make_figures():
    rows = io_utils.read_rows("wp1_adaptive_scan")
    if not rows:
        raise NotImplementedError("no wp1_adaptive_scan.csv yet")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"adaptive": uq.NAVY, "random": uq.GREY, "raster": "#b0b0b0",
              "oracle": uq.ORANGE}
    ls = {"adaptive": "-", "random": "-", "raster": "--", "oracle": ":"}
    lines = [el for el in config.FIG_LINES]
    fig, axes = plt.subplots(1, len(lines), figsize=(3.3 * len(lines), 3.2),
                             sharey=False)
    axes = np.atleast_1d(axes)
    for ax, el in zip(axes, lines):
        for s in STRATEGIES:
            fr = sorted({round(float(r["measured_frac"]), 3) for r in rows
                         if r["strategy"] == s})
            ys = []
            for f in fr:
                v = [float(r["r"]) for r in rows
                     if r["strategy"] == s and r["candidate"] == "ensemble"
                     and r["region"] == "footprint" and r["element"] == el
                     and round(float(r["measured_frac"]), 3) == f]
                ys.append(np.mean(v))
            ax.plot(fr, ys, color=colors[s], ls=ls[s], lw=1.6, label=s)
        ax.set_title(el, fontsize=10)
        ax.set_xlabel("measured fraction of the tilted frame")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("r vs truth (footprint)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = io_utils.fig_path("wp1_adaptive.png")
    fig.savefig(out, dpi=200)
    fig.savefig(out.replace(".png", ".pdf"))
    plt.close(fig)
    print("saved:", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    if args.figures:
        make_figures()
    else:
        run(quick=args.quick)
