"""WP3 - degradation grid, classical controls, regime map.
OWNER: Dimitrije (taken over 2026-08-28)

CLAIM UNDER TEST
    Where exactly does the learned prior pay?  We sweep degradation
    severity (angle x hole size x dose, config.GRID), restore every
    case with deterministic physics, physics+U-Net, the WP1 jitter
    ensemble, the classical inpainting controls and one hybrid, and
    condense the sweep into ONE regime-map figure: the frontier beyond
    which learning beats everything that does not learn.  Without the
    classical controls the paper cannot claim the prior is NEEDED.

METHOD (inference-only sweep - no training)
    1. Cases from prova2 (never in training) via common.restore.degrade
       with the validated instrument emulator (sim="validated").
    2. Candidates
         det                      nominal physics inverse
         net                      MVP checkpoint (nominal simulator)
         ens_jitter               mean of the 12 WP1 jitter members
         classical_nearest / _biharmonic / _telea / _ns
                                  fill in the tilted frame, nominal inverse
         classical_biharmonic+net biharmonic fill, then the MVP net with
                                  validity = footprint (the net is told
                                  the hole is data)
    3. Frozen scoring, footprint + hole regions.
    4. Real-scan anchor (ruotato vs prova2; det / net / ens_jitter)
       and the harsh case repeated with sim="sharp" (acquisition-blur
       sensitivity note).

OUTPUTS
    results/wp3_degradation_grid.csv   angle, hole_px, hole, dose, seed,
        sim, element, candidate, region, r, ssim, bias_pct, cv_ratio, n_px
    results/wp3_regime_summary.txt
    figures/wp3_regime_map.png / .pdf

The run is restartable: cases already in the CSV are skipped.

Run from the repo root:
    python neurips_submission/wp3_degradation_benchmark/exp_degradation_grid.py [--quick] [--figures] [--summary]
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

ELEMENTS = core.ELEMENTS
CSV = "wp3_degradation_grid"
HARSH = dict(angle=20.0, hole=(14, 20), dose=1.0)
NAVY, GREY, ORANGE = "#1f2a44", "#8c8c8c", "#c8641e"


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

def load_jitter_members():
    d = os.path.join(core.RESULTS_DIR, "wp1_ensemble")
    members = []
    for i in range(config.ENSEMBLE_N):
        p = os.path.join(d, f"jitter_{i:02d}.pt")
        if not os.path.exists(p):
            break
        net = core.RestorationUNet()
        net.load_state_dict(torch.load(p, weights_only=True))
        net.eval()
        members.append(net)
    if len(members) < config.ENSEMBLE_N:
        print(f"NOTE: WP1 jitter ensemble incomplete ({len(members)}/"
              f"{config.ENSEMBLE_N} members) - ens_jitter skipped")
        return []
    return members


def ensemble_mean(members, tilted, angle, validity):
    acc = {el: [] for el in ELEMENTS}
    for net in members:
        _, learned = restore.apply_network(net, tilted, angle,
                                           validity=validity)
        for el in ELEMENTS:
            acc[el].append(learned[el])
    return {el: np.mean(np.stack(v), axis=0) for el, v in acc.items()}


def candidates(net, members, tilted, v_tilt, angle, validity):
    """All restorations of one degraded acquisition (frontal maps)."""
    det, learned = restore.apply_network(net, tilted, angle,
                                         validity=validity)
    cands = {"det": det, "net": learned}
    if members:
        cands["ens_jitter"] = ensemble_mean(members, tilted, angle, validity)
    if v_tilt is not None:
        cands.update(classical.classical_restorations(tilted, v_tilt, angle))
        filled = classical.biharmonic_fill(tilted, v_tilt)
        _, hybrid = restore.apply_network(net, filled, angle, validity=None)
        cands["classical_biharmonic+net"] = hybrid
    return cands


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------

def case_key(angle, hole, dose, seed, sim):
    return (f"{float(angle):g}", hole, f"{float(dose):g}", str(int(seed)),
            sim)


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
        raise SystemExit(
            "MVP checkpoint missing (neurips-restore/experiments/"
            "checkpoint.pt) - run neurips-restore/scripts/"
            "03_learned_restoration.py first or fetch the checkpoint")
    members = load_jitter_members()

    grid = config.GRID_QUICK if quick else config.GRID
    plan = [(a, hw, d, s, "validated") for a, hw, d, s in itertools.product(
        grid["angles"], grid["holes"], grid["doses"], grid["seeds"])]
    # acquisition-blur sensitivity note: the harsh case with the sharp
    # (cubic, training-style) simulator
    for s in grid["seeds"]:
        plan.append((HARSH["angle"], HARSH["hole"], HARSH["dose"], s, "sharp"))

    rows = io_utils.read_rows(CSV)
    done = done_keys(rows)
    t0 = time.time()
    n_new = 0
    for i, (angle, (h, w), dose, seed, sim) in enumerate(plan):
        key = case_key(angle, f"{h}x{w}", dose, seed, sim)
        if key in done:
            continue
        print(f"[{i + 1}/{len(plan)}] angle={angle:g} hole={h}x{w} "
              f"dose={dose:g} seed={seed} sim={sim}  [{time.time() - t0:.0f} s]",
              flush=True)
        rows += run_case(net, members, angle, h, w, dose, seed, sim)
        done.add(key)
        n_new += 1
        if n_new % 10 == 0:
            io_utils.write_rows(CSV, rows)

    # real-scan anchor: measured ruotato restored, scored against prova2
    if case_key(core.fm.REF_ANGLE_DEG, "0x0", 1.0, -1, "real") not in done:
        ruo = core.fm.load_summed_maps("ruotato")
        truth2 = core.fm.load_summed_maps("prova2")
        cands = candidates(net, members, ruo, None, core.fm.REF_ANGLE_DEG,
                           None)
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


def _cands(rows):
    out = []
    for r in rows:
        if r["candidate"] not in out:
            out.append(r["candidate"])
    return out


def summarize(path=None) -> str:
    rows = io_utils.read_rows(CSV)
    if not rows:
        print("no rows yet")
        return ""
    sim = _sel(rows, sim="validated")
    holes = [f"{h}x{w}" for (h, w) in config.GRID["holes"]]
    angles = list(config.GRID["angles"])
    cands = _cands(sim)
    classical_c = [c for c in cands if c.startswith("classical_")
                   and not c.endswith("+net")]
    L = []
    P = L.append
    P("WP3 - degradation grid: summary")
    P(f"rows {len(rows)}; validated-sim cases: "
      f"{len({(r['angle'], r['hole'], r['dose'], r['seed']) for r in sim})}"
      f"; candidates: {', '.join(cands)}")
    P("")

    def table(region, dose, lines=None):
        P(f"[{region}, dose {dose:g}, mean r over seeds and "
          f"{'all 8 lines' if lines is None else '/'.join(lines)}]")
        P("    " + f"{'angle':>6s} {'hole':>6s}" + "".join(
            f"{c[:24]:>26s}" for c in cands) + f"{'best':>26s}")
        for a in angles:
            for hole in holes:
                if region == "hole" and hole == "0x0":
                    continue
                vals = {}
                for c in cands:
                    rr = _sel(sim, angle=a, hole=hole, dose=dose,
                              candidate=c, region=region)
                    if lines is not None:
                        rr = [r for r in rr if r["element"] in lines]
                    vals[c] = _mean(rr)
                best = max(vals, key=lambda c: (vals[c] if np.isfinite(
                    vals[c]) else -9))
                P("    " + f"{a:6g} {hole:>6s}" + "".join(
                    f"{vals[c]:26.4f}" for c in cands) + f"{best:>26s}")
        P("")

    table("hole", 1.0)
    table("footprint", 1.0)

    # crossover statements (dose 1, all lines, hole region)
    P("[crossover, hole region, dose 1, mean over seeds and 8 lines]")
    for learned in ("net", "ens_jitter", "classical_biharmonic+net"):
        if learned not in cands:
            continue
        wins, losses = [], []
        for a in angles:
            for hole in holes[1:]:
                rl = _mean(_sel(sim, angle=a, hole=hole, dose=1.0,
                                candidate=learned, region="hole"))
                best_c, best_v = None, -9
                for c in classical_c:
                    v = _mean(_sel(sim, angle=a, hole=hole, dose=1.0,
                                   candidate=c, region="hole"))
                    if v > best_v:
                        best_c, best_v = c, v
                (wins if rl > best_v else losses).append(
                    f"{a:g}deg/{hole} ({rl:+.3f} vs {best_c} {best_v:+.3f})")
        P(f"  {learned}: beats the best classical control in "
          f"{len(wins)}/{len(wins) + len(losses)} cells")
        if wins:
            P("    wins:   " + "; ".join(wins))
        if losses:
            P("    losses: " + "; ".join(losses))
    P("")

    # dose effect
    P("[dose effect: mean r over angles, seeds, 8 lines; hole 14x20 (hole"
      " region) and no hole (footprint)]")
    for d in config.GRID["doses"]:
        parts = []
        for c in cands:
            v_h = _mean(_sel(sim, hole="14x20", dose=d, candidate=c,
                             region="hole"))
            v_f = _mean(_sel(sim, hole="0x0", dose=d, candidate=c,
                             region="footprint"))
            parts.append(f"{c} {v_h:.3f}/{v_f:.3f}")
        P(f"  dose {d:g}: " + "  ".join(parts))
    P("")

    # sharp vs validated harsh case
    P("[acquisition-blur note: harsh case 20 deg / 14x20 / dose 1, mean over"
      " seeds and 8 lines, r hole / r footprint / cv_ratio footprint]")
    for s in ("validated", "sharp"):
        rr = _sel(rows, sim=s, angle=HARSH["angle"], hole="14x20", dose=1.0)
        if not rr:
            continue
        parts = []
        for c in cands:
            parts.append(
                f"{c} {_mean(_sel(rr, candidate=c, region='hole')):.3f}/"
                f"{_mean(_sel(rr, candidate=c, region='footprint')):.3f}/"
                f"{_mean(_sel(rr, candidate=c, region='footprint'), 'cv_ratio'):.3f}")
        P(f"  {s}: " + "  ".join(parts))
    P("")

    # real anchor
    real = _sel(rows, sim="real")
    if real:
        P("[REAL anchor: ruotato vs prova2, footprint, per line r (cv_ratio)]")
        rc = _cands(real)
        P("    " + f"{'line':6s}" + "".join(f"{c:>20s}" for c in rc))
        for el in ELEMENTS:
            P("    " + f"{el:6s}" + "".join(
                f"{_mean(_sel(real, element=el, candidate=c)):12.4f}"
                f" ({_mean(_sel(real, element=el, candidate=c), 'cv_ratio'):.3f})"
                for c in rc))
        P("    mean  " + "".join(
            f"{_mean(_sel(real, candidate=c)):12.4f}"
            f" ({_mean(_sel(real, candidate=c), 'cv_ratio'):.3f})" for c in rc))
    text = "\n".join(L)
    print(text)
    path = path or os.path.join(core.RESULTS_DIR, "wp3_regime_summary.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return text


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------

def make_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    rows = io_utils.read_rows(CSV)
    if not rows:
        raise NotImplementedError("run the grid first (no CSV)")
    sim = _sel(rows, sim="validated")
    holes = [f"{h}x{w}" for (h, w) in config.GRID["holes"]]
    hole_px = [h * w for (h, w) in config.GRID["holes"]]
    angles = list(config.GRID["angles"])
    doses = list(config.GRID["doses"])
    cands = _cands(sim)
    classical_c = [c for c in cands if c.startswith("classical_")
                   and not c.endswith("+net")]
    lines = list(config.FIG_LINES)

    def cell(a, hole, d, cand, region):
        rr = _sel(sim, angle=a, hole=hole, dose=d, candidate=cand,
                  region=region)
        rr = [r for r in rr if r["element"] in lines]
        return _mean(rr)

    # regime map: net minus max(det, best classical)
    maps = {}
    for d in doses:
        M = np.full((len(angles), len(holes)), np.nan)
        for i, a in enumerate(angles):
            for j, hole in enumerate(holes):
                region = "footprint" if hole == "0x0" else "hole"
                ref = max([cell(a, hole, d, "det", region)]
                          + [cell(a, hole, d, c, region) for c in classical_c
                             if hole != "0x0"])
                M[i, j] = cell(a, hole, d, "net", region) - ref
        maps[d] = M
    vmax = float(np.nanmax(np.abs(np.concatenate([m.ravel()
                                                  for m in maps.values()]))))
    vmax = max(vmax, 0.05)

    fig = plt.figure(figsize=(4.0 * len(doses) + 6.4, 3.6))
    gs = fig.add_gridspec(1, len(doses) + 2,
                          width_ratios=[1.0] * len(doses) + [1.15, 1.15],
                          wspace=0.45)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    heat_axes = []
    for k, d in enumerate(doses):
        ax = fig.add_subplot(gs[0, k])
        heat_axes.append(ax)
        M = maps[d]
        im = ax.imshow(M, cmap="RdBu_r", norm=norm, aspect="auto",
                       origin="lower")
        for i in range(len(angles)):
            for j in range(len(holes)):
                v = M[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                            fontsize=7.5,
                            color="white" if abs(v) > 0.6 * vmax else "black")
        ax.set_xticks(range(len(holes)))
        ax.set_xticklabels([str(p) for p in hole_px], fontsize=8)
        ax.set_yticks(range(len(angles)))
        ax.set_yticklabels([f"{a:g}" for a in angles], fontsize=8)
        ax.set_xlabel("hole area (px)")
        if k == 0:
            ax.set_ylabel("tilt angle (deg)")
        ax.set_title(f"dose {d:g}", fontsize=10)
    cbar = fig.colorbar(im, ax=heat_axes, fraction=0.025, pad=0.02)
    cbar.set_label("r(net) minus best non-learned", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # side panels: r vs hole area at 20 deg, dose 1, hole and footprint
    style = {"det": dict(color=ORANGE, ls="-", marker="o"),
             "net": dict(color=NAVY, ls="-", marker="o"),
             "ens_jitter": dict(color=NAVY, ls="--", marker="s"),
             "classical_biharmonic+net": dict(color=NAVY, ls=":", marker="^"),
             "classical_nearest": dict(color=GREY, ls="-", marker="v"),
             "classical_biharmonic": dict(color=GREY, ls="--", marker="D"),
             "classical_telea": dict(color=GREY, ls=":", marker="x"),
             "classical_ns": dict(color=GREY, ls="-.", marker="+")}
    for k, region in enumerate(("hole", "footprint")):
        ax = fig.add_subplot(gs[0, len(doses) + k])
        for c in cands:
            xs, ys = [], []
            for hole, px in zip(holes, hole_px):
                if region == "hole" and hole == "0x0":
                    continue
                v = cell(HARSH["angle"], hole, 1.0, c, region)
                if np.isfinite(v):
                    xs.append(px)
                    ys.append(v)
            if xs:
                st = style.get(c, dict(color=GREY, ls="-", marker="."))
                ax.plot(xs, ys, ms=4, lw=1.4, label=c, **st)
        ax.set_xlabel("hole area (px)")
        ax.set_ylabel(f"r vs truth, {region}")
        ax.set_title(f"{HARSH['angle']:g} deg, dose 1, {region}", fontsize=9)
        ax.grid(alpha=0.25)
        if k == 1:
            ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    out = io_utils.fig_path("wp3_regime_map.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved:", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.figures:
        make_figures()
    elif args.summary:
        summarize()
    else:
        run(quick=args.quick)
