"""main.py - the pipeline that ties all three workpackages together.

    python neurips_submission/main.py --stage smoke     day-1 shared-infra check (~2 min)
    python neurips_submission/main.py --stage status    what exists / what is missing
    python neurips_submission/main.py --stage wp1|wp2|wp3 [--quick]
    python neurips_submission/main.py --stage figures   assemble all paper figures
    python neurips_submission/main.py --stage all [--quick]

Run from the REPO ROOT.  Every stage is restartable: experiments write
CSVs incrementally and figures are rebuilt from CSVs only.
"""

import argparse
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from common import core, io_utils  # noqa: F401  (core bootstraps paths)

# stage -> (module, owner) ; each module exposes run(quick) and
# optionally make_figures()
STAGES = {
    "wp1": [("wp1_uq_ensemble.exp_ensemble_uq", "Dimitrije"),
            ("wp1_uq_ensemble.exp_adaptive_scan", "Dimitrije (stretch)")],
    "wp2": [("wp2_simulator_audit.exp_defect_tolerance", "WP2 owner"),
            ("wp2_simulator_audit.exp_diagnostics", "WP2 owner")],
    "wp3": [("wp3_degradation_benchmark.exp_degradation_grid", "WP3 owner")],
    "wp4": [("wp4_closed_loop.exp_simulator_posterior", "Dimitrije")],
}

EXPECTED_CSVS = {
    "wp1_ensemble_members": "WP1", "wp1_uq_coverage": "WP1",
    "wp1_uq_diagnostics": "WP1", "wp1_uq_accuracy": "WP1",
    "wp1_adaptive_scan": "WP1 (stretch)",
    "wp2_defect_tolerance": "WP2", "wp2_diagnostics": "WP2",
    "wp2_diag_confusion": "WP2", "wp3_degradation_grid": "WP3",
    "wp4_abc_draws": "WP4", "wp4_posterior_coverage": "WP4",
}


def run_stage(stage: str, quick: bool):
    for modname, owner in STAGES[stage]:
        print(f"\n=== {modname}  (owner: {owner}) ===")
        mod = importlib.import_module(modname)
        try:
            mod.run(quick=quick)
        except NotImplementedError as e:
            print(f"SKIPPED - not implemented yet: {e}")


def smoke():
    """Shared-infrastructure check every team member runs on day 1."""
    import numpy as np
    from common import classical, perturb, restore, training

    print("[1/5] nominal sample from datagen ...")
    x, y, lm, vm, _ = core.dg.sample(np.random.default_rng(0))
    assert x.shape == (10, 60, 120) and y.shape == (8, 60, 120)

    print("[2/5] perturbed sample (bilinear-blur defect) ...")
    knobs = perturb.SimKnobs(blur_mode="bilinear", noise_k_scale=2.0,
                             label="smoke")
    x2, *_ = perturb.sample(np.random.default_rng(0), knobs=knobs)
    assert x2.shape == x.shape and not np.allclose(x, x2)

    print("[3/5] tiny training run (30 steps) ...")
    net, hist = training.train_net(
        training.make_batch_fn(knobs=knobs),
        dict(config.QUICK_TRAIN, steps=30, val_every=15), seed=1,
        verbose=False)
    print(f"      val L1 {hist['best_val_l1']:.4f} in {hist['wall_s']:.0f} s")

    print("[4/5] degrade -> restore -> score (+ classical controls) ...")
    case = restore.degrade(source="prova2", angle=20.0,
                           block=restore.centered_block(14, 20), seed=0)
    det, learned = restore.apply_network(net, case["tilted"], case["angle"],
                                         validity=case["validity"])
    cands = {"det": det, "net": learned}
    cands.update(classical.classical_restorations(
        case["tilted"], case["v_tilt"], case["angle"]))
    rows = restore.score_candidates(
        cands, case["truth"], {"footprint": case["fp"],
                               "hole": case["hole"]})
    ca = {r["candidate"]: r for r in rows
          if r["element"] == "Ca" and r["region"] == "hole"}
    for name, r in ca.items():
        print(f"      Ca hole r: {name:22s} {float(r['r']):+.3f}")

    print("[5/5] MVP checkpoint ...")
    print("      found" if restore.load_mvp_net() is not None else
          "      MISSING - WP3 needs neurips-restore/scripts/03 run first")
    print("\nSMOKE PASSED - your environment is ready.")


def verify():
    """Contract checks for team rewrites of the physics/ML code.

    Run this after ANY rewrite of common/perturb.py, common/training.py
    or a model swap - green here means your version is safe to use.
    """
    import numpy as np
    import torch
    from common import classical, perturb, training

    print("[1/4] physics identity: forward_perturbed(NOMINAL) == "
          "datagen.forward_sharp ...")
    p1 = core.dg.prova1_stack()
    maps = {el: np.ascontiguousarray(p1[i])
            for i, el in enumerate(core.ELEMENTS)}
    a = perturb.forward_perturbed(maps, 12.0, np.random.default_rng(5),
                                  perturb.NOMINAL)
    b = core.dg.forward_sharp(maps, 12.0, np.random.default_rng(5))
    d = max(float(np.abs(a[el] - b[el]).max()) for el in core.ELEMENTS)
    assert d == 0.0, f"nominal simulator diverged from forward_sharp: {d}"
    print(f"      OK (max diff {d})")

    print("[2/4] knob monotonicity: noise_k_scale raises the noise ...")
    lo = perturb.forward_perturbed(maps, 12.0, np.random.default_rng(3),
                                   perturb.SimKnobs(noise_k_scale=0.25))
    hi = perturb.forward_perturbed(maps, 12.0, np.random.default_rng(3),
                                   perturb.SimKnobs(noise_k_scale=4.0))
    r_lo = np.mean([np.std(np.diff(lo[el], axis=1)) for el in core.ELEMENTS])
    r_hi = np.mean([np.std(np.diff(hi[el], axis=1)) for el in core.ELEMENTS])
    assert r_hi > r_lo, "higher noise_k_scale must give rougher maps"
    print(f"      OK (roughness {r_lo:.3f} -> {r_hi:.3f})")

    print("[3/4] trainer contract: shapes, zero-init baseline, learning ...")
    bf = training.make_batch_fn()
    x, y, lm, vm = bf(np.random.default_rng(0), 2)
    assert x.shape == (2, 10, 60, 120) and y.shape == (2, 8, 60, 120)
    assert lm.dtype == bool and vm.dtype == bool
    net0 = core.RestorationUNet()
    with torch.no_grad():
        res = net0(torch.from_numpy(x))
    assert float(res.abs().max()) == 0.0, \
        "fresh net must equal the physics baseline (zero-init head)"
    net, hist = training.train_net(
        bf, dict(config.QUICK_TRAIN, steps=30, val_every=15),
        seed=2, verbose=False)
    assert np.isfinite(hist["best_val_l1"])
    print(f"      OK (30-step val L1 {hist['best_val_l1']:.4f})")

    print("[4/4] classical fills touch ONLY the hole ...")
    tilted = {el: maps[el][:45, :80].copy() for el in core.ELEMENTS}
    v = np.ones(core.TILTED_SHAPE)
    v[10:20, 30:44] = 0.0
    for name, fill in classical.CANDIDATES.items():
        f = fill(tilted, v)
        for el in core.ELEMENTS:
            outside = np.abs(f[el] - tilted[el])[v > 0.5].max()
            assert outside < 1e-9, f"{name} modified valid pixels of {el}"
    print("      OK")
    print("\nVERIFY PASSED - contracts intact.")


def status():
    print(f"{'results CSV':32s} {'WP':4s} {'rows':>6s}")
    print("-" * 46)
    for name, wp in EXPECTED_CSVS.items():
        rows = io_utils.read_rows(name)
        state = f"{len(rows):6d}" if rows else "  MISSING"
        print(f"{name:32s} {wp:4s} {state}")
    figs = (os.listdir(core.FIGURES_DIR)
            if os.path.isdir(core.FIGURES_DIR) else [])
    print(f"\nfigures/: {', '.join(sorted(figs)) if figs else '(empty)'}")


def figures():
    for stage, mods in STAGES.items():
        for modname, owner in mods:
            mod = importlib.import_module(modname)
            if not hasattr(mod, "make_figures"):
                continue
            print(f"figures: {modname}")
            try:
                mod.make_figures()
            except NotImplementedError as e:
                print(f"  SKIPPED - {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["smoke", "verify", "status", "wp1", "wp2",
                             "wp3", "wp4", "figures", "all"])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.stage == "smoke":
        smoke()
    elif args.stage == "verify":
        verify()
    elif args.stage == "status":
        status()
    elif args.stage == "figures":
        figures()
    elif args.stage == "all":
        for s in ("wp1", "wp2", "wp3", "wp4"):
            run_stage(s, args.quick)
        figures()
        status()
    else:
        run_stage(args.stage, args.quick)


if __name__ == "__main__":
    main()
