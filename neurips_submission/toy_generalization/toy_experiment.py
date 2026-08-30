"""
toy_experiment.py

Main script for the synthetic, instrument-agnostic sanity check. Runs the
full toy recipe end to end:

  (a) train N jitter nets + N control nets
  (b) held-out ensemble spread / calibration check
  (c) blind diagnostic battery: null distribution + defect-naming trials
  (d) rejection ABC ground-truth recovery (SBC-style rank check)
  (e) save one summary figure and one summary text file

Everything here is fully synthetic: a made-up 32x32 "field" standing in for
a painting, a made-up 3-knob forward model standing in for the real
calibrated XRF simulator, and made-up "calibration sigmas" for those knobs
(see toy_forward.py docstring). No real instrument, dataset, or number from
the main pipeline is used or reproduced; the goal is only to check whether
the same STRUCTURE of result (ensembles, blind diagnostics, ABC) shows up
on an unrelated toy problem.

All inputs/outputs stay inside neurips_submission/toy_generalization/.
"""

import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toy_forward import (
    GRID, K0, GAIN_SCALE_SD, WARP_SHIFT_SD, NOISE_SCALE_LOGSD,
    SimKnobs, nominal_knobs, jitter_knobs, random_field, forward,
)
from toy_model import ResCNN, n_params, train_one_net, predict
from toy_diagnostics import (
    STAT_NAMES, build_null, compute_stats, zscores, whitened_distance, diagnose,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

MASTER_SEED = 12345
N_ENSEMBLE = 6
N_STEPS = 350
BATCH_SIZE = 10
N_HELDOUT = 12
N_NULL_PAIRS = 400
N_DIAG_TRIALS = 18
N_ABC_TRIALS = 15
N_ABC_DRAWS = 800
ABC_ACCEPT_FRAC = 0.10

DEFECTS = {
    "gain": SimKnobs(gain_scale=2.5, warp_shift=(0.0, 0.0), noise_scale=1.0),
    "warp": SimKnobs(gain_scale=1.0, warp_shift=(1.5, 1.5), noise_scale=1.0),
    "noise": SimKnobs(gain_scale=1.0, warp_shift=(0.0, 0.0), noise_scale=6.0),
}
FAMILIES = ["gain", "warp", "noise"]


def log(msg):
    print(msg, flush=True)


def step_a_train_ensembles():
    log("")
    log("=== Step (a): training jitter and control ensembles ===")
    log(f"ResCNN parameter count: {n_params(ResCNN())}")
    jitter_nets, jitter_knobs_used = [], []
    control_nets, control_knobs_used = [], []
    t0 = time.time()
    for seed in range(N_ENSEMBLE):
        t1 = time.time()
        net, knobs = train_one_net("jitter", seed, n_steps=N_STEPS, batch_size=BATCH_SIZE)
        jitter_nets.append(net)
        jitter_knobs_used.append(knobs)
        log(f"  jitter  net seed={seed} trained in {time.time()-t1:.1f}s  "
            f"knobs: gain={knobs.gain_scale:.3f} warp=({knobs.warp_shift[0]:.3f},{knobs.warp_shift[1]:.3f}) "
            f"noise={knobs.noise_scale:.3f}")
    for seed in range(N_ENSEMBLE):
        t1 = time.time()
        net, knobs = train_one_net("control", seed, n_steps=N_STEPS, batch_size=BATCH_SIZE)
        control_nets.append(net)
        control_knobs_used.append(knobs)
        log(f"  control net seed={seed} trained in {time.time()-t1:.1f}s  (nominal knobs)")
    log(f"Total training time: {time.time()-t0:.1f}s")
    return jitter_nets, control_nets


def step_b_heldout_eval(jitter_nets, control_nets, rng):
    log("")
    log("=== Step (b): held-out ensemble spread and calibration ===")
    errs_j, stds_j, errs_c, stds_c = [], [], [], []
    examples = []
    for i in range(N_HELDOUT):
        field = random_field(rng)
        knobs_true = jitter_knobs(rng)  # unknown, calibration-uncertain "real" instrument state
        obs = forward(field, knobs_true, rng)

        preds_j = np.stack([predict(net, obs) for net in jitter_nets])
        preds_c = np.stack([predict(net, obs) for net in control_nets])
        mean_j, std_j = preds_j.mean(axis=0), preds_j.std(axis=0)
        mean_c, std_c = preds_c.mean(axis=0), preds_c.std(axis=0)

        errs_j.append(np.abs(mean_j - field).ravel())
        stds_j.append(std_j.ravel())
        errs_c.append(np.abs(mean_c - field).ravel())
        stds_c.append(std_c.ravel())

        if i < 2:
            examples.append({"field": field, "obs": obs, "mean_j": mean_j, "mean_c": mean_c})

    errs_j = np.concatenate(errs_j); stds_j = np.concatenate(stds_j)
    errs_c = np.concatenate(errs_c); stds_c = np.concatenate(stds_c)

    rms_std_j = float(np.sqrt(np.mean(stds_j ** 2)))
    rms_std_c = float(np.sqrt(np.mean(stds_c ** 2)))
    spread_ratio = rms_std_j / rms_std_c

    cov = {}
    for z in (1, 2):
        cov[("jitter", z)] = float(np.mean(errs_j <= z * stds_j))
        cov[("control", z)] = float(np.mean(errs_c <= z * stds_c))

    log(f"held-out fields evaluated: {N_HELDOUT}")
    log(f"rms std, jitter ensemble : {rms_std_j:.5f}")
    log(f"rms std, control ensemble: {rms_std_c:.5f}")
    log(f"spread ratio (jitter/control): {spread_ratio:.3f}")
    for z in (1, 2):
        log(f"coverage z={z}: jitter={cov[('jitter', z)]:.3f}  control={cov[('control', z)]:.3f}  "
            f"(nominal Gaussian target: {2*0.8413-1 if z==1 else 2*0.9772-1:.3f})")

    return {
        "rms_std_jitter": rms_std_j,
        "rms_std_control": rms_std_c,
        "spread_ratio": spread_ratio,
        "coverage": cov,
        "examples": examples,
    }


def step_c_diagnostics(rng):
    log("")
    log("=== Step (c): blind diagnostic battery ===")
    log(f"building calibration-uncertainty-aware null from {N_NULL_PAIRS} paired draws...")
    null = build_null(rng, n_pairs=N_NULL_PAIRS)
    for name in STAT_NAMES:
        log(f"  null[{name}]: mean={null[name]['mean']:.4f} std={null[name]['std']:.4f}")

    confusion = {fam: {f2: 0 for f2 in FAMILIES + ["none"]} for fam in FAMILIES}
    correct = 0
    trial_log = []
    for trial in range(N_DIAG_TRIALS):
        family = FAMILIES[trial % 3]
        knobs = DEFECTS[family]
        field = random_field(rng)
        meas = forward(field, knobs, rng)
        sim = forward(field, nominal_knobs(), rng)
        stats = compute_stats(meas, sim)
        diagnosed, best_stat, best_z = diagnose(stats, null)
        confusion[family][diagnosed] += 1
        ok = diagnosed == family
        correct += int(ok)
        trial_log.append((trial, family, diagnosed, best_stat, best_z, ok))

    log(f"defect-naming trials: {N_DIAG_TRIALS}  (cycling gain / warp / noise defects)")
    for t, family, diagnosed, best_stat, best_z, ok in trial_log:
        log(f"  trial {t:2d}: true={family:5s} -> diagnosed={diagnosed:5s} "
            f"(via {best_stat}, |z|={best_z:.2f})  {'OK' if ok else 'WRONG'}")
    log(f"confusion matrix (rows=true family, cols=diagnosed):")
    header = "true\\diag  " + " ".join(f"{f:>7s}" for f in FAMILIES + ["none"])
    log("  " + header)
    for fam in FAMILIES:
        row = " ".join(f"{confusion[fam][f2]:7d}" for f2 in FAMILIES + ["none"])
        log(f"  {fam:9s} {row}")
    accuracy = correct / N_DIAG_TRIALS
    log(f"overall defect-naming accuracy: {correct}/{N_DIAG_TRIALS} = {accuracy:.3f}")

    return {"null": null, "confusion": confusion, "accuracy": accuracy, "trial_log": trial_log}


def step_d_abc(null, rng):
    log("")
    log("=== Step (d): rejection ABC ground-truth recovery ===")
    dim_names = ["gain_scale", "warp_dy", "warp_dx", "noise_scale"]
    ranks_pooled = []
    ranks_by_dim = {name: [] for name in dim_names}
    n_accept = max(1, int(round(N_ABC_DRAWS * ABC_ACCEPT_FRAC)))
    log(f"trials={N_ABC_TRIALS}  prior draws/trial={N_ABC_DRAWS}  accepted/trial={n_accept} "
        f"({ABC_ACCEPT_FRAC*100:.0f}%)")

    for trial in range(N_ABC_TRIALS):
        theta_true = jitter_knobs(rng)
        field = random_field(rng)
        measurement = forward(field, theta_true, rng)

        cand = {name: np.empty(N_ABC_DRAWS) for name in dim_names}
        dist = np.empty(N_ABC_DRAWS)
        for i in range(N_ABC_DRAWS):
            k = jitter_knobs(rng)
            sim = forward(field, k, rng)
            stats = compute_stats(measurement, sim)
            dist[i] = whitened_distance(stats, null)
            cand["gain_scale"][i] = k.gain_scale
            cand["warp_dy"][i] = k.warp_shift[0]
            cand["warp_dx"][i] = k.warp_shift[1]
            cand["noise_scale"][i] = k.noise_scale

        order = np.argsort(dist)
        accept_idx = order[:n_accept]

        true_vals = {
            "gain_scale": theta_true.gain_scale,
            "warp_dy": theta_true.warp_shift[0],
            "warp_dx": theta_true.warp_shift[1],
            "noise_scale": theta_true.noise_scale,
        }

        per_dim_ranks = {}
        for name in dim_names:
            accepted_vals = cand[name][accept_idx]
            rank = np.sum(accepted_vals <= true_vals[name])
            normalized_rank = rank / len(accepted_vals)
            ranks_pooled.append(normalized_rank)
            ranks_by_dim[name].append(normalized_rank)
            per_dim_ranks[name] = normalized_rank

        log(f"  trial {trial:2d}: true gain={theta_true.gain_scale:.3f} "
            f"warp=({theta_true.warp_shift[0]:.3f},{theta_true.warp_shift[1]:.3f}) "
            f"noise={theta_true.noise_scale:.3f}  "
            f"ranks: " + " ".join(f"{n}={v:.2f}" for n, v in per_dim_ranks.items()))

    ranks_pooled = np.array(ranks_pooled)
    mean_rank = float(np.mean(ranks_pooled))
    log(f"pooled normalized ranks (n={len(ranks_pooled)}): mean={mean_rank:.3f} "
        f"(near 0.5 => calibrated posterior)")
    for name in dim_names:
        log(f"  mean normalized rank, {name}: {np.mean(ranks_by_dim[name]):.3f}")

    return {"ranks_pooled": ranks_pooled, "ranks_by_dim": ranks_by_dim, "mean_rank": mean_rank}


def step_e_outputs(b_res, c_res, d_res):
    log("")
    log("=== Step (e): saving figure and text summary ===")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(9, 8))

    ex = b_res["examples"][0]
    vmin = min(ex["field"].min(), ex["obs"].min())
    vmax = max(ex["field"].max(), ex["obs"].max())
    im0 = axes[0, 0].imshow(ex["field"], vmin=vmin, vmax=vmax, cmap="viridis")
    axes[0, 0].set_title("example: true field")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    im1 = axes[0, 1].imshow(ex["obs"], vmin=vmin, vmax=vmax, cmap="viridis")
    axes[0, 1].set_title("example: degraded observation")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    z_vals = [1, 2]
    nominal = [2 * 0.8413447 - 1, 2 * 0.9772499 - 1]
    jitter_cov = [b_res["coverage"][("jitter", z)] for z in z_vals]
    control_cov = [b_res["coverage"][("control", z)] for z in z_vals]
    x = np.arange(len(z_vals))
    w = 0.25
    axes[1, 0].bar(x - w, nominal, width=w, label="nominal Gaussian", color="gray")
    axes[1, 0].bar(x, jitter_cov, width=w, label="jitter ensemble", color="tab:blue")
    axes[1, 0].bar(x + w, control_cov, width=w, label="control ensemble", color="tab:orange")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([f"z={z}" for z in z_vals])
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_ylabel("empirical coverage")
    axes[1, 0].set_title("coverage / calibration check")
    axes[1, 0].legend(fontsize=8)

    bins = np.linspace(0, 1, 11)
    axes[1, 1].hist(d_res["ranks_pooled"], bins=bins, color="tab:green", edgecolor="black", alpha=0.8)
    expected = len(d_res["ranks_pooled"]) / (len(bins) - 1)
    axes[1, 1].axhline(expected, color="black", linestyle="--", label="uniform expectation")
    axes[1, 1].set_title(f"ABC SBC-style rank histogram (mean={d_res['mean_rank']:.2f})")
    axes[1, 1].set_xlabel("normalized rank of true knob value")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("Toy generalization check: synthetic non-XRF sanity test", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = os.path.join(RESULTS_DIR, "toy_summary.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    log(f"saved figure: {fig_path}")

    txt_path = os.path.join(RESULTS_DIR, "toy_summary.txt")
    lines = []
    lines.append("Toy generalization check: synthetic, non-X-ray, non-painting sanity test")
    lines.append("=" * 78)
    lines.append("")
    lines.append("PURPOSE")
    lines.append("-" * 78)
    lines.append("This is a fully synthetic, self-contained re-run of the same four-part")
    lines.append("recipe (jitter-vs-control deep ensembles, a blind diagnostic battery, and")
    lines.append("rejection ABC) used in the main pipeline, but on an unrelated toy problem:")
    lines.append("32x32 fields made of random 2D Gaussian blobs plus smooth low-frequency")
    lines.append("noise, degraded by a made-up 3-knob forward model (multiplicative gain,")
    lines.append("sub-pixel warp shift, Poisson-like noise level). No real instrument or")
    lines.append("painting data is used anywhere in this script.")
    lines.append("")
    lines.append("SYNTHETIC CALIBRATION SIGMAS USED (arbitrary, chosen by the author, not")
    lines.append("measured from any real instrument):")
    lines.append(f"  gain_scale:  multiplicative, nominal 1.0, log-sd {GAIN_SCALE_SD:.3f}")
    lines.append(f"  warp_shift:  (dy,dx) pixels, nominal (0,0), sd {WARP_SHIFT_SD:.3f} px/axis")
    lines.append(f"  noise_scale: multiplicative on Var = noise_scale * {K0} * clip(signal,0,None),")
    lines.append(f"               nominal 1.0, log-sd {NOISE_SCALE_LOGSD:.3f}")
    lines.append("")

    lines.append("STEP (b): HELD-OUT ENSEMBLE SPREAD AND CALIBRATION")
    lines.append("-" * 78)
    lines.append(f"held-out fields evaluated: {N_HELDOUT}")
    lines.append(f"ensemble size (each of jitter / control): {N_ENSEMBLE}")
    lines.append(f"rms std, jitter ensemble : {b_res['rms_std_jitter']:.5f}")
    lines.append(f"rms std, control ensemble: {b_res['rms_std_control']:.5f}")
    lines.append(f"spread ratio (jitter/control): {b_res['spread_ratio']:.3f}")
    for z in (1, 2):
        nom = nominal[z - 1]
        lines.append(f"coverage at z={z} (target ~{nom:.3f} if Gaussian and correctly scaled):")
        lines.append(f"    jitter ensemble : {b_res['coverage'][('jitter', z)]:.3f}")
        lines.append(f"    control ensemble: {b_res['coverage'][('control', z)]:.3f}")
    lines.append("")

    lines.append("STEP (c): BLIND DIAGNOSTIC BATTERY")
    lines.append("-" * 78)
    lines.append(f"null distribution built from {N_NULL_PAIRS} pairs, BOTH sides simulated with")
    lines.append("knobs independently drawn within calibration uncertainty of nominal (never")
    lines.append("a noise-free vs noisy comparison):")
    for name in STAT_NAMES:
        nm = c_res["null"][name]
        lines.append(f"    {name:14s} mean={nm['mean']:.4f}  std={nm['std']:.4f}")
    lines.append(f"defect-naming trials: {N_DIAG_TRIALS} (cycling gain / warp / noise, one large")
    lines.append("out-of-calibration defect injected per trial: gain_scale=2.5, warp_shift=")
    lines.append("(1.5,1.5) px, or noise_scale=6.0)")
    lines.append("confusion matrix (rows = true defect family, cols = diagnosed family):")
    header = "  true\\diag  " + " ".join(f"{f:>7s}" for f in FAMILIES + ["none"])
    lines.append(header)
    for fam in FAMILIES:
        row = " ".join(f"{c_res['confusion'][fam][f2]:7d}" for f2 in FAMILIES + ["none"])
        lines.append(f"  {fam:9s}  {row}")
    lines.append(f"overall defect-naming accuracy: "
                  f"{int(round(c_res['accuracy']*N_DIAG_TRIALS))}/{N_DIAG_TRIALS} = {c_res['accuracy']:.3f}")
    lines.append("")

    lines.append("STEP (d): REJECTION ABC GROUND-TRUTH RECOVERY (SBC-STYLE)")
    lines.append("-" * 78)
    lines.append(f"trials: {N_ABC_TRIALS}, prior draws/trial: {N_ABC_DRAWS}, "
                  f"accept fraction: {ABC_ACCEPT_FRAC*100:.0f}%")
    lines.append("true knob vectors drawn from the same calibration-jitter prior used to")
    lines.append("train the jitter ensemble; distance = whitened RMS-z over the 3 blind")
    lines.append("diagnostic statistics, using the null model from step (c).")
    lines.append(f"pooled normalized rank of true value within accepted posterior draws")
    lines.append(f"(n={len(d_res['ranks_pooled'])} = {N_ABC_TRIALS} trials x 4 knob dimensions):")
    lines.append(f"    mean = {d_res['mean_rank']:.3f}  (0.5 = calibrated if uniform)")
    for name, vals in d_res["ranks_by_dim"].items():
        lines.append(f"    mean rank, {name:12s}: {np.mean(vals):.3f}")
    lines.append("")

    lines.append("FILES")
    lines.append("-" * 78)
    lines.append("toy_summary.png : example field/observation, coverage bars, rank histogram")
    lines.append("toy_summary.txt : this file")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log(f"saved text summary: {txt_path}")


def main():
    t_start = time.time()
    log("Toy generalization check starting.")
    log(f"grid={GRID}x{GRID}  ensemble size={N_ENSEMBLE}  steps/net={N_STEPS}  batch={BATCH_SIZE}")

    jitter_nets, control_nets = step_a_train_ensembles()

    rng_b = np.random.default_rng(MASTER_SEED + 1)
    b_res = step_b_heldout_eval(jitter_nets, control_nets, rng_b)

    rng_cd = np.random.default_rng(MASTER_SEED + 2)
    c_res = step_c_diagnostics(rng_cd)
    d_res = step_d_abc(c_res["null"], rng_cd)

    step_e_outputs(b_res, c_res, d_res)

    log("")
    log(f"Done in {time.time()-t_start:.1f}s total.")


if __name__ == "__main__":
    main()
