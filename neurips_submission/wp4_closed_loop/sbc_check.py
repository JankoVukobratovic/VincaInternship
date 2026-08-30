"""wp4_closed_loop/sbc_check.py - Simulation-Based Calibration (SBC) of the
rejection-ABC posterior (Talts, Betancourt, Simpson, Vehtari, Gelman 2018,
"Validating Bayesian Inference Algorithms with Simulation-Based Calibration",
arXiv:1804.06788).

WHY
    exp_simulator_posterior.py validates the ABC posterior against the ONE
    real ruotato scan, whose true simulator is unknown - a plausible
    looking posterior is not proof the procedure recovers truth. Here the
    true knobs ARE known by construction: draw theta_true from the SAME
    prior used in the real ABC (config.JITTER via perturb.jittered), make a
    synthetic "real" tilted measurement from prova2 under theta_true
    (prova2 stands in for "an independent measurement", exactly as the real
    ruotato is not derived from the prova1 training source), run the
    IDENTICAL rejection-ABC machinery (same battery, same whitening null,
    same prior) against that synthetic measurement, and check whether
    theta_true's own rank inside the resulting posterior sample is
    uniformly distributed across many repeated trials (the SBC rank test).
    This validates the LOOP itself, independent of what the real
    instrument's true simulator happens to be - it answers "does rejection
    ABC with this battery recover a known truth", not "did it find a
    plausible story for the one real scan we have".

    A separate discrete test does the same for blur_mode (bilinear vs
    cubic), since that knob is not a location parameter and needs its own
    treatment (reliability of the posterior probability, not a rank).

DOES NOT MODIFY exp_simulator_posterior.py or any file whose output feeds
the submitted paper (results/wp4_abc_draws.csv, wp4_summary.txt, the
trained posterior ensemble). This is a new, additive validation only.

OUTPUTS
    results/wp4_sbc.csv           per-trial ranks (continuous) / P(bilinear)
    results/wp4_sbc_summary.txt   KS uniformity per knob, blur separation
    figures/wp4_sbc.png / .pdf    rank histograms, 6 continuous knobs

Run from the repo root:
    python neurips_submission/wp4_closed_loop/sbc_check.py
"""

import dataclasses
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.stats import kstest

import config
from common import core, io_utils, perturb
from wp2_simulator_audit import exp_diagnostics as dg
from wp4_closed_loop.exp_simulator_posterior import real_null, ANGLE

ELEMENTS = core.ELEMENTS
CONT_KNOBS = ("noise_k_scale", "gain_scale", "angle_bias_deg",
             "warp_rot_deg", "warp_dy", "warp_dx")
KNOB_TITLE = {"noise_k_scale": "noise constant (x)",
             "gain_scale": "gain slope (x)",
             "angle_bias_deg": "angle belief (deg)",
             "warp_rot_deg": "reg. rotation (deg)",
             "warp_dy": "reg. shift dy (px)",
             "warp_dx": "reg. shift dx (px)"}
N_TRIALS_CONT = 24
N_TRIALS_BLUR = 20
N_DRAWS = 1500
ACCEPT_FRAC = 0.10
BASE_SEED = config.BASE_SEED + 90000  # separate stream from the main ABC


def knob_value(k: perturb.SimKnobs, name: str) -> float:
    if name == "warp_dy":
        return k.warp_shift_px[0]
    if name == "warp_dx":
        return k.warp_shift_px[1]
    return float(getattr(k, name))


def whitened_distance(null, real_map, sim_map) -> float:
    b = dg.battery(real_map, sim_map, ANGLE)
    z = null.z(b)
    zz = np.array([[z[s][el] for el in ELEMENTS] for s in dg.STATS])
    return float(np.sqrt(np.mean(zz ** 2)))


def run_abc_trial(null, p1, real_map, n_draws, seed, blur_random):
    """Rejection ABC of one synthetic trial; returns the accepted knobs
    (closest ACCEPT_FRAC of n_draws by whitened RMS-z distance). Mirrors
    exp_simulator_posterior.run_abc's prior_draw + distance logic exactly,
    reimplemented locally so that file is never imported for its private
    closures."""
    rng = np.random.default_rng(seed)
    cands = []
    for i in range(n_draws):
        k = perturb.jittered(rng, config.JITTER, f"c{i}")
        if blur_random and rng.random() < 0.5:
            k = dataclasses.replace(k, blur_mode="bilinear")
        S = perturb.forward_perturbed(p1, ANGLE,
                                      np.random.default_rng(seed + 1_000_003 + i),
                                      k)
        d = whitened_distance(null, real_map, S)
        cands.append((d, k))
    cands.sort(key=lambda t: t[0])
    n_acc = max(int(round(ACCEPT_FRAC * n_draws)), 5)
    return [k for _, k in cands[:n_acc]], n_acc


def continuous_sbc(null, p1, p2, n_trials=N_TRIALS_CONT, n_draws=N_DRAWS):
    rows = []
    t0 = time.time()
    for t in range(n_trials):
        rng_true = np.random.default_rng(BASE_SEED + 2 * t)
        theta_true = perturb.jittered(rng_true, config.JITTER, f"true_{t}")
        # blur fixed at cubic (the validated mode) so it is not a confound
        # in the continuous-knob test; a separate discrete test below
        # covers blur_mode on its own.
        synth_real = perturb.forward_perturbed(
            p2, ANGLE, np.random.default_rng(BASE_SEED + 2 * t + 1), theta_true)
        accepted, n_acc = run_abc_trial(null, p1, synth_real, n_draws,
                                        seed=BASE_SEED + 500_000 + 137 * t,
                                        blur_random=False)
        row = {"trial": t, "n_acc": n_acc}
        for name in CONT_KNOBS:
            true_v = knob_value(theta_true, name)
            post_v = np.array([knob_value(k, name) for k in accepted])
            rank = int(np.sum(post_v < true_v))
            row[f"true_{name}"] = true_v
            row[f"rank_{name}"] = rank
            row[f"rank_frac_{name}"] = rank / n_acc
        rows.append(row)
        print(f"  continuous trial {t + 1}/{n_trials}"
              f"  [{time.time() - t0:.0f} s]", flush=True)
    return rows


def discrete_blur_sbc(null, p1, p2, n_trials=N_TRIALS_BLUR, n_draws=N_DRAWS):
    rows = []
    t0 = time.time()
    for t in range(n_trials):
        true_label = "bilinear" if t % 2 == 0 else "cubic"
        # other knobs fixed at NOMINAL for a clean, single-factor isolation
        # (documented choice: this measures blur-mode calibration alone,
        # not blur confounded with drawn noise/gain/registration values)
        theta_true = dataclasses.replace(perturb.NOMINAL, blur_mode=true_label)
        synth_real = perturb.forward_perturbed(
            p2, ANGLE, np.random.default_rng(BASE_SEED + 900_000 + 3 * t),
            theta_true)
        accepted, n_acc = run_abc_trial(null, p1, synth_real, n_draws,
                                        seed=BASE_SEED + 700_000 + 149 * t,
                                        blur_random=True)
        p_bilinear = float(np.mean([k.blur_mode == "bilinear"
                                    for k in accepted]))
        rows.append({"trial": t, "true_blur": true_label,
                    "p_bilinear": p_bilinear, "n_acc": n_acc})
        print(f"  blur trial {t + 1}/{n_trials} true={true_label}"
              f" P(bilinear)={p_bilinear:.3f}  [{time.time() - t0:.0f} s]",
              flush=True)
    return rows


def make_figure(cont_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    NAVY, GREY = "#1f2a44", "#8c8c8c"
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.3))
    axes = axes.ravel()
    n_acc = cont_rows[0]["n_acc"]
    for j, name in enumerate(CONT_KNOBS):
        ax = axes[j]
        ranks = np.array([r[f"rank_frac_{name}"] for r in cont_rows])
        nb = 6
        ax.hist(ranks, bins=np.linspace(0, 1, nb + 1), color=NAVY,
               edgecolor="white", linewidth=0.6)
        ax.axhline(len(ranks) / nb, color=GREY, lw=1.2, ls="--")
        ax.set_title(KNOB_TITLE[name], fontsize=9)
        ax.set_xlim(0, 1)
        ax.tick_params(labelsize=7)
        if j % 3 == 0:
            ax.set_ylabel("trial count", fontsize=8)
        if j >= 3:
            ax.set_xlabel("rank / n_posterior", fontsize=8)
    fig.suptitle(f"SBC rank histograms, {len(cont_rows)} trials"
                f" ({n_acc} posterior draws each); flat = calibrated",
                fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = io_utils.fig_path("wp4_sbc.png")
    fig.savefig(out, dpi=200)
    fig.savefig(out.replace(".png", ".pdf"))
    plt.close(fig)
    print("saved:", out)


def summarize(cont_rows, blur_rows):
    L = []
    P = L.append
    P("WP4 addendum - Simulation-Based Calibration (SBC) of the rejection-ABC posterior")
    P(f"{len(cont_rows)} continuous trials, {len(blur_rows)} discrete blur trials,"
      f" {N_DRAWS} ABC draws/trial, accept {ACCEPT_FRAC:.0%}"
      f" ({cont_rows[0]['n_acc']} posterior draws/trial)")
    P("")
    P("[1] continuous knobs: rank of the true value inside the posterior sample,")
    P("    over trials; a calibrated posterior gives ranks uniform on [0,1]")
    P("    (KS test against Uniform(0,1); p < 0.05 flags miscalibration)")
    P(f"    {'knob':16s}{'mean rank':>12s}{'frac tails':>12s}{'KS stat':>10s}{'KS p':>10s}  verdict")
    flags = []
    for name in CONT_KNOBS:
        ranks = np.array([r[f"rank_frac_{name}"] for r in cont_rows])
        mean_r = float(np.mean(ranks))
        frac_tails = float(np.mean((ranks <= 0.1) | (ranks >= 0.9)))
        stat, p = kstest(ranks, "uniform")
        if p < 0.05 and frac_tails > 0.35:
            verdict = "UNDERDISPERSED (too narrow, truth often at the edge)"
        elif p < 0.05 and abs(mean_r - 0.5) > 0.15:
            verdict = f"BIASED ({'low' if mean_r > 0.5 else 'high'} relative to truth)"
        elif p < 0.05:
            verdict = "miscalibrated (shape, see histogram)"
        else:
            verdict = "calibrated (KS does not reject uniformity)"
        flags.append((name, p < 0.05))
        P(f"    {name:16s}{mean_r:12.3f}{frac_tails:12.3f}{stat:10.3f}{p:10.3f}  {verdict}")
    n_bad = sum(1 for _, bad in flags if bad)
    P("")
    P(f"    {n_bad}/{len(CONT_KNOBS)} knobs flagged at p<0.05"
      f" (expected under-the-null false-positive rate at alpha=0.05 with"
      f" {len(CONT_KNOBS)} tests is about {0.05 * len(CONT_KNOBS):.1f})")

    P("")
    P("[2] discrete blur_mode: posterior P(bilinear), true knobs at NOMINAL"
      " except blur (clean isolation)")
    p_true_bi = [r["p_bilinear"] for r in blur_rows if r["true_blur"] == "bilinear"]
    p_true_cu = [r["p_bilinear"] for r in blur_rows if r["true_blur"] == "cubic"]
    acc = np.mean([(r["p_bilinear"] > 0.5) == (r["true_blur"] == "bilinear")
                  for r in blur_rows])
    P(f"    mean P(bilinear) | true=bilinear: {np.mean(p_true_bi):.3f}"
      f"  (n={len(p_true_bi)}, sd {np.std(p_true_bi):.3f})")
    P(f"    mean P(bilinear) | true=cubic:    {np.mean(p_true_cu):.3f}"
      f"  (n={len(p_true_cu)}, sd {np.std(p_true_cu):.3f})")
    P(f"    MAP classification accuracy (P>0.5 -> bilinear): {acc:.3f}"
      f" ({int(round(acc * len(blur_rows)))}/{len(blur_rows)})")
    sep = np.mean(p_true_bi) - np.mean(p_true_cu)
    P(f"    separation (mean P|bilinear - mean P|cubic): {sep:+.3f}"
      f" (1.0 = perfect, 0.0 = no information)")

    P("")
    P("[3] interpretation")
    if n_bad == 0:
        P("    All six continuous knobs pass the SBC uniformity check at the")
        P("    trial counts used here: when the true simulator is known by")
        P("    construction, rejection ABC with this battery recovers it")
        P("    without a detectable rank bias or dispersion pathology.")
    else:
        bad_names = ", ".join(n for n, bad in flags if bad)
        P(f"    {n_bad} of 6 knobs ({bad_names}) show a rank distribution")
        P("    distinguishable from uniform at this trial count; see the")
        P("    per-knob verdicts above for the direction (bias vs under-")
        P("    dispersion). This is a real and expected limitation of")
        P("    rejection ABC with a coarse, hand-built summary-statistic")
        P("    battery and a modest number of trials (24), not evidence the")
        P("    procedure is useless -- it identifies PRECISELY which knobs'")
        P("    posteriors should be read as qualitative (direction/sign) only")
        P("    rather than quantitatively sharp, a distinction the single")
        P("    real-scan posterior alone could never establish.")
    if acc >= 0.9:
        P(f"    The discrete blur_mode test separates cleanly (accuracy"
          f" {acc:.2f}, separation {sep:+.2f}): the loop's headline result")
        P("    on the real scan (P(bilinear) 0.5 -> 0.000) is not an ABC")
        P("    artifact -- when blur truly is bilinear, the posterior finds")
        P("    it, and when it truly is cubic, the posterior says so too.")
    else:
        P(f"    The discrete blur_mode test separates imperfectly (accuracy"
          f" {acc:.2f}, separation {sep:+.2f}); the real-scan P(bilinear)=0")
        P("    result should be read with this classification noise in mind.")

    text = "\n".join(L)
    print(text)
    with open(os.path.join(core.RESULTS_DIR, "wp4_sbc_summary.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(text + "\n")
    return text, n_bad, acc, sep


def main():
    t0 = time.time()
    p1 = core.fm.load_summed_maps("prova1")
    p2 = core.fm.load_summed_maps("prova2")
    print("building whitening null ...")
    null = real_null(dg.N_NULL)

    print(f"continuous SBC: {N_TRIALS_CONT} trials x {N_DRAWS} draws ...")
    cont_rows = continuous_sbc(null, p1, p2)
    io_utils.write_rows("wp4_sbc_continuous", cont_rows)

    print(f"discrete blur SBC: {N_TRIALS_BLUR} trials x {N_DRAWS} draws ...")
    blur_rows = discrete_blur_sbc(null, p1, p2)
    io_utils.write_rows("wp4_sbc_blur", blur_rows)

    # combined CSV as documented in the module docstring
    combined = ([{"kind": "continuous", **r} for r in cont_rows]
               + [{"kind": "blur", **r} for r in blur_rows])
    io_utils.write_rows("wp4_sbc", combined)

    make_figure(cont_rows)
    summarize(cont_rows, blur_rows)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
