"""WP2 / experiment 2 - blind simulator diagnostics.
OWNER: Dimitrije (took over the whole WP on 2026-08-28)

CLAIM UNDER TEST
    A small battery of summary statistics, compared between a REAL
    measurement and its simulation, can not only DETECT that a
    simulator is wrong but IDENTIFY WHICH component is broken,
    turning our hand-made blur-mismatch discovery into a procedure.

BATTERY (per element line, tilted frame, "real" R vs simulation S)
    level_ratio   mean(R)/mean(S)                    -> gain-like errors
    cv_ratio      CV(R)/CV(S)                        -> blur and noise
    hf_ratio      power ratio, radial f > 0.6 Nyq    -> blur and noise
    mf_ratio      power ratio, 0.15 < f < 0.40 Nyq   -> blur (the mid
                  band is signal-dominated, so a noise change barely
                  moves it while a resampling blur does)
    k_ratio       single-map noise-constant estimate ratio (residual to
                  a 3x3 mean, median(d^2/m)); texture leaks equally
                  into both sides                    -> noise constant
    shift_dy/dx   signed cross-correlation peak offset R vs S (sub-pixel;
                  signed, because a magnitude makes an unshifted pair look
                  anomalous against a jittered null) -> registration shift
    rot_deg       vertical shift of the right half minus the left half,
                  divided by their separation        -> registration rotation

NULL MODEL AND DECISION RULE (fixed BEFORE looking at the rungs)
    Null = 24 pairs (simulator with knobs drawn WITHIN calibration
    uncertainty, config.JITTER = the WP1 contract) vs (nominal
    simulator) -> per-line mean/sd of every statistic.  A first version
    with same-source noise-only pairs flagged EVERY statistic on the
    real scan (z up to 60 for a 0.3 deg rotation), i.e. it answered
    'is anything different?' instead of 'is the discrepancy larger
    than the simulator's own stated uncertainty?'; the jitter null asks
    the second question.  Defects inside the calibration uncertainty
    are, by construction, undetectable.  A test pair
    gives z per line; the aggregate A = mean_lines |z|.  Threshold per
    statistic = mean_null(A) + 3 sd_null(A).  Rule, in order:
        rot flagged                      -> warp_rot
        shift flagged                    -> warp_shift
        level flagged                    -> gain_like  (gain and
                                            angle_bias are DEGENERATE at
                                            one angle: both scale g-1)
        mf flagged                       -> blur
        hf or k or cv flagged            -> noise_k
        nothing                          -> ok
    Grouped scoring counts gain/angle_bias as one family.  The rule is
    NOT tuned on the rungs; the confusion matrix reports its failures.
    A leave-one-out nearest-centroid classifier on the signed z-vector
    is reported as the separability upper bound (optional polish).

REAL-SCAN VERDICT
    R = the measured ruotato; S = nominal simulation of prova1 at 7.7
    deg, once with the v2 cubic sampling and once with the v1 bilinear
    sampling.  The null for the real scan must contain the SESSION
    difference, so it is built from (sim of prova2, sim of prova1)
    pairs instead of same-source pairs.  Expected: the blur statistics
    fire for v1 and stay quiet for v2.

OUTPUTS
    results/wp2_diagnostics.csv       statistic values + z per line
    results/wp2_diag_confusion.csv    truth x diagnosis per (rung, seed)
    results/wp2_diag_summary.txt      accuracy, confusion, real verdict
    figures/wp2_diag_confusion.png    signature heat map + confusion

Run from the repo root:
    python neurips_submission/wp2_simulator_audit/exp_diagnostics.py [--quick]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.ndimage import uniform_filter

import config
from common import core, io_utils, perturb, restore

ELEMENTS = core.ELEMENTS
STATS = ("level_ratio", "cv_ratio", "hf_ratio", "mf_ratio", "k_ratio",
         "shift_dy", "shift_dx", "rot_deg", "gain_proj")
# gain_proj is POST-HOC (added after the pre-registered rule returned 0/40
# on the gain family): the per-line level deviation projected on the
# simulator's own tilt-response vector g-1 (least-squares slope), i.e. an
# estimate of gain_scale - 1.  A scalar per pair, broadcast to all lines.
PREREG = STATS[:-1]
FAMILIES = ("noise_k", "gain", "angle_bias", "blur", "warp_shift",
            "warp_rot")
GROUP = {"gain": "gain_like", "angle_bias": "gain_like"}
N_NULL = 24
SEEDS_PER_RUNG = 5
Z_T = 3.0


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def _radial_power(img: np.ndarray):
    """(f_radial in Nyquist units, power) of the mean-subtracted map."""
    F = np.abs(np.fft.fft2(img - img.mean())) ** 2
    h, w = img.shape
    fy = np.fft.fftfreq(h)[:, None] * 2.0
    fx = np.fft.fftfreq(w)[None, :] * 2.0
    return np.sqrt(fy ** 2 + fx ** 2), F


def band_power_ratio(real, sim, lo, hi) -> float:
    fr, Pr = _radial_power(real)
    _, Ps = _radial_power(sim)
    m = (fr > lo) & (fr <= hi)
    return float(Pr[m].sum() / max(Ps[m].sum(), 1e-12))


def hf_ratio(real, sim) -> float:
    return band_power_ratio(real, sim, 0.6, 1.5)


def mf_ratio(real, sim) -> float:
    return band_power_ratio(real, sim, 0.15, 0.40)


def level_ratio(real, sim) -> float:
    return float(real.mean() / max(sim.mean(), 1e-12))


def cv_ratio(real, sim) -> float:
    cr = real.std() / max(real.mean(), 1e-12)
    cs = sim.std() / max(sim.mean(), 1e-12)
    return float(cr / max(cs, 1e-12))


def k_single(img) -> float:
    """Single-map estimate of k in Var = k*m: residual to a 3x3 mean
    (variance factor 8/9), median of d^2/m over pixels with m > 0."""
    d = img - uniform_filter(img, 3, mode="nearest")
    ok = img > 0
    if ok.sum() < 20:
        return float("nan")
    return float(np.median(d[ok] ** 2 / img[ok]) / (8.0 / 9.0))


def k_ratio(real, sim) -> float:
    return k_single(real) / max(k_single(sim), 1e-12)


def _xcorr_shift(a, b) -> tuple:
    """(dy, dx) sub-pixel offset of a relative to b via the FFT cross-
    correlation peak with a parabolic refinement."""
    a = a - a.mean()
    b = b - b.mean()
    c = np.real(np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))))
    c = np.fft.fftshift(c)
    h, w = c.shape
    iy, ix = np.unravel_index(np.argmax(c), c.shape)

    def refine(i, axis_len, line):
        if 0 < i < axis_len - 1:
            l, m, r = line[i - 1], line[i], line[i + 1]
            den = (l - 2 * m + r)
            return i + (0.5 * (l - r) / den if den != 0 else 0.0)
        return float(i)

    dy = refine(iy, h, c[:, ix]) - h // 2
    dx = refine(ix, w, c[iy, :]) - w // 2
    return float(dy), float(dx)


def shift_dy(real, sim) -> float:
    """Signed vertical offset (a magnitude would make an UNshifted pair
    look anomalous against the jitter null, which always has some shift)."""
    return _xcorr_shift(real, sim)[0]


def shift_dx(real, sim) -> float:
    return _xcorr_shift(real, sim)[1]


def rot_deg(real, sim) -> float:
    w = real.shape[1]
    half = w // 2
    dy_l, _ = _xcorr_shift(real[:, :half], sim[:, :half])
    dy_r, _ = _xcorr_shift(real[:, half:], sim[:, half:])
    sep = w / 2.0
    return float(np.degrees(np.arctan2(dy_r - dy_l, sep)))


STAT_FN = {"level_ratio": level_ratio, "cv_ratio": cv_ratio,
           "hf_ratio": hf_ratio, "mf_ratio": mf_ratio, "k_ratio": k_ratio,
           "shift_dy": shift_dy, "shift_dx": shift_dx, "rot_deg": rot_deg}


_GVEC = {}


def gain_vector(angle: float) -> np.ndarray:
    if angle not in _GVEC:
        g = core.fm.tilt_gains(angle)
        _GVEC[angle] = np.array([g[el] - 1.0 for el in ELEMENTS])
    return _GVEC[angle]


def battery(real_maps: dict, sim_maps: dict, angle: float) -> dict:
    """{statistic: {element: value}}."""
    out = {}
    for name, fn in STAT_FN.items():
        out[name] = {el: fn(np.asarray(real_maps[el], float),
                            np.asarray(sim_maps[el], float))
                     for el in ELEMENTS}
    dev = np.array([out["level_ratio"][el] - 1.0 for el in ELEMENTS])
    gv = gain_vector(angle)
    proj = float(np.dot(dev, gv) / max(np.dot(gv, gv), 1e-12))
    out["gain_proj"] = {el: proj for el in ELEMENTS}
    return out


# ---------------------------------------------------------------------------
# null model and identification
# ---------------------------------------------------------------------------

def null_pair(src_real: dict, src_sim: dict, angle: float, seed: int):
    """One null pair: a 'real' measurement simulated from src_real with
    knobs drawn WITHIN calibration uncertainty (config.JITTER, the WP1
    contract) vs the nominal simulation of src_sim.  The null therefore
    answers 'is the discrepancy larger than the simulator's own stated
    uncertainty?'; with src_real = prova2 it also contains the session
    difference."""
    rng = np.random.default_rng(seed)
    knobs = perturb.jittered(rng, config.JITTER, "null")
    real = perturb.forward_perturbed(src_real, angle, rng, knobs)
    sim = perturb.forward_perturbed(src_sim, angle,
                                    np.random.default_rng(seed + 1),
                                    perturb.NOMINAL)
    return real, sim


class Null:
    """Per-line mean/sd of every statistic under nominal-vs-nominal (or
    session-vs-session) pairs, plus the null of the aggregate A."""

    def __init__(self, pairs: list, angle: float):
        self.angle = angle
        vals = {s: {el: [] for el in ELEMENTS} for s in STATS}
        self.batteries = []
        for real, sim in pairs:
            b = battery(real, sim, angle)
            self.batteries.append(b)
            for s in STATS:
                for el in ELEMENTS:
                    vals[s][el].append(b[s][el])
        self.mu = {s: {el: float(np.mean(v)) for el, v in d.items()}
                   for s, d in vals.items()}
        self.sd = {s: {el: max(float(np.std(v, ddof=1)), 1e-12)
                       for el, v in d.items()} for s, d in vals.items()}
        # null distribution of the aggregate A = mean_lines |z|
        A = {s: [] for s in STATS}
        for b in self.batteries:
            z = self.z(b)
            for s in STATS:
                A[s].append(float(np.mean(np.abs(list(z[s].values())))))
        self.thr = {s: float(np.mean(A[s]) + Z_T * np.std(A[s], ddof=1))
                    for s in STATS}

    def z(self, b: dict) -> dict:
        return {s: {el: (b[s][el] - self.mu[s][el]) / self.sd[s][el]
                    for el in ELEMENTS} for s in STATS}

    def aggregate(self, b: dict) -> dict:
        z = self.z(b)
        return {s: float(np.mean(np.abs(list(z[s].values()))))
                for s in STATS}, \
               {s: float(np.mean(list(z[s].values()))) for s in STATS}


def identify(stats: dict, null: Null, posthoc: bool = False) -> tuple:
    """(diagnosis, flags dict, A dict, signed-z dict).

    posthoc=False is the pre-registered rule (mean |z| over lines of the
    original battery); posthoc=True additionally lets the gain_proj
    template statistic call gain_like."""
    A, zs = null.aggregate(stats)
    flags = {s: A[s] > null.thr[s] for s in STATS}
    if flags["rot_deg"]:
        d = "warp_rot"
    elif flags["shift_dy"] or flags["shift_dx"]:
        d = "warp_shift"
    elif flags["level_ratio"] or (posthoc and flags["gain_proj"]):
        d = "gain_like"
    elif flags["mf_ratio"]:
        d = "blur"
    elif flags["hf_ratio"] or flags["k_ratio"] or flags["cv_ratio"]:
        d = "noise_k"
    else:
        d = "ok"
    return d, flags, A, zs


def nearest_centroid_loo(samples: list) -> list:
    """samples: [(label, z-vector)] -> LOO nearest-centroid predictions."""
    labels = sorted({s[0] for s in samples})
    X = np.array([s[1] for s in samples])
    y = np.array([s[0] for s in samples])
    preds = []
    for i in range(len(samples)):
        best, bestd = None, np.inf
        for lab in labels:
            m = (y == lab)
            m[i] = False
            if not m.any():
                continue
            c = X[m].mean(axis=0)
            sd = X[m].std(axis=0) + 1e-6
            d = float(np.sum(((X[i] - c) / sd) ** 2))
            if d < bestd:
                best, bestd = lab, d
        preds.append(best)
    return preds


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run(quick: bool = False):
    p1 = core.fm.load_summed_maps("prova1")
    p2 = core.fm.load_summed_maps("prova2")
    angle = core.fm.REF_ANGLE_DEG
    n_null = 8 if quick else N_NULL
    n_seeds = 2 if quick else SEEDS_PER_RUNG

    print(f"null model: {n_null} (jittered within calibration uncertainty)"
          " vs nominal pairs ...")
    pairs = [null_pair(p1, p1, angle, 5000 + 2 * j) for j in range(n_null)]
    null = Null(pairs, angle)
    print("  thresholds A*: " + "  ".join(f"{s} {null.thr[s]:.2f}"
                                          for s in STATS))

    rows, confusion, samples = [], [], []
    # the blind test at the calibration angle AND at an extrapolation
    # angle (20 deg): gain-like defects scale with the tilt, so a defect
    # invisible at 7.7 deg may be detectable where the simulator is
    # actually used
    angles_t = (angle,) if quick else (angle, 20.0)
    nulls = {angle: null}
    for angle_t in angles_t:
        if angle_t not in nulls:
            print(f"null model at {angle_t:g} deg ...")
            nulls[angle_t] = Null([null_pair(p1, p1, angle_t, 5000 + 2 * j)
                                   for j in range(n_null)], angle_t)
        nl = nulls[angle_t]
        # false-positive check: the null pairs themselves
        for j, b in enumerate(nl.batteries):
            d, flags, A, zs = identify(b, nl)
            d2 = identify(b, nl, posthoc=True)[0]
            confusion.append({"angle": angle_t, "truth": "ok", "group": "ok",
                              "defect": "nominal", "seed": j, "diagnosis": d,
                              "diagnosis_posthoc": d2,
                              "correct": d == "ok", "correct_grouped": d == "ok",
                              "correct_posthoc": d2 == "ok"})
            samples.append(("ok", [zs[s_] for s_ in STATS]))
        ladders = list(config.DEFECT_LADDERS.items())
        if quick:
            ladders = ladders[:3]
        for family, rungs in ladders:
            for label, kw in rungs:
                defective = perturb.SimKnobs(label=label, **kw)
                for seed in range(n_seeds):
                    pseudo_real = perturb.forward_perturbed(
                        p1, angle_t, np.random.default_rng(7000 + 10 * seed),
                        defective)
                    reference = perturb.forward_perturbed(
                        p1, angle_t, np.random.default_rng(7001 + 10 * seed),
                        perturb.NOMINAL)
                    b = battery(pseudo_real, reference, angle_t)
                    d, flags, A, zs = identify(b, nl)
                    d2 = identify(b, nl, posthoc=True)[0]
                    g = GROUP.get(family, family)
                    confusion.append({"angle": angle_t, "truth": family,
                                      "group": g, "defect": label,
                                      "seed": seed, "diagnosis": d,
                                      "diagnosis_posthoc": d2,
                                      "correct": d == family,
                                      "correct_grouped": d == g,
                                      "correct_posthoc": d2 == g,
                                      **{f"A_{s_}": A[s_] for s_ in STATS},
                                      **{f"flag_{s_}": int(flags[s_])
                                         for s_ in STATS}})
                    samples.append((g, [zs[s_] for s_ in STATS]))
                    z = nl.z(b)
                    for s_ in STATS:
                        for el in ELEMENTS:
                            rows.append({"angle": angle_t,
                                         "defect_family": family,
                                         "defect": label, "seed": seed,
                                         "statistic": s_, "element": el,
                                         "value": b[s_][el], "z": z[s_][el]})
                print(f"  {angle_t:g} deg {family}/{label}: " + ", ".join(
                    c["diagnosis"] for c in confusion
                    if c["defect"] == label and c["truth"] == family
                    and c["angle"] == angle_t))
    # LOO nearest-centroid as the separability upper bound
    preds = nearest_centroid_loo(samples)
    k = 0
    for c in confusion:
        c["nn_diagnosis"] = preds[k]
        c["nn_correct_grouped"] = preds[k] == c["group"]
        k += 1
    io_utils.write_rows("wp2_diagnostics", rows)
    io_utils.write_rows("wp2_diag_confusion", confusion)

    # ---- the real-scan verdict --------------------------------------------
    print("real-scan verdict ...")
    pairs2 = [null_pair(p2, p1, angle, 9000 + 2 * j) for j in range(n_null)]
    null2 = Null(pairs2, angle)
    ruo = core.fm.load_summed_maps("ruotato")
    real_rows = []
    verdicts = {}
    for tag, knobs in (("v2_cubic", perturb.NOMINAL),
                       ("v1_bilinear", perturb.SimKnobs(
                           blur_mode="bilinear", label="v1"))):
        sim = perturb.forward_perturbed(p1, angle, np.random.default_rng(4242),
                                        knobs)
        b = battery(ruo, sim, angle)
        d, flags, A, zs = identify(b, null2, posthoc=True)
        verdicts[tag] = (d, flags, A, zs, null2.thr,
                         {s_: float(np.mean(list(b[s_].values()))) for s_ in STATS})
        z = null2.z(b)
        for s in STATS:
            for el in ELEMENTS:
                real_rows.append({"angle": angle, "defect_family": "REAL",
                                  "defect": tag, "seed": 0, "statistic": s,
                                  "element": el, "value": b[s][el],
                                  "z": z[s][el]})
        # also the validated bilinear emulator (forward_model.forward) as
        # a third reference: it is the fidelity-gated simulator of MVP-2
    sim_v = core.fm.forward(p1, angle_deg=angle, rng=np.random.default_rng(4242),
                            add_noise=True, input_noise="measured")
    b = battery(ruo, sim_v, angle)
    d, flags, A, zs = identify(b, null2, posthoc=True)
    verdicts["validated_forward"] = (d, flags, A, zs, null2.thr,
                                     {s_: float(np.mean(list(b[s_].values()))) for s_ in STATS})
    z = null2.z(b)
    for s in STATS:
        for el in ELEMENTS:
            real_rows.append({"angle": angle, "defect_family": "REAL",
                              "defect": "validated_forward", "seed": 0,
                              "statistic": s, "element": el,
                              "value": b[s][el], "z": z[s][el]})
    io_utils.write_rows("wp2_diagnostics", rows + real_rows)
    summarize(verdicts)


def summarize(verdicts: dict | None = None):
    conf = io_utils.read_rows("wp2_diag_confusion")
    diag = io_utils.read_rows("wp2_diagnostics")
    L = []
    P = L.append
    P("WP2 - blind simulator diagnostics")
    groups = ["ok"] + sorted({c["group"] for c in conf if c["group"] != "ok"})
    diags = ["ok", "noise_k", "gain_like", "blur", "warp_shift", "warp_rot"]
    angles = sorted({float(c.get("angle", 7.7)) for c in conf})
    for ang in angles:
        ca = [c for c in conf if float(c.get("angle", 7.7)) == ang]
        n = len(ca)
        ok = sum(c["correct"] == "True" for c in ca)
        okg = sum(c["correct_grouped"] == "True" for c in ca)
        nn = sum(c.get("nn_correct_grouped", "") == "True" for c in ca)
        P("")
        P(f"=== blind test at {ang:g} deg: rule exact {ok}/{n}, grouped (gain"
          f" and angle_bias merged) {okg}/{n}; LOO nearest-centroid on the"
          f" z-vector (grouped) {nn}/{n}")
        P("confusion (rows = true family group, cols = rule diagnosis)")
        P("    " + f"{'':12s}" + "".join(f"{d:>11s}" for d in diags))
        for g in groups:
            cnt = [sum(1 for c in ca if c["group"] == g and c["diagnosis"] == d)
                   for d in diags]
            P("    " + f"{g:12s}" + "".join(f"{v:11d}" for v in cnt))
        okp = sum(c.get("correct_posthoc", "") == "True" for c in ca)
        P(f"POST-HOC rule (+ gain_proj template): grouped {okp}/{n}")
        P("    " + f"{'':12s}" + "".join(f"{d:>11s}" for d in diags))
        for g in groups:
            cnt = [sum(1 for c in ca if c["group"] == g
                       and c.get("diagnosis_posthoc") == d) for d in diags]
            P("    " + f"{g:12s}" + "".join(f"{v:11d}" for v in cnt))
        P("per rung (pre-registered rule over seeds | post-hoc rule):")
        seen = []
        for c in ca:
            key = (c["truth"], c["defect"])
            if key in seen or c["truth"] == "ok":
                continue
            seen.append(key)
            ds = [x["diagnosis"] for x in ca if (x["truth"], x["defect"]) == key]
            dp = [x.get("diagnosis_posthoc", "") for x in ca
                  if (x["truth"], x["defect"]) == key]
            P(f"    {c['truth']:12s}{c['defect']:16s} -> " + ", ".join(ds)
              + "  |  " + ", ".join(dp))
    if verdicts:
        P("")
        P("REAL scan (ruotato) vs its simulation from prova1 at 7.7 deg; "
          "null = sim(prova2) vs sim(prova1) pairs (contains the session "
          "difference)")
        for tag, (d, flags, A, zs, thr, raw) in verdicts.items():
            P(f"  simulator {tag:18s} diagnosis: {d}")
            P("    " + "  ".join(
                f"{s} A={A[s]:.2f}/{thr[s]:.2f}{'*' if flags[s] else ''}"
                f"(z={zs[s]:+.2f}, raw {raw[s]:.3f})" for s in STATS))
        hf = {}
        for r in diag:
            if r["defect_family"] == "REAL" and r["statistic"] == "hf_ratio":
                hf.setdefault(r["defect"], {})[r["element"]] = float(r["value"])
        if hf:
            P("  hf_ratio real/sim per line: ")
            for tag, d_ in hf.items():
                P(f"    {tag:18s} " + "  ".join(f"{el} {d_[el]:.2f}"
                                                for el in ELEMENTS))
    P("")
    P("reading: A = mean over lines of |z|, threshold = null mean + 3 sd; "
      "'*' = flagged. gain and angle_bias are degenerate at a single angle "
      "(both scale g-1), hence grouped scoring.  gain_proj is a POST-HOC "
      "statistic (added after the pre-registered rule scored 0/40 on the "
      "gain family: the mean over lines dilutes a level change that sits on "
      "the two or three lines with a large tilt response); the real-scan "
      "verdict uses the post-hoc rule.")
    text = "\n".join(L)
    print(text)
    with open(os.path.join(core.RESULTS_DIR, "wp2_diag_summary.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(text + "\n")
    return text


def make_figures():
    conf = io_utils.read_rows("wp2_diag_confusion")
    if not conf:
        raise NotImplementedError("run exp_diagnostics first")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("nwo", ["#c8641e", "#ffffff",
                                                      "#1f2a44"])
    # signature heat map: rung x statistic, signed mean z over seeds
    diag = io_utils.read_rows("wp2_diagnostics")
    rungs = []
    for r in diag:
        key = (r["defect_family"], r["defect"])
        if key not in rungs and r["defect_family"] != "REAL":
            rungs.append(key)
    real_tags = []
    for r in diag:
        if r["defect_family"] == "REAL" and r["defect"] not in real_tags:
            real_tags.append(r["defect"])
    M = np.zeros((len(rungs) + len(real_tags), len(STATS)))
    for i, (fam, lab) in enumerate(rungs + [("REAL", t) for t in real_tags]):
        for j, s in enumerate(STATS):
            v = [float(r["z"]) for r in diag if r["defect_family"] == fam
                 and r["defect"] == lab and r["statistic"] == s]
            M[i, j] = np.mean(v) if v else np.nan
    groups = ["ok"] + sorted({c["group"] for c in conf if c["group"] != "ok"})
    diags = ["ok", "noise_k", "gain_like", "blur", "warp_shift", "warp_rot"]
    C = np.array([[sum(1 for c in conf if c["group"] == g and c["diagnosis"] == d)
                   for d in diags] for g in groups], float)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 0.26 * M.shape[0] + 2.0),
                                   gridspec_kw={"width_ratios": [1.2, 1]})
    lim = 6.0
    im = ax1.imshow(np.clip(M, -lim, lim), cmap=cmap, vmin=-lim, vmax=lim,
                    aspect="auto")
    ax1.set_xticks(range(len(STATS)))
    ax1.set_xticklabels([s.replace("_", "\n") for s in STATS], fontsize=8)
    ax1.set_yticks(range(M.shape[0]))
    ax1.set_yticklabels([f"{f}/{l}" for f, l in rungs]
                        + [f"REAL vs {t}" for t in real_tags], fontsize=8)
    ax1.axhline(len(rungs) - 0.5, color="k", lw=0.8)
    ax1.set_title("signature: mean z per statistic, both angles (clipped at +-6)",
                  fontsize=9)
    plt.colorbar(im, ax=ax1, fraction=0.03, pad=0.02)
    rowsum = C.sum(axis=1, keepdims=True)
    ax2.imshow(C / np.maximum(rowsum, 1), cmap="Blues", vmin=0, vmax=1,
               aspect="auto")
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if C[i, j] > 0:
                ax2.text(j, i, int(C[i, j]), ha="center", va="center",
                         fontsize=9, color="w" if C[i, j] / rowsum[i, 0] > 0.6
                         else "k")
    ax2.set_xticks(range(len(diags)))
    ax2.set_xticklabels(diags, fontsize=8, rotation=30, ha="right")
    ax2.set_yticks(range(len(groups)))
    ax2.set_yticklabels(groups, fontsize=8)
    ax2.set_xlabel("rule diagnosis")
    ax2.set_ylabel("true family (grouped)")
    ax2.set_title("blind identification, pre-registered rule, both angles", fontsize=9)
    fig.tight_layout()
    out = io_utils.fig_path("wp2_diag_confusion.png")
    fig.savefig(out, dpi=200)
    fig.savefig(out.replace(".png", ".pdf"))
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
