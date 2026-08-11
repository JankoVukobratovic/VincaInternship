"""
07_geometry_fit.py
Forward-model sketch for the dual-detector ratio R(E, tilt).

Stage 1 -- detector part, fitted on the frontal ratios:

    R_det(E) = k * exp(+d_abs * mu_Be(E)) * A_Si(E; t1) / A_Si(E; t2)

    k     : energy-independent factor (solid angles, gain)
    d_abs : differential low-energy absorber between the two channels,
            expressed as Be-equivalent thickness. It lumps together the
            Be windows, air paths and collimator differences, so it is
            an *effective* number, not a physical window spec.
    A_Si  : active-layer absorption 1 - exp(-t * mu_Si(E)). The 19511
            thickness is fixed at 500 um (typical SDD) and only the
            10264 thickness is fitted -- the absolute thicknesses are
            degenerate here, only their ratio is constrained.

Stage 2 -- geometric part, fitted on the tilt-induced relative shift
delta(E) = R_tilt / R_frontal - 1, from a thick-sample fluorescence
model with effective take-off angles psi1 (10264) and psi2 (19511):

    I_d ~ 1 / (mu_in / sin(phi) + mu_out(E) / sin(psi_d))

Tilting the canvas forward by theta is assumed to move the take-off
angles antisymmetrically (psi1 + theta, psi2 - theta) and the beam
incidence to phi0 - theta. The matrix attenuation ratio is modelled as
a photoelectric power law mu_out/mu_in = (E/Ec)^-3 with Ec fitted,
plus an energy-independent offset c for the solid-angle change of the
tilted surface.

With a single tilt angle neither the two take-off angles nor their
difference is identifiable from the shift alone (the tilt term
dominates and the asymmetry only enters at second order; tried and it
returns ~1e5-degree uncertainties). The identifiable quantities are:

    s  : the "lever arm" -- how many degrees the effective take-off
         angles move per degree of canvas tilt (s = 1 would be the
         ideal antisymmetric response psi +- theta);
    Ec : the energy scale of the matrix attenuation;
    c  : the energy-independent solid-angle offset.

The mean take-off angle is fixed at PSI_MEAN_DEG (nominal head
geometry, to be confirmed with the instrument builder).

The tilt angle is estimated from the vertical foreshortening of the
tilted scan (see 07b_tilt_angle.py); the fit takes it as a module
constant so it can be re-run unchanged if the instrument builder
provides the exact mounting angle.

The tilt-shift panel also carries a nonparametric cross-check: a
zero-mean Gaussian process with an RBF kernel, the measurement errors
as noise and hyperparameters set by maximizing the marginal
likelihood. If the parametric curve stays inside the GP band, the
observed shape (a monotonic positive decay, ~+9% at Ca Ka falling to
~+1% at the Pb L lines) is a property of the data, not of the model
choice.

Handoff 2 (A -> B, PLAN amendment 8.4): the script also exports the
smooth response-ratio curve R(E) that the Noise2Noise dataloader needs
for channel-by-channel target scaling. It is built as

    R(E) = R_det(E) * exp(GP residual)

i.e. the parametric stage-1 curve corrected by a zero-mean GP fitted to
the log-residuals of the eight measured lines. The GP alone would revert
to R = 1 outside the data range; anchoring it on the physics model keeps
the extrapolation sane, while inside 3.7-14.8 keV the curve follows the
measurements. The tilted-scan column R_tilt(E) = R(E) * (1 + delta(E))
scales the ruotato pixels, whose ratio is up to 9.5% higher.

Input : results/registration/overlap_ratios.csv  (from script 08).
        The ratios are restricted to the registered overlap region:
        the full-frame ratios of script 06 mix in a field-of-view
        artifact (the apparent zero crossing of the tilt shift) that
        the overlap crop removes.
Output: results/detector_diff/geometry_fit.png
        results/detector_diff/geometry_fit.txt
        results/detector_diff/handoff2_ratio_curve.csv
        results/detector_diff/handoff2_ratio_curve.md

Run from the project root:
    python scripts/07_geometry_fit.py
"""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, minimize

OUTPUT_DIR = os.path.join("results", "detector_diff")
CSV_IN     = os.path.join("results", "registration", "overlap_ratios.csv")

# Estimated from the vertical foreshortening of the ruotato scan:
# anisotropic registration onto the frontal scans gives sy/sx = 1/cos
# (theta) with the unknown step sizes cancelling in the ratio.
# Per-element spread gives +-1 deg; the prova1<->prova2 scale drift
# (0.43%) adds ~+-2 deg systematic. A no-tilt control pair returns
# "5.3 deg", so foreshortening at this angle is near the resolution
# floor: treat the value as an upper bound (theta <~ 8 deg) until the
# instrument builder confirms the mounting angle.
TILT_DEG = 7.7
PHI0_DEG = 90.0   # frontal beam incidence (assumed normal)

# handoff-2 export grid (keV). The upper end is where the NIST grid below
# stops; the N2N loss window is 3.5-15.5 keV, well inside.
HANDOFF_LO_KEV   = 2.0
HANDOFF_HI_KEV   = 20.0
HANDOFF_STEP_KEV = 0.02
# regularization of the GP correction (see the handoff block below):
# white noise pinned at the frontal repeatability floor, length scale
# floored so the curve cannot chase line-window systematics.
HANDOFF_JITTER      = 0.005
HANDOFF_ELL_MIN_KEV = 1.5
PSI_MEAN_DEG = 45.0  # nominal mean take-off angle, to be confirmed
T_SI2_UM = 500.0  # fixed reference Si thickness for det 19511, um
RHO_BE   = 1.848  # g/cm3
RHO_SI   = 2.33   # g/cm3

# NIST mass attenuation coefficients (cm2/g), log-log interpolated;
# verified against the Hubbell & Seltzer tables (physics.nist.gov,
# XrayMassCoef, Z=4 and Z=14). Si has its K edge at 1.839 keV, so the
# grid is only valid above 2 keV -- fine, every line used here is
# >= 3.3 keV.
E_GRID = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0])
MU_BE  = np.array([74.69, 21.27, 8.685, 4.369, 2.527,
                   1.124, 0.6466, 0.307, 0.2251])
MU_SI  = np.array([2777.0, 978.4, 452.9, 245.0, 147.0,
                   64.68, 33.89, 10.34, 4.464])


def mu_be(E):
    return np.exp(np.interp(np.log(E), np.log(E_GRID), np.log(MU_BE)))


def mu_si(E):
    return np.exp(np.interp(np.log(E), np.log(E_GRID), np.log(MU_SI)))


def det_ratio(E, k, d_abs_um, t_si1_um):
    """Stage 1: frontal ratio R = eff(10264) / eff(19511)."""
    t_be = np.exp(d_abs_um * 1e-4 * RHO_BE * mu_be(E))
    a1 = 1.0 - np.exp(-t_si1_um * 1e-4 * RHO_SI * mu_si(E))
    a2 = 1.0 - np.exp(-T_SI2_UM  * 1e-4 * RHO_SI * mu_si(E))
    return k * t_be * a1 / a2


def geom_ratio(E, s, ec, tilt_deg):
    x   = (E / ec) ** -3.0
    phi = np.radians(PHI0_DEG - tilt_deg)
    p1  = np.radians(PSI_MEAN_DEG + s * tilt_deg)
    p2  = np.radians(PSI_MEAN_DEG - s * tilt_deg)
    return ((1.0 / np.sin(phi) + x / np.sin(p2))
            / (1.0 / np.sin(phi) + x / np.sin(p1)))


def tilt_shift(E, s, ec, c):
    """Stage 2: delta(E) = R_tilt / R_frontal - 1."""
    g  = geom_ratio(E, s, ec, TILT_DEG)
    g0 = geom_ratio(E, s, ec, 0.0)
    return (1.0 + c) * g / g0 - 1.0


def gp_regress(x, y, yerr, xs, fixed_jit=None, ell_min=0.0):
    """Zero-mean GP regression with RBF kernel.

    The noise is yerr plus a fitted white-noise term: the bootstrap
    errors only capture counting statistics, not the scan-to-scan
    systematics (cf. the 0.4% frontal scale drift), and without the
    extra term the maximum-likelihood GP collapses to a sub-keV length
    scale that just interpolates the points. Hyperparameters
    (amplitude, length scale, white noise) by maximum marginal
    likelihood over a few starts. Returns mean and sigma on xs.

    ``fixed_jit`` pins the white-noise term instead of fitting it and
    ``ell_min`` puts a floor under the length scale; both are used for
    the handoff-2 curve, where the free fit degenerates (all structure
    absorbed into white noise, amplitude -> 0) and where length scales
    below ~1 keV would fit line-window systematics rather than detector
    response.
    """
    d2 = (x[:, None] - x[None, :]) ** 2

    def unpack(logp):
        if fixed_jit is None:
            amp, ell, jit = np.exp(logp)
        else:
            amp, ell = np.exp(logp)
            jit = fixed_jit
        return amp, ell_min + ell, jit

    def nll(logp):
        amp, ell, jit = unpack(logp)
        K = amp ** 2 * np.exp(-0.5 * d2 / ell ** 2)
        K[np.diag_indices_from(K)] += yerr ** 2 + jit ** 2
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10
        a = np.linalg.solve(L.T, np.linalg.solve(L, y))
        return 0.5 * y @ a + np.log(np.diag(L)).sum()

    starts = ([0.05, 2.0, 0.01], [0.05, 5.0, 0.005], [0.2, 3.0, 0.02])
    if fixed_jit is not None:
        starts = tuple(p[:2] for p in starts)
    best = None
    for p0 in starts:
        r = minimize(nll, np.log(p0), method="Nelder-Mead")
        if best is None or r.fun < best.fun:
            best = r
    amp, ell, jit = unpack(best.x)

    K = amp ** 2 * np.exp(-0.5 * d2 / ell ** 2)
    K[np.diag_indices_from(K)] += yerr ** 2 + jit ** 2
    L = np.linalg.cholesky(K)
    a = np.linalg.solve(L.T, np.linalg.solve(L, y))
    ks = amp ** 2 * np.exp(
        -0.5 * (xs[:, None] - x[None, :]) ** 2 / ell ** 2)
    mean = ks @ a
    v = np.linalg.solve(L, ks.T)
    sig = np.sqrt(np.maximum(
        amp ** 2 - np.sum(v * v, axis=0), 0.0) + jit ** 2)
    return mean, sig, amp, ell, jit


if __name__ == "__main__":
    rows = []
    with open(CSV_IN, newline="") as f:
        for row in csv.DictReader(f):
            if row["reliable"] == "True":
                rows.append(row)

    els = [r["element"] for r in rows]
    E   = np.array([float(r["kev"]) for r in rows])
    rf     = np.array([float(r["R_frontal_overlap"]) for r in rows])
    rf_err = np.array([float(r["sig_frontal_overlap"]) for r in rows])
    rr  = np.array([float(r["R_ruotato"]) for r in rows])
    sr  = np.array([float(r["sig_ruotato"]) for r in rows])

    delta     = rr / rf - 1.0
    delta_err = (rr / rf) * np.hypot(sr / rr, rf_err / rf)

    # ---- stage 1: detector part on the frontal curve -------------------
    p1_opt, p1_cov = curve_fit(
        det_ratio, E, rf, p0=(0.9, 800.0, 300.0), sigma=rf_err,
        absolute_sigma=True,
        bounds=([0.1, 0.0, 50.0], [5.0, 3000.0, 1500.0]), maxfev=20000,
    )
    p1_err  = np.sqrt(np.diag(p1_cov))
    chi2_1  = np.sum(((rf - det_ratio(E, *p1_opt)) / rf_err) ** 2)
    dof_1   = len(E) - len(p1_opt)

    # ---- stage 2: geometric part on the tilt shift ---------------------
    p2_opt, p2_cov = curve_fit(
        tilt_shift, E, delta, p0=(0.3, 6.0, -0.02),
        sigma=delta_err, absolute_sigma=True,
        bounds=([-3.0, 1.0, -0.2], [3.0, 20.0, 0.2]),
        maxfev=20000,
    )
    p2_err  = np.sqrt(np.diag(p2_cov))
    chi2_2  = np.sum(((delta - tilt_shift(E, *p2_opt)) / delta_err) ** 2)
    dof_2   = len(E) - len(p2_opt)

    # ---- GP cross-check of the tilt-shift shape ------------------------
    Ef = np.linspace(3.0, 16.0, 300)
    gp_mean, gp_sig, gp_amp, gp_ell, gp_jit = gp_regress(
        E, delta, delta_err, Ef)
    in_data = (Ef >= E.min()) & (Ef <= E.max())
    z_gp = np.max(np.abs(
        (tilt_shift(Ef[in_data], *p2_opt) - gp_mean[in_data])
        / gp_sig[in_data]))

    # ---- handoff 2: smooth R(E) for the N2N target scaling -------------
    # GP on the log-residuals of the stage-1 model: the parametric curve
    # carries the extrapolation, the GP absorbs what the three-parameter
    # detector model cannot reproduce (chi2/dof above is large because
    # the bootstrap errors are ~0.1%).
    #
    # The residuals reach +6.1% at Cu and -5.2% at Pb Ll, i.e. 11% across
    # 1.1 keV -- too fast for an efficiency ratio and consistent with the
    # suspected contamination of the Ll window (PLAN 8.2). The GP is
    # therefore regularized: white noise pinned at the frontal
    # repeatability floor and length scales below HANDOFF_ELL_MIN_KEV
    # forbidden, so the exported curve smooths through such pairs instead
    # of writing them into the target scaling.
    log_res     = np.log(rf) - np.log(det_ratio(E, *p1_opt))
    log_res_err = rf_err / rf
    Eh = np.arange(HANDOFF_LO_KEV, HANDOFF_HI_KEV + 1e-9, HANDOFF_STEP_KEV)
    res_mean, res_sig, res_amp, res_ell, res_jit = gp_regress(
        E, log_res, log_res_err, Eh,
        fixed_jit=HANDOFF_JITTER, ell_min=HANDOFF_ELL_MIN_KEV)
    r_model  = det_ratio(Eh, *p1_opt)
    r_smooth = r_model * np.exp(res_mean)
    r_sigma  = r_smooth * res_sig
    r_tilt   = r_smooth * (1.0 + tilt_shift(Eh, *p2_opt))
    # closure test: the exported curve against the eight measured points
    r_at_lines = np.interp(E, Eh, r_smooth)
    closure_pct = 100.0 * np.abs(r_at_lines / rf - 1.0)

    lines = [
        f"tilt angle: {TILT_DEG:.1f} deg"
        " (from foreshortening; upper bound, see 07b)",
        "",
        "stage 1 -- detector part R_det(E), fit on frontal overlap ratio:",
        f"  k      = {p1_opt[0]:8.4f} +- {p1_err[0]:.4f}",
        f"  d_abs  = {p1_opt[1]:8.1f} +- {p1_err[1]:.1f} um Be-equivalent",
        f"  t_Si1  = {p1_opt[2]:8.1f} +- {p1_err[2]:.1f} um"
        f"  (t_Si2 fixed at {T_SI2_UM:.0f} um)",
        f"  chi2/dof = {chi2_1:.1f}/{dof_1}",
        "",
        "stage 2 -- geometric part delta(E), fit on tilt shift:",
        f"  s (lever arm) = {p2_opt[0]:+6.3f} +- {p2_err[0]:.3f} deg/deg"
        f"  (mean take-off fixed at {PSI_MEAN_DEG:.0f} deg)",
        f"  Ec            = {p2_opt[1]:6.2f} +- {p2_err[1]:.2f} keV",
        f"  c (offset)    = {p2_opt[2]:+6.4f} +- {p2_err[2]:.4f}",
        f"  chi2/dof = {chi2_2:.1f}/{dof_2}",
        "",
        "GP cross-check of the tilt shift (RBF kernel, max marginal"
        " likelihood):",
        f"  amp = {100 * gp_amp:.2f}%  length scale = {gp_ell:.1f} keV"
        f"  fitted white noise = {100 * gp_jit:.2f}%",
        f"  max |model - GP mean| / GP sigma over"
        f" {E.min():.1f}-{E.max():.1f} keV: {z_gp:.2f}",
        "",
        "handoff 2 -- smooth R(E) = R_det(E) * exp(GP residual):",
        f"  GP on log-residuals: amp = {100 * res_amp:.2f}%"
        f"  length scale = {res_ell:.1f} keV"
        f"  (floor {HANDOFF_ELL_MIN_KEV:.1f} keV)"
        f"  white noise pinned at {100 * res_jit:.1f}%",
        f"  closure at the 8 measured lines: max"
        f" {closure_pct.max():.2f}%, mean {closure_pct.mean():.2f}%",
        f"  exported on {HANDOFF_LO_KEV:.0f}-{HANDOFF_HI_KEV:.0f} keV"
        f" ({len(Eh)} points, {HANDOFF_STEP_KEV:.2f} keV step);"
        " R_tilt column for the ruotato pixels",
    ]
    print("\n".join(lines))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    txt_path = os.path.join(OUTPUT_DIR, "geometry_fit.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---- handoff 2 files -----------------------------------------------
    curve_path = os.path.join(OUTPUT_DIR, "handoff2_ratio_curve.csv")
    with open(curve_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kev", "R", "R_sigma", "R_model", "R_tilt"])
        for e, r, s, rm, rt in zip(Eh, r_smooth, r_sigma, r_model, r_tilt):
            w.writerow([f"{e:.4f}", f"{r:.6f}", f"{s:.6f}",
                        f"{rm:.6f}", f"{rt:.6f}"])

    md = [
        "# Handoff 2 (A -> B): response ratio R(E) = det10264 / det19511",
        "",
        "Table 1 rows (registered overlap region, `results/registration/"
        "overlap_ratios.csv`) and the smooth curve derived from them.",
        "",
        "| line | E (keV) | R frontal | sigma | R tilted | tilt shift |",
        "|------|---------|-----------|-------|----------|------------|",
    ]
    for i, name in enumerate(els):
        md.append(f"| {name} | {E[i]:.2f} | {rf[i]:.4f} | {rf_err[i]:.4f}"
                  f" | {rr[i]:.4f} | {100 * delta[i]:+.2f}% |")
    md += [
        "",
        "## Curve file",
        "",
        "`handoff2_ratio_curve.csv`, columns:",
        "",
        "- `kev`     - energy grid,"
        f" {HANDOFF_LO_KEV:.0f}-{HANDOFF_HI_KEV:.0f} keV in"
        f" {HANDOFF_STEP_KEV:.2f} keV steps;",
        "- `R`       - smooth frontal ratio, stage-1 detector model"
        " corrected by a GP on the log-residuals of the eight lines"
        f" (closure at the measured points: max {closure_pct.max():.2f}%);",
        "- `R_sigma` - 1-sigma band of that correction;",
        "- `R_model` - the bare parametric stage-1 curve (physics only);",
        "- `R_tilt`  - R times the fitted tilt shift, i.e. the ratio that"
        " applies to the tilted (ruotato) scan.",
        "",
        "## How to use it (N2N target scaling, PLAN 8.4)",
        "",
        "```python",
        "from src.data.cross_detector import ratio_curve_from_csv",
        "R = ratio_curve_from_csv(path, n_channels, slope, intercept)"
        "   # kev, R",
        "```",
        "",
        "Use `R` for prova1/prova2 pixels and `R_tilt` (`r_col=\"R_tilt\"`)"
        " for the ruotato pixels; keep the loss masked to 3.5-15.5 keV,"
        " outside which the curve is model extrapolation rather than"
        " measurement.",
        "",
        "Supersedes the provisional curve interpolated from"
        " `efficiency_ratios.csv`: those are full-frame ratios, which"
        " PLAN 8.7 showed to carry a field-of-view artifact (up to 4% on"
        " R itself, sign flip on the tilt shift).",
    ]
    md_path = os.path.join(OUTPUT_DIR, "handoff2_ratio_curve.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")

    # ---- figure --------------------------------------------------------
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4.4))

    axa.errorbar(E, rf, yerr=rf_err, fmt="o", capsize=3,
                 label="frontal (data)")
    axa.plot(Ef, det_ratio(Ef, *p1_opt), "-",
             label="detector model fit")
    in_grid = (Eh >= Ef.min()) & (Eh <= Ef.max())
    axa.plot(Eh[in_grid], r_smooth[in_grid], ":", color="0.35", lw=1.4,
             label="handoff-2 curve (model x GP residual)")
    for x, y, name in zip(E, rf, els):
        axa.annotate(name, (x, y), textcoords="offset points",
                     xytext=(4, 4), fontsize=8)
    axa.set_yscale("log")
    axa.set_xlabel("line energy (keV)")
    axa.set_ylabel("R = det10264 / det19511")
    axa.set_title("Stage 1: detector part (frontal)")
    axa.legend()
    axa.grid(alpha=0.3)

    axb.fill_between(Ef, 100 * (gp_mean - 2 * gp_sig),
                     100 * (gp_mean + 2 * gp_sig), color="0.85",
                     label="GP regression (2$\\sigma$)")
    axb.plot(Ef, 100 * gp_mean, "--", color="0.45", lw=1)
    axb.errorbar(E, 100 * delta, yerr=100 * delta_err, fmt="s",
                 capsize=3, label="tilt shift (data)")
    axb.plot(Ef, 100 * tilt_shift(Ef, *p2_opt), "-",
             label=f"geometric model fit ({TILT_DEG:.1f}\N{DEGREE SIGN} tilt)")
    axb.axhline(0, color="gray", lw=0.8)
    for x, y, name in zip(E, 100 * delta, els):
        axb.annotate(name, (x, y), textcoords="offset points",
                     xytext=(4, 4), fontsize=8)
    axb.set_xlabel("line energy (keV)")
    axb.set_ylabel("tilt-induced shift of R (%)")
    axb.set_title("Stage 2: geometric part (tilted vs frontal)")
    axb.legend()
    axb.grid(alpha=0.3)

    fig.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "geometry_fit.png")
    fig.savefig(fig_path, dpi=200)

    print(f"\nSaved: {txt_path}")
    print(f"Saved: {fig_path}")
    print(f"Saved: {curve_path}")
    print(f"Saved: {md_path}")
