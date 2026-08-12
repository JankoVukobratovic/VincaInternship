"""
16_closure_test.py
Closure test: does the two-stage characterization of script 07 TRANSFER
to the tilted scan?

BEFORE: compare the measured tilted-scan ratios R_ruotato(E) against the
frontal detector model alone, i.e. residuals

    delta(E) = R_ruotato / R_frontal_overlap - 1

(this is exactly the raw tilt shift of 07). It is several percent and
its chi2 against the measurement errors (dof = 8, no free parameters)
is far above 1: the frontal characterization alone does NOT describe
the tilted scan.

AFTER: divide out the fitted stage-2 geometric factor

    F(E) = (1 + c) * geom_ratio(E, s, Ec, 7.7) / geom_ratio(E, s, Ec, 0)
         = 1 + tilt_shift(E, s, Ec, c)

and recompute the residuals

    delta'(E) = (R_ruotato / R_frontal_overlap) / F(E) - 1

with errors delta_err / F(E). The tilt signature should collapse to the
noise level (dof = 8 - 3 = 5, the three stage-2 parameters were fitted
on these same data).

HONESTY NOTE: the "after" residuals are correlated with the fit -- the
stage-2 parameters (s, Ec, c) were fitted to exactly these eight points.
This is a consistency / closure check of the parametrization, not an
independent validation; an independent validation would need a scan at
a second tilt angle. The chi2_after is by construction identical to the
stage-2 fit chi2 of 07.

Cross-check: the smooth handoff-2 curve R(E) interpolated at the eight
line energies is compared against R_frontal_overlap (expected < 0.5%,
cf. "closure at the 8 measured lines" in geometry_fit.txt).

The stage-1 and stage-2 fits are re-run here exactly as in the __main__
of 07 (deterministic curve_fit), importing 07 as a module for the model
functions and constants.

Input : results/registration/overlap_ratios.csv
        results/detector_diff/handoff2_ratio_curve.csv
Output: results/detector_diff/closure_test.png
        results/detector_diff/closure_test.txt

Run from the project root:
    python scripts/16_closure_test.py
"""

import csv
import importlib
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
gf = importlib.import_module("07_geometry_fit")  # name starts with a digit

CSV_RATIOS = os.path.join("results", "registration", "overlap_ratios.csv")
CSV_CURVE  = os.path.join("results", "detector_diff",
                          "handoff2_ratio_curve.csv")
OUTPUT_DIR = os.path.join("results", "detector_diff")
PNG_OUT    = os.path.join(OUTPUT_DIR, "closure_test.png")
TXT_OUT    = os.path.join(OUTPUT_DIR, "closure_test.txt")

# figure colors (validated categorical palette, light mode)
C_BEFORE = "#2a78d6"   # slot 1 blue  -- before, filled markers
C_AFTER  = "#eb6834"   # slot 2 orange -- after, open markers
C_MODEL  = "#898781"   # muted ink    -- fitted stage-2 curve
C_GRID   = "#e1e0d9"
C_AXIS   = "#c3c2b7"
C_INK    = "#0b0b0b"


def main():
    # ---- data: the 8 reliable lines, as in 07 ---------------------------
    rows = []
    with open(CSV_RATIOS, newline="") as f:
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

    # ---- refit stage 1 and stage 2 exactly as 07 __main__ does ----------
    p1_opt, _ = curve_fit(
        gf.det_ratio, E, rf, p0=(0.9, 800.0, 300.0), sigma=rf_err,
        absolute_sigma=True,
        bounds=([0.1, 0.0, 50.0], [5.0, 3000.0, 1500.0]), maxfev=20000,
    )
    p2_opt, p2_cov = curve_fit(
        gf.tilt_shift, E, delta, p0=(0.3, 6.0, -0.02),
        sigma=delta_err, absolute_sigma=True,
        bounds=([-3.0, 1.0, -0.2], [3.0, 20.0, 0.2]),
        maxfev=20000,
    )
    p2_err = np.sqrt(np.diag(p2_cov))
    s_fit, ec_fit, c_fit = p2_opt

    # ---- BEFORE: frontal detector model alone ---------------------------
    # residual = R_ruotato / R_frontal - 1, tested against zero shift;
    # no parameter of this comparison was fitted to the tilted scan.
    res_before = delta
    err_before = delta_err
    chi2_before = np.sum((res_before / err_before) ** 2)
    dof_before  = len(E)                    # no free parameters

    # ---- AFTER: divide out the fitted stage-2 geometric factor ----------
    geom_factor = ((1.0 + c_fit)
                   * gf.geom_ratio(E, s_fit, ec_fit, gf.TILT_DEG)
                   / gf.geom_ratio(E, s_fit, ec_fit, 0.0))
    res_after = (rr / rf) / geom_factor - 1.0
    err_after = delta_err / geom_factor
    chi2_after = np.sum((res_after / err_after) ** 2)
    dof_after  = len(E) - len(p2_opt)       # 3 params fitted on same data

    # ---- cross-check: handoff-2 smooth curve at the line energies -------
    kev_c, r_c = [], []
    with open(CSV_CURVE, newline="") as f:
        for row in csv.DictReader(f):
            kev_c.append(float(row["kev"]))
            r_c.append(float(row["R"]))
    r_at_lines  = np.interp(E, np.asarray(kev_c), np.asarray(r_c))
    closure_pct = 100.0 * (r_at_lines / rf - 1.0)

    # ---- text report -----------------------------------------------------
    lines = [
        "Closure test of the two-stage characterization (scripts 07/16)",
        f"tilt angle: {gf.TILT_DEG:.1f} deg;"
        f" stage-2 refit: s = {s_fit:+.3f} +- {p2_err[0]:.3f},"
        f" Ec = {ec_fit:.2f} +- {p2_err[1]:.2f} keV,"
        f" c = {c_fit:+.4f} +- {p2_err[2]:.4f}",
        "",
        "BEFORE -- tilted ratios vs frontal detector model alone:",
        "  residual = R_ruotato / R_frontal_overlap - 1"
        "  (raw tilt shift)",
        f"  chi2/dof = {chi2_before:.1f}/{dof_before}"
        f" = {chi2_before / dof_before:.1f}"
        f"   max |residual| = {100 * np.max(np.abs(res_before)):.2f}%",
        "",
        "AFTER -- stage-2 geometric factor"
        " (1+c) * geom(7.7 deg)/geom(0) divided out:",
        f"  chi2/dof = {chi2_after:.1f}/{dof_after}"
        f" = {chi2_after / dof_after:.1f}"
        f"   max |residual| = {100 * np.max(np.abs(res_after)):.2f}%",
        "",
        "per-line residuals (percent):",
        "  line    E(keV)   before +- err     after +- err",
    ]
    for i, name in enumerate(els):
        lines.append(
            f"  {name:<6}  {E[i]:5.2f}  {100 * res_before[i]:+7.2f}"
            f" +- {100 * err_before[i]:4.2f}   {100 * res_after[i]:+7.2f}"
            f" +- {100 * err_after[i]:4.2f}"
        )
    lines += [
        "",
        "cross-check -- handoff-2 smooth curve R(E) interpolated at the"
        " 8 lines vs R_frontal_overlap:",
        f"  max |closure| = {np.max(np.abs(closure_pct)):.2f}%,"
        f"  mean |closure| = {np.mean(np.abs(closure_pct)):.2f}%"
        "  (expected < 0.5%, cf. geometry_fit.txt)",
    ]
    for i, name in enumerate(els):
        lines.append(f"  {name:<6}  {E[i]:5.2f}  {closure_pct[i]:+5.2f}%")
    lines += [
        "",
        "CAVEAT: the 'after' residuals are correlated with the fit --"
        " the three stage-2",
        "parameters (s, Ec, c) were fitted on these same eight points,"
        " so chi2_after",
        "reproduces the stage-2 fit chi2 by construction. This is a"
        " consistency /",
        "closure check of the parametrization, not an independent"
        " validation;",
        "independent validation would require a scan at a second tilt"
        " angle.",
        "The residual chi2/dof > 1 after correction reflects the"
        " bootstrap errors",
        "capturing only counting statistics, not the ~0.4-0.6%"
        " scan-to-scan",
        "systematics (cf. 07).",
    ]
    print("\n".join(lines))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(TXT_OUT, "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---- figure ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    Ef = np.linspace(3.2, 15.5, 400)
    ax.plot(Ef, 100 * gf.tilt_shift(Ef, *p2_opt), "--", color=C_MODEL,
            lw=1.2, zorder=1,
            label="fitted stage-2 geometric model (divided out)")
    ax.axhline(0.0, color=C_AXIS, lw=1.0, zorder=1)

    ax.errorbar(E, 100 * res_before, yerr=100 * err_before, fmt="o",
                color=C_BEFORE, markersize=8, capsize=3, lw=1.4,
                zorder=3, label="before: frontal detector model alone")
    ax.errorbar(E, 100 * res_after, yerr=100 * err_after, fmt="o",
                markerfacecolor="white", markeredgecolor=C_AFTER,
                markeredgewidth=1.6, ecolor=C_AFTER, color=C_AFTER,
                markersize=8, capsize=3, lw=1.4, zorder=4,
                label="after: geometric factor divided out")

    for x, y, name in zip(E, 100 * res_before, els):
        ax.annotate(name, (x, y), textcoords="offset points",
                    xytext=(5, 6), fontsize=8, color=C_INK)

    ax.text(0.97, 0.80,
            "$\\chi^2$/dof: "
            f"{chi2_before / dof_before:.0f} $\\rightarrow$ "
            f"{chi2_after / dof_after:.1f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, color=C_INK)

    ax.set_xlabel("line energy (keV)")
    ax.set_ylabel("residual of $R_{\\mathrm{ruotato}}$ vs model (%)")
    ax.set_title("Closure test: the fitted geometry removes the tilt"
                 f" signature ({gf.TILT_DEG:.1f}\N{DEGREE SIGN} tilt)")
    ax.grid(color=C_GRID, lw=0.8)
    for spine in ax.spines.values():
        spine.set_color(C_AXIS)
    ax.tick_params(colors=C_INK)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(PNG_OUT, dpi=200)

    print(f"\nSaved: {TXT_OUT}")
    print(f"Saved: {PNG_OUT}")


if __name__ == "__main__":
    main()
