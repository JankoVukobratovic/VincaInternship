"""config.py - single source of truth for all three workpackages.

Only stdlib here (no imports from common/) so every module can import it
without cycles.  If you change a grid or a ladder, announce it - CSVs
produced under different configs must not be silently mixed.
"""

# ---------------------------------------------------------------------------
# training defaults (adapted from the MVP: best val was at step ~450,
# wall clock ~4 min on CPU; time_budget_s is a hard cap per net)
# ---------------------------------------------------------------------------
TRAIN = dict(steps=1500, batch=8, lr=1e-3, val_every=50,
             patience=10, time_budget_s=900.0)
QUICK_TRAIN = dict(steps=120, batch=8, lr=1e-3, val_every=30,
                   patience=4, time_budget_s=180.0)

VAL_ANGLES = (4.0, 7.7, 10.0, 13.0, 16.0, 19.0, 22.0, 25.0)
VAL_REPS = 3
BASE_SEED = 20260821

# ---------------------------------------------------------------------------
# WP1 - calibration-uncertainty jitter (1-sigma of each SimKnobs field).
# Every number below is traced to a measured quantity of the calibration
# chain (scripts 08/11 registration + sensitivity, MVP-2 simulator check);
# nothing is a guess.  Sources, in the order of the fields:
#
#   noise_k_log_sd = 0.33
#       Transfer uncertainty of the calibrated noise law Var = k*counts
#       from the frontal pair (where k was fitted) to the tilted session.
#       MVP-2 check [3] (neurips-restore/results/simulator_check.txt)
#       measures sd_real/sd_pred on the real tilted scan; on the six
#       lines where texture does not leak into sd_real (Ca, Ti, Fe, Cu,
#       PbLl, PbLg) the log variance ratio 2*ln(sd_real/sd_pred) is
#       0.65, 0.50, -0.06, 0.22, -0.12, -0.13 -> sample sd 0.33.
#   gain_scale_sd = 0.20
#       The per-degree tilt slopes are extrapolated from ONE reference
#       mounting at 7.7 deg, and 7.7 deg is an UPPER bound on that angle
#       (results/registration/positioning_sensitivity.txt); the
#       foreshortening fit cannot resolve it (handoff1: alpha 1.8 deg,
#       range 0-6.4).  A +-20 % slope uncertainty corresponds to the
#       true mounting angle being 6.4 deg instead of 7.7 (the upper end
#       of the foreshortening range); it is a symmetric proxy for a
#       one-sided uncertainty and therefore a lower bound on it.
#   gain_pct_offset_sd = 1.0   (percent, independent per line)
#       Per-line level uncertainty of the measured gains: the fit error
#       tilt_sig_sum is 0.11-0.45 % per line and the session
#       repeatability floor of the levels is 0.96 % RMS (prova1 vs
#       prova2, same geometry; positioning_sensitivity.txt).  The
#       quadrature sum is 0.97-1.06 %, rounded to 1.0 % for all lines.
#   angle_bias_sd_deg = 0.5
#       Reading uncertainty of the goniometer mounting angle (assumed;
#       the registration itself cannot check angles below ~5 deg).
#       Enters the simulator only through the gain extrapolation, so it
#       is subdominant to gain_pct_offset_sd at every angle.
#   warp_shift_sd_px = 0.1
#       Spread of the translation across the five per-element affine
#       fits of results/registration/affine_params.csv: sd(tx) = 0.045
#       px, sd(ty) = 0.10 px (Ca, Ti, Fe, Cu, Pb).
#   warp_rot_sd_deg = 0.3
#       Spread of the rotation across the same five fits: 0.75, 1.44,
#       1.55, 1.59, 1.37 deg -> sd 0.33 (0.10 without the Ca outlier).
#       0.3 deg moves the far edge of the 80-px tilted frame by 0.2 px.
# ---------------------------------------------------------------------------
JITTER = dict(noise_k_log_sd=0.33,     # k multiplier ~ lognormal
              gain_scale_sd=0.20,      # g' = 1 + s*(g-1), s ~ N(1, sd)
              gain_pct_offset_sd=1.0,  # per-line additive on tilt_pct_sum, %
              angle_bias_sd_deg=0.5,   # simulator's belief about the tilt
              warp_shift_sd_px=0.1,    # registration translation error, px
              warp_rot_sd_deg=0.3)     # registration rotation error, deg
ENSEMBLE_N = 12
ENSEMBLE_N_QUICK = 3
COVERAGE_Z = (0.5, 1.0, 1.5, 2.0, 3.0)   # |err| <= z*sigma coverage probes
# held-out test cases for the UQ analysis (source prova2, never trained
# on): angle x hole x test-simulator x dose.  "validated" = the bilinear
# instrument emulator (MVP continuity, differs from the cubic training
# simulator = a realistic train/test simulator mismatch); "sharp" = the
# same sampling as the training simulator.
WP1_CASES = dict(angles=(8.0, 14.0, 20.0, 25.0),
                 holes=((0, 0), (14, 20)),
                 sims=("validated", "sharp"),
                 doses=(1.0, 0.5),
                 seed=0)
WP1_CASES_QUICK = dict(angles=(20.0,), holes=((0, 0), (14, 20)),
                       sims=("validated",), doses=(1.0,), seed=0)
WP1_NOISE_REPS = 8        # noise replicates per case for the aleatoric sigma
WP1_HARSH_CASE = dict(angle=20.0, hole=(14, 20), sim="validated", dose=1.0)

# ---------------------------------------------------------------------------
# WP2 - defect ladders: named corruptions BEYOND calibration uncertainty.
# Each entry: (label, SimKnobs kwargs).  'nominal' is the control and is
# always run first.  Ladders extended on 2026-08-28 (Dimitrije): the
# registration rotation family uses the SimKnobs.warp_rot_deg knob added
# for WP1; every rung trains ONE net with the seed of WP1's control_00 so
# that rung-to-rung differences are the simulator, not the init.
# ---------------------------------------------------------------------------
DEFECT_LADDERS = {
    "noise_k": [(f"k_x{v}", dict(noise_k_scale=v))
                for v in (0.25, 0.5, 2.0, 4.0)],
    "gain": [(f"gain_x{v}", dict(gain_scale=v))
             for v in (0.0, 0.5, 1.5, 2.0)],
    # +-2 deg enters only through the gain extrapolation (a 0.3 % level
    # effect at 20 deg) so the ladder reaches +-5 deg to find any damage
    "angle_bias": [(f"angle_{v:+.0f}deg", dict(angle_bias_deg=v))
                   for v in (-5.0, -2.0, 2.0, 5.0)],
    "blur": [("bilinear_v1", dict(blur_mode="bilinear"))],
    "warp_shift": [(f"shift_{v}px", dict(warp_shift_px=(v, 0.0)))
                   for v in (0.5, 1.0, 2.0)],
    "warp_rot": [(f"rot_{v}deg", dict(warp_rot_deg=v))
                 for v in (1.0, 2.0)],
}
# numeric x for the tolerance curves (nominal value first)
DEFECT_X = {"noise_k": ("noise_k_scale", 1.0), "gain": ("gain_scale", 1.0),
            "angle_bias": ("angle_bias_deg", 0.0), "blur": ("blur_mode", "cubic"),
            "warp_shift": ("warp_shift_px", 0.0), "warp_rot": ("warp_rot_deg", 0.0)}
# WP2 testbed: held-out simulated cases (prova2, validated emulator) + real
WP2_TEST = dict(angles=(8.0, 14.0, 20.0, 25.0), holes=((0, 0), (14, 20)),
                seed=0)
WP2_TEST_QUICK = dict(angles=(20.0,), holes=((0, 0), (14, 20)), seed=0)

# ---------------------------------------------------------------------------
# WP3 - degradation grid (inference-only sweep, no training):
# source prova2 (never in training), simulate tilt+hole+dose, restore
# with physics / physics+U-Net / classical controls, score vs prova2.
# ---------------------------------------------------------------------------
GRID = dict(
    angles=(8.0, 14.0, 20.0, 25.0),
    # (h, w) hole in the tilted frame, centered blocks; (0, 0) = no hole
    holes=((0, 0), (6, 8), (10, 14), (14, 20), (18, 26)),
    doses=(1.0, 0.5, 0.25),
    seeds=(0, 1, 2),
)
GRID_QUICK = dict(angles=(20.0,), holes=((0, 0), (14, 20)),
                  doses=(1.0,), seeds=(0,))

# elements used for headline figures (largest measured headroom)
FIG_LINES = ("Ca", "Cu", "PbLa")
