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
# WP1 - calibration-uncertainty jitter (1-sigma of each knob).
# PLACEHOLDER values: TODO(WP1) replace with the actual calibration
# uncertainties from the dual-detector fits (noise k from script MVP-1
# spread, gains from positioning_sensitivity.csv errors, warp from the
# affine-fit residuals) and document the source of each number.
# ---------------------------------------------------------------------------
JITTER = dict(noise_k_log_sd=0.15,     # k multiplier ~ lognormal
              gain_scale_sd=0.15,      # g' = 1 + s*(g-1), s ~ N(1, sd)
              angle_bias_sd_deg=0.4,   # simulator's belief about the tilt
              warp_shift_sd_px=0.3)    # registration error, px
ENSEMBLE_N = 12
ENSEMBLE_N_QUICK = 3
COVERAGE_Z = (0.5, 1.0, 1.5, 2.0, 3.0)   # |err| <= z*sigma coverage probes

# ---------------------------------------------------------------------------
# WP2 - defect ladders: named corruptions BEYOND calibration uncertainty.
# Each entry: (label, SimKnobs kwargs).  'nominal' is the control and is
# always run first.  TODO(WP2): confirm/extend the ladders - the design
# of the corruptions is the scientific content of WP2.
# ---------------------------------------------------------------------------
DEFECT_LADDERS = {
    "noise_k": [(f"k_x{v}", dict(noise_k_scale=v))
                for v in (0.25, 0.5, 2.0, 4.0)],
    "gain": [(f"gain_x{v}", dict(gain_scale=v))
             for v in (0.0, 0.5, 1.5, 2.0)],
    "angle_bias": [(f"angle_{v:+.0f}deg", dict(angle_bias_deg=v))
                   for v in (-2.0, -1.0, 1.0, 2.0)],
    "blur": [("bilinear_v1", dict(blur_mode="bilinear"))],
    "warp": [(f"shift_{v}px", dict(warp_shift_px=(v, 0.0)))
             for v in (1.0, 2.0)],
}

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
