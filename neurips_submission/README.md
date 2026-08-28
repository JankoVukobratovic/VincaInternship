# neurips_submission — Sim2Science @ NeurIPS 2026

Team submission package built ON TOP of the validated `neurips-restore/`
infrastructure (measured forward model, residual U-Net, honest eval).
**Deadline: Aug 29, 2026, 23:59 AoE** (OpenReview, 5 pages, double-blind,
reproducibility checklist mandatory).

Paper story: *we got burned by an imperfect simulator (blur mismatch) →
we turn that into a systematic audit: how imperfect can a simulator be
(WP2), which statistics catch the imperfection blind (WP2), and how to
quantify the uncertainty it leaves behind (WP1) — with the regime map of
where learning pays at all (WP3), everything anchored on a REAL scan.*

## Who owns what

| WP | Owner | Folder | Deliverable |
|----|-------|--------|-------------|
| WP1 | Dimitrije | `wp1_uq_ensemble/` | Simulator-uncertainty ensemble UQ: N nets trained on jittered simulators, coverage/calibration analysis. Stretch: uncertainty-guided adaptive scanning. |
| WP2 | Dimitrije (from Aug 28) | `wp2_simulator_audit/` | Defect-tolerance curves (train on deliberately broken simulators, measure degradation) + blind diagnostics battery that identifies WHICH component is broken. |
| WP3 | Dimitrije (from Aug 28) | `wp3_degradation_benchmark/` | Degradation grid (angle x hole x dose), classical inpainting controls, the "when does ML pay?" regime-map figure. |
| infra + paper glue | Claude/Dimitrije | `common/`, `main.py`, `paper/` | Shared simulator knobs, trainer, eval, figures assembly, LaTeX skeleton, checklist, anonymization. |

Each experiment file states its own contract (inputs, outputs, definition
of done) in the module docstring, and marks the scientific core with
`TODO(WP-owner)`. The glue (loops, saving, plumbing) is already written —
you fill in the science, run it, and commit the CSV.

## How to run

Everything from the REPO ROOT:

    python neurips_submission/main.py --stage smoke    # day-1 check, ~2 min: everyone runs this first
    python neurips_submission/main.py --stage status   # what exists, what is missing, per WP
    python neurips_submission/main.py --stage wp1      # or wp2 / wp3: run your experiment
    python neurips_submission/main.py --stage figures  # assemble paper figures from results/*.csv

Each experiment also runs standalone with `--quick` (small grids, minutes)
before the full overnight run.

## Shared contracts (do not break these)

1. **`common/perturb.SimKnobs`** is THE interface to the simulator.
   WP1 samples knobs *within* calibration uncertainty (`config.JITTER`),
   WP2 sets them *beyond* it (`config.DEFECT_LADDERS`), WP3 keeps them
   nominal and sweeps the degradation (`config.GRID`) instead. If you
   need a new knob, add it to `SimKnobs` + `config.py` and tell the team —
   never fork a private simulator variant.
2. **Results schema**: every experiment appends rows to
   `results/<experiment>.csv` via `common.io_utils` with the shared
   columns (`element, candidate, region, r, ssim, bias_pct, cv_ratio,
   n_px` + experiment-specific keys). Figures are built from CSVs only,
   never from in-memory state, so runs are restartable and mergeable.
3. **Evaluation is frozen**: metrics come from `neurips-restore/src/eval.py`
   through `common.restore.score_candidates`. Nobody edits metrics after
   the first full run.
4. **Honest-evaluation rules** of `neurips-restore/README.md` hold:
   prova2 and the real ruotato are NEVER training data; every WP's final
   table includes the real-scan anchor row; matching-but-not-beating a
   baseline is a negative result and gets reported as such.
5. `neurips-restore/src/` is FROZEN — bug fixes only with team sign-off.

## WP1 status (Dimitrije, 2026-08-28): IMPLEMENTED, full run in progress

`wp1_uq_ensemble/exp_ensemble_uq.py` is complete: 12 jitter + 12 control
members, coverage for three uncertainty bands (ensemble spread only /
spread + propagated measurement noise / + reference-map noise), variance
decomposition (jitter minus control = the simulator's share), error-ranking
diagnostics (Spearman, AUSE), accuracy of the ensemble means, the real-scan
anchor, three figures and `results/wp1_summary.txt`.  The stretch
`exp_adaptive_scan.py` (uncertainty-guided tile acquisition vs random /
raster / oracle) is implemented too.

Simulator contract additions (announced here, `--stage verify` still green):
`SimKnobs.warp_rot_deg` (registration rotation error) and
`SimKnobs.gain_pct_offset` (per-line additive error on the measured
tilt response, %).  `config.JITTER` now carries the MEASURED calibration
uncertainties with the source of every number in the comment block.

Parallel training on a many-core machine (4 shells, ~1 h total, then the
evaluation picks up the cached members):

    python neurips_submission/wp1_uq_ensemble/exp_ensemble_uq.py --train-only --kind jitter  --members 0-5  --threads 3
    python neurips_submission/wp1_uq_ensemble/exp_ensemble_uq.py --train-only --kind jitter  --members 6-11 --threads 3
    python neurips_submission/wp1_uq_ensemble/exp_ensemble_uq.py --train-only --kind control --members 0-5  --threads 3
    python neurips_submission/wp1_uq_ensemble/exp_ensemble_uq.py --train-only --kind control --members 6-11 --threads 3
    python neurips_submission/main.py --stage wp1

WP1 CSVs: `wp1_ensemble_members`, `wp1_uq_coverage` (ensemble, band,
case, region, z, coverage, expected), `wp1_uq_diagnostics` (var_control,
var_jitter, var_sim, var_alea, spearman_*, ause_*), `wp1_uq_accuracy`
(shared schema), `wp1_adaptive_scan`.

## WP2 and WP3 status (Dimitrije took both over on 2026-08-28)

WP3 is DONE: `wp3_degradation_benchmark/exp_degradation_grid.py` (180-case
grid + sharp note + real anchor, 8 candidates incl. the WP1 ensemble mean,
four classical fills with OpenCV Telea/NS added to `common/classical.py`,
and a biharmonic+net hybrid), `results/wp3_regime_summary.txt`,
`figures/wp3_regime_map.png`, paper text `results/wp3_paper_section.md`.
Headline: the biharmonic fill beats the net inside the hole in 16/16 cells;
learned candidates beat everything on the footprint; the hybrid wins every
hole cell by 0.005 to 0.03.

WP2 diagnostics is DONE: `wp2_simulator_audit/exp_diagnostics.py`
(8-statistic battery, calibration-uncertainty null shared with WP1's
JITTER, pre-registered rule + a labelled post-hoc gain template),
`results/wp2_diag_summary.txt`, `figures/wp2_diag_confusion.png`, text in
`results/wp2_paper_section.md`.  WP2 tolerance
(`exp_defect_tolerance.py`, 18 rungs cached in `results/wp2_rungs/`,
nominal band = WP1's 12 control members) runs like WP1 in parallel shells:

    python neurips_submission/wp2_simulator_audit/exp_defect_tolerance.py --list
    python neurips_submission/wp2_simulator_audit/exp_defect_tolerance.py --train-only --rungs 1-5 --threads 3   # x4 shells
    python neurips_submission/main.py --stage wp2

Instrument facts for the setup paragraph: `INSTRUMENT.md`.

## WP2 + WP3 + WP4 status (Dimitrije, 2026-08-28): ALL DONE AND RUN

WP2: 18-rung defect-tolerance ladder (results/wp2_tolerance_summary.txt with
the visibility x damage crosstab) + blind diagnostics with the
calibration-uncertainty null (results/wp2_diag_summary.txt).  WP3: full
180-case grid with four classical controls, the hybrid, the regime map and
the real anchor (results/wp3_regime_summary.txt).  WP4 (new folder
wp4_closed_loop/): ABC posterior over SimKnobs from the single real scan
using the WP2 battery as summaries, posterior predictive check, and a
12-member posterior ensemble (results/wp4_summary.txt).  Paper text drafts
with every number: results/wp*_paper_section.md.  Figures: wp2_tolerance_
curves, wp2_diag_confusion, wp3_regime_map, wp4_prior_posterior (+ WP1's
four).  `--stage verify` green throughout.

## Outputs

    results/    *.csv (committed), *.pt / *.npy (gitignored, big)
    figures/    paper-ready .png/.pdf assembled by main.py --stage figures
    paper/      LaTeX (NeurIPS 2026 template, dblblindworkshop) — added
                separately; drop the official style files in paper/ when
                downloaded from the CfP page

## Timeline to the deadline (submitted 23:59 AoE Aug 29)

| Date | Milestone |
|------|-----------|
| Aug 21–22 | everyone: smoke passes; WP cores implemented; `--quick` runs look sane |
| Aug 23–24 | full runs (overnight is fine — one training is ~4–8 min, whole ladders fit in hours); first figures |
| Aug 25 | EXPERIMENT FREEZE; regime map + tolerance curves + coverage plots final; sections drafted |
| Aug 26–28 | (EuCAIFCon) paper text only; reruns only for bugs |
| Aug 28–29 | assemble, checklist, anonymize, reciprocal reviewer agreed, SUBMIT |
