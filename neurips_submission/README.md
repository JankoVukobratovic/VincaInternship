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
| WP2 | ______ | `wp2_simulator_audit/` | Defect-tolerance curves (train on deliberately broken simulators, measure degradation) + blind diagnostics battery that identifies WHICH component is broken. |
| WP3 | ______ | `wp3_degradation_benchmark/` | Degradation grid (angle x hole x dose), classical inpainting controls, the "when does ML pay?" regime-map figure. |
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
