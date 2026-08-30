# Sim2Science submission: how wrong can your simulator be?

Workshop paper for Sim2Science (NeurIPS 2026, Paris). The full paper is
`paper/main.pdf` (built from `paper/main.tex`); this package holds the
code and results behind every number in it.

The study is built on `neurips-restore/` (a validated, measured forward
model and residual U-Net for restoring tilted MA-XRF scans of a
painting) and asks a different question of it: how imperfect is that
simulator, what does the imperfection cost a network trained on it, can
the imperfection be detected without ground truth, how should it be
represented in predictive uncertainty, and can one real measurement
correct it. Four work packages, all anchored on the same real scan.

## Layout

Run everything from the repository root.

```
config.py                    knob ladders, calibration sigmas (with sources), grids
common/                      shared simulator (SimKnobs), trainer, scoring, CSV schema
main.py                      pipeline: smoke | verify | status | wp1..wp4 | figures
wp1_uq_ensemble/              jitter-vs-control ensemble UQ, adaptive-scan pilot
wp2_simulator_audit/          defect-tolerance ladder, blind diagnostics
wp3_degradation_benchmark/    degradation grid vs classical inpainting controls
wp4_closed_loop/              rejection-ABC posterior over the simulator, SBC check
toy_generalization/           synthetic, XRF-unrelated replication of the recipe
paper/                        NeurIPS 2026 LaTeX source, compiled PDF, checklist
results/                      CSVs (committed) and *_summary.txt / *_paper_section.md
figures/                      figures assembled from the CSVs
```

## Running

```
python main.py --stage smoke     # ~2 min sanity check
python main.py --stage verify    # contract checks after touching common/*.py
python main.py --stage status    # what result CSVs exist
python main.py --stage wp1       # or wp2 / wp3, --quick for a fast pass
python main.py --stage figures   # rebuild every figure from its CSV
```

Training scripts also accept `--train-only --members a-b --threads n` so
an ensemble can be split across several parallel shells; see the
docstring at the top of each `exp_*.py` file for the exact commands and
what each stage produces.

## Shared contracts

- `common.perturb.SimKnobs` is the only interface to the simulator.
  WP1 draws knobs within the calibration uncertainty (`config.JITTER`),
  WP2 sets them beyond it (`config.DEFECT_LADDERS`), WP3 keeps them
  nominal, WP4 infers them. A new knob is drawn last in `jittered()`,
  gated on its own sigma key, so old callers keep a bit-identical RNG
  stream; `--stage verify` checks this after any change.
- Every experiment writes to `results/<name>.csv` through
  `common.io_utils`; figures are rebuilt from those CSVs only, never
  from in-memory state, so any stage is restartable.
- Scoring is frozen in `neurips-restore/src/eval.py` via
  `common.restore.score_candidates`. `F2` and the real tilted scan are
  never used in training; every result table carries the real-scan
  anchor row.

## What each work package found

- **WP1**: an ensemble trained across simulators drawn within the
  calibration uncertainty spreads far wider than a fixed-simulator
  control on the real scan and is close to calibrated where the
  control is not; its mean is the first learned restoration that does
  not lose accuracy relative to the physics baseline on the real scan.
- **WP2**: an 18-rung defect ladder shows the training simulator
  tolerates large noise and gain errors but fails sharply on geometry;
  a blind diagnostic battery, thresholded on the calibration
  uncertainty, catches every defect that causes real damage.
- **WP3**: the learned prior wins everywhere on measured pixels but
  loses to a plain biharmonic fill inside missing data; a hybrid
  (classical fill, then the network) is the only candidate that wins
  everywhere.
- **WP4**: rejection ABC turns the real scan into a posterior over the
  simulator, rejecting the project's historical blur defect on its
  own; the posterior-trained ensemble is the most accurate restoration
  in the study, and simulation-based calibration on synthetic ground
  truth confirms the mechanism is not an artifact of that one scan.

Full numbers, honest caveats and proposed paper text for each of these
live in `results/wp*_summary.txt` and `results/wp*_paper_section.md`.
