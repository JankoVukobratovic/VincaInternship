# neurips-restore

Physics-guided restoration of degraded MA-XRF scans (NeurIPS workshop
project, Paris). Third paper in the series: the dual-detector paper
measured the instrument physics; here that measured physics becomes the
guidance of a generative/learned restoration model.

**Claim under test:** a scan acquired under degraded geometry (tilted /
rotated canvas) can be restored to its frontal equivalent by combining
the measured forward model with a learned prior - with per-pixel
uncertainty, and without hallucinating pigment.

## Assets inherited from the dual-detector work

| Asset | Where |
|---|---|
| Affine registration frontal <-> tilted | `results/registration/affine_params.csv` (script 08) |
| Per-element tilt gains, %/deg | `results/registration/positioning_sensitivity.csv` (script 11) |
| Geometric model (s, Ec, c) + detector model | `scripts/07_geometry_fit.py`, `results/detector_diff/geometry_fit.txt` |
| Flat-field of the detector ratio | `results/detector_diff/flatfield_combined.npy` (script 10) |
| Smooth R(E) curve | `results/detector_diff/handoff2_ratio_curve.csv` |
| Per-element maps, all scans x detectors | `results/detector_diff/_npy_cache/{scan}_{det}_{el}.npy` |
| Raw cubes, frontal scans | `results/vulnerability_mapping/ablation_cube_*.npy` |

Scans: prova1, prova2 (frontal, 60x120 px, 7 days apart - the noise
floor pair) and ruotato (tilted, 45x80 px). 8 reliable lines: Ca, Ti,
Fe, Cu, PbLl, PbLa, PbLb, PbLg. Working representation for the MVP:
detector-summed per-element maps.

## MVP (feasibility gate for the paper)

1. **Forward simulator** (`src/forward_model.py`): frontal maps ->
   simulated tilted maps: affine warp + per-element tilt gain +
   Poisson-consistent noise. Fidelity test: simulate the tilt from
   prova1 and compare against the REAL ruotato scan.
2. **Deterministic baseline** (`src/eval.py`,
   `scripts/02_deterministic_baseline.py`): invert the operator on the
   real ruotato (warp back + divide gains) and score against the
   registered frontal truth. Any learned model must beat this.
3. **Learned restoration MVP**: conditional U-Net trained purely on
   physics-simulated pairs, tested on the REAL ruotato -> frontal task.
   Generative upgrade (diffusion posterior sampling + UQ) only if the
   MVP shows headroom over the baseline.

## Honest-evaluation rules (non-negotiable)

- Test case is the REAL tilted scan; simulated degradations are for
  training/validation only.
- Metrics vs the registered frontal truth on the common footprint:
  per-element Pearson r, SSIM, and **absolute level bias per line**
  (scale-invariant metrics cannot see calibration errors - lesson from
  the N2N fusion, where Ca/Ti carried +-30% bias invisible to SNR).
- Always report the prova1<->prova2 noise floor as the upper bound.
- Contrast guard (cv ratio vs input) against restoration-by-blurring.
- A learned model that only matches the deterministic baseline is a
  negative result and gets reported as such.

## Layout

    src/        forward model, evaluation harness, models
    scripts/    numbered entry points (run from the REPO ROOT)
    results/    figures, tables, txt reports (npy is gitignored)
