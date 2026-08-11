# Handoff 2 (A -> B): response ratio R(E) = det10264 / det19511

Table 1 rows (registered overlap region, `results/registration/overlap_ratios.csv`) and the smooth curve derived from them.

| line | E (keV) | R frontal | sigma | R tilted | tilt shift |
|------|---------|-----------|-------|----------|------------|
| Ca | 3.69 | 5.7764 | 0.0130 | 6.3259 | +9.51% |
| Ti | 4.51 | 2.4219 | 0.0079 | 2.6130 | +7.89% |
| Fe | 6.40 | 1.2065 | 0.0026 | 1.2484 | +3.48% |
| Cu | 8.04 | 1.0116 | 0.0032 | 1.0370 | +2.51% |
| PbLl | 9.19 | 0.8280 | 0.0035 | 0.8620 | +4.11% |
| PbLa | 10.54 | 0.7647 | 0.0008 | 0.7813 | +2.16% |
| PbLb | 12.61 | 0.6989 | 0.0007 | 0.7113 | +1.78% |
| PbLg | 14.77 | 0.6302 | 0.0012 | 0.6338 | +0.58% |

## Curve file

`handoff2_ratio_curve.csv`, columns:

- `kev`     - energy grid, 2-20 keV in 0.02 keV steps;
- `R`       - smooth frontal ratio, stage-1 detector model corrected by a GP on the log-residuals of the eight lines (closure at the measured points: max 0.31%);
- `R_sigma` - 1-sigma band of that correction;
- `R_model` - the bare parametric stage-1 curve (physics only);
- `R_tilt`  - R times the fitted tilt shift, i.e. the ratio that applies to the tilted (ruotato) scan.

## How to use it (N2N target scaling, PLAN 8.4)

```python
from src.data.cross_detector import ratio_curve_from_csv
R = ratio_curve_from_csv(path, n_channels, slope, intercept)   # kev, R
```

Use `R` for prova1/prova2 pixels and `R_tilt` (`r_col="R_tilt"`) for the ruotato pixels; keep the loss masked to 3.5-15.5 keV, outside which the curve is model extrapolation rather than measurement.

Supersedes the provisional curve interpolated from `efficiency_ratios.csv`: those are full-frame ratios, which PLAN 8.7 showed to carry a field-of-view artifact (up to 4% on R itself, sign flip on the tilt shift).
