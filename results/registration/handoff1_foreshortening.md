# Handoff 1 (B -> A): vertical foreshortening from registration

**f = s_x / s_y (ruotato -> prova1) = 0.9995 +/- 0.0058**

- per-element spread (Ca, Ti, Fe, Cu, Pb): [0.991 , 0.9976, 1.004 , 1.0057, 0.9982]
- control (prova2 -> prova1, expected 1.0): f = 0.9963 -> pipeline
  noise floor |f-1| = 0.0037
- joint affine [sx, sy, rot_deg, shear, tx, ty] = [9.98400e-01, 9.98900e-01, 1.43620e+00, 5.20000e-03, 1.83327e+01,
 3.99580e+00]

## Implied tilt angle

Assuming equal motor pitch (mm/px) in both scans and pure forward tilt,
f = cos(alpha):

**alpha = 1.8 deg  (range 0.0-6.4 deg from the element spread)**

## Caveats

- If the ruotato scan used a different step size, the *isotropic* part
  of the scale is absorbed by (sx, sy) jointly and f is unaffected; an
  *anisotropic* pitch difference would contaminate f directly. Cross-check
  against Ridolfi's number (PLAN §3.1.2) before using alpha in the fit.
- The registration measures |cos| only; it cannot distinguish tilt
  forward from backward.
- Small angles are at the edge of resolution: the control run puts the
  noise floor at |f-1| ~ 0.004, i.e. angles below
  ~5 deg are not distinguishable from zero
  by foreshortening alone.
