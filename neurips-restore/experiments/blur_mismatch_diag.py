"""Diagnostic: does the sim training input carry more blur than the real
restoration input?  Compare cv_ratio of the deterministic inverse on
(a) bilinear-forward sim, (b) nearest-neighbor-forward sim, (c) the
REAL ruotato (vs prova2, already published) -- all at 7.7 deg."""
import os
import sys

import numpy as np

REPO = r"C:\everything\projekti\VincaInternship"
sys.path.insert(0, os.path.join(REPO, "neurips-restore", "src"))
import eval as ev
import forward_model as fm

p1 = fm.load_summed_maps("prova1")
p2 = fm.load_summed_maps("prova2")
ruo = fm.load_summed_maps("ruotato")
mask = fm.frontal_footprint_mask()
gains = fm.tilt_gains(7.7)
ks = fm.calibrate_noise()

# nearest-neighbor forward sampling (sharp acquisition emulation)
A, t = fm.affine_ruotato_to_frontal()
hr, wr = fm.TILTED_SHAPE
hf, wf = fm.FRONTAL_SHAPE
cxr, cyr = (wr - 1) / 2.0, (hr - 1) / 2.0
cxf, cyf = (wf - 1) / 2.0, (hf - 1) / 2.0
yy, xx = np.meshgrid(np.arange(hr), np.arange(wr), indexing="ij")
xr = xx - cxr
yr = yy - cyr
xf = A[0, 0] * xr + A[0, 1] * yr + t[0] + cxf
yf = A[1, 0] * xr + A[1, 1] * yr + t[1] + cyf
rows = np.clip(np.round(yf).astype(int), 0, hf - 1)
cols = np.clip(np.round(xf).astype(int), 0, wf - 1)

from scipy.ndimage import map_coordinates

rng = np.random.default_rng(123)
print("cv_ratio of deterministic inverse vs its own truth (prova1 for sim,")
print("prova2 for real) and r; 7.7 deg, averaged over 8 noise draws (sim)")
print(f"{'line':6s} {'cv_bil':>8s} {'cv_nn':>8s} {'cv_cub':>8s}"
      f" {'cv_real':>8s} {'r_bil':>8s} {'r_nn':>8s} {'r_cub':>8s}"
      f" {'r_real':>8s}")
for el in fm.ELEMENTS:
    m = p1[el]
    g = gains[el]
    k = ks[el]
    cvb, cvn, cvc, rb, rn, rc = [], [], [], [], [], []
    for _ in range(8):
        # (a) bilinear forward, validated simulator
        sim_b = fm.forward({el: m}, 7.7, rng=rng, add_noise=True)[el]
        inv_b = fm.inverse({el: sim_b}, 7.7)[el]
        # (b) NN forward: sharp copy + gain + complement noise
        T = m[rows, cols] * g
        var = k * np.clip(T, 0, None) * max(1.0 - g, 0.0)
        T = np.clip(T + rng.normal(size=T.shape) * np.sqrt(var), 0, None)
        inv_n = fm.inverse({el: T}, 7.7)[el]
        # (c) cubic-spline forward at exact positions + complement noise
        Tc = map_coordinates(m, [yf, xf], order=3, mode="nearest") * g
        Tc = np.clip(Tc + rng.normal(size=Tc.shape)
                     * np.sqrt(k * np.clip(Tc, 0, None)
                               * max(1.0 - g, 0.0)), 0, None)
        inv_c = fm.inverse({el: Tc}, 7.7)[el]
        cvb.append(ev.cv_ratio(inv_b, m, mask))
        cvn.append(ev.cv_ratio(inv_n, m, mask))
        cvc.append(ev.cv_ratio(inv_c, m, mask))
        rb.append(ev.pearson_r(inv_b, m, mask))
        rn.append(ev.pearson_r(inv_n, m, mask))
        rc.append(ev.pearson_r(inv_c, m, mask))
    det_real = fm.inverse(ruo, 7.7)[el]
    cvr = ev.cv_ratio(det_real, p2[el], mask)
    rr = ev.pearson_r(det_real, p2[el], mask)
    print(f"{el:6s} {np.mean(cvb):8.4f} {np.mean(cvn):8.4f}"
          f" {np.mean(cvc):8.4f} {cvr:8.4f}"
          f" {np.mean(rb):8.4f} {np.mean(rn):8.4f} {np.mean(rc):8.4f}"
          f" {rr:8.4f}")
