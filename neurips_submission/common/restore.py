"""restore.py - degrade / restore / score plumbing shared by all WPs.

The degradation side deliberately uses the VALIDATED instrument
emulator (forward_model.forward, bilinear + measured noise) as the
default test-case generator - continuity with the MVP harsh-demo
numbers; `sim="sharp"` is available for the acquisition-blur ablation
(a real tilted scan carries one resampling blur, see datagen.forward_sharp).
"""

import numpy as np
import torch

from common import core

fm, dg, ev = core.fm, core.dg, core.ev
ELEMENTS = core.ELEMENTS


def load_mvp_net():
    """The MVP checkpoint (nominal-simulator net) or None if missing."""
    import os
    if not os.path.exists(core.MVP_CKPT):
        return None
    ck = torch.load(core.MVP_CKPT, weights_only=False)
    net = core.RestorationUNet()
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net


def apply_network(net, tilted_maps, angle_deg, validity=None):
    """inverse() + network refinement -> (det dict, learned dict).

    Identical to scripts/03: both are frontal maps, NaN off-footprint.
    """
    det = fm.inverse(tilted_maps, angle_deg=angle_deg)
    x = dg.build_input(det, angle_deg, validity=validity)
    with torch.no_grad():
        rest = net.restore(torch.from_numpy(x[None]))[0].numpy()
    scales = dg.norm_scales()
    fp = dg.footprint()
    learned = {el: np.where(fp, rest[i] * scales[i], np.nan)
               for i, el in enumerate(ELEMENTS)}
    return det, learned


def centered_block(h, w):
    """(r0, c0, h, w) centered in the tilted frame; None if h*w == 0."""
    if h == 0 or w == 0:
        return None
    th, tw = core.TILTED_SHAPE
    return ((th - h) // 2, (tw - w) // 2, h, w)


def degrade(source="prova2", angle=20.0, block=None, dose=1.0, seed=0,
            sim="validated"):
    """Simulate one degraded acquisition from an UNSEEN source scan.

    Returns dict with: truth (dose-scaled frontal maps - the target a
    perfect restoration would reproduce), tilted (degraded measurement),
    validity (frontal float mask in [0,1]), hole (frontal bool mask of
    the dropped block), fp (footprint), and the case parameters.
    """
    rng = np.random.default_rng(seed)
    src = fm.load_summed_maps(source)
    truth = {el: m * dose for el, m in src.items()}
    if sim == "validated":
        tilted = fm.forward(truth, angle_deg=angle, rng=rng,
                            add_noise=True, input_noise="measured")
    elif sim == "sharp":
        tilted = dg.forward_sharp(truth, angle, rng)
    else:
        raise ValueError(f"unknown sim '{sim}'")

    v_tilt = np.ones(core.TILTED_SHAPE)
    if block is not None:
        r0, c0, h, w = block
        v_tilt[r0:r0 + h, c0:c0 + w] = 0.0
        for el in ELEMENTS:
            tilted[el][r0:r0 + h, c0:c0 + w] = 0.0
    validity = np.nan_to_num(fm.warp_tilted_to_frontal(v_tilt),
                             nan=0.0).astype(np.float32)
    fp = dg.footprint()
    hole = fp & (validity < 0.5)
    return {"truth": truth, "tilted": tilted, "validity": validity,
            "v_tilt": v_tilt, "hole": hole, "fp": fp, "angle": angle,
            "dose": dose, "block": block, "seed": seed, "sim": sim,
            "source": source}


def score_candidates(cands: dict, truth: dict, regions: dict,
                     min_px: int = 9) -> list:
    """Frozen scoring -> rows for io_utils.

    cands   : {candidate_name: {element: frontal map}}
    truth   : {element: frontal map}
    regions : {region_name: bool mask}; 'footprint' gets the full
              score_pair (r, ssim, bias, cv_ratio); other regions
              (e.g. 'hole') get r + bias only (too small for SSIM).
    """
    rows = []
    fp = regions.get("footprint")
    for el in ELEMENTS:
        drange = float(np.nanmax(truth[el][fp]) - np.nanmin(truth[el][fp])) \
            if fp is not None else None
        for name, maps in cands.items():
            for reg, mask in regions.items():
                if mask is None or int(mask.sum()) < min_px:
                    continue
                if reg == "footprint":
                    s = ev.score_pair(maps[el], truth[el], mask,
                                      data_range=drange)
                else:
                    a = np.nan_to_num(maps[el], nan=0.0)[mask]
                    t = truth[el][mask]
                    s = {"r": ev.pearson_r(maps[el], truth[el], mask),
                         "bias_pct": float(100.0 * (a.mean() - t.mean())
                                           / max(t.mean(), 1e-12)),
                         "n_px": int(mask.sum())}
                rows.append({"element": el, "candidate": name,
                             "region": reg, **s})
    return rows
