"""
Phase B (dual-detector paper): build and smoke-test the cross-detector
Noise2Noise pairs (PLAN §4.3 + amendment 8.4).

Loads full spectral cubes for prova1, prova2 and the tilted ruotato
scan (both detectors each), assembles the per-scan split - training on
prova1 + ruotato with spatial-block validation, external evaluation on
prova2 - loads the R(E) target-scaling curves (handoff 2 if present,
otherwise the provisional per-element table), and sanity-checks batches
through UNet1D.

Outputs:
    data/processed/<scan>_<det>_raw.npy         spectral cube caches
    data/processed/cross_detector_split.json    split record
    data/processed/ratio_curve_handoff2.npz     R(E) frontal + tilted
    data/processed/ratio_curve_provisional.npy  fallback R(E), (C,)

Run from xrf-denoise/:
    python scripts/06_cross_detector_pairs.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
VINCA_ROOT = PROJECT_ROOT.parent

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.data.loader import load_datacube
from src.data.cross_detector import (
    XRFCrossDetectorDataset,
    build_cross_detector_splits,
    clamp_curve_to_mask,
    make_channel_mask,
    ratio_curve_from_csv,
    ratio_curve_from_table,
)
from src.models.unet1d import UNet1D

cfg = Config()

# scan -> (accepted folder names, rows, cols)
SCANS = {
    "prova1":  (["aurora-antico1-prova1"], 60, 120),
    "prova2":  (["aurora-antico1-prova2"], 60, 120),
    "ruotato": (["antico1-prova4-ruotato", "aurora-antico1-ruotato"], 45, 80),
}
RATIO_TABLE = VINCA_ROOT / "results" / "detector_diff" / "efficiency_ratios.csv"
HANDOFF2_CURVE = (VINCA_ROOT / "results" / "detector_diff"
                  / "handoff2_ratio_curve.csv")


def resolve_scan_dir(names: list[str]) -> Path:
    for root in (VINCA_ROOT / "Resources", VINCA_ROOT):
        for name in names:
            p = root / name
            if (p / cfg.detector_a).is_dir() and (p / cfg.detector_b).is_dir():
                return p
    sys.exit(f"ERROR: raw data not found for {names} "
             "(see README: Setup / Raw MCA data).")


if __name__ == "__main__":
    processed = cfg.abs_path(cfg.processed_dir)
    processed.mkdir(parents=True, exist_ok=True)

    # ── 1. load spectral cubes (cached) ─────────────────────────
    scan_specs = {}
    for scan, (names, rows, cols) in SCANS.items():
        scan_dir = resolve_scan_dir(names)
        print(f"[{scan}] {scan_dir}")
        cubes = []
        for det in (cfg.detector_a, cfg.detector_b):
            cube, meta = load_datacube(
                scan_dir, det, rows=rows, cols=cols,
                cache_path=processed / f"{scan}_{det}_raw.npy",
            )
            cubes.append(cube)
        scan_specs[scan] = (cubes[0], cubes[1])
        s_a = cubes[0].sum(axis=2).mean()
        s_b = cubes[1].sum(axis=2).mean()
        print(f"  mean total counts/px: A={s_a:.0f}  B={s_b:.0f}  "
              f"A/B={s_a / s_b:.3f}")

    n_ch = scan_specs["prova1"][0].shape[2]
    assert all(v[0].shape[2] == n_ch for v in scan_specs.values()), \
        "channel counts differ between scans"

    # ── 2. split (train: prova1+ruotato, val: spatial blocks, test: prova2)
    splits, record = build_cross_detector_splits(
        scan_specs, seed=cfg.seed,
    )

    # ── 3. ratio curve(s) + loss mask ───────────────────────────
    # Handoff 2 if it is there (smooth R(E) from the registered-overlap
    # fit, with a separate column for the tilted scan), otherwise the
    # provisional per-element interpolation of the full-frame table.
    mask = make_channel_mask(n_ch, cfg.cal_slope, cfg.cal_intercept)

    if HANDOFF2_CURVE.exists():
        frontal = clamp_curve_to_mask(ratio_curve_from_csv(
            HANDOFF2_CURVE, n_ch, cfg.cal_slope, cfg.cal_intercept), mask)
        tilted = clamp_curve_to_mask(ratio_curve_from_csv(
            HANDOFF2_CURVE, n_ch, cfg.cal_slope, cfg.cal_intercept,
            r_col="R_tilt"), mask)
        curve = {"prova1": frontal, "prova2": frontal, "ruotato": tilted}
        np.savez(processed / "ratio_curve_handoff2.npz",
                 frontal=frontal, tilted=tilted)
        print(f"\nR(E) from handoff 2 ({HANDOFF2_CURVE.name}):")
        print(f"  frontal: min={frontal.min():.3f} max={frontal.max():.3f}")
        print(f"  tilted : min={tilted.min():.3f} max={tilted.max():.3f}"
              f"  (max tilt shift {100 * (tilted / frontal - 1).max():+.1f}%)")
        curve_ref = frontal
    else:
        curve = ratio_curve_from_table(
            RATIO_TABLE, n_ch, cfg.cal_slope, cfg.cal_intercept,
        )
        np.save(processed / "ratio_curve_provisional.npy", curve)
        print(f"\nR(E) PROVISIONAL curve (handoff 2 missing): "
              f"min={curve.min():.3f} max={curve.max():.3f}")
        curve_ref = curve

    print(f"loss mask: {mask.sum()}/{n_ch} channels "
          f"({cfg.cal_intercept + 0 * cfg.cal_slope:.1f}"
          f"-{cfg.cal_intercept + (n_ch - 1) * cfg.cal_slope:.1f} keV axis, "
          f"window 3.5-15.5 keV)")

    # global scale: keep inputs O(1) - 99.9th percentile of train counts
    sample = splits["train"][0][1][::37].ravel()
    global_scale = float(np.percentile(sample, 99.9)) or 1.0
    print(f"global scale (p99.9 of train counts): {global_scale:.1f}")

    datasets = {
        name: XRFCrossDetectorDataset(
            parts, ratio_curve=curve, global_scale=global_scale,
            loss_mask=mask,
        )
        for name, parts in splits.items()
    }
    for name, ds in datasets.items():
        print(f"  {name:5s}: {len(ds):6d} examples "
              f"({len(ds) // 2} px x 2 directions)")
    ds_total = sum(len(d) for d in datasets.values())
    print(f"  total: {ds_total} (PLAN: ~36 000)")

    datasets["train"].save_record(
        record, processed / "cross_detector_split.json")

    # ── 4. smoke test: batches through UNet1D ───────────────────
    model = UNet1D(in_channels=1, base_filters=cfg.base_filters,
                   n_blocks=cfg.n_encoder_blocks, dropout=cfg.dropout)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"\nUNet1D: {n_par / 1e6:.2f} M params")

    for name, ds in datasets.items():
        dl = DataLoader(ds, batch_size=8, shuffle=(name == "train"))
        x, y = next(iter(dl))
        assert x.shape == y.shape == (8, 1, n_ch), (name, x.shape)
        assert torch.isfinite(x).all() and torch.isfinite(y).all(), name
        with torch.no_grad():
            out = model(x)
        assert out.shape == x.shape, (name, out.shape)
        m = ds.loss_mask
        print(f"  {name:5s}: batch OK  x in [{x.min():.3f},{x.max():.3f}]  "
              f"masked-loss channels={int(m.sum())}  "
              f"unet out {tuple(out.shape)}")

    # direction/scaling sanity on a fixed pixel of prova1
    scan, fa, fb, idx = splits["test"][0]
    j = idx[len(idx) // 2]
    a, b = fa[j], fb[j]
    ca_ch = cfg.kev_to_channel(3.69)
    sl = slice(ca_ch - 5, ca_ch + 6)
    print(f"\nCa-window sanity (pixel {j}, {scan}): "
          f"A={a[sl].sum():.0f}  B={b[sl].sum():.0f}  "
          f"BxR={float((b * curve_ref)[sl].sum()):.0f}  (should approach A)")

    print("\nSmoke test passed.")
