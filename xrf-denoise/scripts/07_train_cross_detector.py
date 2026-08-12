"""
Phase B (dual-detector paper): train the cross-detector Noise2Noise
fusion network and export fused cubes for the benchmark (PLAN 4.4-4.5).

The two detectors see the same pixel at the same time, so each is a
noisy realization of the other's expected spectrum once the response
ratio R(E) is taken out (handoff 2). Training on such pairs is
Noise2Noise: no clean target exists, and the network cannot do better
than predicting the conditional mean, which is exactly the denoised
spectrum.

    direction 0 : input = det A, target = det B * R(E)   (A response scale)
    direction 1 : input = det B, target = det A / R(E)   (B response scale)

Loss: MSE masked to 3.5-15.5 keV. Scaled targets are no longer integer
counts, so Poisson NLL does not apply; outside the window the ratio
curve is model extrapolation rather than measurement.

Split (PLAN 4.3): train on prova1 + ruotato with spatial-block
validation carved out of them, prova2 never seen. The exported fused
cubes and the held-out pixel list let 09_fusion.py add the learned
variant to the same table as summing and inverse-variance weighting.

Inputs : data/processed/<scan>_<det>_raw.npy      (from script 06)
         ../results/detector_diff/handoff2_ratio_curve.csv
Outputs: experiments/cross_detector/checkpoints/best_model.pt
         experiments/cross_detector/results/history.json
         data/processed/fused_<scan>.npy          (A-scale counts)
         data/processed/fused_heldout_px.json     (unseen pixels)

Run from xrf-denoise/:
    python scripts/07_train_cross_detector.py            # full run
    python scripts/07_train_cross_detector.py --smoke    # 30 s check
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
VINCA_ROOT = PROJECT_ROOT.parent

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.data.cross_detector import (
    XRFCrossDetectorDataset,
    build_cross_detector_splits,
    clamp_curve_to_mask,
    make_channel_mask,
    ratio_curve_from_csv,
)
from src.models.unet1d import UNet1D

cfg = Config()

SCANS = {"prova1": (60, 120), "prova2": (60, 120), "ruotato": (45, 80)}
HANDOFF2_CURVE = (VINCA_ROOT / "results" / "detector_diff"
                  / "handoff2_ratio_curve.csv")
WARM_START = PROJECT_ROOT / "experiments" / "A_scratch" / "checkpoints" / "best_model.pt"
EXP_DIR = PROJECT_ROOT / "experiments" / "cross_detector"


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_cubes(processed: Path) -> dict:
    """Cached cubes from script 06; no MCA parsing here."""
    scan_specs = {}
    for scan in SCANS:
        cubes = []
        for det in (cfg.detector_a, cfg.detector_b):
            p = processed / f"{scan}_{det}_raw.npy"
            if not p.exists():
                sys.exit(f"ERROR: {p} missing - run "
                         "scripts/06_cross_detector_pairs.py first.")
            cubes.append(np.load(p))
        scan_specs[scan] = (cubes[0], cubes[1])
    return scan_specs


def line_window_mask(n_ch, slope, intercept):
    """Channels inside the +-hw windows of the eight reliable lines.

    The trained loss covers 3.5-15.5 keV, most of which is continuum and
    scatter. Map quality depends on the line windows alone, and the two
    criteria do not rank models the same way -- a model can fit the bulk
    spectrum better while flattening the Ca line. Model selection uses
    this mask; see the ablation in scripts/13.
    """
    with open(VINCA_ROOT / "src" / "elements.json") as fh:
        els = json.load(fh)
    keys = ["Ca", "Ti", "Fe", "Cu", "PbLl", "PbLa", "PbLb", "PbLg"]
    energy = np.arange(n_ch) * slope + intercept
    m = np.zeros(n_ch, dtype=bool)
    for k in keys:
        hw = els[k].get("hw", 0.30)
        m |= np.abs(energy - els[k]["kev"]) <= hw
    return m


def masked_mse(pred, target, mask, weight=None):
    """MSE over the loss window only (mask broadcast over the batch).

    ``weight`` (per example, per channel) compensates the variance that
    the R(E) rescaling puts into the target; without it the loss is
    dominated by the low-energy channels where R is large and the
    network under-fits exactly the lines that need it most.
    """
    w = mask if weight is None else mask * weight
    return ((pred - target) ** 2 * w).sum() / w.sum()


def run_epoch(model, loader, mask, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    for batch in loader:
        x, y = batch[0], batch[1]
        w = batch[2].to(device, non_blocking=True) if len(batch) > 2 else None
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            loss = masked_mse(model(x), y, mask, w)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total += loss.item() * x.shape[0]
        n += x.shape[0]
    return total / max(n, 1)


@torch.no_grad()
def fuse_scan(model, cube_a, cube_b, curve, global_scale, device,
              batch=256, weights="invvar"):
    """Fused spectra for one scan, in detector-A counts scale.

    Both directions are run: the A-input prediction is already in the A
    scale, the B-input prediction is brought over with R(E). Combining
    the two is what makes this a *fusion* rather than a denoiser applied
    to one channel.

    The two are not equally reliable. Bringing B over multiplies its
    variance by R^2 while its mean only grows by R, so at the Ca line
    (R ~ 5.8) the B-side estimate carries ~6x the variance of the A-side
    one. Inverse-variance weighting is therefore R : 1 per channel --
    the same logic as the classical weighted fusion, which independently
    puts 89% of the Ca weight on detector 10264. Equal averaging
    ("equal") is kept for the ablation.
    """
    model.eval()
    rows, cols, n_ch = cube_a.shape
    fa = cube_a.reshape(-1, n_ch).astype(np.float32)
    fb = cube_b.reshape(-1, n_ch).astype(np.float32)
    out = np.empty_like(fa)
    r = torch.from_numpy(np.asarray(curve, dtype=np.float32)).to(device)
    wa = r / (r + 1.0) if weights == "invvar" else torch.full_like(r, 0.5)
    wb = 1.0 - wa
    for i in range(0, fa.shape[0], batch):
        xa = torch.from_numpy(fa[i:i + batch] / global_scale).unsqueeze(1).to(device)
        xb = torch.from_numpy(fb[i:i + batch] / global_scale).unsqueeze(1).to(device)
        pa = model(xa).squeeze(1)               # A scale
        pb = model(xb).squeeze(1) * r           # B scale -> A scale
        out[i:i + batch] = ((wa * pa + wb * pb)
                            * global_scale).cpu().numpy()
    return out.reshape(rows, cols, n_ch)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=cfg.batch_size)
    ap.add_argument("--lr", type=float, default=cfg.lr)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-warm-start", action="store_true")
    ap.add_argument("--loss-weight", default="invvar",
                    choices=("invvar", "poisson", "none"),
                    help="per-channel loss weights: 'invvar' compensates "
                         "R(E), 'poisson' the full inverse target variance")
    ap.add_argument("--fuse-weights", default="invvar",
                    choices=("invvar", "equal"),
                    help="how the two directions are combined at export")
    ap.add_argument("--tag", default="",
                    help="suffix for checkpoint and fused-cube names, so "
                         "ablation runs do not overwrite the main model")
    ap.add_argument("--init-from", default="",
                    help="checkpoint to load instead of this tag's own "
                         "(for re-exporting one model under another config)")
    ap.add_argument("--eval-only", action="store_true",
                    help="score a checkpoint on the validation split with "
                         "both criteria (full window and line windows)")
    ap.add_argument("--export-only", action="store_true",
                    help="skip training, re-export cubes from the checkpoint")
    ap.add_argument("--smoke", action="store_true",
                    help="2 epochs on a small subset, no export")
    args = ap.parse_args()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = pick_device(args.device)
    processed = cfg.abs_path(cfg.processed_dir)
    (EXP_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
    (EXP_DIR / "results").mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  CROSS-DETECTOR NOISE2NOISE FUSION")
    print("=" * 70)
    print(f"device: {device}")

    # ---- data ----------------------------------------------------------
    scan_specs = load_cubes(processed)
    n_ch = scan_specs["prova1"][0].shape[2]
    splits, record = build_cross_detector_splits(scan_specs, seed=cfg.seed)

    mask_np = make_channel_mask(n_ch, cfg.cal_slope, cfg.cal_intercept)
    if not HANDOFF2_CURVE.exists():
        sys.exit(f"ERROR: {HANDOFF2_CURVE} missing - run "
                 "scripts/07_geometry_fit.py in the main repo (handoff 2).")
    frontal = clamp_curve_to_mask(ratio_curve_from_csv(
        HANDOFF2_CURVE, n_ch, cfg.cal_slope, cfg.cal_intercept), mask_np)
    tilted = clamp_curve_to_mask(ratio_curve_from_csv(
        HANDOFF2_CURVE, n_ch, cfg.cal_slope, cfg.cal_intercept,
        r_col="R_tilt"), mask_np)
    curves = {"prova1": frontal, "prova2": frontal, "ruotato": tilted}

    sample = splits["train"][0][1][::37].ravel()
    global_scale = float(np.percentile(sample, 99.9)) or 1.0

    datasets = {
        name: XRFCrossDetectorDataset(
            parts, ratio_curve=curves, global_scale=global_scale,
            loss_mask=mask_np,
            return_weight={"invvar": "ratio", "poisson": "poisson",
                           "none": False}[args.loss_weight])
        for name, parts in splits.items()
    }
    if args.smoke:
        args.epochs = 2
        for name in ("train", "val"):
            ds = datasets[name]
            ds._table = ds._table[:512]
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size,
                            shuffle=True, drop_last=True),
        "val": DataLoader(datasets["val"], batch_size=args.batch_size),
    }
    print(f"train {len(datasets['train'])} / val {len(datasets['val'])}"
          f" / test {len(datasets['test'])} examples"
          f"   global scale {global_scale:.1f}")
    print(f"loss window: {int(mask_np.sum())}/{n_ch} channels (3.5-15.5 keV)")

    suffix = f"_{args.tag}" if args.tag else ""

    # ---- model ---------------------------------------------------------
    model = UNet1D(in_channels=1, base_filters=cfg.base_filters,
                   n_blocks=cfg.n_encoder_blocks, dropout=cfg.dropout)
    warm = "scratch"
    if not args.no_warm_start and WARM_START.exists():
        state = torch.load(WARM_START, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"  warm start partial: {len(missing)} missing,"
                  f" {len(unexpected)} unexpected keys")
        warm = str(WARM_START.relative_to(PROJECT_ROOT))
        print(f"  warm start from {warm}")
    model = model.to(device)
    mask = torch.from_numpy(mask_np).to(device).float().view(1, 1, -1)

    if args.eval_only:
        src = Path(args.init_from) if args.init_from else (
            EXP_DIR / "checkpoints" / f"best_model{suffix}.pt")
        model.load_state_dict(torch.load(src, map_location=device,
                                         weights_only=False)["model_state_dict"])
        lm = line_window_mask(n_ch, cfg.cal_slope, cfg.cal_intercept)
        full = torch.from_numpy(mask_np).to(device).float().view(1, 1, -1)
        lines_m = torch.from_numpy(lm & mask_np).to(device).float().view(1, 1, -1)
        val = DataLoader(datasets["val"], batch_size=args.batch_size)
        v_full = run_epoch(model, val, full, device)
        v_line = run_epoch(model, val, lines_m, device)
        print(f"  {src.name}:  val(full window) = {v_full:.6f}"
              f"   val(line windows, {int((lm & mask_np).sum())} ch)"
              f" = {v_line:.6f}")
        sys.exit(0)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3)

    # ---- training ------------------------------------------------------
    history, best_val, best_epoch, t0 = [], float("inf"), -1, time.time()
    # a smoke run must never overwrite a real checkpoint
    ckpt_path = EXP_DIR / "checkpoints" / (
        "smoke_model.pt" if args.smoke else f"best_model{suffix}.pt")
    load_path = Path(args.init_from) if args.init_from else ckpt_path
    for epoch in range(0 if not args.export_only else args.epochs, args.epochs):
        tr = run_epoch(model, loaders["train"], mask, device, optimizer)
        va = run_epoch(model, loaders["val"], mask, device)
        sched.step(va)
        history.append({"epoch": epoch, "train": tr, "val": va,
                        "lr": optimizer.param_groups[0]["lr"]})
        flag = ""
        if va < best_val:
            best_val, best_epoch, flag = va, epoch, "  *"
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch, "val_loss": va,
                        "global_scale": global_scale,
                        "warm_start": warm,
                        "seed": cfg.seed}, ckpt_path)
        print(f"  epoch {epoch:3d}  train {tr:.6f}  val {va:.6f}"
              f"  ({time.time() - t0:5.0f}s){flag}", flush=True)
        if epoch - best_epoch >= args.patience:
            print(f"  early stop (no improvement in {args.patience} epochs)")
            break

    if args.export_only or args.smoke:  # noqa: keep the record intact
        # a smoke run must not overwrite the record of a real run either
        print(f"{'export-only' if args.export_only else 'smoke'}:"
              f" {ckpt_path.relative_to(PROJECT_ROOT)},"
              " history.json left untouched")
    else:
        print(f"best val {best_val:.6f} at epoch {best_epoch}"
              f"  -> {ckpt_path.relative_to(PROJECT_ROOT)}")
        with open(EXP_DIR / "results" / f"history{suffix}.json", "w") as fh:
            json.dump({"history": history, "best_epoch": best_epoch,
                       "best_val": best_val, "global_scale": global_scale,
                       "warm_start": warm, "epochs_run": len(history),
                       "batch_size": args.batch_size, "lr": args.lr,
                       "fuse_weights": args.fuse_weights,
                       "loss_weight": args.loss_weight,
                       "device": str(device)}, fh, indent=1)

    if args.smoke:
        print("\nSmoke run: no cubes exported.")
        sys.exit(0)

    # ---- export fused cubes for the benchmark --------------------------
    model.load_state_dict(torch.load(load_path, map_location=device,
                                     weights_only=False)["model_state_dict"])
    print(f"  exporting from {Path(load_path).name}"
          f"  (fuse weights: {args.fuse_weights})")
    for scan in ("prova1", "prova2"):
        ca, cb = scan_specs[scan]
        fused = fuse_scan(model, ca, cb, curves[scan], global_scale, device,
                          weights=args.fuse_weights)
        out = processed / f"fused_{scan}{suffix}.npy"
        np.save(out, fused.astype(np.float32))
        print(f"  fused {scan}: {fused.shape} -> {out.name}"
              f"  (mean counts/px {fused.sum(axis=2).mean():.0f})")

    # pixels of the frontal grid the network never saw: the prova1
    # validation blocks. 09 evaluates the learned variant there.
    heldout = record["scans"]["prova1"]["val_indices"]
    with open(processed / "fused_heldout_px.json", "w") as fh:
        json.dump({"scan": "prova1", "rows": 60, "cols": 120,
                   "val_indices": heldout,
                   "note": "prova1 spatial validation blocks; prova2 is "
                           "fully held out (test scan)"}, fh)
    print(f"  held-out pixels: {len(heldout)} (prova1 val blocks)")
    print("\nDone. Next: python scripts/09_fusion.py in the main repo.")
