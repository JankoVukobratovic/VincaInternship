"""
03_learned_restoration.py - the learned restoration MVP (item 3).

A small residual U-Net (src/model.py, ~0.47M params, CPU) refines the
deterministic physics inversion of forward_model.py.  It is trained
PURELY on physics-simulated pairs generated from the prova1 maps
(src/datagen.py): random angle 4-25 deg, calibrated noise, optional
low-dose scaling and dropout blocks; input = inverse(forward(prova1)),
target = prova1.  The network predicts residuals on top of the
deterministic inverse, so the absolute level is protected by
construction (zero-init head = exact baseline at step 0).

HONEST TEST PROTOCOL
--------------------
The network never sees prova2 or the real ruotato scan.  Final test:
restore the REAL ruotato (deterministic inverse + network) and score
against prova2 ALONE as truth (not the mean - prova1 was training
data), on the footprint of the warped tilted scan, with src/eval.py.
The plain deterministic baseline and the noise floor (prova1 scored
against prova2) are computed the same way, so the three rows per line
are apples-to-apples.  NOTE: the deterministic numbers here differ
slightly from results/deterministic_baseline.txt, which scored against
mean(prova1, prova2) and used the common-mode-free per_deg gains; here
both baseline and network use forward_model.inverse (tilt_pct_sum
gains, common mode included) so training and test share one operator.

Spatial holdout for early stopping: 4 fixed 15x30 blocks of prova1
(datagen.VAL_BLOCKS) never enter the training loss; validation L1 is
measured there on a fixed 24-sample simulated set.

Success criteria on the real test (per line):
  - cv_ratio moves toward 1.0 vs the deterministic baseline
  - no bias regression: |bias| within 1.5 pp of the baseline's
  - r and SSIM at or above the baseline
Matching-but-not-beating = negative result, reported as such.

Outputs:
    neurips-restore/results/learned_restoration.txt / .csv / .png
    neurips-restore/results/harsh_demo.png       (20 deg + dropout block)
    neurips-restore/experiments/checkpoint.pt
    neurips-restore/experiments/history.json

Run from the repo root:
    python neurips-restore/scripts/03_learned_restoration.py
    (--smoke for a 2-minute pipeline check; --skip-train to re-run the
     test/figures from the saved checkpoint)
"""

import argparse
import copy
import csv
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))
import datagen as dg
import eval as ev
import forward_model as fm
from model import RestorationUNet, count_params

REPO_ROOT = fm.REPO_ROOT
OUT_DIR = os.path.join(REPO_ROOT, "neurips-restore", "results")
EXP_DIR = os.path.join(REPO_ROOT, "neurips-restore", "experiments")
TXT_PATH = os.path.join(OUT_DIR, "learned_restoration.txt")
CSV_PATH = os.path.join(OUT_DIR, "learned_restoration.csv")
PNG_PATH = os.path.join(OUT_DIR, "learned_restoration.png")
HARSH_PATH = os.path.join(OUT_DIR, "harsh_demo.png")
CKPT_PATH = os.path.join(EXP_DIR, "checkpoint.pt")
HIST_PATH = os.path.join(EXP_DIR, "history.json")

ELEMENTS = fm.ELEMENTS
FIG_LINES = ("Ca", "Cu")          # largest measured headroom (MVP-2)
HARSH_LINES = ("Ca", "PbLa")
VAL_ANGLES = (4.0, 7.7, 10.0, 13.0, 16.0, 19.0, 22.0, 25.0)
VAL_REPS = 3
HARSH_ANGLE = 20.0
HARSH_BLOCK = (14, 28, 14, 20)    # (r0, c0, h, w) in the tilted frame


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def masked_l1(pred, target, mask):
    """L1 over mask (B, H, W) broadcast across the 8 element channels."""
    diff = (pred - target).abs()
    m = mask.unsqueeze(1).expand_as(diff)
    return diff[m].mean()


def build_val_set(style, seed=777):
    """Fixed simulated validation set: angles x reps, clean protocol."""
    rng = np.random.default_rng(seed)
    xs, ys, vms = [], [], []
    for angle in VAL_ANGLES:
        for _ in range(VAL_REPS):
            x, y, _, vm, _ = dg.sample(rng, angle=angle, dose=1.0,
                                       flip=(False, False), blocks=[],
                                       input_style=style)
            xs.append(x)
            ys.append(y)
            vms.append(vm)
    return (torch.from_numpy(np.stack(xs)),
            torch.from_numpy(np.stack(ys)),
            torch.from_numpy(np.stack(vms)))


def train(net, cfg):
    """Training loop; returns (best_state, history dict)."""
    rng = np.random.default_rng(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    xv, yv, vmv = build_val_set(cfg["style"])
    print(f"val set: {xv.shape[0]} samples, "
          f"{int(vmv[0].sum())} holdout px each, style={cfg['style']}")

    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=4, min_lr=5e-5)

    history = []
    best_val, best_state, best_step = np.inf, None, 0
    bad_vals = 0
    t0 = time.time()
    run_loss, run_n = 0.0, 0

    for step in range(1, cfg["steps"] + 1):
        x, y, lm, _, _ = dg.make_batch(rng, cfg["batch"],
                                       input_style=cfg["style"])
        xt = torch.from_numpy(x)
        yt = torch.from_numpy(y)
        lmt = torch.from_numpy(lm)
        opt.zero_grad()
        loss = masked_l1(net.restore(xt), yt, lmt)
        loss.backward()
        opt.step()
        run_loss += float(loss.detach())
        run_n += 1

        if step % cfg["val_every"] == 0 or step == cfg["steps"]:
            net.eval()
            with torch.no_grad():
                vloss = float(masked_l1(net.restore(xv), yv, vmv))
            net.train()
            sched.step(vloss)
            lr_now = opt.param_groups[0]["lr"]
            elapsed = time.time() - t0
            history.append({"step": step, "train_l1": run_loss / max(run_n, 1),
                            "val_l1": vloss, "lr": lr_now,
                            "elapsed_s": elapsed})
            marker = ""
            if vloss < best_val - 1e-6:
                best_val, best_step = vloss, step
                best_state = copy.deepcopy(net.state_dict())
                bad_vals = 0
                marker = "  *best*"
            else:
                bad_vals += 1
            print(f"step {step:5d}  train L1 {run_loss / max(run_n, 1):.5f}  "
                  f"val L1 {vloss:.5f}  lr {lr_now:.1e}  "
                  f"[{elapsed:6.1f} s]{marker}")
            run_loss, run_n = 0.0, 0
            if bad_vals >= cfg["patience"]:
                print(f"early stop: no val improvement in "
                      f"{cfg['patience']} checks")
                break
            if elapsed > cfg["time_budget_s"]:
                print(f"time budget reached ({cfg['time_budget_s']} s)")
                break

    wall = time.time() - t0
    print(f"training done: {wall:.1f} s, best val L1 {best_val:.5f} "
          f"at step {best_step}")
    if best_state is None:
        best_state = copy.deepcopy(net.state_dict())
    return best_state, {"entries": history, "best_val_l1": best_val,
                        "best_step": best_step, "wall_s": wall}


# --------------------------------------------------------------------------
# restoration of a tilted map set with the trained network
# --------------------------------------------------------------------------

def apply_network(net, tilted_maps, angle_deg, validity=None):
    """inverse() + network refinement -> (det dict, learned dict).

    Both dicts are frontal-frame maps with NaN outside the footprint.
    """
    det = fm.inverse(tilted_maps, angle_deg=angle_deg)
    x = dg.build_input(det, angle_deg, validity=validity)
    xt = torch.from_numpy(x[None])
    net.eval()
    with torch.no_grad():
        rest = net.restore(xt)[0].numpy()
    scales = dg.norm_scales()
    fp = dg.footprint()
    learned = {}
    for i, el in enumerate(ELEMENTS):
        learned[el] = np.where(fp, rest[i] * scales[i], np.nan)
    return det, learned


# --------------------------------------------------------------------------
# final test on the REAL ruotato scan
# --------------------------------------------------------------------------

def real_test(net):
    """Score baseline / learned / floor against prova2 alone."""
    ruo = fm.load_summed_maps("ruotato")
    truth = fm.load_summed_maps("prova2")     # never seen in training
    p1 = fm.load_summed_maps("prova1")        # training data -> floor row
    mask = dg.footprint()

    det, learned = apply_network(net, ruo, fm.REF_ANGLE_DEG)

    results = {}
    for el in ELEMENTS:
        drange = float(truth[el][mask].max() - truth[el][mask].min())
        results[el] = {
            "baseline": ev.score_pair(det[el], truth[el], mask,
                                      data_range=drange),
            "learned": ev.score_pair(learned[el], truth[el], mask,
                                     data_range=drange),
            "floor": ev.score_pair(p1[el], truth[el], mask,
                                   data_range=drange),
        }
    return results, det, learned, truth, mask


def criteria_check(results):
    """Per-line pass/fail of the success criteria; returns rows + verdict."""
    rows = []
    for el in ELEMENTS:
        b = results[el]["baseline"]
        l = results[el]["learned"]
        cv_ok = abs(1.0 - l["cv_ratio"]) < abs(1.0 - b["cv_ratio"])
        bias_ok = abs(l["bias_pct"]) <= abs(b["bias_pct"]) + 1.5
        r_ok = l["r"] >= b["r"] - 1e-4
        ssim_ok = l["ssim"] >= b["ssim"] - 1e-4
        rows.append({"element": el, "cv_toward_1": cv_ok, "bias_ok": bias_ok,
                     "r_ok": r_ok, "ssim_ok": ssim_ok,
                     "all": cv_ok and bias_ok and r_ok and ssim_ok})
    return rows


# --------------------------------------------------------------------------
# report + figures
# --------------------------------------------------------------------------

def yn(flag):
    return "yes" if flag else "NO"


def write_report(results, crit, hist, cfg, harsh_info):
    cand_label = {"baseline": "deterministic (physics)",
                  "learned": "physics + U-Net",
                  "floor": "noise floor (p1 vs p2)"}
    n_px = results[ELEMENTS[0]]["baseline"]["n_px"]
    style = cfg["style"]
    if style == "cubic":
        style_lines = [
            "    Tilted measurements are simulated with datagen.",
            "    forward_sharp (cubic sampling at the exact positions of",
            "    the measured warp + the measured gains + noise",
            "    k*s*(max(1-g,0) + 0.25)): a real tilted scan is a direct",
            "    measurement and carries NO resampling blur, so the",
            "    restoration input inverse(tilted) must carry ONE bilinear",
            "    blur as at test time - the validated bilinear forward()",
            "    would add a second one (see ITERATION NOTE below).",
        ]
    else:
        style_lines = [
            "    Tilted measurements are simulated with the validated",
            "    bilinear forward() (v1 style; carries a second resampling",
            "    blur the real restoration input does not have).",
        ]
    lines = [
        "Learned restoration (MVP-3) - residual U-Net on top of the",
        "deterministic physics inversion, REAL ruotato -> frontal",
        "",
        "PROTOCOL (honest-evaluation rules of the README):",
        "  - Training data: physics-simulated pairs from the prova1 maps",
        "    ONLY: random angle 4-25 deg, calibrated noise Var = k*counts,",
        "    random flips; 25% low-dose samples (x0.35-1.0 before noise);",
        "    30% with 1-2 dropout blocks zeroed in the tilted frame",
        "    (validity mask channel).",
        *style_lines,
        "    Input = inverse(simulated tilted); target = prova1; the net",
        "    predicts residuals on the deterministic inverse (zero-init",
        "    head = exact physics baseline at step 0).",
        "  - Spatial holdout: 4 fixed 15x30 blocks of prova1 (r0,c0 =",
        "    " + ", ".join(str(b) for b in dg.VAL_BLOCKS) + ") never in",
        "    the training loss; early stopping on their simulated val L1.",
        "  - The network NEVER sees prova2 or the real ruotato.",
        "  - Test: restore the REAL ruotato (deterministic inverse +",
        "    network) and score against prova2 ALONE as truth (prova1 was",
        "    training data), on the tilted-scan footprint"
        f" ({n_px} px of 60x120),",
        "    with eval.py: Pearson r, SSIM (shared data range per line),",
        "    bias %, contrast ratio cv_ratio.  Baseline and floor rows",
        "    are computed identically (floor = prova1 scored vs prova2).",
        "  - Both baseline and network input use forward_model.inverse",
        "    (tilt_pct_sum gains, session common mode included), so these",
        "    baseline numbers differ slightly from",
        "    deterministic_baseline.txt (truth=mean, per_deg gains).",
        "",
        f"Model: RestorationUNet base=32, {cfg['n_params']} params, input",
        "8 element channels + validity + angle, residual output.",
        f"Training: {cfg['steps_done']} steps of batch {cfg['batch']}"
        f" (Adam lr {cfg['lr']:.0e}, plateau decay),",
        f"input style '{style}', "
        f"best val L1 {hist['best_val_l1']:.5f} at step {hist['best_step']},"
        f" wall clock {hist['wall_s']:.0f} s.",
        "",
        f"{'line':6s} {'candidate':<26s} {'r':>7s} {'SSIM':>7s}"
        f" {'bias %':>8s} {'cv_ratio':>9s}",
        "-" * 66,
    ]
    for el in ELEMENTS:
        for cand in ("baseline", "learned", "floor"):
            s = results[el][cand]
            lines.append(f"{el:6s} {cand_label[cand]:<26s} {s['r']:7.4f}"
                         f" {s['ssim']:7.4f} {s['bias_pct']:+8.2f}"
                         f" {s['cv_ratio']:9.4f}")
        lines.append("")

    lines += [
        "SUCCESS CRITERIA (learned vs deterministic, per line):",
        "  cv->1   : |1 - cv_ratio| strictly smaller than baseline's",
        "  bias ok : |bias| within 1.5 pp of baseline's |bias|",
        "  r, SSIM : at or above baseline (tolerance 1e-4)",
        "",
        f"{'line':6s} {'cv->1':>6s} {'bias':>6s} {'r':>6s} {'SSIM':>6s}"
        f" {'ALL':>6s}",
        "-" * 40,
    ]
    for c in crit:
        lines.append(f"{c['element']:6s} {yn(c['cv_toward_1']):>6s}"
                     f" {yn(c['bias_ok']):>6s} {yn(c['r_ok']):>6s}"
                     f" {yn(c['ssim_ok']):>6s} {yn(c['all']):>6s}")
    n_all = sum(c["all"] for c in crit)
    n_cv = sum(c["cv_toward_1"] for c in crit)
    n_bias = sum(c["bias_ok"] for c in crit)
    lines += [
        "",
        f"Lines meeting ALL criteria: {n_all}/8 "
        f"(cv->1: {n_cv}/8, bias ok: {n_bias}/8)",
        "",
        "ITERATION NOTE (v1 -> v2): the first training run used the",
        "validated bilinear forward() to simulate the tilted scan.  Its",
        "restoration inputs carry TWO bilinear resampling blurs (forward",
        "warp + inverse warp) where the real restoration input carries",
        "only ONE - the real ruotato is a direct measurement on the",
        "tilted grid, not a resampled frontal image (consistent with",
        "MVP-2 check [3], HF_sim < HF_real on all 8 lines).  Measured at",
        "7.7 deg, the deterministic inverse of bilinear-forward sims has",
        "cv_ratio 0.90-0.96 vs prova1 while the real one has 0.93-0.96",
        "vs prova2; a v1-trained net learned to undo the doubled blur",
        "and OVER-sharpened the real scan: cv_ratio overshoot up to",
        "1.051 (PbLa/PbLb), r 0.002-0.012 below baseline on all lines,",
        "0/8 criteria (cv->1 6/8, bias 8/8, r/SSIM 0/8).  v2 replaces",
        "the training-input simulation with cubic sampling at the exact",
        "warp positions (datagen.forward_sharp): mean |cv_sim - cv_real|",
        "drops from 0.023 to 0.012 (closer on 6/8 lines).  The test",
        "protocol and operator are unchanged.",
    ]
    if harsh_info:
        lines += ["", "HARSH-REGIME DEMO (simulated, prova2 source - never in"
                      " training):",
                  f"  {HARSH_ANGLE:.0f} deg tilt + {HARSH_BLOCK[2]}x"
                  f"{HARSH_BLOCK[3]} dropout block at (r0,c0)="
                  f"({HARSH_BLOCK[0]},{HARSH_BLOCK[1]}) in the tilted frame;",
                  "  r vs prova2 on the footprint and inside the warped hole"
                  " (deterministic",
                  "  physics cannot fill the hole; the U-Net inpaints from"
                  " context):",
                  f"  {'line':6s} {'r det':>8s} {'r net':>8s}"
                  f" {'r det(hole)':>12s} {'r net(hole)':>12s}"]
        for el, v in harsh_info.items():
            lines.append(f"  {el:6s} {v['r_det']:8.4f} {v['r_net']:8.4f}"
                         f" {v['r_det_hole']:12.4f} {v['r_net_hole']:12.4f}")
    with open(TXT_PATH, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))

    with open(CSV_PATH, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["element", "candidate", "r",
                                           "ssim", "bias_pct", "cv_ratio",
                                           "n_px"])
        w.writeheader()
        for el in ELEMENTS:
            for cand in ("baseline", "learned", "floor"):
                w.writerow({"element": el, "candidate": cand,
                            **results[el][cand]})


def panel(ax, img, vmin, vmax, title, cmap):
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal",
                   interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def make_main_figure(results, det, learned, truth, mask):
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("0.82")
    fig, axes = plt.subplots(len(FIG_LINES), 3,
                             figsize=(12.5, 2.4 * len(FIG_LINES) + 0.9),
                             dpi=120, layout="constrained")
    for i, el in enumerate(FIG_LINES):
        vmin, vmax = np.percentile(truth[el][mask], [1, 99])
        b, l = results[el]["baseline"], results[el]["learned"]
        panels = [
            (np.where(mask, truth[el], np.nan),
             f"{el} - truth (prova2, unseen)"),
            (det[el], f"{el} - deterministic  r={b['r']:.4f}"
                      f"  cv={b['cv_ratio']:.3f}"),
            (learned[el], f"{el} - physics + U-Net  r={l['r']:.4f}"
                          f"  cv={l['cv_ratio']:.3f}"),
        ]
        for j, (img, title) in enumerate(panels):
            im = panel(axes[i, j], img, vmin, vmax, title, cmap)
        cb = fig.colorbar(im, ax=axes[i, :], fraction=0.025, pad=0.01)
        cb.set_label("counts / s", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle("Learned restoration - REAL tilted scan vs prova2 "
                 "(shared color scale per row; gray = outside footprint)",
                 fontsize=11, fontweight="bold")
    fig.savefig(PNG_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


def harsh_demo(net):
    """20 deg tilt + dropout block, simulated from prova2 (unseen)."""
    truth = fm.load_summed_maps("prova2")
    rng = np.random.default_rng(4242)
    tilted = fm.forward(truth, angle_deg=HARSH_ANGLE, rng=rng,
                        add_noise=True, input_noise="measured")
    r0, c0, h, w = HARSH_BLOCK
    v_tilt = np.ones(fm.TILTED_SHAPE)
    v_tilt[r0:r0 + h, c0:c0 + w] = 0.0
    for el in ELEMENTS:
        tilted[el][r0:r0 + h, c0:c0 + w] = 0.0
    validity = np.nan_to_num(fm.warp_tilted_to_frontal(v_tilt),
                             nan=0.0).astype(np.float32)
    det, learned = apply_network(net, tilted, HARSH_ANGLE,
                                 validity=validity)

    fp = dg.footprint()
    hole = fp & (validity < 0.5)
    info = {}
    for el in HARSH_LINES:
        info[el] = {
            "r_det": ev.pearson_r(det[el], truth[el], fp),
            "r_net": ev.pearson_r(learned[el], truth[el], fp),
            "r_det_hole": ev.pearson_r(det[el], truth[el], hole),
            "r_net_hole": ev.pearson_r(learned[el], truth[el], hole),
        }

    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("0.82")
    fig, axes = plt.subplots(len(HARSH_LINES), 4,
                             figsize=(15.0, 2.4 * len(HARSH_LINES) + 0.9),
                             dpi=120, layout="constrained")
    for i, el in enumerate(HARSH_LINES):
        vmin, vmax = np.percentile(truth[el][fp], [1, 99])
        v = info[el]
        panels = [
            (tilted[el], f"{el} - degraded ({HARSH_ANGLE:.0f} deg tilt "
                         "+ block, tilted frame)"),
            (det[el], f"{el} - deterministic inverse  r={v['r_det']:.3f}"),
            (learned[el], f"{el} - physics + U-Net  r={v['r_net']:.3f}"),
            (np.where(fp, truth[el], np.nan), f"{el} - truth (prova2)"),
        ]
        for j, (img, title) in enumerate(panels):
            im = panel(axes[i, j], img, vmin, vmax, title, cmap)
            if j == 0:
                axes[i, j].add_patch(Rectangle((c0 - 0.5, r0 - 0.5), w, h,
                                               fill=False, edgecolor="cyan",
                                               linewidth=1.2,
                                               linestyle="--"))
        cb = fig.colorbar(im, ax=axes[i, :], fraction=0.025, pad=0.01)
        cb.set_label("counts / s", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle("Harsh-regime demo (SIMULATED from prova2, unseen in "
                 "training): 20 deg tilt + dropout block - deterministic "
                 "physics cannot fill the hole", fontsize=11,
                 fontweight="bold")
    fig.savefig(HARSH_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return info


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=12,
                    help="early stop after N val checks without improvement")
    ap.add_argument("--time-budget", type=float, default=2400.0,
                    help="training wall-clock cap in seconds")
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument("--style", choices=dg.INPUT_STYLES, default="cubic",
                    help="training-input simulation style (cubic = "
                         "sharp-acquisition v2 default; bilinear = v1 "
                         "ablation)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run to verify the pipeline")
    ap.add_argument("--skip-train", action="store_true",
                    help="load experiments/checkpoint.pt and only re-run "
                         "the test + figures")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(EXP_DIR, exist_ok=True)

    cfg = {"steps": args.steps, "batch": args.batch,
           "val_every": args.val_every, "lr": args.lr,
           "patience": args.patience, "time_budget_s": args.time_budget,
           "seed": args.seed, "style": args.style}
    if args.smoke:
        cfg.update(steps=150, val_every=25, time_budget_s=300.0)
        print("SMOKE MODE: 150 steps only - results are NOT the deliverable")

    net = RestorationUNet()
    cfg["n_params"] = count_params(net)
    print(f"RestorationUNet: {cfg['n_params']} params")

    if args.skip_train:
        ck = torch.load(CKPT_PATH, weights_only=False)
        net.load_state_dict(ck["state_dict"])
        hist = {"best_val_l1": ck.get("best_val_l1", float("nan")),
                "best_step": ck.get("best_step", -1),
                "wall_s": ck.get("wall_s", float("nan"))}
        cfg["steps_done"] = ck.get("steps_done", -1)
        cfg["style"] = ck.get("config", {}).get("style", args.style)
        print(f"loaded checkpoint: best val L1 {hist['best_val_l1']:.5f} "
              f"at step {hist['best_step']}, style {cfg['style']}")
    else:
        best_state, hist = train(net, cfg)
        net.load_state_dict(best_state)
        cfg["steps_done"] = (hist["entries"][-1]["step"]
                             if hist["entries"] else 0)
        torch.save({"state_dict": best_state,
                    "norm_scales": dg.norm_scales(),
                    "elements": list(ELEMENTS),
                    "best_val_l1": hist["best_val_l1"],
                    "best_step": hist["best_step"],
                    "wall_s": hist["wall_s"],
                    "steps_done": cfg["steps_done"],
                    "config": {k: v for k, v in cfg.items()}}, CKPT_PATH)
        with open(HIST_PATH, "w") as fh:
            json.dump({"config": cfg, **hist}, fh, indent=1)
        print(f"saved: {CKPT_PATH}")
        print(f"saved: {HIST_PATH}")

    # ---- honest final test on the REAL ruotato --------------------------
    print("\nFinal test: REAL ruotato restored, scored vs prova2 alone")
    results, det, learned, truth, mask = real_test(net)
    crit = criteria_check(results)

    # ---- harsh demo ------------------------------------------------------
    harsh_info = harsh_demo(net)

    # ---- report + figures ------------------------------------------------
    write_report(results, crit, hist, cfg, harsh_info)
    make_main_figure(results, det, learned, truth, mask)

    for p in (TXT_PATH, CSV_PATH, PNG_PATH, HARSH_PATH):
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()
