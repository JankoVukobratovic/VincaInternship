"""training.py - one reusable trainer for every workpackage.

OWNERSHIP: TEAM (ML code).  Working first version, adapted from the MVP
script - rewrite freely (loss, schedule, architecture swaps).  The
contract that must survive: make_batch_fn() and train_net() signatures
and return shapes; `main.py --stage verify` checks them.  New model
architectures (e.g. a mask-aware partial-conv U-Net) go in a new module
under your WP folder and plug in through the same train_net(batch_fn).

Adapted from neurips-restore/scripts/03 (same loss, schedule, early
stopping) but generic over the batch source, so:

    WP1:  train_net(batch_fn=make_batch_fn(knobs=jittered_knobs))  x N
    WP2:  train_net(batch_fn=make_batch_fn(knobs=defect_knobs))    per rung
    WP3:  no training - reuses the MVP checkpoint (common.restore)

Early stopping validates on the SAME (possibly perturbed) simulator the
net trains on - that is what a practitioner with a wrong simulator
would do, and it is the honest way to measure the damage a wrong
simulator causes downstream.
"""

import copy
import time

import numpy as np
import torch

import config
from common import core, perturb

dg = core.dg


def make_batch_fn(knobs: perturb.SimKnobs | None = None, **fixed):
    """Batch source: (rng, n, **overrides) -> (x, y, loss_mask, val_mask).

    knobs=None uses the nominal datagen.sample; a SimKnobs instance uses
    the perturbed simulator.  `fixed` / `overrides` are forwarded to the
    sample function (angle, dose, flip, blocks, ...).
    """
    def fn(rng, n, **overrides):
        kw = dict(fixed)
        kw.update(overrides)
        xs, ys, lms, vms = [], [], [], []
        for _ in range(n):
            if knobs is None:
                x, y, lm, vm, _ = dg.sample(rng, **kw)
            else:
                x, y, lm, vm, _ = perturb.sample(rng, knobs=knobs, **kw)
            xs.append(x)
            ys.append(y)
            lms.append(lm)
            vms.append(vm)
        return (np.stack(xs), np.stack(ys), np.stack(lms), np.stack(vms))
    return fn


def masked_l1(pred, target, mask):
    diff = (pred - target).abs()
    m = mask.unsqueeze(1).expand_as(diff)
    return diff[m].mean()


def build_val_set(batch_fn, seed=777):
    """Fixed simulated validation set: config.VAL_ANGLES x VAL_REPS."""
    rng = np.random.default_rng(seed)
    xs, ys, vms = [], [], []
    for angle in config.VAL_ANGLES:
        x, y, _, vm = batch_fn(rng, config.VAL_REPS, angle=angle, dose=1.0,
                               flip=(False, False), blocks=[])
        xs.append(x)
        ys.append(y)
        vms.append(vm)
    return (torch.from_numpy(np.concatenate(xs)),
            torch.from_numpy(np.concatenate(ys)),
            torch.from_numpy(np.concatenate(vms)))


def train_net(batch_fn, train_cfg: dict | None = None, seed: int = 0,
              verbose: bool = True):
    """Train a fresh RestorationUNet on batch_fn; returns (net, history).

    train_cfg defaults to config.TRAIN; pass config.QUICK_TRAIN for
    smoke runs.  The returned net carries the best-validation weights.
    """
    cfg = dict(config.TRAIN)
    if train_cfg:
        cfg.update(train_cfg)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    net = core.RestorationUNet()
    xv, yv, vmv = build_val_set(batch_fn, seed=777)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=4, min_lr=5e-5)

    best_val, best_state, best_step, bad = np.inf, None, 0, 0
    t0 = time.time()
    for step in range(1, cfg["steps"] + 1):
        x, y, lm, _ = batch_fn(rng, cfg["batch"])
        opt.zero_grad()
        loss = masked_l1(net.restore(torch.from_numpy(x)),
                         torch.from_numpy(y), torch.from_numpy(lm))
        loss.backward()
        opt.step()

        if step % cfg["val_every"] == 0 or step == cfg["steps"]:
            net.eval()
            with torch.no_grad():
                vloss = float(masked_l1(net.restore(xv), yv, vmv))
            net.train()
            sched.step(vloss)
            if vloss < best_val - 1e-6:
                best_val, best_step, bad = vloss, step, 0
                best_state = copy.deepcopy(net.state_dict())
            else:
                bad += 1
            if verbose:
                print(f"  step {step:5d}  val L1 {vloss:.5f}"
                      f"  [{time.time() - t0:6.1f} s]"
                      f"{'  *best*' if best_step == step else ''}")
            if bad >= cfg["patience"] or time.time() - t0 > cfg["time_budget_s"]:
                break

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net, {"best_val_l1": float(best_val), "best_step": best_step,
                 "wall_s": time.time() - t0, "seed": seed}
