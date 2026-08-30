"""Cross-detector Noise2Noise dataset (dual-detector paper, PLAN §4.3 + 8.4).

The two SDD detectors observe the same pixel simultaneously, so their
spectra are conditionally independent Poisson realizations - but of
DIFFERENT expected signals: the response ratio R(E) = E[A]/E[B] runs
from ~6 at Ca Kα to ~0.65 at Pb Lγ. A valid Noise2Noise target must
therefore be rescaled into the input detector's response scale,
channel by channel, before training (PLAN amendment 8.4); otherwise
the network learns detector B's response instead of the clean signal.

Handoff 2 delivers that curve as
``results/detector_diff/handoff2_ratio_curve.csv`` (script 07): column
``R`` for the frontal scans, ``R_tilt`` for the tilted one, read with
:func:`ratio_curve_from_csv`. The older
:func:`ratio_curve_from_table` builds a PROVISIONAL curve from the
full-frame per-element ratios of ``efficiency_ratios.csv`` and is kept
only for reproducing pre-handoff runs - those ratios carry the
field-of-view artifact of PLAN §8.7. Trial trainings may also run
entirely unscaled (``ratio_curve=None``), as PLAN §4.3 allows.

Scaling caveat for the loss: a target multiplied by R(E) is no longer
integer Poisson counts - use MSE (or a variance-weighted loss) on
scaled targets, not ``poisson_nll`` on raw counts.

Split policy (PLAN §4.3): training on prova1 + ruotato, evaluation on
prova2 - per-scan split with spatial-block validation carved out of
the training scans. Never random per-pixel.
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Loss window (keV). 3.0 was tried, to bring the Ca background sidebands
# (down to 3.14 keV) inside the supervised range: it fails badly. Those
# channels carry R up to 34, so the rescaled target's variance explodes
# there and the reconstruction degrades exactly where it was meant to
# improve (fused Ca intensity -90%, cv 5.0). The window stays at 3.5 and
# the Ca bias is handled downstream instead.
LOSS_LO_KEV = 3.5
LOSS_HI_KEV = 15.5


# ── ratio curve ─────────────────────────────────────────────────────────


def make_channel_mask(
    n_channels: int,
    slope: float,
    intercept: float,
    lo_kev: float = LOSS_LO_KEV,
    hi_kev: float = LOSS_HI_KEV,
) -> np.ndarray:
    """Boolean per-channel mask for the loss window (True = use channel)."""
    energy = np.arange(n_channels) * slope + intercept
    return (energy >= lo_kev) & (energy <= hi_kev)


def clamp_curve_to_mask(curve: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Hold R(E) constant outside the loss window.

    Below ~3 keV detector B sees almost nothing, so the physical ratio
    runs to 1e5 and scaled targets would carry astronomical values in
    channels the loss discards anyway. Freezing the curve at the window
    edges keeps every tensor bounded without touching any channel that
    enters the loss.
    """
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise ValueError("empty loss mask")
    out = np.array(curve, dtype=np.float32, copy=True)
    out[:idx[0]] = out[idx[0]]
    out[idx[-1] + 1:] = out[idx[-1]]
    return out


def _interp_ratio(kev_pts, r_pts, n_channels, slope, intercept) -> np.ndarray:
    """Log-linear interpolation of R over energy, constant beyond the ends."""
    energy = np.arange(n_channels) * slope + intercept
    order = np.argsort(kev_pts)
    kev_pts = np.asarray(kev_pts, float)[order]
    logr = np.log(np.asarray(r_pts, float)[order])
    return np.exp(np.interp(energy, kev_pts, logr)).astype(np.float32)


def ratio_curve_from_table(
    csv_path: str | Path,
    n_channels: int,
    slope: float,
    intercept: float,
) -> np.ndarray:
    """PROVISIONAL R(E): log-linear interpolation of the per-element
    frontal-mean ratios (reliable rows only) from 06's CSV. Replace with
    the GP curve from handoff 2 via :func:`ratio_curve_from_csv`."""
    kev, r = [], []
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("reliable", "True") != "True":
                continue
            kev.append(float(row["kev"]))
            r.append(0.5 * (float(row["R_prova1"]) + float(row["R_prova2"])))
    if len(kev) < 2:
        raise ValueError(f"no reliable rows in {csv_path}")
    return _interp_ratio(kev, r, n_channels, slope, intercept)


def ratio_curve_from_csv(
    csv_path: str | Path,
    n_channels: int,
    slope: float,
    intercept: float,
    kev_col: str = "kev",
    r_col: str = "R",
) -> np.ndarray:
    """R(E) from a generic (kev, R) table - the handoff-2 GP curve."""
    kev, r = [], []
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            kev.append(float(row[kev_col]))
            r.append(float(row[r_col]))
    return _interp_ratio(kev, r, n_channels, slope, intercept)


# ── line-integral operator (integral anchor, script 07) ────────────────


def net_line_operator(
    n_channels: int,
    slope: float,
    intercept: float,
    elements: dict,
    keys: list[str],
    bg_hw: float = 0.25,
) -> np.ndarray:
    """Benchmark net line integrals as one (K, C) linear operator.

    Row k applied to a spectrum reproduces the main repo's
    ``xrf_core._integrate_fixed_hw`` for line ``keys[k]`` exactly - peak window of ±hw around the line center, minus the linear
    background interpolated between the means of the two ``bg_hw``-wide
    sidebands - except for the final ``max(0, ·)`` clamp, which is
    dropped so the functional stays linear and differentiable (at real
    line pixels the net integral is positive anyway). Because
    ``sum(linspace(l, r, n)) = n*(l+r)/2``, the background subtraction
    is itself linear: -(n_peak/2)/n_side on every sideband channel.

    ``slope``/``intercept`` must be the calibration the benchmark
    integrator uses, so the anchored quantity is channel-identical to
    the maps that scripts/09_fusion.py evaluates.
    """
    energy = np.arange(n_channels) * slope + intercept
    op = np.zeros((len(keys), n_channels), dtype=np.float32)
    for k_i, key in enumerate(keys):
        cfg_el = elements[key]
        hw = cfg_el.get("hw", 0.30)
        idx = int(np.argmin(np.abs(energy - cfg_el["kev"])))
        half = max(1, int(round(hw / slope)))
        bg = max(1, int(round(bg_hw / slope)))
        lo = max(0, idx - half)
        hi = min(n_channels - 1, idx + half)
        n_peak = hi - lo + 1
        op[k_i, lo:hi + 1] = 1.0
        bl_l, bl_r = max(0, lo - bg), lo                    # [bl_l, bl_r)
        br_l, br_r = hi + 1, min(n_channels - 1, hi + 1 + bg)
        if bl_r > bl_l:
            op[k_i, bl_l:bl_r] -= (n_peak / 2.0) / (bl_r - bl_l)
        else:                                # xrf_core fallback: counts[lo]
            op[k_i, lo] -= n_peak / 2.0
        if br_r > br_l:
            op[k_i, br_l:br_r] -= (n_peak / 2.0) / (br_r - br_l)
        else:                                # xrf_core fallback: counts[hi]
            op[k_i, hi] -= n_peak / 2.0
    return op


# ── split ───────────────────────────────────────────────────────────────


def make_block_split(
    rows: int,
    cols: int,
    val_frac: float = 0.15,
    block_size: int = 15,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Two-way spatial block split (train/val) of one scan grid."""
    rng = np.random.default_rng(seed)
    n_blocks_r = (rows + block_size - 1) // block_size
    n_blocks_c = (cols + block_size - 1) // block_size
    block_ids = np.arange(n_blocks_r * n_blocks_c)
    rng.shuffle(block_ids)
    n_val = max(1, int(round(len(block_ids) * val_frac)))
    val_blocks = set(block_ids[:n_val])

    r_idx, c_idx = np.divmod(np.arange(rows * cols), cols)
    bid = (r_idx // block_size) * n_blocks_c + (c_idx // block_size)
    is_val = np.isin(bid, list(val_blocks))
    return {
        "train": np.flatnonzero(~is_val),
        "val": np.flatnonzero(is_val),
    }


def build_cross_detector_splits(
    scan_specs: dict[str, tuple[np.ndarray, np.ndarray]],
    train_scans: tuple[str, ...] = ("prova1", "ruotato"),
    test_scan: str = "prova2",
    val_frac: float = 0.15,
    block_size: int = 15,
    seed: int = 42,
) -> tuple[dict[str, list], dict]:
    """Assemble per-split (spec_a, spec_b, indices) triples.

    Parameters
    ----------
    scan_specs : dict scan -> (cube_a, cube_b), cubes (rows, cols, C)

    Returns
    -------
    splits : {'train'|'val'|'test': [(scan, spec_a, spec_b, indices), ...]}
        spec_* flattened to (rows*cols, C).
    index_record : JSON-serializable record of the split for the repo.
    """
    splits: dict[str, list] = {"train": [], "val": [], "test": []}
    record: dict = {"seed": seed, "val_frac": val_frac,
                    "block_size": block_size, "scans": {}}

    for scan, (ca, cb) in scan_specs.items():
        rows, cols, n_ch = ca.shape
        assert cb.shape == ca.shape, f"{scan}: detector cube shapes differ"
        fa = ca.reshape(-1, n_ch)
        fb = cb.reshape(-1, n_ch)
        if scan == test_scan:
            idx = np.arange(rows * cols)
            splits["test"].append((scan, fa, fb, idx))
            record["scans"][scan] = {"role": "test", "n": int(idx.size)}
        elif scan in train_scans:
            sp = make_block_split(rows, cols, val_frac, block_size, seed)
            splits["train"].append((scan, fa, fb, sp["train"]))
            splits["val"].append((scan, fa, fb, sp["val"]))
            record["scans"][scan] = {
                "role": "train",
                "n_train": int(sp["train"].size),
                "n_val": int(sp["val"].size),
                "val_indices": sp["val"].tolist(),
            }
    return splits, record


# ── dataset ─────────────────────────────────────────────────────────────


class XRFCrossDetectorDataset(Dataset):
    """Noise2Noise pairs across the two detectors of one or more scans.

    Each pixel yields up to two examples (PLAN: 18 000 px × 2 directions):

        direction 0 : input = detector A, target = detector B × R(E)
        direction 1 : input = detector B, target = detector A / R(E)

    so the target always lives in the input detector's response scale.
    With ``ratio_curve=None`` the targets stay raw (trial mode).

    Parameters
    ----------
    parts : list of (scan, spec_a, spec_b, indices)
        As produced by :func:`build_cross_detector_splits`.
    ratio_curve : np.ndarray (C,), dict scan -> np.ndarray (C,), or None
        R(E) = E[A]/E[B] per channel. A dict selects the curve per scan,
        which is how the tilted scan gets its own ratio (handoff 2
        column ``R_tilt``: the tilt raises R by up to 9.5%).
    global_scale : float
        Divide both input and target by this to keep values O(1).
    both_directions : bool
        If False, only direction 0 (A -> B) is served.
    loss_mask : np.ndarray (C,) bool, optional
        Exposed as ``self.loss_mask`` (torch.bool) for masked losses.
    return_weight : {False, 'ratio', 'poisson'}
        Per-channel loss weights, returned as a third item ``(x, y, w)``.

        ``'ratio'`` compensates only the rescaling: scaling detector B up
        by R multiplies its variance by R^2 while the mean grows by R, so
        a plain MSE is dominated by the low-energy channels where R ~ 6.
        Weights are 1/R (direction 0) and R (direction 1).

        ``'poisson'`` is the full inverse target variance, which also
        accounts for the count level: Var = R * E[a] in direction 0 and
        E[b] / R in direction 1, so w = 1/(R*(a+1)) and R/(b+1). The
        expectation is estimated from the *input* spectrum, which is
        independent of the target, so the weights carry no bias. Without
        this the Pb lines, two orders of magnitude brighter than Ca,
        take over the gradient and the Ca line is left under-fitted --
        visible as a collapsed spatial contrast in the fused Ca map.
    integral_anchor : dict or None
        Enables the integrated-line-intensity anchor behind script 07's
        ``--integral-loss-weight`` (default None = off, previous
        behavior unchanged). Keys: ``"op"``, a (K, C) array from
        :func:`net_line_operator` (the benchmark's net line integrals
        as linear functionals), and ``"r_line"``, a (K,) array or dict
        scan -> (K,) with the scalar response ratio at each line
        center. Every example then carries two extra tensors
        ``(anchor, var)``: the per-line inverse-variance combination of
        BOTH raw detectors in the input's response scale,

            direction 0 (A scale): I_k = wa*N_k(a) + (1-wa)*r_k*N_k(b)
            direction 1 (B scale): the same divided by r_k,

        with wa = r/(r+1), and its variance estimated from the observed
        counts (Var N_k(x) ~ sum(op_k^2 * (x+1))).

        Design reasoning. The anchor is NOT the noisy N2N target: in
        direction 0 the target is R*b, whose net integral at Ca carries
        R^2 ~ 34x detector B's window variance - a very noisy level
        reference that also uses only one detector. The per-channel N2N
        argument (target must be independent of the input) does not
        carry over to the level term either, because the level error
        being corrected is common-mode across pixels while any anchor
        noise is zero-mean: the systematic gradient survives averaging,
        the noise does not. The invvar combination is instead the
        minimum-variance unbiased per-pixel estimate of the line level
        (E[a] = r*E[b], so E[I_k] is the clean level in the input
        scale) and is the exact reference of the benchmark's
        ``bias_learned_pct`` column (09_fusion ``level_bias``), so the
        loss pins precisely the statistic that column measures. Caveat:
        the anchor contains the input's own noise, so a LARGE weight
        teaches the network to copy the raw per-pixel map noise back
        into the fused map (the cross-scan SNR gain would collapse to
        the classical weighted fusion); the weight must stay moderate.

        When set, ``__getitem__`` always returns 5-tuples
        ``(x, y, w, anchor, var)`` - ``w`` falls back to ones when
        ``return_weight`` is False.
    """

    def __init__(
        self,
        parts: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
        ratio_curve: np.ndarray | None = None,
        global_scale: float = 1.0,
        both_directions: bool = True,
        loss_mask: np.ndarray | None = None,
        return_weight: bool = False,
        integral_anchor: dict | None = None,
    ):
        self.return_weight = return_weight
        self.parts = parts
        self.integral_anchor = integral_anchor
        if integral_anchor is not None:
            self._int_op = np.asarray(integral_anchor["op"],
                                      dtype=np.float32)
            self._int_op2 = self._int_op ** 2
            rl = integral_anchor["r_line"]
            if isinstance(rl, dict):
                self._int_r = [np.asarray(rl[scan], dtype=np.float32)
                               for scan, _, _, _ in parts]
            else:
                rl = np.asarray(rl, dtype=np.float32)
                self._int_r = [rl] * len(parts)
        if ratio_curve is None:
            self.ratio_curve = None
            self._curves = [None] * len(parts)
        elif isinstance(ratio_curve, dict):
            self.ratio_curve = {
                k: np.asarray(v, dtype=np.float32)
                for k, v in ratio_curve.items()
            }
            missing = {scan for scan, _, _, _ in parts} - set(self.ratio_curve)
            if missing:
                raise KeyError(f"no ratio curve for scan(s): {sorted(missing)}")
            self._curves = [self.ratio_curve[scan] for scan, _, _, _ in parts]
        else:
            self.ratio_curve = np.asarray(ratio_curve, dtype=np.float32)
            self._curves = [self.ratio_curve] * len(parts)
        self.global_scale = float(global_scale)
        self.n_directions = 2 if both_directions else 1
        self.loss_mask = (
            None if loss_mask is None
            else torch.from_numpy(np.asarray(loss_mask, bool))
        )
        # flat example table: (part_i, pixel_row)
        self._table = np.array(
            [(pi, j) for pi, (_, _, _, idx) in enumerate(parts) for j in idx],
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return self._table.shape[0] * self.n_directions

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        direction, flat = divmod(idx, self._table.shape[0])
        pi, j = self._table[flat]
        _, fa, fb, _ = self.parts[pi]
        a = fa[j].astype(np.float32)
        b = fb[j].astype(np.float32)
        curve = self._curves[pi]

        if direction == 0:
            x, y = a, (b * curve if curve is not None else b)
        else:
            x, y = b, (a / curve if curve is not None else a)

        x = torch.from_numpy(x / self.global_scale).unsqueeze(0)
        y = torch.from_numpy(y / self.global_scale).unsqueeze(0)
        if not self.return_weight and self.integral_anchor is None:
            return x, y
        if not self.return_weight or curve is None:
            w = np.ones_like(a)
        elif self.return_weight == "poisson":
            w = (1.0 / (curve * (a + 1.0)) if direction == 0
                 else curve / (b + 1.0))
        else:
            w = (1.0 / curve) if direction == 0 else curve
        w = torch.from_numpy(w.astype(np.float32)).unsqueeze(0)
        if self.integral_anchor is None:
            return x, y, w

        # per-line invvar level anchor + its variance, input scale
        rl = self._int_r[pi]
        na, nb = self._int_op @ a, self._int_op @ b
        va = self._int_op2 @ (a + 1.0)
        vb = self._int_op2 @ (b + 1.0)
        wa = rl / (rl + 1.0)
        anchor = wa * na + (1.0 - wa) * rl * nb
        var = wa ** 2 * va + (1.0 - wa) ** 2 * rl ** 2 * vb
        if direction == 1:                       # bring to the B scale
            anchor, var = anchor / rl, var / rl ** 2
        gs = self.global_scale
        return (x, y, w,
                torch.from_numpy((anchor / gs).astype(np.float32)),
                torch.from_numpy((var / (gs * gs)).astype(np.float32)))

    def save_record(self, record: dict, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(record, fh)
