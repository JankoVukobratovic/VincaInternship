"""Cross-detector Noise2Noise dataset (dual-detector paper, PLAN §4.3 + 8.4).

The two SDD detectors observe the same pixel simultaneously, so their
spectra are conditionally independent Poisson realizations — but of
DIFFERENT expected signals: the response ratio R(E) = E[A]/E[B] runs
from ~6 at Ca Kα to ~0.65 at Pb Lγ. A valid Noise2Noise target must
therefore be rescaled into the input detector's response scale,
channel by channel, before training (PLAN amendment 8.4); otherwise
the network learns detector B's response instead of the clean signal.

Until Person A delivers the smooth GP ratio curve (handoff 2), a
PROVISIONAL curve interpolated from the 8 reliable line ratios of
``results/detector_diff/efficiency_ratios.csv`` is available via
:func:`ratio_curve_from_table`; trial trainings may also run entirely
unscaled (``ratio_curve=None``), as PLAN §4.3 allows.

Scaling caveat for the loss: a target multiplied by R(E) is no longer
integer Poisson counts — use MSE (or a variance-weighted loss) on
scaled targets, not ``poisson_nll`` on raw counts.

Split policy (PLAN §4.3): training on prova1 + ruotato, evaluation on
prova2 — per-scan split with spatial-block validation carved out of
the training scans. Never random per-pixel.
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# loss window (keV): outside this range the ratio curve is extrapolation
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
    """R(E) from a generic (kev, R) table — the handoff-2 GP curve."""
    kev, r = [], []
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            kev.append(float(row[kev_col]))
            r.append(float(row[r_col]))
    return _interp_ratio(kev, r, n_channels, slope, intercept)


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
    ratio_curve : np.ndarray (C,) or None
        R(E) = E[A]/E[B] per channel.
    global_scale : float
        Divide both input and target by this to keep values O(1).
    both_directions : bool
        If False, only direction 0 (A -> B) is served.
    loss_mask : np.ndarray (C,) bool, optional
        Exposed as ``self.loss_mask`` (torch.bool) for masked losses.
    """

    def __init__(
        self,
        parts: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
        ratio_curve: np.ndarray | None = None,
        global_scale: float = 1.0,
        both_directions: bool = True,
        loss_mask: np.ndarray | None = None,
    ):
        self.parts = parts
        self.ratio_curve = (
            None if ratio_curve is None
            else np.asarray(ratio_curve, dtype=np.float32)
        )
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

        if direction == 0:
            x, y = a, (b * self.ratio_curve if self.ratio_curve is not None else b)
        else:
            x, y = b, (a / self.ratio_curve if self.ratio_curve is not None else a)

        x = torch.from_numpy(x / self.global_scale).unsqueeze(0)
        y = torch.from_numpy(y / self.global_scale).unsqueeze(0)
        return x, y

    def save_record(self, record: dict, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(record, fh)
