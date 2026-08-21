"""io_utils.py - one CSV convention for the whole team.

Rules:
  - every experiment writes results/<name>.csv through write_rows();
  - shared columns (element, candidate, region, r, ssim, bias_pct,
    cv_ratio, n_px) keep their names EXACTLY; experiment-specific keys
    (angle, dose, defect, defect_family, hole_px, z, ...) are added
    freely - the writer unions all keys;
  - figures are built from the CSVs, never from in-memory state, so
    partial runs can be merged and everything is restartable;
  - CSVs are committed to git; checkpoints/npy stay out (size).
"""

import csv
import os

from common import core


def _path(name: str) -> str:
    os.makedirs(core.RESULTS_DIR, exist_ok=True)
    return os.path.join(core.RESULTS_DIR, name if name.endswith(".csv")
                        else name + ".csv")


def write_rows(name: str, rows: list, append: bool = False) -> str:
    """Write/append dict rows; the header is the union of all keys."""
    path = _path(name)
    if not rows:
        return path
    old = read_rows(name) if (append and os.path.exists(path)) else []
    allrows = old + rows
    fields = []
    for r in allrows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(allrows)
    return path


def read_rows(name: str) -> list:
    path = _path(name)
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def fig_path(name: str) -> str:
    os.makedirs(core.FIGURES_DIR, exist_ok=True)
    return os.path.join(core.FIGURES_DIR, name)
