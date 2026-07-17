"""
01_run_analysis.py
Driver for xrf_core.run_scan — element maps for prova1 (and prova2 when
available) on detector 10264.

Run from the project root:
    python scripts/01_run_analysis.py

Data resolution order per dataset:
  1. complete .npy cache in results/10264/_npy_cache/   (fast, no raw data needed)
  2. raw MCA files in Resources/aurora-antico1-<dataset>/10264/

Datasets with neither source are skipped with a warning. If both prova1
and prova2 are available the script also renders the difference maps
(mode="compare_scans"); with a single dataset it falls back to
mode="standard".
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from xrf_core import load_elements, run_scan

WIDTH, HEIGHT = 120, 60
DETECTOR      = "10264"
OUTPUT_DIR    = os.path.join("results", DETECTOR)
CACHE_DIR     = os.path.join(OUTPUT_DIR, "_npy_cache")


def resolve_dataset_dir(dataset_name: str) -> str:
    """Return the first existing detector folder for a dataset, searching
    Resources/ first and then the repository root."""
    for root in ("Resources", "."):
        p = os.path.join(root, dataset_name, DETECTOR)
        if os.path.isdir(p):
            return p
    return os.path.join("Resources", dataset_name, DETECTOR)


CANDIDATES = {
    "prova1": resolve_dataset_dir("aurora-antico1-prova1"),
    "prova2": resolve_dataset_dir("aurora-antico1-prova2"),
}


def dataset_available(label: str, folder: str) -> bool:
    """True if a complete npy cache or the raw MCA folder exists."""
    el_keys = load_elements().keys()
    cache_complete = all(
        os.path.exists(os.path.join(CACHE_DIR, f"{label}_{k}.npy"))
        for k in el_keys
    )
    if cache_complete:
        return True
    return os.path.isdir(folder) and any(
        f.endswith(".mca") for f in os.listdir(folder)
    )


if __name__ == "__main__":
    datasets = {
        label: folder
        for label, folder in CANDIDATES.items()
        if dataset_available(label, folder)
    }
    skipped = sorted(set(CANDIDATES) - set(datasets))
    for label in skipped:
        print(f"NOTE: '{label}' skipped — no npy cache and no raw data in "
              f"{CANDIDATES[label]} (see README: Setup / Raw MCA data).")

    if not datasets:
        sys.exit(
            "ERROR: no dataset available. Place the raw MCA scans under "
            "Resources/ (see README) or provide the .npy cache in "
            f"{CACHE_DIR}."
        )

    run_scan(
        datasets=datasets,
        width=WIDTH,
        height=HEIGHT,
        output_dir=OUTPUT_DIR,
        mode="compare_scans" if len(datasets) >= 2 else "standard",
        integrator="fixed_hw",
    )
