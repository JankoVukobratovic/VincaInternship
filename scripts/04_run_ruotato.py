"""
04_run_ruotato.py
Driver for the rotated scan ("ruotato", 80x45 px): per-detector element
maps, detector sum (improved SNR), detector difference, individual maps
and annotated summed spectra.

Run from the project root:
    python scripts/04_run_ruotato.py

Accepts the dataset folder under either of its historical names,
searching Resources/ first and then the repository root:
    antico1-prova4-ruotato   |   aurora-antico1-ruotato
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from xrf_core import run_scan

WIDTH, HEIGHT = 80, 45
DETECTORS     = ["10264", "19511"]
OUTPUT_DIR    = "results_rotated"

DATASET_NAMES = ["antico1-prova4-ruotato", "aurora-antico1-ruotato"]


def resolve_dataset_dir() -> str | None:
    for root in ("Resources", "."):
        for name in DATASET_NAMES:
            p = os.path.join(root, name)
            if all(os.path.isdir(os.path.join(p, det)) for det in DETECTORS):
                return p
    return None


if __name__ == "__main__":
    dataset_dir = resolve_dataset_dir()
    if dataset_dir is None:
        sys.exit(
            "ERROR: rotated-scan data not found. Expected one of "
            f"{DATASET_NAMES} under Resources/ (see README: Setup)."
        )
    print(f"Rotated scan data: {dataset_dir}")

    run_scan(
        datasets={det: os.path.join(dataset_dir, det) for det in DETECTORS},
        width=WIDTH,
        height=HEIGHT,
        output_dir=OUTPUT_DIR,
        mode="compare_detectors",
        integrator="fixed_hw",
        detectors=DETECTORS,
        individual_maps=True,
        stacked_spectrum=True,
    )
