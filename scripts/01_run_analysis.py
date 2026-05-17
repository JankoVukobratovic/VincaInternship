"""
01_run_analysis.py
Example driver for xrf_core.run_scan — compares prova1 vs prova2 on detector 10264.
Run from project root: `python scripts/01_run_analysis.py`
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from xrf_core import run_scan


if __name__ == "__main__":
    run_scan(
        datasets={
            "prova1": "Resources/aurora-antico1-prova1/10264",
            "prova2": "Resources/aurora-antico1-prova2/10264",
        },
        width=120,
        height=60,
        output_dir="results/10264",
        mode="compare_scans",
        integrator="fixed_hw",
    )
