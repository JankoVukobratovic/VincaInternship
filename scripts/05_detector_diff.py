"""
05_detector_diff.py
Per-element detector difference maps (detector 10264 − detector 19511),
computed separately for each scan: prova1, prova2 and the rotated scan.

For every available scan the script processes both detector folders,
applies the standard correction pipeline, builds the display cube
(combined Pb, corrected Zn/S) and saves:

    results/detector_diff/<scan>/diff_<El>.png     one PNG per element
    results/detector_diff/<scan>/diff_grid.png     overview grid

RED = more counts in 10264  |  BLUE = more counts in 19511.

Scans with missing raw data (and no .npy cache) are skipped with a
warning. Run from the project root:
    python scripts/05_detector_diff.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from xrf_core import (
    apply_corrections,
    build_calibration,
    build_display_cube,
    load_elements,
    process_folder,
    render_display_grid,
    render_individual,
)

DETECTORS  = ["10264", "19511"]
OUTPUT_DIR = os.path.join("results", "detector_diff")
CACHE_DIR  = os.path.join(OUTPUT_DIR, "_npy_cache")

# scan label -> (accepted dataset folder names, width, height)
SCANS = {
    "prova1":  (["aurora-antico1-prova1"], 120, 60),
    "prova2":  (["aurora-antico1-prova2"], 120, 60),
    "ruotato": (["antico1-prova4-ruotato", "aurora-antico1-ruotato"], 80, 45),
}


def resolve_scan_dir(names: list[str]) -> str | None:
    """First existing dataset folder containing both detector subfolders,
    searching Resources/ first and then the repository root."""
    for root in ("Resources", "."):
        for name in names:
            p = os.path.join(root, name)
            if all(os.path.isdir(os.path.join(p, det)) for det in DETECTORS):
                return p
    return None


def cache_complete(label: str, el_keys) -> bool:
    return all(
        os.path.exists(os.path.join(CACHE_DIR, f"{label}_{k}.npy"))
        for k in el_keys
    )


if __name__ == "__main__":
    slope, intercept = build_calibration()
    elements = load_elements()
    el_keys  = list(elements.keys())
    done     = []

    for scan, (names, width, height) in SCANS.items():
        scan_dir = resolve_scan_dir(names)
        cached   = all(cache_complete(f"{scan}_{det}", el_keys) for det in DETECTORS)
        if scan_dir is None and not cached:
            print(f"NOTE: '{scan}' skipped - no npy cache and no raw data for "
                  f"{names} (see README: Setup / Raw MCA data).")
            continue

        print(f"\n=== {scan}  ({width}x{height} px) ===")
        disp_cubes, disp_keys = {}, None
        for det in DETECTORS:
            label  = f"{scan}_{det}"
            folder = os.path.join(scan_dir or names[0], det)
            cube, keys = process_folder(
                folder, width, height, label, elements,
                "fixed_hw", slope, intercept, CACHE_DIR,
            )
            cube_c = apply_corrections(cube, keys, label)
            disp_cubes[det], disp_keys = build_display_cube(cube_c, keys, elements, label)

        d1, d2 = DETECTORS
        diff = disp_cubes[d1] - disp_cubes[d2]
        out_dir = os.path.join(OUTPUT_DIR, scan)

        render_individual(
            diff, disp_keys, elements, width, height,
            out_dir=out_dir, prefix="diff_", is_diff=True,
        )
        render_display_grid(
            diff, disp_keys, elements, width, height,
            save_path=os.path.join(out_dir, "diff_grid.png"),
            title=f"Detector difference — {scan}:  {d1} − {d2}\n"
                  f"RED = more in {d1}  |  BLUE = more in {d2}",
            is_diff=True,
        )
        done.append(scan)

    if not done:
        sys.exit("ERROR: no scan available. Place the raw MCA data under "
                 "Resources/ (see README: Setup / Raw MCA data).")
    print(f"\nDone: {', '.join(done)} -> {os.path.abspath(OUTPUT_DIR)}/")
