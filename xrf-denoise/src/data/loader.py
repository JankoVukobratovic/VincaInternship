"""Load raw XRF datacubes from .mca files."""

import numpy as np
from pathlib import Path
from typing import Optional


def parse_mca_file(filepath: str | Path) -> dict:
    """
    Parse a single .mca spectrum file.

    Parameters
    ----------
    filepath : str or Path
        Path to the .mca file.

    Returns
    -------
    dict
        'counts': np.ndarray of shape (n_channels,), dtype float64
        'time': float, acquisition time in seconds (REAL_TIME)
    """
    meta, counts, in_data = {}, [], False
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == "<<DATA>>":
                in_data = True
                continue
            if line == "<<END>>":
                in_data = False
                break
            if in_data:
                try:
                    counts.append(int(line))
                except ValueError:
                    pass
            elif " - " in line:
                k, v = line.split(" - ", 1)
                meta[k.strip()] = v.strip()
    return {
        "counts": np.array(counts, dtype=np.float64),
        "time": float(meta.get("REAL_TIME", 1.0)),
    }


def load_datacube(
    dataset_dir: str | Path,
    detector: str,
    rows: int = 60,
    cols: int = 120,
    normalize_cps: bool = False,
    cache_path: Optional[str | Path] = None,
) -> tuple[np.ndarray, dict]:
    """
    Load all spectra from a detector folder into a 3D datacube.

    Parameters
    ----------
    dataset_dir : str or Path
        Root dataset directory (e.g., 'aurora-antico1-prova1').
    detector : str
        Detector ID ('10264' or '19511').
    rows, cols : int
        Scan grid dimensions.
    normalize_cps : bool
        If True, divide counts by acquisition time (counts per second).
    cache_path : str or Path, optional
        If provided, save/load cached .npy file.

    Returns
    -------
    cube : np.ndarray, shape (rows, cols, n_channels), float32
    metadata : dict with 'n_channels', 'mean_time', 'total_counts_mean'
    """
    if cache_path and Path(cache_path).exists():
        cube = np.load(cache_path).astype(np.float32)
        n_ch = cube.shape[2]
        return cube, {'n_channels': n_ch, 'from_cache': True}

    folder = Path(dataset_dir) / detector
    total = rows * cols

    # Determine channel count from first file
    first = parse_mca_file(folder / "None_1.mca")
    n_ch = len(first['counts'])

    cube = np.zeros((rows, cols, n_ch), dtype=np.float64)
    times = []

    for i in range(1, total + 1):
        data = parse_mca_file(folder / f"None_{i}.mca")
        r = (i - 1) // cols
        c = (i - 1) % cols
        spectrum = data['counts']
        t = data['time']
        times.append(t)

        if normalize_cps:
            cube[r, c, :len(spectrum)] = spectrum / max(t, 0.1)
        else:
            cube[r, c, :len(spectrum)] = spectrum

    cube = cube.astype(np.float32)

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, cube)

    metadata = {
        'n_channels': n_ch,
        'rows': rows,
        'cols': cols,
        'detector': detector,
        'mean_time': float(np.mean(times)),
        'total_counts_mean': float(cube.sum(axis=2).mean()),
        'from_cache': False,
    }
    return cube, metadata


def load_both_detectors(
    dataset_dir: str | Path,
    detector_a: str = "10264",
    detector_b: str = "19511",
    rows: int = 60,
    cols: int = 120,
    cache_dir: Optional[str | Path] = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Load datacubes from both detectors.

    Returns
    -------
    cube_a, cube_b : np.ndarray, shape (rows, cols, n_channels)
    metadata : dict
    """
    cache_a = Path(cache_dir) / f"{detector_a}_raw.npy" if cache_dir else None
    cache_b = Path(cache_dir) / f"{detector_b}_raw.npy" if cache_dir else None

    print(f"  Loading detector {detector_a}...", end=" ", flush=True)
    cube_a, meta_a = load_datacube(dataset_dir, detector_a, rows, cols,
                                   cache_path=cache_a)
    print(f"OK ({meta_a['n_channels']} ch)")

    print(f"  Loading detector {detector_b}...", end=" ", flush=True)
    cube_b, meta_b = load_datacube(dataset_dir, detector_b, rows, cols,
                                   cache_path=cache_b)
    print(f"OK ({meta_b['n_channels']} ch)")

    metadata = {
        'n_channels': meta_a['n_channels'],
        'rows': rows, 'cols': cols,
        'detector_a': detector_a,
        'detector_b': detector_b,
        'mean_counts_a': float(cube_a.sum(axis=2).mean()),
        'mean_counts_b': float(cube_b.sum(axis=2).mean()),
    }
    return cube_a, cube_b, metadata
