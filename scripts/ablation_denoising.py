"""
ablation_denoising.py
===============================================================================
Denoising ablation (paper Sec. "Effect of Self-Supervised Denoising",
Table VII): cross-scan CVI robustness with and without the U-Net stage.

Both arms share the IDENTICAL element extraction (fixed_hw windows,
detector 10264) and the identical CVI computation from 02_vulnerability.py;
the only difference is whether the raw counts pass through the trained
1D U-Net (xrf-denoise/experiments/A_scratch/checkpoints/best_model.pt)
before extraction.

    raw  : counts             -> element maps -> CVI
    unet : counts -> U-Net    -> element maps -> CVI

Metrics between prova1 and prova2: composite Wasserstein-1, pixel-wise
Pearson r, SSIM (7x7), plus per-element cross-scan Pearson r.

Run from the project root:
    python scripts/ablation_denoising.py
"""

import importlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XRFDEN = os.path.join(ROOT, "xrf-denoise")
sys.path.insert(0, XRFDEN)                       # for src.config / src.models
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

import xrf_core                                   # noqa: E402
vuln = importlib.import_module("02_vulnerability")

DET = "10264"
ROWS, COLS = vuln.ROWS, vuln.COLS
TOTAL = ROWS * COLS
CACHE_DIR = os.path.join("results", "vulnerability_mapping")
os.makedirs(CACHE_DIR, exist_ok=True)

_ELEMENTS_JSON = xrf_core.load_elements()


def _json_key(el):
    """Map a pipeline element name (e.g. Pb_La) to its elements.json key."""
    for cand in vuln._ELEMENT_ALIASES.get(el, [el]):
        if cand in _ELEMENTS_JSON:
            return cand
    raise KeyError(f"no elements.json entry for {el}")


def load_cube(dataset):
    """Raw counts cube (ROWS, COLS, n_ch) for detector 10264."""
    cache = os.path.join(CACHE_DIR, f"ablation_cube_{dataset}_{DET}.npy")
    if os.path.exists(cache):
        print(f"  [{dataset}] cube from cache")
        return np.load(cache)

    if dataset == "prova1":
        p = os.path.join(XRFDEN, "data", "processed", f"{DET}_raw.npy")
        if os.path.exists(p):
            print(f"  [{dataset}] cube from xrf-denoise cache")
            cube = np.load(p).astype(np.float64)
            np.save(cache, cube)
            return cube

    folder = os.path.join(
        vuln.resolve_dataset_dir(f"aurora-antico1-{dataset}"), DET)
    print(f"  [{dataset}] parsing {TOTAL} MCA files from {folder} ...")
    probe = xrf_core.parse_mca_file(os.path.join(folder, "None_1.mca"))
    n_ch = len(probe["counts"])
    cube = np.zeros((ROWS, COLS, n_ch), dtype=np.float64)
    for i in range(1, TOTAL + 1):
        data = xrf_core.parse_mca_file(os.path.join(folder, f"None_{i}.mca"))
        cube[(i - 1) // COLS, (i - 1) % COLS] = data["counts"]
        if i % 2000 == 0:
            print(f"    {i}/{TOTAL}", flush=True)
    np.save(cache, cube)
    return cube


def extract_maps(cube):
    """Element maps via the same integrator as the cached pipeline maps."""
    n_ch = cube.shape[2]
    en = xrf_core.energy_axis(n_ch, vuln._SLOPE, vuln._INTERCEPT)
    maps = {}
    for el in vuln.ELEMENTS:
        key = _json_key(el)
        cfg_el = _ELEMENTS_JSON[key]
        m = np.zeros((ROWS, COLS))
        for r in range(ROWS):
            for c in range(COLS):
                m[r, c] = xrf_core.integrate(cube[r, c], en, key, cfg_el,
                                             "fixed_hw")
        maps[el] = m
    return maps


def load_unet():
    import torch
    from src.config import Config
    from src.models.unet1d import UNet1D

    c = Config()
    model_path = c.abs_path(c.exp_a_dir) / "checkpoints" / "best_model.pt"
    with open(c.abs_path(c.exp_a_dir) / "results" / "phase4a_summary.json") as f:
        gscale = json.load(f)["global_scale"]
    model = UNet1D(base_filters=c.base_filters, n_blocks=c.n_encoder_blocks,
                   dropout=0).to(c.device)
    model.load_state_dict(torch.load(model_path, map_location=c.device,
                                     weights_only=True))
    model.eval()
    return model, gscale, c.device


def denoise(model, cube, gscale, device, bs=256):
    import torch
    flat = cube.reshape(-1, cube.shape[2])
    out = np.zeros_like(flat)
    with torch.no_grad():
        for i in range(0, flat.shape[0], bs):
            x = torch.from_numpy(flat[i:i + bs] / gscale).float()
            x = x.unsqueeze(1).to(device)
            y = model(x).squeeze(1).cpu().numpy() * gscale
            out[i:i + bs] = np.maximum(y, 0)
    return out.reshape(cube.shape)


def corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


if __name__ == "__main__":
    print("=" * 70)
    print("  DENOISING ABLATION (detector 10264, prova1 vs prova2)")
    print("=" * 70)

    print("\nLoading U-Net...")
    model, gscale, device = load_unet()
    print(f"  global_scale={gscale:.1f}, device={device}")

    maps = {"raw": {}, "unet": {}}
    for ds in ("prova1", "prova2"):
        print(f"\n[{ds}]")
        cube = load_cube(ds)
        print("  extracting element maps (raw)...")
        maps["raw"][ds] = extract_maps(cube)
        print("  denoising...")
        cube_d = denoise(model, cube, gscale, device)
        print("  extracting element maps (denoised)...")
        maps["unet"][ds] = extract_maps(cube_d)

    # Sanity: the raw arm must reproduce the cached pipeline maps
    print("\nSanity check: raw-arm maps vs cached pipeline maps (prova1):")
    cached = vuln.load_element_maps("prova1")
    for el in vuln.ELEMENTS:
        print(f"  {el:6s} r = {corr(maps['raw']['prova1'][el], cached[el]):.4f}")

    results = {}
    for arm, label in (("unet", "With U-Net denoising"),
                       ("raw", "Without denoising")):
        v1 = vuln.phase2_vulnerability(maps[arm]["prova1"], f"prova1-{arm}")
        v2 = vuln.phase2_vulnerability(maps[arm]["prova2"], f"prova2-{arm}")
        val = vuln.phase3_validation(v1["cvi"], v2["cvi"],
                                     v1["risk_maps"], v2["risk_maps"])
        el_r = {el: corr(maps[arm]["prova1"][el], maps[arm]["prova2"][el])
                for el in vuln.ELEMENTS}
        results[arm] = {"label": label, "w1": val["w1_cvi"],
                        "r": val["pearson"], "ssim": val["ssim"],
                        "el_r": el_r}

    print("\n" + "=" * 70)
    print("  ABLATION SUMMARY (Table VII)")
    print("=" * 70)
    print(f"  {'Configuration':28s}  {'W1':>8s}  {'r':>8s}  {'SSIM':>8s}")
    for arm in ("unet", "raw"):
        r = results[arm]
        print(f"  {r['label']:28s}  {r['w1']:8.4f}  {r['r']:8.4f}"
              f"  {r['ssim']:8.4f}")

    print("\n  Per-element cross-scan Pearson r:")
    print(f"  {'Element':8s}  {'with U-Net':>10s}  {'raw':>10s}")
    for el in vuln.ELEMENTS:
        print(f"  {el:8s}  {results['unet']['el_r'][el]:10.4f}"
              f"  {results['raw']['el_r'][el]:10.4f}")

    out = {arm: {k: results[arm][k] for k in ("w1", "r", "ssim", "el_r")}
           for arm in results}
    out_path = os.path.join(CACHE_DIR, "ablation_denoising.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {out_path}")
