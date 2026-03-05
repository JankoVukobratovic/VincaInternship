"""
analiza_novi.py
────────────────────────────────────────────────────────────────────
XRF analiza – prova1 i prova2, detektori 10264 i 19511.
Generise:
  - mape elemenata za svaki detektor posebno (prova1 i prova2)
  - razlike prova1 − prova2 za svaki detektor posebno

Svi rezultati u: rezultati_novi/

Popravka u odnosu na better_main.py:
  - window smanjen sa ±1.0 keV na ±0.3 keV
    (FWHM pikova je ~0.23 keV, dakle 0.3 keV ≈ 2.6σ uhvata >99% pika)
  - NumPy cache – ne mora ponovo da se cita ako vec postoji
"""

import os, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import linregress


def _mk(name, dark_color):
    """Pravi colormapu: crna (0 = nema elementa) → boja (1 = max signal)."""
    return LinearSegmentedColormap.from_list(name, ["#000000", dark_color])


# ─── Elementi: ukloneni S, Ni, Cr (nema pikova); dodati K, Zn ─────
# Sve kolormape: crno = nema elementa, svetlo/boja = ima elementa
ELEMENT_MAP = {
    "K":  {"name": "Kalium",    "kev": 3.00,  "cmap": _mk("K",  "#FFD700")},  # crno→zlatno
    "Ca": {"name": "Calcium",   "kev": 3.69,  "cmap": _mk("Ca", "#FFFFFF")},  # crno→belo
    "Ti": {"name": "Titanium",  "kev": 4.51,  "cmap": _mk("Ti", "#FF6666")},  # crno→svetlocrveno
    "Fe": {"name": "Iron",      "kev": 6.40,  "cmap": _mk("Fe", "#FF2200")},  # crno→crveno
    "Cu": {"name": "Copper",    "kev": 8.04,  "cmap": _mk("Cu", "#00FFAA")},  # crno→tirkizno
    "Zn": {"name": "Zinc",      "kev": 8.64,  "cmap": _mk("Zn", "#66CCFF")},  # crno→plavo
    "As": {"name": "Arsenic",   "kev": 10.54, "cmap": _mk("As", "#FF8800")},  # crno→narandzasto
    "Pb": {"name": "Lead",      "kev": 12.61, "cmap": _mk("Pb", "#CC66FF")},  # crno→ljubicasto
}

ELEMENT_DIFF_MAP = {
    key: {"name": f"Δ {val['name']}", "kev": val["kev"], "cmap": "RdBu_r"}
    for key, val in ELEMENT_MAP.items()
}

# ─── Kalibracija ──────────────────────────────────────────────────
_CAL = np.array([[219, 6.4], [278, 8.0], [363, 10.5], [436, 12.6], [869, 25.3]])
_SLOPE, _INTERCEPT, *_ = linregress(_CAL[:, 0], _CAL[:, 1])

# ─── Dataset definicije ───────────────────────────────────────────
DATASETS = {
    "prova1": {"dir": "aurora-antico1-prova1", "w": 120, "h": 60},
    "prova2": {"dir": "aurora-antico1-prova2", "w": 120, "h": 60},
}
DETEKTORI = ["10264", "19511"]

IZLAZ     = "rezultati_novi"
NPY_CACHE = os.path.join(IZLAZ, "_npy_cache")
os.makedirs(NPY_CACHE, exist_ok=True)


# ══════════════════════════════════════════════════════════════════
#  POMOCNE FUNKCIJE
# ══════════════════════════════════════════════════════════════════

def parse_mca_file(filepath):
    meta, counts, in_data = {}, [], False
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line == "<<DATA>>":  in_data = True;  continue
            if line == "<<END>>":   in_data = False;  break
            if in_data:
                try: counts.append(int(line))
                except ValueError: pass
            elif " - " in line:
                k, v = line.split(" - ", 1)
                meta[k.strip()] = v.strip()
    return {
        "counts": np.array(counts, dtype=np.float64),
        "time":   float(meta.get("REAL_TIME", 1.0)),
    }


_CH_STEP = _SLOPE  # keV po kanalu ≈ 0.029

def fixed_window_integral(counts, energy_axis, target_kev, half_width_kev=0.3):
    """
    Fiksni pravougaoni prozor ±half_width_kev oko centra pika.
    Isti opseg za svaki piksel → stabilan piksel-to-piksel.
    half_width_kev=0.3 keV ≈ ±10 kanala (FWHM ~0.23 keV, pokriva >99% pika).
    """
    idx      = int(np.argmin(np.abs(energy_axis - target_kev)))
    half_ch  = max(1, int(round(half_width_kev / _CH_STEP)))
    lo       = max(0, idx - half_ch)
    hi       = min(len(counts) - 1, idx + half_ch)
    return float(counts[lo:hi + 1].sum())


def process_dataset(folder_path, w, h, label):
    """
    Cita sve None_N.mca fajlove i vraca 3D matricu (n_elem, h, w).
    Ako postoji npy cache, ucitava ga umesto ponovnog citanja.
    """
    el_keys = list(ELEMENT_MAP.keys())
    n_el    = len(el_keys)
    total   = w * h

    # ── Cache ──────────────────────────────────────────────────────
    cache_paths = {k: os.path.join(NPY_CACHE, f"{label}_{k}.npy") for k in el_keys}
    if all(os.path.exists(p) for p in cache_paths.values()):
        print(f"  [{label}] Ucitavam iz cache-a...")
        cube = np.stack([np.load(cache_paths[k]) for k in el_keys], axis=0)
        return cube, el_keys

    cube = np.zeros((n_el, h, w), dtype=np.float64)

    for i in range(1, total + 1):
        path = os.path.join(folder_path, f"None_{i}.mca")
        if not os.path.exists(path):
            continue

        data   = parse_mca_file(path)
        counts = data["counts"]
        energy = np.arange(len(counts)) * _SLOPE + _INTERCEPT
        row    = (i - 1) // w
        col    = (i - 1) % w

        for ei, key in enumerate(el_keys):
            cube[ei, row, col] = fixed_window_integral(counts, energy, ELEMENT_MAP[key]["kev"])

        if i % 500 == 0 or i == total:
            print(f"  [{label}] {i}/{total} ({100*i//total}%)")

    # Sacuvaj cache
    for ei, k in enumerate(el_keys):
        np.save(cache_paths[k], cube[ei])
    print(f"  [{label}] Cache sacuvan.")
    return cube, el_keys


def render_grid(cube, el_keys, el_map, w, h, save_path, title=""):
    """Grid slika svih elemenata. Bela pozadina, mape crno→boja."""
    is_diff = (el_map is ELEMENT_DIFF_MAP)
    n  = len(el_keys)
    nc = 4
    nr = math.ceil(n / nc)

    fig, axes = plt.subplots(nr, nc, figsize=(5.5 * nc, 4.2 * nr + 1.2), dpi=110)
    fig.patch.set_facecolor("white")
    axes = axes.flatten()

    for idx, key in enumerate(el_keys):
        cfg = el_map[key]
        img = cube[idx]
        ax  = axes[idx]

        if is_diff:
            abs_max = np.percentile(np.abs(img), 99)
            abs_max = abs_max if abs_max > 0 else 1.0
            vmin, vmax = -abs_max, abs_max
        else:
            vmin = max(0, np.percentile(img, 1))
            vmax = np.percentile(img, 99)
            vmax = vmax if vmax > vmin else vmin + 1.0

        im = ax.imshow(
            img, cmap=cfg["cmap"], aspect="auto", origin="upper",
            interpolation="nearest", vmin=vmin, vmax=vmax
        )
        ax.set_title(f"{cfg['name']}  ({key}, {cfg['kev']} keV)",
                     fontsize=9, fontweight="bold", color="black", pad=4)
        ax.set_xticks([0, w // 2, w])
        ax.set_yticks([0, h // 2, h])
        ax.tick_params(labelsize=7, colors="black")

        cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label(
            "Δ counts (prova1−prova2)" if is_diff else "counts",
            fontsize=7, color="black"
        )
        cbar.ax.tick_params(labelsize=7, colors="black")

    for j in range(len(el_keys), len(axes)):
        axes[j].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", color="black")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Sacuvano: {os.path.relpath(save_path)}")


# ══════════════════════════════════════════════════════════════════
#  OBRADA
# ══════════════════════════════════════════════════════════════════

cubes = {}  # cubes[(prova, det)] = (cube, el_keys)

for prova, cfg in DATASETS.items():
    for det in DETEKTORI:
        folder = os.path.join(cfg["dir"], det)
        label  = f"{prova}_{det}"
        print(f"\n{'='*60}")
        print(f"  Obrada: {label}  ({cfg['w']}x{cfg['h']} = {cfg['w']*cfg['h']} tacaka)")
        print(f"{'='*60}")
        cube, keys = process_dataset(folder, cfg["w"], cfg["h"], label)
        cubes[(prova, det)] = (cube, keys)


# ══════════════════════════════════════════════════════════════════
#  SLIKE: mape elemenata po detektoru
# ══════════════════════════════════════════════════════════════════

print("\n--- Generisanje slika mapa elemenata ---")

for prova, cfg in DATASETS.items():
    for det in DETEKTORI:
        cube, keys = cubes[(prova, det)]
        save = os.path.join(IZLAZ, prova, f"elementi_{det}.png")
        render_grid(
            cube, keys, ELEMENT_MAP,
            cfg["w"], cfg["h"], save,
            title=f"Mape elemenata – {prova}  |  Detektor {det}\n"
                  f"(fiksni window ±0.3 keV, {cfg['w']}×{cfg['h']} tacaka)"
        )


# ══════════════════════════════════════════════════════════════════
#  SLIKE: razlike prova1 − prova2 po detektoru
# ══════════════════════════════════════════════════════════════════

print("\n--- Generisanje slika razlika prova1 − prova2 ---")

for det in DETEKTORI:
    cube1, keys = cubes[("prova1", det)]
    cube2, _    = cubes[("prova2", det)]
    diff = cube1 - cube2

    cfg1 = DATASETS["prova1"]
    save = os.path.join(IZLAZ, "razlike_prova1_vs_prova2", f"diff_{det}.png")
    render_grid(
        diff, keys, ELEMENT_DIFF_MAP,
        cfg1["w"], cfg1["h"], save,
        title=f"Razlike mapa: prova1 − prova2  |  Detektor {det}\n"
              f"CRVENO = vise u prova1  |  PLAVO = vise u prova2  |  BELO = jednako"
    )

print(f"\n{'='*60}")
print(f"  Sve sacuvano u: {os.path.abspath(IZLAZ)}/")
print(f"{'='*60}")
