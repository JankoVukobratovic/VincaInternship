"""
analiza_korigovana.py
──────────────────────────────────────────────────────────────────────
Korigovana XRF analiza – prova1 i prova2, detektori 10264 i 19511.

Ispravke u odnosu na analiza_novi.py:
  1. K target: 3.31 keV (ispravno), hw=0.15 keV (uzak, izbegava Ca)
  2. Pozadinsko oduzimanje: linearna interpolacija ispod svakog pika
  3. Zn korekcija: oduzima Cu Kβ doprinos (Kβ/Kα ≈ 0.17)
  4. As: obelezena kao "As+PbLα (inseparabilno)" – uz napomenu
  5. Pb: cisti Lβ signal (validan)
  6. Sve vrednosti klipirane na 0 (net signal ≥ 0)

Izlaz: rezultati_korigovani/
"""

import os, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import linregress

# ── Colormap helper ────────────────────────────────────────────
def _mk(name, color):
    return LinearSegmentedColormap.from_list(name, ["#000000", color])

# ── Elementi (sa ispravnim energijama) ─────────────────────────
#
#  K  : target pomeren na 3.31 keV, hw=0.15 (uzak prozor, izbegava Ca Kα)
#  As : zadrzano, ali ce biti korigovano za Pb Lα doprinos
#  Pb : Lβ1 linija (12.61 keV) – cista, nije zahvacena drugim linijama
#
ELEMENT_MAP = {
    "S":    {"name": "S Kα",   "kev": 2.31,  "hw": 0.20, "cmap": _mk("S",   "#FFFF66")},
    "K":    {"name": "K Kα",   "kev": 3.31,  "hw": 0.15, "cmap": _mk("K",   "#FFD700")},
    "Ca":   {"name": "Ca Kα",  "kev": 3.69,  "hw": 0.30, "cmap": _mk("Ca",  "#FFFFFF")},
    "Ti":   {"name": "Ti Kα",  "kev": 4.51,  "hw": 0.30, "cmap": _mk("Ti",  "#FF6666")},
    "Fe":   {"name": "Fe Kα",  "kev": 6.40,  "hw": 0.30, "cmap": _mk("Fe",  "#FF2200")},
    "Cu":   {"name": "Cu Kα",  "kev": 8.04,  "hw": 0.30, "cmap": _mk("Cu",  "#00FFAA")},
    "Zn":   {"name": "Zn Kα",  "kev": 8.64,  "hw": 0.20, "cmap": _mk("Zn",  "#66CCFF")},  # hw 0.25→0.20: hi+1 bi pao u Cu Kβ (8.903 keV)
    "PbLl": {"name": "Pb Ll",  "kev": 9.185, "hw": 0.28, "cmap": _mk("PbLl","#BB88FF")},
    "As":   {"name": "Pb Lα",  "kev": 10.54, "hw": 0.30, "cmap": _mk("As",  "#FF8800")},
    "Pb":   {"name": "Pb Lβ",  "kev": 12.61, "hw": 0.30, "cmap": _mk("Pb",  "#CC66FF")},
    "PbLg": {"name": "Pb Lγ",  "kev": 14.77, "hw": 0.35, "cmap": _mk("PbLg","#9933AA")},
}

ELEMENT_DIFF_MAP = {
    k: {"name": f"Δ {v['name']}", "kev": v["kev"], "hw": v["hw"], "cmap": "RdBu_r"}
    for k, v in ELEMENT_MAP.items()
}

# ── Prikaz: kombinovani Pb + odvojeni As ───────────────────────
DISPLAY_MAP = {
    "S":   {"name": "S Kα",              "cmap": _mk("S",  "#FFFF66")},
    "K":   {"name": "K Kα",              "cmap": _mk("K",  "#FFD700")},
    "Ca":  {"name": "Ca Kα",             "cmap": _mk("Ca", "#FFFFFF")},
    "Ti":  {"name": "Ti Kα",             "cmap": _mk("Ti", "#FF6666")},
    "Fe":  {"name": "Fe Kα",             "cmap": _mk("Fe", "#FF2200")},
    "Cu":  {"name": "Cu Kα",             "cmap": _mk("Cu", "#00FFAA")},
    "Zn":  {"name": "Zn Kα",             "cmap": _mk("Zn", "#66CCFF")},
    "Pb":  {"name": "Pb (Lα+Lβ+Ll+Lγ)", "cmap": _mk("Pb", "#CC66FF")},
    "As":  {"name": "As Kα (kor.)",      "cmap": _mk("As", "#FF8800")},
}
DISPLAY_DIFF_MAP = {
    k: {"name": f"Δ {v['name']}", "cmap": "RdBu_r"}
    for k, v in DISPLAY_MAP.items()
}

# ── Kalibracija ────────────────────────────────────────────────
_CAL = np.array([[219, 6.4], [278, 8.0], [363, 10.5], [436, 12.6], [869, 25.3]])
_SLOPE, _INTERCEPT, *_ = linregress(_CAL[:, 0], _CAL[:, 1])

# ── Korekcioni faktori ─────────────────────────────────────────
# Zn: Cu Kβ/Kα ≈ 0.17 (standardni odnos za Cu fluorescenciju)
CU_KB_KA_RATIO = 0.17

# As: Pb Lα / Pb Lβ1 odnos – odredjujemo iz samih podataka (vidi dole)
# Inicijalna procena: 1.40 (tipicna vrednost za Pb pri 30-40 kV ekscitaciji)
PB_LA_LB_RATIO_INIT = 1.40

DATASETS  = {
    "prova1": {"dir": "aurora-antico1-prova1", "w": 120, "h": 60},
    "prova2": {"dir": "aurora-antico1-prova2", "w": 120, "h": 60},
}
DETEKTORI = ["10264", "19511"]
IZLAZ     = "rezultati_korigovani"
NPY_CACHE = os.path.join(IZLAZ, "_npy_cache")
os.makedirs(NPY_CACHE, exist_ok=True)


# ══════════════════════════════════════════════════════════════
#  POMOCNE FUNKCIJE
# ══════════════════════════════════════════════════════════════

def parse_mca_file(filepath):
    meta, counts, in_data = {}, [], False
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line == "<<DATA>>":  in_data = True;  continue
            if line == "<<END>>":   break
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


def bg_subtracted_integral(counts, energy, target_kev, hw=0.30, bg_hw=0.25):
    """
    Neto signal = integral pika − linearna pozadina.

    Peak prozor : [lo, hi]  = [idx-half_ch, idx+half_ch]
    BG prozori  : [lo-bg_ch, lo)  i  (hi, hi+1+bg_ch]
                  Oba prozora ISKLJUCUJU ivicne kanale pika (lo i hi),
                  simetricno sa Python slice semantikom.

    Vraca max(0, neto_signal) – nikad negativno.
    """
    idx      = int(np.argmin(np.abs(energy - target_kev)))
    half_ch  = max(1, int(round(hw   / _SLOPE)))
    bg_ch    = max(1, int(round(bg_hw / _SLOPE)))

    lo = max(0,               idx - half_ch)
    hi = min(len(counts) - 1, idx + half_ch)

    bg_lo_l = max(0,               lo - bg_ch)
    bg_lo_r = lo
    bg_hi_l = hi + 1
    bg_hi_r = min(len(counts) - 1, hi + 1 + bg_ch)

    bg_left  = counts[bg_lo_l:bg_lo_r].mean() if bg_lo_r > bg_lo_l else counts[lo]
    bg_right = counts[bg_hi_l:bg_hi_r].mean() if bg_hi_r > bg_hi_l else counts[hi]

    n_peak  = hi - lo + 1
    bg_line = np.linspace(bg_left, bg_right, n_peak)

    return max(0.0, float((counts[lo : hi + 1] - bg_line).sum()))


def k_signal(counts, energy):
    """
    Specijalna K Kα metoda (3.3138 keV).

    Problem: K je okruzen Ar Kα (2.957 keV) sleva i Ca Kα (3.69 keV) zdesna.
    Standardni bg prozor zahvata rep Ar pika (levo) i nagib Ca pika (desno),
    sto daje lazno visoku pozadinu i net K = 0.

    Resenje: tesni sideband ODMAH uz K pik, u cistoj dolini izmedju pikova:
      BG levo : [3.23, 3.28] keV  – posle Ar repa, pre K pika
      BG desno: [3.35, 3.42] keV  – posle K pika, pre Ca nagiba
      K prozor: ±2 kanala oko 3.314 keV (5 kanala = 0.146 keV)
    """
    k_ch     = int(np.argmin(np.abs(energy - 3.3138)))
    pk_lo    = k_ch - 2
    pk_hi    = k_ch + 2

    ch_ll = int(np.argmin(np.abs(energy - 3.23)))
    ch_lr = int(np.argmin(np.abs(energy - 3.28)))
    ch_rl = int(np.argmin(np.abs(energy - 3.35)))
    ch_rr = int(np.argmin(np.abs(energy - 3.42)))

    bg_left  = counts[ch_ll:ch_lr + 1].mean() if ch_lr >= ch_ll else counts[pk_lo]
    bg_right = counts[ch_rl:ch_rr + 1].mean() if ch_rr >= ch_rl else counts[pk_hi]
    bg_per_ch = (bg_left + bg_right) / 2.0

    n   = pk_hi - pk_lo + 1
    net = max(0.0, float(counts[pk_lo:pk_hi + 1].sum()) - bg_per_ch * n)
    return net


def process_dataset(folder_path, w, h, label):
    """
    Cita sve None_N.mca fajlove i vraca 3D matricu (n_elem, h, w).
    Koristi NPY cache ako postoji.
    """
    el_keys = list(ELEMENT_MAP.keys())
    n_el    = len(el_keys)
    total   = w * h

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
            if key == "K":
                val = k_signal(counts, energy)
            else:
                el  = ELEMENT_MAP[key]
                val = bg_subtracted_integral(counts, energy, el["kev"], el["hw"])
            cube[ei, row, col] = val

        if i % 500 == 0 or i == total:
            print(f"  [{label}] {i}/{total} ({100*i//total}%)")

    for ei, k in enumerate(el_keys):
        np.save(cache_paths[k], cube[ei])
    print(f"  [{label}] Cache sacuvan.")
    return cube, el_keys


def apply_corrections(cube, el_keys):
    """
    Primenjuje:
      1. Zn korekcija: Zn_net = Zn - CU_KB_KA_RATIO × Cu
      2. As korekcija: As_net = As - PB_LA_LB_RATIO × Pb
         Ratio se odredjuje iz podataka (korelacijom) ako je moguce.
    """
    cube = cube.copy()

    ku = {k: i for i, k in enumerate(el_keys)}

    # ── Zn: oduzmi Cu Kβ doprinos ────────────────────────────
    if "Cu" in ku and "Zn" in ku:
        cube[ku["Zn"]] = np.maximum(0, cube[ku["Zn"]] - CU_KB_KA_RATIO * cube[ku["Cu"]])

    # ── Pb Lα (prozor 10.54 keV): As nije prisutan u uzorku,
    #    sirovi integral direktno prikazuje Pb Lα distribuciju — bez korekcije.

    # ── S: oduzmi Pb Mα doprinos ─────────────────────────────
    # Pb Mα (2.346 keV) i S Kα (2.308 keV) se preklapaju u S prozoru.
    # Koristimo donji kvartal S vrednosti (pikseli bez S) da odredimo
    # Pb Mα / Pb Lβ odnos, pa oduzimamo Pb doprinos.
    if "S" in ku and "Pb" in ku:
        s_flat  = cube[ku["S"]].flatten()
        pb_flat = cube[ku["Pb"]].flatten()

        threshold = np.percentile(s_flat, 25)
        mask      = s_flat <= threshold
        if mask.sum() > 50 and pb_flat[mask].std() > 1:
            from scipy.stats import linregress as lr
            slope_est, _, r, *_ = lr(pb_flat[mask], s_flat[mask])
            ratio_s = max(0.0, min(2.0, slope_est))
            print(f"  [korekcija] Pb Ma/Lb iz podataka: {ratio_s:.4f}  (r={r:.3f})")
        else:
            ratio_s = 0.005
            print(f"  [korekcija] Pb Ma/Lb = inicijalna vrednost: {ratio_s:.4f}")

        cube[ku["S"]] = np.maximum(0, cube[ku["S"]] - ratio_s * cube[ku["Pb"]])

    return cube


def render_grid(cube, el_keys, el_map, w, h, save_path, title=""):
    n  = len(el_keys)
    nc = 4
    nr = math.ceil(n / nc)
    is_diff = (el_map is ELEMENT_DIFF_MAP)

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
            vmin = 0
            vmax = np.percentile(img, 99)
            vmax = vmax if vmax > 0 else 1.0

        im = ax.imshow(
            img, cmap=cfg["cmap"], aspect="auto", origin="upper",
            interpolation="nearest", vmin=vmin, vmax=vmax
        )
        ax.set_title(f"{cfg['name']}  ({cfg['kev']} keV)",
                     fontsize=9, fontweight="bold", color="black", pad=4)
        ax.set_xticks([0, w // 2, w])
        ax.set_yticks([0, h // 2, h])
        ax.tick_params(labelsize=7, colors="black")

        cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label(
            "Δ counts (prova1−prova2)" if is_diff else "net counts (bg oduzeta)",
            fontsize=7, color="black"
        )
        cbar.ax.tick_params(labelsize=7, colors="black")

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", color="black")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Sacuvano: {os.path.relpath(save_path)}")


def build_display_cube(cube, el_keys, label=""):
    ku = {k: i for i, k in enumerate(el_keys)}
    pb_combined = (cube[ku["PbLl"]] + cube[ku["As"]] +
                   cube[ku["Pb"]]   + cube[ku["PbLg"]])
    as_raw = cube[ku["As"]]
    pb_lb  = cube[ku["Pb"]]
    as_f, pb_f = as_raw.flatten(), pb_lb.flatten()
    mask = as_f <= np.percentile(as_f, 25)
    if mask.sum() > 50 and pb_f[mask].std() > 1:
        from scipy.stats import linregress as lr
        s, _, r, *_ = lr(pb_f[mask], as_f[mask])
        ratio = max(0.5, min(2.5, s))
        print(f"  [{label}] As: Pb Lα/Lβ = {ratio:.3f}  (r={r:.3f})")
    else:
        ratio = PB_LA_LB_RATIO_INIT
    as_corr = np.maximum(0, as_raw - ratio * pb_lb)
    disp_keys = ["S", "K", "Ca", "Ti", "Fe", "Cu", "Zn", "Pb", "As"]
    disp = [cube[ku[k]] for k in ["S", "K", "Ca", "Ti", "Fe", "Cu", "Zn"]]
    disp += [pb_combined, as_corr]
    return np.stack(disp, axis=0), disp_keys


def render_display(cube, disp_keys, w, h, save_path, title="", is_diff=False):
    el_map = DISPLAY_DIFF_MAP if is_diff else DISPLAY_MAP
    n  = len(disp_keys)
    nc = 4
    nr = math.ceil(n / nc)
    fig, axes = plt.subplots(nr, nc, figsize=(5.5*nc, 4.2*nr+1.2), dpi=110)
    fig.patch.set_facecolor("white")
    axes = np.array(axes).flatten()
    for idx, key in enumerate(disp_keys):
        cfg = el_map[key]
        img = cube[idx]
        ax  = axes[idx]
        if is_diff:
            am = np.percentile(np.abs(img), 99) or 1.0
            vmin, vmax = -am, am
        else:
            vmin = 0
            vmax = np.percentile(img, 99) or 1.0
        im = ax.imshow(img, cmap=cfg["cmap"], aspect="auto", origin="upper",
                       interpolation="nearest", vmin=vmin, vmax=vmax)
        ax.set_title(cfg["name"], fontsize=9, fontweight="bold", color="black", pad=3)
        ax.set_xticks([0, w//2, w]);  ax.set_yticks([0, h//2, h])
        ax.tick_params(labelsize=7, colors="black")
        cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Δ counts" if is_diff else "net counts", fontsize=7, color="black")
        cbar.ax.tick_params(labelsize=7, colors="black")
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", color="black")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Sacuvano: {os.path.relpath(save_path)}")


# ══════════════════════════════════════════════════════════════
#  OBRADA
# ══════════════════════════════════════════════════════════════

cubes  = {}
dcubes = {}

for prova, cfg in DATASETS.items():
    for det in DETEKTORI:
        folder = os.path.join(cfg["dir"], det)
        label  = f"{prova}_{det}"
        print(f"\n{'='*60}")
        print(f"  Obrada: {label}  ({cfg['w']}x{cfg['h']})")
        print(f"{'='*60}")
        cube, keys = process_dataset(folder, cfg["w"], cfg["h"], label)
        print(f"  Primena korekcija...")
        cube_corr = apply_corrections(cube, keys)
        cubes[(prova, det)]  = (cube_corr, keys)
        disp_cube, dkeys = build_display_cube(cube_corr, keys, label)
        dcubes[(prova, det)] = (disp_cube, dkeys)


# ── Mape elemenata ────────────────────────────────────────────
print("\n--- Generisanje slika ---")

for prova, cfg in DATASETS.items():
    for det in DETEKTORI:
        disp_cube, dkeys = dcubes[(prova, det)]
        save = os.path.join(IZLAZ, prova, f"elementi_{det}.png")
        render_display(
            disp_cube, dkeys, cfg["w"], cfg["h"], save,
            title=f"Mape elemenata – {prova}  |  Detektor {det}\n"
                  f"bg oduzeta · Zn−CuKβ · S−PbMα · Pb kombinovano · As korigovano"
        )

# ── Razlike prova1 − prova2 ───────────────────────────────────
for det in DETEKTORI:
    d1, dkeys = dcubes[("prova1", det)]
    d2, _     = dcubes[("prova2", det)]
    diff = d1 - d2
    cfg1 = DATASETS["prova1"]
    save = os.path.join(IZLAZ, "razlike_prova1_vs_prova2", f"diff_{det}.png")
    render_display(
        diff, dkeys, cfg1["w"], cfg1["h"], save, is_diff=True,
        title=f"Razlike: prova1 − prova2  |  Detektor {det}\n"
              f"CRVENO = vise u prova1  |  PLAVO = vise u prova2"
    )

print(f"\n{'='*60}")
print(f"  Sve sacuvano u: {os.path.abspath(IZLAZ)}/")
print(f"{'='*60}")
