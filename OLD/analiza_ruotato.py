"""
analiza_ruotato.py
──────────────────────────────────────────────────────────────────────
XRF analiza ruotato skeniranja – detektori 10264 i 19511.
Grid: 80 kolona × 45 redova = 3600 piksela, dwell=3.0s

Elementi mapirani (statisticki validni):
  S (Kα 2.31), Ca (Kα 3.69), Ti (Kα 4.51),
  Fe (Kα 6.40), Cu (Kα 8.05), Zn (Kα 8.64),
  As+PbLα (10.54), Pb (Lβ 12.61), PbLl (9.185), PbLγ (14.77)
  K (Kα 3.31) – marginalan signal

Napomena: Cr (5.42 keV) iskljucen – korelacija izmedju detektora r=0.21 (suma je sum)
Napomena: Mn, Co, Ni – iskljuceni, nema neto signala posle bg oduzimanja

Sve korekcije iz analiza_korigovana.py:
  - Linearna bg subtrakcija
  - K: tesni sideband
  - Zn: oduzima Cu Kβ (×0.17)
  - As: oduzima Pb Lα procenjeno iz Pb Lβ
  - S:  oduzima Pb Mα procenjeno iz Pb Lβ

Generise:
  1. Mape elemenata za svaki detektor
  2. Suma oba detektora (bolji SNR)
  3. Razlika det10264 − det19511
  4. Sumovani spektar sa anotiranim pikovima
"""

import os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import linregress

# ── Colormap helper ────────────────────────────────────────────
def _mk(name, color):
    return LinearSegmentedColormap.from_list(name, ["#000000", color])

# ── Elementi sa pikovima u ruotato spektru ─────────────────────
ELEMENT_MAP = {
    "S":    {"name": "S Kα",    "kev": 2.31,  "hw": 0.20, "cmap": _mk("S",   "#FFFF66")},
    "K":    {"name": "K Kα",    "kev": 3.31,  "hw": 0.15, "cmap": _mk("K",   "#FFD700")},
    "Ca":   {"name": "Ca Kα",   "kev": 3.69,  "hw": 0.30, "cmap": _mk("Ca",  "#FFFFFF")},
    "Ti":   {"name": "Ti Kα",   "kev": 4.51,  "hw": 0.30, "cmap": _mk("Ti",  "#FF9966")},
    "Fe":   {"name": "Fe Kα",   "kev": 6.40,  "hw": 0.30, "cmap": _mk("Fe",  "#FF2200")},
    "Cu":   {"name": "Cu Kα",   "kev": 8.05,  "hw": 0.30, "cmap": _mk("Cu",  "#00FFAA")},
    "Zn":   {"name": "Zn Kα",   "kev": 8.64,  "hw": 0.20, "cmap": _mk("Zn",  "#66CCFF")},  # hw 0.25→0.20: hi+1 bi pao u Cu Kβ (8.903 keV)
    "PbLl": {"name": "Pb Ll",   "kev": 9.185, "hw": 0.28, "cmap": _mk("PbLl","#BB88FF")},
    "As":   {"name": "Pb Lα",   "kev": 10.54, "hw": 0.30, "cmap": _mk("As",  "#FF8800")},
    "Pb":   {"name": "Pb Lβ",   "kev": 12.61, "hw": 0.30, "cmap": _mk("Pb",  "#CC66FF")},
    "PbLg": {"name": "Pb Lγ",   "kev": 14.77, "hw": 0.35, "cmap": _mk("PbLg","#9933AA")},
}
# Napomena: Ar Kα (2.957 keV) vidljiv samo na det 10264 (tanji Be prozor) – Ar iz vazduha u putu zraka, nije element uzorka
# Napomena: Cr Kα (5.415 keV) – iskljucen; korelacija izmedju detektora r=0.21, reproducibilnost r=0.37 → statisticki sum

ELEMENT_DIFF_MAP = {
    k: {"name": f"Δ {v['name']}", "kev": v["kev"], "hw": v["hw"], "cmap": "RdBu_r"}
    for k, v in ELEMENT_MAP.items()
}

# ── Prikaz mapa: kombinovani Pb + odvojeni As ──────────────────
DISPLAY_MAP = {
    "S":  {"name": "S Kα",              "cmap": _mk("S",  "#FFFF66")},
    "K":  {"name": "K Kα",              "cmap": _mk("K",  "#FFD700")},
    "Ca": {"name": "Ca Kα",             "cmap": _mk("Ca", "#FFFFFF")},
    "Ti": {"name": "Ti Kα",             "cmap": _mk("Ti", "#FF9966")},
    "Fe": {"name": "Fe Kα",             "cmap": _mk("Fe", "#FF2200")},
    "Cu": {"name": "Cu Kα",             "cmap": _mk("Cu", "#00FFAA")},
    "Zn": {"name": "Zn Kα",             "cmap": _mk("Zn", "#66CCFF")},
    "Pb": {"name": "Pb (Lα+Lβ+Ll+Lγ)", "cmap": _mk("Pb", "#CC66FF")},
    "As": {"name": "As Kα (kor.)",      "cmap": _mk("As", "#FF8800")},
}
DISPLAY_DIFF_MAP = {
    k: {"name": f"Δ {v['name']}", "cmap": "RdBu_r"}
    for k, v in DISPLAY_MAP.items()
}

# ── Kalibracija ────────────────────────────────────────────────
_CAL   = np.array([[219,6.4],[278,8.0],[363,10.5],[436,12.6],[869,25.3]])
_SLOPE, _INTERCEPT, *_ = linregress(_CAL[:,0], _CAL[:,1])

CU_KB_KA_RATIO = 0.17
PB_LA_LB_RATIO_INIT = 1.40

DETEKTORI = ["10264", "19511"]
RUOTATO   = "aurora-antico1-ruotato"
W, H      = 80, 45
IZLAZ     = "rezultati_ruotato"
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
    return {"counts": np.array(counts, dtype=np.float64),
            "time":   float(meta.get("REAL_TIME", 1.0))}


def bg_subtracted_integral(counts, energy, target_kev, hw=0.30, bg_hw=0.25):
    idx     = int(np.argmin(np.abs(energy - target_kev)))
    half_ch = max(1, int(round(hw   / _SLOPE)))
    bg_ch   = max(1, int(round(bg_hw / _SLOPE)))
    lo = max(0,               idx - half_ch)
    hi = min(len(counts) - 1, idx + half_ch)
    bg_lo_l = max(0,               lo - bg_ch)
    bg_lo_r = lo
    bg_hi_l = hi + 1
    bg_hi_r = min(len(counts) - 1, hi + 1 + bg_ch)
    bg_left  = counts[bg_lo_l:bg_lo_r].mean() if bg_lo_r > bg_lo_l else counts[lo]
    bg_right = counts[bg_hi_l:bg_hi_r].mean() if bg_hi_r > bg_hi_l else counts[hi]
    n_peak   = hi - lo + 1
    bg_line  = np.linspace(bg_left, bg_right, n_peak)
    return max(0.0, float((counts[lo:hi+1] - bg_line).sum()))


def k_signal(counts, energy):
    """K Kα sa tesnim sidebandzima (izbegava Ar i Ca)."""
    k_ch  = int(np.argmin(np.abs(energy - 3.3138)))
    pk_lo = k_ch - 2
    pk_hi = k_ch + 2
    ch_ll = int(np.argmin(np.abs(energy - 3.23)))
    ch_lr = int(np.argmin(np.abs(energy - 3.28)))
    ch_rl = int(np.argmin(np.abs(energy - 3.35)))
    ch_rr = int(np.argmin(np.abs(energy - 3.42)))
    bg_left  = counts[ch_ll:ch_lr+1].mean() if ch_lr >= ch_ll else counts[pk_lo]
    bg_right = counts[ch_rl:ch_rr+1].mean() if ch_rr >= ch_rl else counts[pk_hi]
    bg_per_ch = (bg_left + bg_right) / 2.0
    return max(0.0, float(counts[pk_lo:pk_hi+1].sum()) - bg_per_ch * (pk_hi - pk_lo + 1))


def integral_for(counts, energy, key):
    if key == "K":
        return k_signal(counts, energy)
    el = ELEMENT_MAP[key]
    return bg_subtracted_integral(counts, energy, el["kev"], el["hw"])


def process_dataset(folder, label):
    el_keys = list(ELEMENT_MAP.keys())
    total   = W * H

    cache_paths = {k: os.path.join(NPY_CACHE, f"{label}_{k}.npy") for k in el_keys}
    if all(os.path.exists(p) for p in cache_paths.values()):
        print(f"  [{label}] Ucitavam iz cache-a...")
        return np.stack([np.load(cache_paths[k]) for k in el_keys], axis=0), el_keys

    cube = np.zeros((len(el_keys), H, W), dtype=np.float64)
    _ec  = {}

    for i in range(1, total + 1):
        path = os.path.join(folder, f"None_{i}.mca")
        if not os.path.exists(path):
            continue
        data   = parse_mca_file(path)
        counts = data["counts"]
        n_ch   = len(counts)
        if n_ch not in _ec:
            _ec[n_ch] = np.arange(n_ch) * _SLOPE + _INTERCEPT
        energy = _ec[n_ch]
        row, col = (i-1) // W, (i-1) % W
        for ei, key in enumerate(el_keys):
            cube[ei, row, col] = integral_for(counts, energy, key)
        if i % 500 == 0 or i == total:
            print(f"  [{label}] {i}/{total} ({100*i//total}%)")

    for ei, k in enumerate(el_keys):
        np.save(cache_paths[k], cube[ei])
    print(f"  [{label}] Cache sacuvan.")
    return cube, el_keys


def apply_corrections(cube, el_keys, label=""):
    cube = cube.copy()
    ku   = {k: i for i, k in enumerate(el_keys)}

    # Zn − Cu Kβ
    if "Cu" in ku and "Zn" in ku:
        cube[ku["Zn"]] = np.maximum(0, cube[ku["Zn"]] - CU_KB_KA_RATIO * cube[ku["Cu"]])

    # Pb Lα (prozor 10.54 keV): As nije prisutan u uzorku,
    # sirovi integral direktno prikazuje Pb Lα distribuciju — bez korekcije.

    # S − Pb Mα
    if "S" in ku and "Pb" in ku:
        s_f, pb_f = cube[ku["S"]].flatten(), cube[ku["Pb"]].flatten()
        mask = s_f <= np.percentile(s_f, 25)
        if mask.sum() > 30 and pb_f[mask].std() > 1:
            s2, _, r, *_ = linregress(pb_f[mask], s_f[mask])
            r_s = max(0.0, min(2.0, s2))
            print(f"  [{label}] Pb Ma/Lb = {r_s:.4f}  (r={r:.3f})")
        else:
            r_s = 0.005
        cube[ku["S"]] = np.maximum(0, cube[ku["S"]] - r_s * cube[ku["Pb"]])

    return cube


def build_display_cube(cube, el_keys, label=""):
    """
    Gradi display cube (8 kanala) od raw cube (10 kanala):
      - S, Ca, Ti, Fe, Cu, Zn  ->  direktno (vec korigovani)
      - Pb  ->  suma svih Pb linija (PbLl + PbLα + PbLβ + PbLγ)
      - As  ->  korigovani As signal (PbLα prozor − ratio × PbLβ)
    """
    ku = {k: i for i, k in enumerate(el_keys)}

    # Kombinovani Pb: suma svih cetiri Pb linija
    pb_combined = (cube[ku["PbLl"]] + cube[ku["As"]] +
                   cube[ku["Pb"]]   + cube[ku["PbLg"]])

    # Odvojeni As: oduzmi Pb Lα doprinos iz As/PbLα prozora
    as_raw = cube[ku["As"]]
    pb_lb  = cube[ku["Pb"]]
    as_f, pb_f = as_raw.flatten(), pb_lb.flatten()
    mask = as_f <= np.percentile(as_f, 25)
    if mask.sum() > 30 and pb_f[mask].std() > 1:
        s, _, r, *_ = linregress(pb_f[mask], as_f[mask])
        ratio = max(0.5, min(2.5, s))
        print(f"  [{label}] As: Pb Lα/Lβ ratio = {ratio:.3f}  (r={r:.3f})")
    else:
        ratio = PB_LA_LB_RATIO_INIT
    as_corr = np.maximum(0, as_raw - ratio * pb_lb)

    disp_keys = ["S", "K", "Ca", "Ti", "Fe", "Cu", "Zn", "Pb", "As"]
    disp = [cube[ku[k]] for k in ["S", "K", "Ca", "Ti", "Fe", "Cu", "Zn"]]
    disp += [pb_combined, as_corr]
    return np.stack(disp, axis=0), disp_keys


def render_display(cube, disp_keys, save_path, title="", is_diff=False):
    """render_grid za DISPLAY_MAP (nema 'hw' — koristi 'cmap' direktno)."""
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
        ax.set_xticks([0, W//2, W]);  ax.set_yticks([0, H//2, H])
        ax.tick_params(labelsize=6, colors="black")
        cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("Δ counts" if is_diff else "net counts", fontsize=6, color="black")
        cbar.ax.tick_params(labelsize=6, colors="black")

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", color="black")
    plt.tight_layout(rect=[0,0,1,0.95])
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Sacuvano: {os.path.relpath(save_path)}")


def render_individual(cube, disp_keys, out_dir, prefix="", is_diff=False):
    """Svaki element kao posebna PNG slika."""
    el_map = DISPLAY_DIFF_MAP if is_diff else DISPLAY_MAP
    os.makedirs(out_dir, exist_ok=True)
    for idx, key in enumerate(disp_keys):
        cfg = el_map[key]
        img = cube[idx]

        if is_diff:
            am = np.percentile(np.abs(img), 99) or 1.0
            vmin, vmax = -am, am
        else:
            vmin = 0
            vmax = np.percentile(img, 99) or 1.0

        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=130)
        fig.patch.set_facecolor("white")
        im = ax.imshow(img, cmap=cfg["cmap"], aspect="auto", origin="upper",
                       interpolation="nearest", vmin=vmin, vmax=vmax)
        ax.set_title(cfg["name"], fontsize=13, fontweight="bold", color="black", pad=6)
        ax.set_xticks([0, W//2, W]);  ax.set_yticks([0, H//2, H])
        ax.tick_params(labelsize=8, colors="black")
        cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("Δ counts" if is_diff else "net counts", fontsize=8, color="black")
        cbar.ax.tick_params(labelsize=7, colors="black")
        plt.tight_layout()
        fname = f"{prefix}{key}.png" if prefix else f"{key}.png"
        save_path = os.path.join(out_dir, fname)
        plt.savefig(save_path, dpi=140, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Sacuvano: {os.path.relpath(save_path)}")


def render_grid(cube, el_keys, el_map, save_path, title="", is_diff=False):
    n  = len(el_keys)
    nc = 4 if n > 6 else 3
    nr = math.ceil(n / nc)

    fig, axes = plt.subplots(nr, nc, figsize=(5.5*nc, 4.2*nr+1.2), dpi=110)
    fig.patch.set_facecolor("white")
    axes = np.array(axes).flatten()

    for idx, key in enumerate(el_keys):
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
        ax.set_title(f"{cfg['name']}  ({cfg['kev']} keV)",
                     fontsize=8, fontweight="bold", color="black", pad=3)
        ax.set_xticks([0, W//2, W]);  ax.set_yticks([0, H//2, H])
        ax.tick_params(labelsize=6, colors="black")
        cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        label_cb = "Δ counts" if is_diff else "net counts"
        cbar.set_label(label_cb, fontsize=6, color="black")
        cbar.ax.tick_params(labelsize=6, colors="black")

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", color="black")
    plt.tight_layout(rect=[0,0,1,0.95])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Sacuvano: {os.path.relpath(save_path)}")


def plot_stacked_spectrum(det_folder, det_label, save_path):
    """Sumirani spektar sa anotiranim pikovima."""
    stacked = np.zeros(1024)
    n = 0
    for i in range(1, W*H+1):
        path = os.path.join(det_folder, f"None_{i}.mca")
        if not os.path.exists(path): continue
        counts = []
        in_data = False
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line == "<<DATA>>": in_data = True; continue
                if line == "<<END>>": break
                if in_data:
                    try: counts.append(int(line))
                    except: pass
        if len(counts) >= 1024:
            stacked[:1024] += counts[:1024]
            n += 1

    energy = np.arange(1024) * _SLOPE + _INTERCEPT

    fig, ax = plt.subplots(figsize=(14, 5), dpi=110)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F8F8F8")

    ax.semilogy(energy, np.maximum(stacked, 1), color="#1155AA",
                linewidth=0.7, alpha=0.85)

    # Annotacije
    annotations = [
        ("S/Pb Mα", 2.31,  "#CCCC00"),
        ("Cl",      2.62,  "#44BB44"),
        ("Ar",      2.96,  "#888888"),
        ("Ca",      3.69,  "#BBBBBB"),
        ("Ti",      4.51,  "#FF9966"),
        ("Mn?",     5.90,  "#884400"),
        ("Fe",      6.40,  "#FF2200"),
        ("Co",      6.93,  "#0099FF"),
        ("Ni",      7.47,  "#FF44FF"),
        ("Cu",      8.05,  "#00FFAA"),
        ("Zn",      8.64,  "#66CCFF"),
        ("As/Pb Lα",10.54, "#FF8800"),
        ("Pb Lβ",  12.61,  "#CC66FF"),
        ("Pb Lγ",  14.77,  "#9944AA"),
    ]
    ymax = stacked.max() * 2
    for label, kev, color in annotations:
        if kev > energy[-1]: continue
        ax.axvline(kev, color=color, linewidth=1.0, alpha=0.7, linestyle="--")
        idx = int(np.argmin(np.abs(energy - kev)))
        y_pos = max(stacked[max(0,idx-3):idx+4].max() * 1.8, ymax * 0.02)
        ax.text(kev, y_pos, label, color=color, fontsize=7, ha="center",
                va="bottom", rotation=90, clip_on=True)

    ax.set_xlim(1, 16)
    ax.set_xlabel("Energija (keV)", fontsize=9, color="black")
    ax.set_ylabel("Counts (log skala)", fontsize=9, color="black")
    ax.tick_params(colors="black", labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#AAAAAA")
    ax.set_title(f"Sumovani spektar – Ruotato  |  Detektor {det_label}  ({n} piksela)",
                 fontsize=11, fontweight="bold", color="black")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Sacuvano: {os.path.relpath(save_path)}")


# ══════════════════════════════════════════════════════════════
#  OBRADA
# ══════════════════════════════════════════════════════════════

cubes  = {}   # (cube_corr, keys)   — 10 kanala, za internu upotrebu
dcubes = {}   # (disp_cube, dkeys)  — 8 kanala, za prikaz

for det in DETEKTORI:
    folder = os.path.join(RUOTATO, det)
    label  = f"ruotato_{det}"
    print(f"\n{'='*60}")
    print(f"  Obrada: {label}  ({W}×{H} = {W*H} tacaka)")
    print(f"{'='*60}")
    cube, keys = process_dataset(folder, label)
    print(f"  Korekcije...")
    cube_corr = apply_corrections(cube, keys, label)
    cubes[det] = (cube_corr, keys)
    disp_cube, dkeys = build_display_cube(cube_corr, keys, det)
    dcubes[det] = (disp_cube, dkeys)


# ── Sumirani spektri ──────────────────────────────────────────
print("\n--- Sumirani spektri ---")
for det in DETEKTORI:
    plot_stacked_spectrum(os.path.join(RUOTATO, det), det,
        os.path.join(IZLAZ, "spektri", f"stacked_{det}.png"))


# ── Mape po detektoru ─────────────────────────────────────────
print("\n--- Mape elemenata po detektoru ---")
for det in DETEKTORI:
    disp_cube, dkeys = dcubes[det]
    render_display(disp_cube, dkeys,
        os.path.join(IZLAZ, f"elementi_{det}.png"),
        title=f"Mape elemenata – Ruotato  |  Detektor {det}\n"
              f"bg oduzeta  ·  Zn−CuKβ  ·  S−PbMα  ·  Pb kombinovano  ·  As korigovano  ({W}×{H})")


# ── Suma oba detektora (bolji SNR) ────────────────────────────
print("\n--- Suma oba detektora ---")
disp_sum = dcubes["10264"][0] + dcubes["19511"][0]
render_display(disp_sum, dkeys,
    os.path.join(IZLAZ, "elementi_suma_det.png"),
    title="Mape elemenata – Ruotato  |  DET 10264 + DET 19511  (bolji SNR)")


# ── Razlika detektora ─────────────────────────────────────────
print("\n--- Razlika det10264 − det19511 ---")
disp_diff = dcubes["10264"][0] - dcubes["19511"][0]
render_display(disp_diff, dkeys,
    os.path.join(IZLAZ, "diff_detektora.png"),
    title="Razlika detektora: 10264 − 19511  |  Ruotato\n"
          "CRVENO = vise u 10264  |  PLAVO = vise u 19511",
    is_diff=True)


# ── Individualne mape po elementu ────────────────────────────
print("\n--- Individualne mape elemenata ---")
for det in DETEKTORI:
    disp_cube, dkeys = dcubes[det]
    render_individual(disp_cube, dkeys,
        out_dir=os.path.join(IZLAZ, "individualne", det),
        prefix=f"det{det}_")

render_individual(disp_sum, dkeys,
    out_dir=os.path.join(IZLAZ, "individualne", "suma"),
    prefix="suma_")

render_individual(disp_diff, dkeys,
    out_dir=os.path.join(IZLAZ, "individualne", "diff"),
    prefix="diff_",
    is_diff=True)


print(f"\n{'='*60}")
print(f"  Sve sacuvano u: {os.path.abspath(IZLAZ)}/")
print(f"{'='*60}")
