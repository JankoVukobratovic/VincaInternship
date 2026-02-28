"""
poredj_i_spoji.py
─────────────────────────────────────────────────────────────────────────────
Spajanje svih elementnih mapa u kompozitne slike i poređenje detektora.

Pokretanje:
  python3 poredj_i_spoji.py prova1   (podrazumevano)
  python3 poredj_i_spoji.py prova2

Izlaz  →  rezultati/{dataset}/poredj_i_spoji/
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import linregress, pearsonr

# ─── Dimenzije mreže ──────────────────────────────────────────────────────────
ROWS, COLS = 60, 120
UKUPNO     = ROWS * COLS          # 7200 tačaka

# ─── Kalibracija ──────────────────────────────────────────────────────────────
KALIB  = np.array([[219, 6.400], [278, 8.046],
                   [363, 10.551], [436, 12.614], [869, 25.271]])
_a, _b, _r, _, _ = linregress(KALIB[:, 0], KALIB[:, 1])

# ─── Elementi koji se analiziraju ─────────────────────────────────────────────
# Za kompozit koristimo 6 elemenata (Pb_Lb je duplikat informacije od Pb_La)
# boja = RGB triplet koji se koristi u aditivnom mešanju
ELEMENTI = [
    {"naziv": "Ca",    "puni": "Kalcijum",  "kanal": 127, "pola": 15,
     "boja": np.array([0.00, 0.65, 1.00])},   # nebo-plava
    {"naziv": "Ti",    "puni": "Titanijum", "kanal": 156, "pola": 10,
     "boja": np.array([0.65, 0.00, 1.00])},   # ljubičasta
    {"naziv": "Fe",    "puni": "Gvozdje",   "kanal": 219, "pola": 10,
     "boja": np.array([1.00, 0.20, 0.00])},   # crveno-narandžasta
    {"naziv": "Cu",    "puni": "Bakar",     "kanal": 278, "pola": 10,
     "boja": np.array([0.00, 0.90, 0.15])},   # zelena
    {"naziv": "Pb_La", "puni": "Olovo La",  "kanal": 363, "pola": 10,
     "boja": np.array([1.00, 0.90, 0.00])},   # žuta
    {"naziv": "Sn",    "puni": "Kalaj",     "kanal": 869, "pola": 15,
     "boja": np.array([1.00, 0.00, 0.80])},   # magenta
]

# ─── Dataset selekcija ────────────────────────────────────────────────────────
DATASET_LABEL = sys.argv[1] if len(sys.argv) > 1 else 'prova1'
_DATASET_MAP  = {'prova1': 'aurora-antico1-prova1', 'prova2': 'aurora-antico1-prova2'}
BAZA          = _DATASET_MAP.get(DATASET_LABEL, DATASET_LABEL)
NPY_DIR       = os.path.join("rezultati", "_npy_cache", DATASET_LABEL)
IZLAZ_DIR     = os.path.join("rezultati", DATASET_LABEL, "poredj_i_spoji")
print(f"Dataset: {DATASET_LABEL}  →  {BAZA}")
os.makedirs(NPY_DIR,   exist_ok=True)
os.makedirs(IZLAZ_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PARSIRANJE I KEŠ
# ══════════════════════════════════════════════════════════════════════════════

def parse_counts(filepath):
    """Čita DATA sekciju .mca fajla i vraća niz odbroja."""
    counts, in_data = [], False
    with open(filepath) as f:
        for line in f:
            s = line.strip()
            if s == "<<DATA>>": in_data = True;  continue
            if s == "<<END>>":  break
            if in_data and s:   counts.append(int(s))
    return np.array(counts, dtype=np.int32)


def ucitaj_detektor(ime):
    """
    Čita sve None_N.mca fajlove za zadati detektor.
    Rezultat se kešuje u .npy fajlove – drugi poziv je trenutan.
    """
    # Provera keša: ako svi .npy fajlovi postoje, učitavamo ih
    sve_u_kesu = all(
        os.path.exists(os.path.join(NPY_DIR, f"{ime}_{el['naziv']}.npy"))
        for el in ELEMENTI
    )
    if sve_u_kesu:
        print(f"  [{ime}] Učitavam iz keša (rezultati/_npy_cache/)...")
        return {
            el["naziv"]: np.load(os.path.join(NPY_DIR, f"{ime}_{el['naziv']}.npy"))
            for el in ELEMENTI
        }

    # Inicijalizacija praznih mapa
    mape = {el["naziv"]: np.zeros((ROWS, COLS)) for el in ELEMENTI}
    data_dir = os.path.join(BAZA, ime)

    for n in range(1, UKUPNO + 1):
        row, col = (n - 1) // COLS, (n - 1) % COLS
        try:
            c = parse_counts(os.path.join(data_dir, f"None_{n}.mca"))
            for el in ELEMENTI:
                lo = max(0, el["kanal"] - el["pola"])
                hi = min(len(c) - 1, el["kanal"] + el["pola"])
                mape[el["naziv"]][row, col] = c[lo : hi + 1].sum()
        except Exception:
            pass
        if n % 1000 == 0:
            print(f"  [{ime}] Obrađeno: {n}/{UKUPNO} ({n * 100 // UKUPNO}%)")

    # Čuvanje u keš
    for el in ELEMENTI:
        np.save(os.path.join(NPY_DIR, f"{ime}_{el['naziv']}.npy"),
                mape[el["naziv"]])
    return mape


# ══════════════════════════════════════════════════════════════════════════════
#  NORMALIZACIJA
# ══════════════════════════════════════════════════════════════════════════════

def norm_pct(mapa, lo=1, hi=99):
    """
    Percentilna normalizacija na [0, 1].
    Robust na ekstremne vrednosti – koristi 1. i 99. percentil umesto min/max.
    """
    v_lo = np.percentile(mapa, lo)
    v_hi = np.percentile(mapa, hi)
    return np.clip((mapa - v_lo) / (v_hi - v_lo + 1e-10), 0, 1)


# ══════════════════════════════════════════════════════════════════════════════
#  KOMPOZITNE SLIKE
# ══════════════════════════════════════════════════════════════════════════════

def aditivni_kompozit(mape, pojacanje=1.3):
    """
    Svaki element normalizujemo na [0,1], množimo sa bojom elementa
    i sabiramo. Rezultat se klipuje na [0,1].
    → Oblasti gde se više elemenata poklapa mešaju se aditivno.
    """
    kompozit = np.zeros((ROWS, COLS, 3))
    for el in ELEMENTI:
        norm = norm_pct(mape[el["naziv"]])                      # [0,1]
        kompozit += norm[:, :, np.newaxis] * el["boja"]        # bojeni sloj
    return np.clip(kompozit * pojacanje, 0, 1)


def rgb_triplet(mape, kljucevi=("Fe", "Cu", "Pb_La")):
    """
    Klasični 3-kanalni RGB kompozit.
    Podrazumevano: R=Fe (okre), G=Cu (azurit/malahit), B=Pb (olovna bela).
    """
    return np.stack([norm_pct(mape[k]) for k in kljucevi], axis=2)


# ══════════════════════════════════════════════════════════════════════════════
#  UČITAVANJE PODATAKA
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("  Učitavanje podataka")
print("═" * 60)
print("Detektor 10264:")
mape_A = ucitaj_detektor("10264")
print("Detektor 19511:")
mape_B = ucitaj_detektor("19511")
print("Podaci učitani.\n")


# ══════════════════════════════════════════════════════════════════════════════
#  SLIKA 1 – ADITIVNI KOMPOZIT SVIH ELEMENATA (oba detektora)
# ══════════════════════════════════════════════════════════════════════════════

print("Crtam sliku 1: Aditivni kompozit...")

komp_A = aditivni_kompozit(mape_A)
komp_B = aditivni_kompozit(mape_B)

fig, axes = plt.subplots(1, 2, figsize=(24, 8))

for ax, komp, ime in zip(axes, [komp_A, komp_B], ["10264", "19511"]):
    ax.imshow(komp, origin="upper", aspect="equal", interpolation="nearest")
    ax.set_title(f"Aditivni kompozit – Detektor {ime}\n"
                 f"(svi elementi preklopljeni prema intenzitetu)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Kolona (0–119)", fontsize=10)
    ax.set_ylabel("Red (0–59)", fontsize=10)

# Legenda boja elemenata
patches = [
    mpatches.Patch(color=el["boja"],
                   label=f"  {el['naziv']}  –  {el['puni']}")
    for el in ELEMENTI
]
fig.legend(handles=patches, loc="lower center", ncol=6,
           fontsize=11, frameon=True,
           bbox_to_anchor=(0.5, -0.02),
           title="Dodeljene boje elemenata", title_fontsize=11)

fig.suptitle("Aditivni kompozit – preklapanje svih 6 elemenata prema intenzitetu\n"
             "Mešane boje = više elemenata prisutno na istoj tački",
             fontsize=14, fontweight="bold", y=1.01)

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(os.path.join(IZLAZ_DIR, "1_aditivni_kompozit.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Sačuvano: 1_aditivni_kompozit.png")


# ══════════════════════════════════════════════════════════════════════════════
#  SLIKA 2 – KLASIČNI RGB: Fe=R, Cu=G, Pb=B
# ══════════════════════════════════════════════════════════════════════════════

print("Crtam sliku 2: RGB Fe/Cu/Pb kompozit...")

fig, axes = plt.subplots(1, 2, figsize=(24, 8))

for ax, mape, ime in zip(axes, [mape_A, mape_B], ["10264", "19511"]):
    rgb = rgb_triplet(mape)
    ax.imshow(rgb, origin="upper", aspect="equal", interpolation="nearest")
    ax.set_title(f"RGB kompozit – Detektor {ime}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Kolona", fontsize=10)
    ax.set_ylabel("Red", fontsize=10)

patches_rgb = [
    mpatches.Patch(color="red",   label="R = Fe (Gvožđe)  –  okre, hematit, umbra"),
    mpatches.Patch(color="lime",  label="G = Cu (Bakar)    –  azurit, malahit, verdigris"),
    mpatches.Patch(color="blue",  label="B = Pb (Olovo)    –  olovna bela, minijum"),
]
fig.legend(handles=patches_rgb, loc="lower center", ncol=3,
           fontsize=12, frameon=True, bbox_to_anchor=(0.5, -0.02))

fig.suptitle("Klasični 3-kanalni RGB kompozit: Fe→Crvena | Cu→Zelena | Pb→Plava\n"
             "Žuta = Fe+Cu zajedno | Cijan = Cu+Pb | Bela = svi prisutni",
             fontsize=14, fontweight="bold", y=1.01)

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(os.path.join(IZLAZ_DIR, "2_rgb_Fe_Cu_Pb.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Sačuvano: 2_rgb_Fe_Cu_Pb.png")


# ══════════════════════════════════════════════════════════════════════════════
#  SLIKA 3 – POREĐENJE ELEMENT PO ELEMENT: mape + razlika
# ══════════════════════════════════════════════════════════════════════════════

print("Crtam sliku 3: Poređenje po elementu...")

fig, axes = plt.subplots(len(ELEMENTI), 3,
                         figsize=(20, len(ELEMENTI) * 3.8))

for i, el in enumerate(ELEMENTI):
    nA = norm_pct(mape_A[el["naziv"]])   # normalizovana mapa detektora A
    nB = norm_pct(mape_B[el["naziv"]])   # normalizovana mapa detektora B
    diff = nA - nB                        # razlika: + = A jači, − = B jači

    # Pearsonov koeficijent korelacije između A i B
    r_val, _ = pearsonr(nA.ravel(), nB.ravel())

    opts = dict(origin="upper", aspect="auto",
                interpolation="nearest", vmin=0, vmax=1)

    # Kolona 0 – detektor A (10264)
    im0 = axes[i, 0].imshow(nA, cmap="viridis", **opts)
    axes[i, 0].set_title(f"{el['puni']}  ({el['naziv']}) – 10264",
                         fontsize=9, fontweight="bold")
    plt.colorbar(im0, ax=axes[i, 0], shrink=0.88, fraction=0.04,
                 label="Norm. intenzitet")

    # Kolona 1 – detektor B (19511)
    im1 = axes[i, 1].imshow(nB, cmap="viridis", **opts)
    axes[i, 1].set_title(f"{el['puni']}  ({el['naziv']}) – 19511\n"
                         f"r = {r_val:.4f}", fontsize=9, fontweight="bold")
    plt.colorbar(im1, ax=axes[i, 1], shrink=0.88, fraction=0.04,
                 label="Norm. intenzitet")

    # Kolona 2 – mapa razlike (divergentna paleta crvena–bela–plava)
    lim = max(abs(diff.min()), abs(diff.max())) + 1e-6
    im2 = axes[i, 2].imshow(diff, cmap="RdBu_r", origin="upper",
                             aspect="auto", interpolation="nearest",
                             vmin=-lim, vmax=lim)
    axes[i, 2].set_title("Razlika  (10264 − 19511)\n"
                         "Crvena = 10264 jači  |  Plava = 19511 jači",
                         fontsize=9)
    plt.colorbar(im2, ax=axes[i, 2], shrink=0.88, fraction=0.04,
                 label="Δ norm.")

    for ax in axes[i]:
        ax.set_xlabel("Kolona", fontsize=7)
        ax.set_ylabel("Red", fontsize=7)
        ax.tick_params(labelsize=6)

fig.suptitle("Poređenje detektora 10264 vs 19511 – normalizovane mape i razlike\n"
             "(percentilna normalizacija p1–p99 po elementu i po detektoru)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(IZLAZ_DIR, "3_poredj_svi_elementi.png"),
            dpi=120, bbox_inches="tight")
plt.close(fig)
print("  Sačuvano: 3_poredj_svi_elementi.png")


# ══════════════════════════════════════════════════════════════════════════════
#  SLIKA 4 – PREKLOPLJENI PRIKAZ: 10264=Crvena, 19511=Zelena
# ══════════════════════════════════════════════════════════════════════════════

print("Crtam sliku 4: Preklopljeni R/G prikaz...")

n_el  = len(ELEMENTI)
n_col = 3
n_row = (n_el + n_col - 1) // n_col

fig, axes = plt.subplots(n_row, n_col, figsize=(20, n_row * 5.5))
axes_flat = axes.flatten()

for i, el in enumerate(ELEMENTI):
    nA = norm_pct(mape_A[el["naziv"]])
    nB = norm_pct(mape_B[el["naziv"]])

    # R=detektor A (10264), G=detektor B (19511), B=0
    # Žuta tamo gde oba detektora registruju visok intenzitet (saglasnost)
    # Crvena tamo gde samo 10264 "vidi" više
    # Zelena tamo gde samo 19511 "vidi" više
    overlay = np.stack([nA, nB, np.zeros_like(nA)], axis=2)

    r_val, _ = pearsonr(nA.ravel(), nB.ravel())
    axes_flat[i].imshow(overlay, origin="upper", aspect="equal",
                        interpolation="nearest")
    axes_flat[i].set_title(
        f"{el['puni']}  ({el['naziv']})\nr = {r_val:.4f}",
        fontsize=10, fontweight="bold"
    )
    axes_flat[i].set_xlabel("Kolona", fontsize=8)
    axes_flat[i].set_ylabel("Red", fontsize=8)

# Sakrivanje praznih panela
for j in range(n_el, len(axes_flat)):
    axes_flat[j].set_visible(False)

# Legenda
patches_rg = [
    mpatches.Patch(color="red",    label="Samo detektor 10264"),
    mpatches.Patch(color="yellow", label="Oba detektora se slažu"),
    mpatches.Patch(color="lime",   label="Samo detektor 19511"),
    mpatches.Patch(color="black",  label="Oba detektora – nizak signal"),
]
fig.legend(handles=patches_rg, loc="lower center", ncol=4,
           fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.01),
           title="Interpretacija boja", title_fontsize=11)

fig.suptitle("Preklopljeni prikaz: Crvena = 10264  |  Zelena = 19511  |  Žuta = saglasnost\n"
             "(normalizovane mape, percentilna skala p1–p99)",
             fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(os.path.join(IZLAZ_DIR, "4_preklopljeni_RG.png"),
            dpi=130, bbox_inches="tight")
plt.close(fig)
print("  Sačuvano: 4_preklopljeni_RG.png")


# ══════════════════════════════════════════════════════════════════════════════
#  SLIKA 5 – SCATTER KORELACIJA A vs B (7200 tačaka po elementu)
# ══════════════════════════════════════════════════════════════════════════════

print("Crtam sliku 5: Korelacioni scatter plotovi...")

fig, axes = plt.subplots(n_row, n_col, figsize=(18, n_row * 5.5))
axes_flat = axes.flatten()

for i, el in enumerate(ELEMENTI):
    A_flat = mape_A[el["naziv"]].ravel()
    B_flat = mape_B[el["naziv"]].ravel()

    r_val, _ = pearsonr(A_flat, B_flat)
    s_slope, s_int, *_ = linregress(A_flat, B_flat)

    # Scatter svake od 7200 tačaka – alpha nizak jer je tačaka mnogo
    axes_flat[i].scatter(A_flat, B_flat,
                         alpha=0.08, s=2.5, c="steelblue", rasterized=True)

    # Linija savršene korelacije (y = x)
    mn = min(A_flat.min(), B_flat.min())
    mx = max(A_flat.max(), B_flat.max())
    axes_flat[i].plot([mn, mx], [mn, mx], "r--", lw=1.2,
                      label="Idealna korelacija (y = x)")

    # Stvarna linearna regresija
    x_fit = np.linspace(A_flat.min(), A_flat.max(), 200)
    axes_flat[i].plot(x_fit, s_slope * x_fit + s_int,
                      color="orange", lw=2,
                      label=f"Regresija: y = {s_slope:.3f}x + {s_int:.0f}")

    axes_flat[i].set_title(
        f"{el['puni']}  ({el['naziv']})\nr = {r_val:.4f}",
        fontsize=10, fontweight="bold"
    )
    axes_flat[i].set_xlabel("Detektor 10264 (odbroji)", fontsize=8)
    axes_flat[i].set_ylabel("Detektor 19511 (odbroji)", fontsize=8)
    axes_flat[i].legend(fontsize=7)
    axes_flat[i].grid(True, alpha=0.3)
    axes_flat[i].tick_params(labelsize=7)

for j in range(n_el, len(axes_flat)):
    axes_flat[j].set_visible(False)

fig.suptitle("Korelacija detektora 10264 vs 19511 – svaka tačka = jedno od 7200 skeniranih polja\n"
             "r = Pearsonov koeficijent korelacije  |  Narandžasta = stvarna regresija",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(IZLAZ_DIR, "5_korelacija_scatter.png"),
            dpi=130, bbox_inches="tight")
plt.close(fig)
print("  Sačuvano: 5_korelacija_scatter.png")


# ══════════════════════════════════════════════════════════════════════════════
#  README.md – Zaključci
# ══════════════════════════════════════════════════════════════════════════════

print("Generišem README.md...")

redovi = []
redovi.append("# Poređenje i kompozit – Detektori 10264 vs 19511\n\n")
redovi.append("## Sadržaj foldera\n\n")
redovi.append("| Fajl | Opis |\n")
redovi.append("|------|------|\n")
redovi.append("| `1_aditivni_kompozit.png` | Svih 6 elemenata preklopljeno bojama prema intenzitetu |\n")
redovi.append("| `2_rgb_Fe_Cu_Pb.png` | Klasični RGB: Fe→Crvena, Cu→Zelena, Pb→Plava |\n")
redovi.append("| `3_poredj_svi_elementi.png` | Normalizovane mape A i B + mapa razlike |\n")
redovi.append("| `4_preklopljeni_RG.png` | 10264=Crvena, 19511=Zelena, saglasnost=Žuta |\n")
redovi.append("| `5_korelacija_scatter.png` | Scatter korelacija 7200 tačaka: A vs B |\n\n")

redovi.append("## Statistike korelacije između detektora\n\n")
redovi.append("| Element | Korelacija r | Nagib regresije | Zaključak |\n")
redovi.append("|---------|-------------|-----------------|----------|\n")

for el in ELEMENTI:
    A_flat = mape_A[el["naziv"]].ravel()
    B_flat = mape_B[el["naziv"]].ravel()
    r_val, _ = pearsonr(A_flat, B_flat)
    slope_s, *_ = linregress(A_flat, B_flat)

    if r_val > 0.95:
        zakljucak = "Odlična saglasnost – oba detektora vide istu strukturu"
    elif r_val > 0.85:
        zakljucak = "Dobra saglasnost – manje razlike u efikasnosti detektora"
    elif r_val > 0.70:
        zakljucak = "Umerena saglasnost – strukturni obrasci slični"
    else:
        zakljucak = "Slaba korelacija – moguće razlike u uglu/poziciji detektora"

    redovi.append(f"| **{el['puni']} ({el['naziv']})** | r = {r_val:.4f} | "
                  f"y = {slope_s:.3f}×x | {zakljucak} |\n")

redovi.append("\n## Opis slika\n\n")
redovi.append("### 1. Aditivni kompozit\n")
redovi.append("Svaki element dobija svoju boju i normalizuje se na [0,1].\n"
              "Boje se aditivno mešaju – onde gde ima više elemenata boje se kombinuju.\n"
              "Osnova interpretacije: oblasti sličnih boja = slični hemijski sastav.\n\n")
redovi.append("**Dodeljene boje:**\n")
for el in ELEMENTI:
    rgb_hex = "#{:02X}{:02X}{:02X}".format(
        int(el["boja"][0]*255), int(el["boja"][1]*255), int(el["boja"][2]*255))
    redovi.append(f"- `{rgb_hex}` → **{el['puni']} ({el['naziv']})**\n")

redovi.append("\n### 2. RGB kompozit (Fe / Cu / Pb)\n")
redovi.append("Klasični XRF false-color prikaz sa 3 kanala:\n"
              "- **Crvene** oblasti = gvožđe (okre) dominira\n"
              "- **Zelene** oblasti = bakar (azurit, malahit) dominira\n"
              "- **Plave** oblasti = olovo (olovna bela, minijum) dominira\n"
              "- **Žuta** = gvožđe + bakar zajedno\n"
              "- **Bela** = svi elementi prisutni u sličnoj meri\n\n")
redovi.append("### 3. Mape razlike\n")
redovi.append("Normalizovane mape (p1–p99) oduzimaju se: **10264 − 19511**.\n"
              "- **Crvene** zone → detektor 10264 meri više signala\n"
              "- **Plave** zone → detektor 19511 meri više signala\n"
              "- **Bele** zone → detektori se savršeno slažu\n\n"
              "Razlike mogu nastati zbog: ugla detektora, senčenja uzorka, "
              "različite energetske efikasnosti.\n\n")
redovi.append("### 4. Preklopljeni R/G prikaz\n")
redovi.append("Najintuitivniji prikaz saglasnosti:\n"
              "- **Žuta** → oba detektora registruju visok signal (saglasnost)\n"
              "- **Crvena** → samo 10264 vidi signal (lokalna razlika)\n"
              "- **Zelena** → samo 19511 vidi signal\n"
              "- **Crna** → oba detektora imaju nizak signal\n\n")
redovi.append("### 5. Scatter korelacija\n")
redovi.append("Svaka od 7200 skeniranih tačaka prikazana kao tačka u dijagramu.\n"
              "Nagib regresione prave < 1 → detektor 19511 generalno meri više.\n"
              "Nagib > 1 → detektor 10264 meri više.\n\n")
redovi.append("---\n")
redovi.append("*Generisano skriptom `poredj_i_spoji.py`*\n")

with open(os.path.join(IZLAZ_DIR, "README.md"), "w", encoding="utf-8") as f:
    f.writelines(redovi)
print("  Sačuvano: README.md")

print("\n" + "═" * 60)
print("  Sve slike generisane!")
print(f"  Rezultati: {os.path.abspath(IZLAZ_DIR)}/")
print("═" * 60)
