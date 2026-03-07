"""
SAM/sam_pipeline.py
===============================================================================
Segmentacija freske pomocu Meta SAM (Segment Anything Model) +
agregirana analiza hemijskog rizika po regionima.

PIPELINE:
  1. Ucitavanje XRF element mapa (Ca, Ti, Fe, Cu, Pb_La) — BEZ Sn
  2. Priprema ulazne slike za SAM (multi-element kompozit)
  3. SAM automatska segmentacija (Automatic Mask Generator)
  4. Filtriranje i spajanje segmenata
  5. Per-region CVI analiza (Chemical Vulnerability Index)
  6. Generisanje izvestaja za restauratore
  7. Vizualizacija

NAPOMENA O Sn (KALAJU):
  Kalaj je iskljucen iz analize jer je identifikovan kao artefakt
  u podacima — ne predstavlja stvarni pigment na fresci.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter, label as ndlabel
from scipy.stats import linregress
import torch

# Dodaj parent dir za pristup podacima
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ═══════════════════════════════════════════════════════════════════════════════
#  KONFIGURACIJA
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAM_DIR  = os.path.dirname(os.path.abspath(__file__))
IZLAZ    = os.path.join(SAM_DIR, 'rezultati')
os.makedirs(IZLAZ, exist_ok=True)

ROWS, COLS = 60, 120
DATASET_LABEL = sys.argv[1] if len(sys.argv) > 1 else 'prova1'

# Elementi — BEZ Sn (artefakt u podacima)
ELEMENTI = ['Ca', 'Ti', 'Fe', 'Cu', 'Pb_La']

# SAM checkpoint
SAM_CHECKPOINT = os.path.join(BASE_DIR, 'modeli', 'sam_vit_b_01ec64.pth')
SAM_MODEL_TYPE = 'vit_b'

# Rizik colormap
RISK_CMAP = LinearSegmentedColormap.from_list('risk', [
    (0.0, '#1a9641'), (0.3, '#a6d96a'), (0.5, '#ffffbf'),
    (0.7, '#fdae61'), (1.0, '#d7191c'),
])

# Pigmentne boje za vizualizaciju
PIGMENT_BOJE = {
    'Ca':    np.array([0.93, 0.87, 0.72]),
    'Ti':    np.array([0.96, 0.96, 0.94]),
    'Fe':    np.array([0.68, 0.20, 0.03]),
    'Cu':    np.array([0.09, 0.27, 0.70]),
    'Pb_La': np.array([0.94, 0.91, 0.80]),
}

# Pravila hemijskog rizika (bez Sn)
PRAVILA_RIZIKA = [
    {'id': 'R1', 'el_a': 'Ti',    'el_b': 'Ca',    'w': 1.00,
     'naziv': 'Termalna inkompatibilnost',
     'opis': 'TiO2 restauracija preko CaCO3 maltera'},
    {'id': 'R2', 'el_a': 'Cu',    'el_b': 'Cu',    'w': 0.85,
     'naziv': 'Degradacija azurita',
     'opis': 'Azurit -> malahit transformacija'},
    {'id': 'R3', 'el_a': 'Pb_La', 'el_b': 'Pb_La', 'w': 0.70,
     'naziv': 'Potamnjivanje olovne bele',
     'opis': 'PbCO3 -> PbS u prisustvu sumpora'},
    {'id': 'R4', 'el_a': 'Ti',    'el_b': 'Cu',    'w': 0.90,
     'naziv': 'Zarobljena vlaga pod restauracijom',
     'opis': 'TiO2 blokira difuziju vlage iz Cu-pigmenta'},
    {'id': 'R5', 'el_a': 'Fe',    'el_b': 'Pb_La', 'w': 0.55,
     'naziv': 'Fe/Pb interfejs degradacija',
     'opis': 'Fe2O3 katalizuje oksidaciju Pb-pigmenta'},
]


# ═══════════════════════════════════════════════════════════════════════════════
#  UCITAVANJE PODATAKA
# ═══════════════════════════════════════════════════════════════════════════════

def norm_percentil(mapa, q_lo=8, q_hi=99):
    bg   = np.percentile(mapa, q_lo)
    peak = np.percentile(mapa, q_hi)
    return np.clip((mapa - bg) / (peak - bg + 1e-10), 0, 1)


def ucitaj_mape(dataset_label):
    npy_dir = os.path.join(BASE_DIR, 'rezultati', '_npy_cache', dataset_label)
    mape = {}
    for el in ELEMENTI:
        m10 = np.load(os.path.join(npy_dir, f'10264_{el}.npy'))
        m19 = np.load(os.path.join(npy_dir, f'19511_{el}.npy'))
        mape[el] = (m10 + m19) / 2.0
    return mape


print(f"Ucitavam element mape ({DATASET_LABEL})...")
mape_raw = ucitaj_mape(DATASET_LABEL)
norm = {el: norm_percentil(mape_raw[el]) for el in ELEMENTI}
print(f"  Ucitano {len(ELEMENTI)} elemenata: {', '.join(ELEMENTI)}")
print(f"  Sn (kalaj) ISKLJUCEN — artefakt u podacima")


# ═══════════════════════════════════════════════════════════════════════════════
#  PRIPREMA SLIKE ZA SAM
# ═══════════════════════════════════════════════════════════════════════════════

print("\nPripremam ulaznu sliku za SAM...")

# SAM ocekuje RGB uint8 sliku. Pravimo multi-element kompozit:
# R = Fe (konture, figure), G = Cu (azurit, pozadina), B = Pb (olovna bela)
# Ovo daje SAM-u maksimalan kontrast izmedju razlicitih materijala.
rgb_input = np.stack([norm['Fe'], norm['Cu'], norm['Pb_La']], axis=2)
rgb_input = np.clip(rgb_input, 0, 1)

# SAM radi bolje na vecim slikama — skaliramo na 480x960 (8x)
from scipy.ndimage import zoom as ndi_zoom
SCALE = 8
rgb_upscaled = np.stack([
    ndi_zoom(rgb_input[:, :, c], SCALE, order=3)
    for c in range(3)
], axis=2)
rgb_upscaled = np.clip(rgb_upscaled, 0, 1)

# Konverzija u uint8
sam_input = (rgb_upscaled * 255).astype(np.uint8)
print(f"  SAM ulaz: {sam_input.shape} (skalirano {SCALE}x)")


# ═══════════════════════════════════════════════════════════════════════════════
#  SAM SEGMENTACIJA
# ═══════════════════════════════════════════════════════════════════════════════

print("\nUcitavam SAM model...")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.eval()
print(f"  Model: {SAM_MODEL_TYPE}, device: cpu")

# Automatic Mask Generator — SAM sam pronalazi sve regione
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,         # Gustina grid-a za inicijalne prompt tacke
    pred_iou_thresh=0.86,       # Minimalan IoU za prihvatanje maske
    stability_score_thresh=0.92, # Stabilnost maske
    min_mask_region_area=100,   # Minimalna velicina regiona (pikseli na uvecanoj slici)
)

print("Pokrecem SAM segmentaciju...")
masks = mask_generator.generate(sam_input)
print(f"  SAM pronasao {len(masks)} segmenata")

# Sortiranje po velicini (najveci prvi)
masks = sorted(masks, key=lambda x: x['area'], reverse=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  POST-PROCESIRANJE MASKI
# ═══════════════════════════════════════════════════════════════════════════════

print("\nPost-procesiranje maski...")

# Downscale maske nazad na originalnu rezoluciju (60x120)
segments_original = np.zeros((ROWS, COLS), dtype=int)
segment_info = []

valid_id = 0
for i, mask_data in enumerate(masks):
    mask_hr = mask_data['segmentation']  # bool array (480, 960)

    # Downscale na 60x120 — piksel pripada segmentu ako >50% uvecanog regiona
    mask_lr = np.zeros((ROWS, COLS), dtype=bool)
    for r in range(ROWS):
        for c in range(COLS):
            patch = mask_hr[r*SCALE:(r+1)*SCALE, c*SCALE:(c+1)*SCALE]
            mask_lr[r, c] = patch.mean() > 0.5

    n_pixels = mask_lr.sum()
    if n_pixels < 5:  # Preskoci premale segmente
        continue

    valid_id += 1
    # Samo pikseli koji nisu vec dodeljeni vecem segmentu
    unassigned = (segments_original == 0) & mask_lr
    segments_original[unassigned] = valid_id

    segment_info.append({
        'id': valid_id,
        'area_px': int(n_pixels),
        'area_pct': n_pixels / (ROWS * COLS) * 100,
        'stability': mask_data['stability_score'],
        'iou': mask_data['predicted_iou'],
        'mask': mask_lr,
    })

# Dodeli nedodeljene piksele najblizem segmentu
from scipy.ndimage import distance_transform_edt
unassigned_mask = segments_original == 0
if unassigned_mask.any():
    # Za svaki segment racunamo distancu, dodelimo najblizem
    min_dist = np.full((ROWS, COLS), np.inf)
    for seg in segment_info:
        dist = distance_transform_edt(~seg['mask'])
        closer = dist < min_dist
        segments_original[unassigned_mask & closer] = seg['id']
        min_dist[unassigned_mask & closer] = dist[unassigned_mask & closer]

n_segments = len(segment_info)
print(f"  Validnih segmenata: {n_segments}")
for seg in segment_info[:10]:
    print(f"    Segment {seg['id']:2d}: {seg['area_px']:4d} px "
          f"({seg['area_pct']:5.1f}%), IoU={seg['iou']:.3f}")
if n_segments > 10:
    print(f"    ... i jos {n_segments - 10} manjih segmenata")


# ═══════════════════════════════════════════════════════════════════════════════
#  PER-REGION CVI ANALIZA
# ═══════════════════════════════════════════════════════════════════════════════

print("\nRacunam CVI po regionima...")

def racunaj_cvi_piksel(norm_maps):
    """Racuna CVI za svaki piksel."""
    cvi = np.zeros((ROWS, COLS))
    dom = np.zeros((ROWS, COLS), dtype=int)
    risk_maps = {}

    for pi, pravilo in enumerate(PRAVILA_RIZIKA):
        a = norm_maps[pravilo['el_a']]
        b = norm_maps[pravilo['el_b']]
        w = pravilo['w']

        if pravilo['el_a'] == pravilo['el_b']:
            risk = w * a
        else:
            risk = w * np.sqrt(a * b)

        risk = gaussian_filter(risk, sigma=1.0)
        risk = np.clip(risk, 0, 1)
        risk_maps[pravilo['id']] = risk

        mask = risk > cvi
        cvi[mask] = risk[mask]
        dom[mask] = pi

    return cvi, dom, risk_maps

cvi, dominant_risk, risk_maps = racunaj_cvi_piksel(norm)

# Agregacija po regionima
region_reports = []
for seg in segment_info:
    mask = segments_original == seg['id']
    n_px = mask.sum()
    if n_px == 0:
        continue

    # Srednji intenziteti elemenata u regionu
    el_means = {el: float(norm[el][mask].mean()) for el in ELEMENTI}

    # CVI statistika u regionu
    cvi_vals = cvi[mask]
    region_cvi_mean = float(cvi_vals.mean())
    region_cvi_max  = float(cvi_vals.max())

    # Dominantni rizik u regionu
    dom_vals = dominant_risk[mask]
    dom_counts = np.bincount(dom_vals, minlength=len(PRAVILA_RIZIKA))
    dom_idx = dom_counts.argmax()

    # Klasifikacija
    pct_elevated = float(np.mean(cvi_vals >= 0.5) * 100)
    pct_critical = float(np.mean(cvi_vals >= 0.75) * 100)

    # Opis materijala (koji element dominira)
    dominant_el = max(el_means, key=el_means.get)
    el_desc = {
        'Ca': 'Krecni malter (intonaco)',
        'Ti': 'Moderna restauracija (TiO2)',
        'Fe': 'Okra/hematit (Fe-pigment)',
        'Cu': 'Azurit (Cu-pigment)',
        'Pb_La': 'Olovna bela (Pb-pigment)',
    }

    report = {
        'id': seg['id'],
        'area_px': n_px,
        'area_pct': n_px / (ROWS * COLS) * 100,
        'el_means': el_means,
        'dominant_el': dominant_el,
        'material': el_desc.get(dominant_el, '?'),
        'cvi_mean': region_cvi_mean,
        'cvi_max': region_cvi_max,
        'dominant_risk': PRAVILA_RIZIKA[dom_idx],
        'pct_elevated': pct_elevated,
        'pct_critical': pct_critical,
    }
    region_reports.append(report)

# Sortiraj po riziku (najrizicniji prvi)
region_reports.sort(key=lambda r: r['cvi_mean'], reverse=True)

print(f"\n{'='*75}")
print(f"  IZVESTAJ PO REGIONIMA — SAM segmentacija + CVI analiza")
print(f"{'='*75}")
for r in region_reports[:15]:
    risk_level = 'KRITICAN' if r['cvi_mean'] >= 0.5 else \
                 'UMEREN' if r['cvi_mean'] >= 0.25 else 'NIZAK'
    print(f"\n  Region {r['id']:2d}  |  {r['area_px']:4d} px ({r['area_pct']:5.1f}%)"
          f"  |  CVI: {r['cvi_mean']:.3f} (max {r['cvi_max']:.3f})  |  [{risk_level}]")
    print(f"    Materijal: {r['material']}")
    print(f"    Dominantan rizik: {r['dominant_risk']['id']} — {r['dominant_risk']['naziv']}")
    print(f"    Elementi: " + "  ".join(
        f"{el}={r['el_means'][el]:.2f}" for el in ELEMENTI))
    if r['pct_elevated'] > 0:
        print(f"    Povisen rizik: {r['pct_elevated']:.1f}% piksela"
              f"  |  Kritican: {r['pct_critical']:.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
#  VIZUALIZACIJA
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*75}")
print(f"  VIZUALIZACIJA")
print(f"{'='*75}")

# ─── Slika 1: SAM segmentacija ──────────────────────────────────────────────
print("\nCrtam sliku 1: SAM segmentacija...")

fig, axes = plt.subplots(1, 3, figsize=(28, 8))

# Ulazna slika za SAM
axes[0].imshow(rgb_input, origin='upper', aspect='equal', interpolation='bicubic')
axes[0].set_title('SAM ulaz\n(R=Fe, G=Cu, B=Pb)', fontsize=12, fontweight='bold')
axes[0].axis('off')

# Segmentacija — random boje po segmentu
rng = np.random.default_rng(42)
seg_rgb = np.zeros((ROWS, COLS, 3))
for seg in segment_info:
    color = rng.random(3) * 0.7 + 0.3  # svetle boje
    mask = segments_original == seg['id']
    seg_rgb[mask] = color

axes[1].imshow(seg_rgb, origin='upper', aspect='equal', interpolation='nearest')
axes[1].set_title(f'SAM segmentacija\n({n_segments} regiona)',
                  fontsize=12, fontweight='bold')
axes[1].axis('off')

# Segmentacija sa granicama na XRF slici
axes[2].imshow(rgb_input, origin='upper', aspect='equal', interpolation='bicubic')
# Crtaj granice segmenata
from scipy.ndimage import binary_dilation
boundaries = np.zeros((ROWS, COLS), dtype=bool)
for seg_id in range(1, n_segments + 1):
    mask = segments_original == seg_id
    dilated = binary_dilation(mask, iterations=1)
    boundary = dilated & ~mask
    boundaries |= boundary
axes[2].imshow(boundaries, origin='upper', aspect='equal',
               cmap='Reds', alpha=0.6, interpolation='nearest')
axes[2].set_title('SAM granice na XRF mapi', fontsize=12, fontweight='bold')
axes[2].axis('off')

fig.suptitle('Meta SAM — automatska segmentacija freske iz XRF podataka',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IZLAZ, '1_sam_segmentacija.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print("  Sacuvano: 1_sam_segmentacija.png")


# ─── Slika 2: CVI po regionima ──────────────────────────────────────────────
print("Crtam sliku 2: CVI po regionima...")

fig, axes = plt.subplots(1, 3, figsize=(28, 8))

# Piksel-CVI mapa
im0 = axes[0].imshow(cvi, origin='upper', aspect='equal', cmap=RISK_CMAP,
                     interpolation='bicubic', vmin=0, vmax=1)
axes[0].set_title('CVI piksel-mapa', fontsize=12, fontweight='bold')
axes[0].axis('off')
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

# Region-CVI mapa (srednji CVI po regionu)
region_cvi_map = np.zeros((ROWS, COLS))
for r in region_reports:
    mask = segments_original == r['id']
    region_cvi_map[mask] = r['cvi_mean']

im1 = axes[1].imshow(region_cvi_map, origin='upper', aspect='equal',
                     cmap=RISK_CMAP, interpolation='nearest', vmin=0, vmax=1)
axes[1].set_title('CVI po SAM regionima\n(agregirani rizik)',
                  fontsize=12, fontweight='bold')
axes[1].axis('off')
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

# Numerirani regioni sa CVI
axes[2].imshow(region_cvi_map, origin='upper', aspect='equal',
               cmap=RISK_CMAP, interpolation='nearest', vmin=0, vmax=1)
for r in region_reports:
    if r['area_pct'] < 2:  # Ne oznacavaj premale
        continue
    mask = segments_original == r['id']
    ys, xs = np.where(mask)
    cy, cx = ys.mean(), xs.mean()
    axes[2].text(cx, cy, f"R{r['id']}\n{r['cvi_mean']:.2f}",
                 ha='center', va='center', fontsize=7, fontweight='bold',
                 color='white' if r['cvi_mean'] > 0.4 else 'black',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='black',
                           alpha=0.5, edgecolor='none'))
axes[2].set_title('Oznaceni regioni sa CVI skorom',
                  fontsize=12, fontweight='bold')
axes[2].axis('off')

fig.suptitle('Chemical Vulnerability Index — piksel vs. region analiza',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IZLAZ, '2_cvi_regioni.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print("  Sacuvano: 2_cvi_regioni.png")


# ─── Slika 3: Dominantni materijal po regionu ───────────────────────────────
print("Crtam sliku 3: Dominantni materijal po regionu...")

fig, axes = plt.subplots(1, 2, figsize=(24, 9))

# Materijal mapa
material_colors = {
    'Ca':    [0.93, 0.87, 0.72],
    'Ti':    [0.96, 0.96, 0.94],
    'Fe':    [0.68, 0.20, 0.03],
    'Cu':    [0.09, 0.27, 0.70],
    'Pb_La': [0.94, 0.91, 0.80],
}
mat_rgb = np.zeros((ROWS, COLS, 3))
for r in region_reports:
    mask = segments_original == r['id']
    mat_rgb[mask] = material_colors[r['dominant_el']]

axes[0].imshow(mat_rgb, origin='upper', aspect='equal', interpolation='nearest')
axes[0].set_title('Dominantni materijal po SAM regionu',
                  fontsize=12, fontweight='bold')
axes[0].axis('off')
el_labels = {'Ca': 'Malter', 'Ti': 'Restauracija', 'Fe': 'Okra',
             'Cu': 'Azurit', 'Pb_La': 'Olovna bela'}
patches = [mpatches.Patch(color=material_colors[el],
                          label=f"{el}: {el_labels[el]}")
           for el in ELEMENTI]
axes[0].legend(handles=patches, loc='lower left', fontsize=9)

# Dominantni rizik po regionu
risk_colors = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#984ea3']
risk_rgb = np.full((ROWS, COLS, 3), 0.94)
for r in region_reports:
    if r['cvi_mean'] < 0.2:
        continue
    mask = segments_original == r['id']
    ri = PRAVILA_RIZIKA.index(r['dominant_risk'])
    c = risk_colors[ri]
    risk_rgb[mask] = [int(c[1:3], 16)/255, int(c[3:5], 16)/255, int(c[5:7], 16)/255]

axes[1].imshow(risk_rgb, origin='upper', aspect='equal', interpolation='nearest')
axes[1].set_title('Dominantni mehanizam degradacije po regionu',
                  fontsize=12, fontweight='bold')
axes[1].axis('off')
patches = [mpatches.Patch(color=risk_colors[i],
                          label=f"{PRAVILA_RIZIKA[i]['id']}: {PRAVILA_RIZIKA[i]['naziv']}")
           for i in range(len(PRAVILA_RIZIKA))]
patches.append(mpatches.Patch(color='#f0f0f0', label='Nizak rizik'))
axes[1].legend(handles=patches, loc='lower left', fontsize=8)

fig.suptitle('SAM segmentacija — identifikacija materijala i mehanizama rizika',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IZLAZ, '3_materijal_i_rizik.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print("  Sacuvano: 3_materijal_i_rizik.png")


# ─── Slika 4: Heatmapa element intenziteta po regionu ───────────────────────
print("Crtam sliku 4: Heatmapa intenziteta po regionu...")

top_n = min(15, len(region_reports))
top_reports = region_reports[:top_n]

fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.5)))

matrix = np.array([[r['el_means'][el] for el in ELEMENTI] for r in top_reports])
im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

ax.set_xticks(range(len(ELEMENTI)))
ax.set_xticklabels(ELEMENTI, fontsize=11, fontweight='bold')
ax.set_yticks(range(top_n))
ax.set_yticklabels([f"R{r['id']} ({r['material'][:20]})\nCVI={r['cvi_mean']:.2f}"
                    for r in top_reports], fontsize=9)

for i in range(top_n):
    for j in range(len(ELEMENTI)):
        val = matrix[i, j]
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=9,
                color='white' if val > 0.5 else 'black', fontweight='bold')

plt.colorbar(im, ax=ax, label='Normirani intenzitet', fraction=0.03, pad=0.04)
ax.set_title('Hemijski profil svakog SAM regiona\n'
             '(sortirano po CVI riziku, najrizicniji gore)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IZLAZ, '4_heatmapa_po_regionu.png'),
            dpi=180, bbox_inches='tight')
plt.close()
print("  Sacuvano: 4_heatmapa_po_regionu.png")


# ─── Slika 5: Bar chart — CVI po regionu ────────────────────────────────────
print("Crtam sliku 5: CVI bar chart...")

fig, ax = plt.subplots(figsize=(14, 6))
ids = [f"R{r['id']}" for r in top_reports]
cvi_vals = [r['cvi_mean'] for r in top_reports]
bar_colors = [RISK_CMAP(v) for v in cvi_vals]

bars = ax.barh(range(top_n), cvi_vals, color=bar_colors, edgecolor='black', alpha=0.9)
for i, (bar, r) in enumerate(zip(bars, top_reports)):
    ax.text(bar.get_width() + 0.01, i,
            f"{r['dominant_risk']['id']}: {r['material'][:25]}",
            va='center', fontsize=9)

ax.set_yticks(range(top_n))
ax.set_yticklabels(ids, fontsize=10)
ax.set_xlabel('Srednji CVI skor', fontsize=12)
ax.set_title('Rangiranje SAM regiona po hemijskom riziku',
             fontsize=13, fontweight='bold')
ax.axvline(0.5, color='red', linestyle='--', alpha=0.6, label='Prag povisen rizik')
ax.axvline(0.25, color='orange', linestyle='--', alpha=0.6, label='Prag umeren rizik')
ax.legend(fontsize=10)
ax.set_xlim(0, 1.1)
ax.grid(True, alpha=0.2, axis='x')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(IZLAZ, '5_cvi_ranking.png'),
            dpi=180, bbox_inches='tight')
plt.close()
print("  Sacuvano: 5_cvi_ranking.png")


# ─── Slika 6: Kompletni pregled (summary panel) ─────────────────────────────
print("Crtam sliku 6: Kompletni pregled...")

fig, axes = plt.subplots(2, 3, figsize=(28, 16))

# (0,0) XRF false-color
axes[0,0].imshow(rgb_input, origin='upper', aspect='equal', interpolation='bicubic')
axes[0,0].set_title('XRF False-color\n(R=Fe, G=Cu, B=Pb)', fontsize=11, fontweight='bold')
axes[0,0].axis('off')

# (0,1) SAM segmentacija
axes[0,1].imshow(seg_rgb, origin='upper', aspect='equal', interpolation='nearest')
axes[0,1].set_title(f'SAM segmentacija\n({n_segments} regiona)', fontsize=11, fontweight='bold')
axes[0,1].axis('off')

# (0,2) Materijal po regionu
axes[0,2].imshow(mat_rgb, origin='upper', aspect='equal', interpolation='nearest')
axes[0,2].set_title('Dominantni materijal', fontsize=11, fontweight='bold')
axes[0,2].axis('off')

# (1,0) CVI piksel mapa
im = axes[1,0].imshow(cvi, origin='upper', aspect='equal', cmap=RISK_CMAP,
                      interpolation='bicubic', vmin=0, vmax=1)
axes[1,0].set_title('CVI piksel-mapa', fontsize=11, fontweight='bold')
axes[1,0].axis('off')
plt.colorbar(im, ax=axes[1,0], fraction=0.046, pad=0.04)

# (1,1) CVI po regionu
im = axes[1,1].imshow(region_cvi_map, origin='upper', aspect='equal',
                      cmap=RISK_CMAP, interpolation='nearest', vmin=0, vmax=1)
axes[1,1].set_title('CVI po SAM regionu', fontsize=11, fontweight='bold')
axes[1,1].axis('off')
plt.colorbar(im, ax=axes[1,1], fraction=0.046, pad=0.04)

# (1,2) Dominantni rizik
axes[1,2].imshow(risk_rgb, origin='upper', aspect='equal', interpolation='nearest')
axes[1,2].set_title('Dominantni mehanizam degradacije', fontsize=11, fontweight='bold')
axes[1,2].axis('off')

fig.suptitle(f'SAM + CVI Pipeline — Kompletna analiza freske ({DATASET_LABEL})\n'
             f'Meta Segment Anything Model + Chemical Vulnerability Index',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IZLAZ, '6_kompletni_pregled.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print("  Sacuvano: 6_kompletni_pregled.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  TEKSTUALNI IZVESTAJ ZA RESTAURATORE
# ═══════════════════════════════════════════════════════════════════════════════

print("\nGenerisem tekstualni izvestaj...")

report_text = f"""{'='*75}
  IZVESTAJ O HEMIJSKOJ RANJIVOSTI FRESKE
  Automatski generisan: SAM segmentacija + CVI analiza
  Dataset: {DATASET_LABEL}
{'='*75}

METODOLOGIJA:
  - Segmentacija: Meta SAM (Segment Anything Model, ViT-B)
  - Rizik: Chemical Vulnerability Index (CVI) — 5 pravila degradacije
  - Ulaz: XRF mape 5 elemenata (Ca, Ti, Fe, Cu, Pb) — Sn iskljucen

STATISTIKA:
  - Ukupno piksela: {ROWS * COLS}
  - SAM segmenata: {n_segments}
  - Srednji CVI: {cvi.mean():.3f}
  - Povisen rizik (CVI >= 0.5): {np.mean(cvi >= 0.5)*100:.1f}% povrsine
  - Kritican rizik (CVI >= 0.75): {np.mean(cvi >= 0.75)*100:.1f}% povrsine

{'='*75}
  TOP REGIONI PO RIZIKU (preporuke za restauratore)
{'='*75}
"""

for i, r in enumerate(region_reports[:10]):
    risk_level = 'KRITICAN' if r['cvi_mean'] >= 0.5 else \
                 'UMEREN' if r['cvi_mean'] >= 0.25 else 'NIZAK'

    if risk_level == 'KRITICAN':
        preporuka = 'HITNA konzervatorska intervencija'
    elif risk_level == 'UMEREN':
        preporuka = 'Preventivni monitoring, planirati konsolidaciju'
    else:
        preporuka = 'Bez neposrednog rizika, standardni monitoring'

    report_text += f"""
  [{i+1}] Region {r['id']}  —  {risk_level}
      Povrsina: {r['area_px']} piksela ({r['area_pct']:.1f}%)
      Materijal: {r['material']}
      CVI: {r['cvi_mean']:.3f} (max {r['cvi_max']:.3f})
      Rizik: {r['dominant_risk']['id']} — {r['dominant_risk']['naziv']}
             {r['dominant_risk']['opis']}
      Preporuka: {preporuka}
"""

report_text += f"""
{'='*75}
  LEGENDA RIZIKA
{'='*75}
"""
for p in PRAVILA_RIZIKA:
    report_text += f"  {p['id']}: {p['naziv']} (w={p['w']})\n"
    report_text += f"      {p['opis']}\n\n"

report_path = os.path.join(IZLAZ, 'izvestaj_restauratori.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_text)
print(f"  Sacuvano: izvestaj_restauratori.txt")


# ═══════════════════════════════════════════════════════════════════════════════
#  ZAVRSETAK
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*75}")
print(f"  SAM + CVI PIPELINE ZAVRSEN")
print(f"  Rezultati: {os.path.abspath(IZLAZ)}/")
print(f"  Slike: 6 vizualizacija")
print(f"  Izvestaj: izvestaj_restauratori.txt")
print(f"{'='*75}")
