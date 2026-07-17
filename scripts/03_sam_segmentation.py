"""
03_sam_segmentation.py
===============================================================================
Region segmentation of the painting with Meta SAM (Segment Anything Model)
+ per-region aggregation of the Chemical Vulnerability Index (CVI).

Companion code to Pešić et al., ICETRAN 2026 (ICETRAN.pdf).

PIPELINE
  1. Load cached XRF element maps (Ca, Ti, Fe, Cu, Pb) — Sn excluded
  2. Build the SAM input image (false-color element composite, upscaled 8x)
  3. SAM automatic mask generation (prompt-free)
  4. Mask post-processing (downscale, majority vote, small-segment filter)
  5. Per-region CVI statistics and dominant degradation mechanism
  6. Conservator-facing report + figures

NOTE ON Sn:
  Tin is excluded from the analysis — it was identified as an acquisition
  artifact in the data, not a real pigment.

Usage (from the project root):
    python scripts/03_sam_segmentation.py [prova1|prova2]

Requires: torch, segment-anything, and the ViT-B checkpoint at
models/sam_vit_b_01ec64.pth (see README: Setup).
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import (binary_dilation, distance_transform_edt,
                           gaussian_filter, zoom as ndi_zoom)

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(BASE_DIR, "results", "sam_segmentation")
os.makedirs(OUT_DIR, exist_ok=True)

ROWS, COLS = 60, 120
DATASET    = sys.argv[1] if len(sys.argv) > 1 else "prova1"

# Sn excluded (acquisition artifact)
ELEMENTS  = ["Ca", "Ti", "Fe", "Cu", "Pb_La"]
# Paper: "All analyses are performed on a single detector (10264)."
# Add "19511" to average both detectors instead.
DETECTORS = ["10264"]

SAM_CHECKPOINT = os.path.join(BASE_DIR, "models", "sam_vit_b_01ec64.pth")
SAM_MODEL_TYPE = "vit_b"
SCALE          = 8            # upscale factor for the SAM input image

RISK_CMAP = LinearSegmentedColormap.from_list("risk", [
    (0.0, "#1a9641"), (0.3, "#a6d96a"), (0.5, "#ffffbf"),
    (0.7, "#fdae61"), (1.0, "#d7191c"),
])

# Fixed rule identity colors (Okabe-Ito, colorblind-safe)
RULE_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]

# Degradation rules — paper Table II, chemistry-first severity weights
RISK_RULES = [
    {"id": "R1", "el_a": "Ti",    "el_b": "Ca",    "w": 0.40,
     "name": "TiO2/CaCO3 thermal mismatch",
     "desc": "TiO2 layer over a CaCO3-bearing ground"},
    {"id": "R2", "el_a": "Cu",    "el_b": "Cu",    "w": 1.00,
     "name": "Cu-based green pigment degradation",
     "desc": "Basic copper carbonates transform in humid air"},
    {"id": "R3", "el_a": "Pb_La", "el_b": "Pb_La", "w": 1.00,
     "name": "Lead white darkening (PbS)",
     "desc": "PbCO3 -> PbS in the presence of sulfur compounds"},
    {"id": "R4", "el_a": "Ti",    "el_b": "Cu",    "w": 0.60,
     "name": "Trapped moisture under TiO2",
     "desc": "TiO2 blocks moisture diffusion out of the Cu pigment"},
    {"id": "R5", "el_a": "Fe",    "el_b": "Pb_La", "w": 0.80,
     "name": "Fe-catalyzed Pb oxidation",
     "desc": "Fe2O3 catalyses oxidation of the Pb pigment"},
]

MATERIAL_DESC = {
    "Ca":    "Lime/chalk layer (CaCO3)",
    "Ti":    "Titanium white ground (TiO2)",
    "Fe":    "Ochre (Fe-based pigment)",
    "Cu":    "Cu-based green pigment",
    "Pb_La": "Lead white (Pb-based pigment)",
}
MATERIAL_COLORS = {
    "Ca":    [0.93, 0.87, 0.72],
    "Ti":    [0.96, 0.96, 0.94],
    "Fe":    [0.68, 0.20, 0.03],
    "Cu":    [0.09, 0.27, 0.70],
    "Pb_La": [0.80, 0.75, 0.95],
}
MATERIAL_SHORT = {"Ca": "Lime/chalk", "Ti": "Ti-white ground", "Fe": "Ochre",
                  "Cu": "Cu green", "Pb_La": "Lead white"}

_MAP_CB = dict(fraction=0.046 * ROWS / COLS, pad=0.02)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING (same flexible cache resolution as 02_vulnerability.py)
# ═══════════════════════════════════════════════════════════════════════════════

_ELEMENT_ALIASES = {"Pb_La": ["Pb_La", "PbLa"]}

def _cache_candidates(dataset, det, el):
    paths = []
    for name in _ELEMENT_ALIASES.get(el, [el]):
        paths += [
            os.path.join(BASE_DIR, "results", det, "_npy_cache", f"{dataset}_{name}.npy"),
            os.path.join(BASE_DIR, "results", "_npy_cache", dataset, f"{det}_{name}.npy"),
            os.path.join(BASE_DIR, "results", "_npy_cache", f"{dataset}_{det}_{name}.npy"),
        ]
    return paths


def load_element_maps(dataset):
    maps, used = {}, set()
    for el in ELEMENTS:
        per_det = []
        for det in DETECTORS:
            for path in _cache_candidates(dataset, det, el):
                if os.path.exists(path):
                    per_det.append(np.load(path))
                    used.add(det)
                    break
        if not per_det:
            sys.exit(f"ERROR: element map '{el}' for '{dataset}' not found in "
                     f"any cache — run scripts/01_run_analysis.py first.")
        maps[el] = np.mean(per_det, axis=0)
    print(f"  [{dataset}] element maps loaded (detectors: {', '.join(sorted(used))})")
    return maps


def norm_percentile(m, q_lo=8, q_hi=99):
    bg   = np.percentile(m, q_lo)
    peak = np.percentile(m, q_hi)
    return np.clip((m - bg) / (peak - bg + 1e-10), 0, 1)


print(f"Loading element maps ({DATASET})...")
raw_maps = load_element_maps(DATASET)
norm = {el: norm_percentile(raw_maps[el]) for el in ELEMENTS}
print(f"  {len(ELEMENTS)} elements: {', '.join(ELEMENTS)}  (Sn excluded — artifact)")


# ═══════════════════════════════════════════════════════════════════════════════
#  SAM INPUT IMAGE
# ═══════════════════════════════════════════════════════════════════════════════

print("\nPreparing the SAM input image...")

# SAM expects an RGB uint8 image. False-color composite:
# R = Fe (contours/figure), G = Cu (green pigment), B = Pb (lead white)
# maximises contrast between the painting's materials.
rgb_input = np.clip(np.stack([norm["Fe"], norm["Cu"], norm["Pb_La"]], axis=2), 0, 1)

rgb_upscaled = np.clip(np.stack([
    ndi_zoom(rgb_input[:, :, c], SCALE, order=3) for c in range(3)
], axis=2), 0, 1)
sam_input = (rgb_upscaled * 255).astype(np.uint8)
print(f"  SAM input: {sam_input.shape} (upscaled {SCALE}x)")


# ═══════════════════════════════════════════════════════════════════════════════
#  SAM SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

if not os.path.exists(SAM_CHECKPOINT):
    sys.exit(
        f"ERROR: SAM checkpoint not found: {SAM_CHECKPOINT}\n"
        "Download sam_vit_b_01ec64.pth from "
        "https://github.com/facebookresearch/segment-anything#model-checkpoints "
        "and place it in models/."
    )

print("\nLoading the SAM model...")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.eval()
print(f"  Model: {SAM_MODEL_TYPE}, device: cpu")

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,           # prompt-point grid density
    pred_iou_thresh=0.86,         # minimum predicted IoU to accept a mask
    stability_score_thresh=0.92,  # mask stability threshold
    min_mask_region_area=100,     # minimum region size (upscaled pixels)
)

print("Running SAM segmentation...")
masks = mask_generator.generate(sam_input)
print(f"  SAM proposed {len(masks)} masks")
masks = sorted(masks, key=lambda x: x["area"], reverse=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  MASK POST-PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

print("\nPost-processing masks...")

segments = np.zeros((ROWS, COLS), dtype=int)
segment_info = []
valid_id = 0

for mask_data in masks:
    mask_hr = mask_data["segmentation"]          # bool (ROWS*SCALE, COLS*SCALE)

    # Downscale by majority vote: a native pixel belongs to the mask if
    # more than half of its upscaled patch does.
    mask_lr = np.zeros((ROWS, COLS), dtype=bool)
    for r in range(ROWS):
        for c in range(COLS):
            patch = mask_hr[r * SCALE:(r + 1) * SCALE, c * SCALE:(c + 1) * SCALE]
            mask_lr[r, c] = patch.mean() > 0.5

    n_pixels = mask_lr.sum()
    if n_pixels < 5:                              # discard tiny segments
        continue

    valid_id += 1
    unassigned = (segments == 0) & mask_lr        # larger segments win overlaps
    segments[unassigned] = valid_id

    segment_info.append({
        "id": valid_id,
        "area_px": int(n_pixels),
        "area_pct": n_pixels / (ROWS * COLS) * 100,
        "stability": mask_data["stability_score"],
        "iou": mask_data["predicted_iou"],
        "mask": mask_lr,
    })

# Assign leftover pixels to the nearest segment
unassigned_mask = segments == 0
if unassigned_mask.any() and segment_info:
    min_dist = np.full((ROWS, COLS), np.inf)
    for seg in segment_info:
        dist = distance_transform_edt(~seg["mask"])
        closer = dist < min_dist
        segments[unassigned_mask & closer] = seg["id"]
        min_dist[unassigned_mask & closer] = dist[unassigned_mask & closer]

n_segments = len(segment_info)
print(f"  Valid segments: {n_segments}")
for seg in segment_info[:10]:
    print(f"    Segment {seg['id']:2d}: {seg['area_px']:4d} px "
          f"({seg['area_pct']:5.1f}%), IoU={seg['iou']:.3f}")
if n_segments > 10:
    print(f"    ... and {n_segments - 10} smaller segments")


# ═══════════════════════════════════════════════════════════════════════════════
#  PER-REGION CVI
# ═══════════════════════════════════════════════════════════════════════════════

print("\nComputing per-region CVI...")

def compute_cvi(norm_maps):
    """Per-pixel CVI + dominant rule (paper Eq. 2)."""
    cvi = np.zeros((ROWS, COLS))
    dom = np.zeros((ROWS, COLS), dtype=int)
    risk_maps = {}

    for ri, rule in enumerate(RISK_RULES):
        a, b, w = norm_maps[rule["el_a"]], norm_maps[rule["el_b"]], rule["w"]
        risk = w * a if rule["el_a"] == rule["el_b"] else w * np.sqrt(a * b)
        risk = np.clip(gaussian_filter(risk, sigma=1.0), 0, 1)
        risk_maps[rule["id"]] = risk

        mask = risk > cvi
        cvi[mask] = risk[mask]
        dom[mask] = ri

    return cvi, dom, risk_maps


cvi, dominant_risk, risk_maps = compute_cvi(norm)

region_reports = []
for seg in segment_info:
    mask = segments == seg["id"]
    n_px = mask.sum()
    if n_px == 0:
        continue

    el_means = {el: float(norm[el][mask].mean()) for el in ELEMENTS}
    cvi_vals = cvi[mask]
    dom_idx  = np.bincount(dominant_risk[mask],
                           minlength=len(RISK_RULES)).argmax()
    dominant_el = max(el_means, key=el_means.get)

    region_reports.append({
        "id": seg["id"],
        "area_px": int(n_px),
        "area_pct": n_px / (ROWS * COLS) * 100,
        "el_means": el_means,
        "dominant_el": dominant_el,
        "material": MATERIAL_DESC[dominant_el],
        "cvi_mean": float(cvi_vals.mean()),
        "cvi_max": float(cvi_vals.max()),
        "dominant_risk": RISK_RULES[dom_idx],
        "pct_elevated": float(np.mean(cvi_vals >= 0.5) * 100),
        "pct_critical": float(np.mean(cvi_vals >= 0.75) * 100),
    })

region_reports.sort(key=lambda r: r["cvi_mean"], reverse=True)


def region_level(mean_cvi):
    if mean_cvi >= 0.75: return "CRITICAL"
    if mean_cvi >= 0.50: return "ELEVATED"
    if mean_cvi >= 0.25: return "MODERATE"
    return "LOW"


print(f"\n{'=' * 75}")
print("  PER-REGION REPORT — SAM segmentation + CVI")
print(f"{'=' * 75}")
for r in region_reports[:15]:
    print(f"\n  Region {r['id']:2d}  |  {r['area_px']:4d} px ({r['area_pct']:5.1f}%)"
          f"  |  CVI: {r['cvi_mean']:.3f} (max {r['cvi_max']:.3f})"
          f"  |  [{region_level(r['cvi_mean'])}]")
    print(f"    Material: {r['material']}")
    print(f"    Dominant risk: {r['dominant_risk']['id']} — {r['dominant_risk']['name']}")
    print("    Elements: " + "  ".join(f"{el}={r['el_means'][el]:.2f}" for el in ELEMENTS))
    if r["pct_elevated"] > 0:
        print(f"    Elevated: {r['pct_elevated']:.1f}% of pixels"
              f"  |  Critical: {r['pct_critical']:.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 75}")
print("  FIGURES")
print(f"{'=' * 75}")


def _style_cb(cb, label=None):
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_edgecolor("#CCCCCC")
    if label:
        cb.set_label(label, fontsize=9)


def _blank(ax):
    ax.set_xticks([]); ax.set_yticks([])


# ─── Figure 1: SAM segmentation ─────────────────────────────────────────────
print("\nFigure 1: SAM segmentation...")

rng = np.random.default_rng(42)
seg_rgb = np.zeros((ROWS, COLS, 3))
for seg in segment_info:
    seg_rgb[segments == seg["id"]] = rng.random(3) * 0.7 + 0.3

boundaries = np.zeros((ROWS, COLS), dtype=bool)
for seg_id in range(1, n_segments + 1):
    m = segments == seg_id
    boundaries |= binary_dilation(m, iterations=1) & ~m

fig, axes = plt.subplots(1, 3, figsize=(18, 3.9), layout="constrained")

axes[0].imshow(rgb_input, origin="upper", aspect="equal", interpolation="bilinear")
axes[0].set_title("SAM input — false-color composite\n(R=Fe, G=Cu, B=Pb)",
                  fontsize=11, fontweight="bold")

axes[1].imshow(seg_rgb, origin="upper", aspect="equal", interpolation="nearest")
axes[1].set_title(f"SAM segmentation\n({n_segments} regions)",
                  fontsize=11, fontweight="bold")

axes[2].imshow(rgb_input, origin="upper", aspect="equal", interpolation="bilinear")
overlay = np.zeros((ROWS, COLS, 4))
overlay[boundaries] = [1.0, 1.0, 1.0, 0.9]
axes[2].imshow(overlay, origin="upper", aspect="equal", interpolation="nearest")
axes[2].set_title("Segment boundaries on the XRF composite\n(white outlines)",
                  fontsize=11, fontweight="bold")

for ax in axes:
    _blank(ax)

fig.suptitle("Meta SAM — automatic region segmentation from XRF data",
             fontsize=13, fontweight="bold")
plt.savefig(os.path.join(OUT_DIR, "1_sam_segmentation.png"),
            dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved: 1_sam_segmentation.png")


# ─── Figure 2: CVI per region ───────────────────────────────────────────────
print("Figure 2: CVI per region...")

region_cvi_map = np.zeros((ROWS, COLS))
for r in region_reports:
    region_cvi_map[segments == r["id"]] = r["cvi_mean"]

fig, axes = plt.subplots(1, 3, figsize=(18, 3.9), layout="constrained")

im0 = axes[0].imshow(cvi, origin="upper", aspect="equal", cmap=RISK_CMAP,
                     interpolation="bilinear", vmin=0, vmax=1)
axes[0].set_title("CVI — pixel level", fontsize=11, fontweight="bold")
_style_cb(plt.colorbar(im0, ax=axes[0], **_MAP_CB), "CVI score")

im1 = axes[1].imshow(region_cvi_map, origin="upper", aspect="equal",
                     cmap=RISK_CMAP, interpolation="nearest", vmin=0, vmax=1)
axes[1].set_title("Mean CVI per SAM region", fontsize=11, fontweight="bold")
_style_cb(plt.colorbar(im1, ax=axes[1], **_MAP_CB), "mean CVI")

axes[2].imshow(region_cvi_map, origin="upper", aspect="equal",
               cmap=RISK_CMAP, interpolation="nearest", vmin=0, vmax=1)
for r in region_reports:
    if r["area_pct"] < 2:
        continue
    ys, xs = np.where(segments == r["id"])
    axes[2].text(xs.mean(), ys.mean(), f"R{r['id']}\n{r['cvi_mean']:.2f}",
                 ha="center", va="center", fontsize=7, fontweight="bold",
                 color="white",
                 bbox=dict(boxstyle="round,pad=0.22", facecolor="black",
                           alpha=0.55, edgecolor="none"))
axes[2].set_title("Regions labelled with mean CVI", fontsize=11, fontweight="bold")

for ax in axes:
    _blank(ax)

fig.suptitle("Chemical Vulnerability Index — pixel vs region aggregation",
             fontsize=13, fontweight="bold")
plt.savefig(os.path.join(OUT_DIR, "2_cvi_regions.png"),
            dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved: 2_cvi_regions.png")


# ─── Figure 3: dominant material and risk per region ────────────────────────
print("Figure 3: dominant material and risk per region...")

mat_rgb = np.zeros((ROWS, COLS, 3))
for r in region_reports:
    mat_rgb[segments == r["id"]] = MATERIAL_COLORS[r["dominant_el"]]

risk_rgb = np.full((ROWS, COLS, 3), 0.95)
for r in region_reports:
    if r["cvi_mean"] < 0.2:
        continue
    ri = RISK_RULES.index(r["dominant_risk"])
    c = RULE_COLORS[ri]
    risk_rgb[segments == r["id"]] = [int(c[j:j+2], 16) / 255 for j in (1, 3, 5)]

fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), layout="constrained")

axes[0].imshow(mat_rgb, origin="upper", aspect="equal", interpolation="nearest")
axes[0].set_title("Dominant material per SAM region", fontsize=11, fontweight="bold")
axes[0].legend(handles=[mpatches.Patch(color=MATERIAL_COLORS[el],
                                       label=f"{el.replace('_La', '')}: {MATERIAL_SHORT[el]}")
                        for el in ELEMENTS],
               loc="upper center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.03), frameon=False)

axes[1].imshow(risk_rgb, origin="upper", aspect="equal", interpolation="nearest")
axes[1].set_title("Dominant degradation mechanism per region",
                  fontsize=11, fontweight="bold")
patches = [mpatches.Patch(color=RULE_COLORS[i], label=f"{r['id']}: {r['name']}")
           for i, r in enumerate(RISK_RULES)]
patches.append(mpatches.Patch(color="#f0f0f0", label="Low risk"))
axes[1].legend(handles=patches, loc="upper center", ncol=2, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.03), frameon=False)

for ax in axes:
    _blank(ax)

fig.suptitle("SAM regions — material identification and risk mechanisms",
             fontsize=13, fontweight="bold")
plt.savefig(os.path.join(OUT_DIR, "3_material_and_risk.png"),
            dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved: 3_material_and_risk.png")


# ─── Figure 4: element-intensity heatmap per region ─────────────────────────
print("Figure 4: per-region chemical profile...")

top_n = min(15, len(region_reports))
top_reports = region_reports[:top_n]

fig, ax = plt.subplots(figsize=(9, max(4.5, top_n * 0.42 + 1.2)),
                       layout="constrained")

matrix = np.array([[r["el_means"][el] for el in ELEMENTS] for r in top_reports])
im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

ax.set_xticks(range(len(ELEMENTS)))
ax.set_xticklabels([el.replace("_La", "") for el in ELEMENTS],
                   fontsize=10, fontweight="bold")
ax.set_yticks(range(top_n))
ax.set_yticklabels([f"R{r['id']}  ({MATERIAL_SHORT[r['dominant_el']]}), "
                    f"CVI {r['cvi_mean']:.2f}" for r in top_reports], fontsize=8.5)

for i in range(top_n):
    for j in range(len(ELEMENTS)):
        v = matrix[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                color="white" if v > 0.55 else "black")

_style_cb(plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02),
          "normalised intensity")
ax.set_title("Chemical profile of each SAM region\n"
             "(sorted by mean CVI, highest risk first)",
             fontsize=12, fontweight="bold")
plt.savefig(os.path.join(OUT_DIR, "4_region_heatmap.png"),
            dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved: 4_region_heatmap.png")


# ─── Figure 5: region CVI ranking ───────────────────────────────────────────
print("Figure 5: region ranking...")

fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.38 + 1.5)),
                       layout="constrained")
cvi_vals = [r["cvi_mean"] for r in top_reports]

bars = ax.barh(range(top_n), cvi_vals,
               color=[RISK_CMAP(v) for v in cvi_vals], height=0.62)
for i, r in enumerate(top_reports):
    ax.text(cvi_vals[i] + 0.012, i,
            f"{r['dominant_risk']['id']} — {MATERIAL_SHORT[r['dominant_el']]}",
            va="center", fontsize=8.5)

ax.set_yticks(range(top_n))
ax.set_yticklabels([f"R{r['id']}" for r in top_reports], fontsize=9)
ax.set_xlabel("Mean CVI score", fontsize=10)
ax.set_title("SAM regions ranked by chemical risk",
             fontsize=12, fontweight="bold")
ax.axvline(0.25, color="#999999", linestyle="--", alpha=0.8, linewidth=1,
           label="Moderate threshold (0.25)")
ax.axvline(0.50, color="#555555", linestyle="--", alpha=0.8, linewidth=1,
           label="Elevated threshold (0.50)")
ax.legend(fontsize=8.5, frameon=False, loc="lower right")
ax.set_xlim(0, 1.0)
ax.grid(True, alpha=0.15, axis="x")
ax.set_axisbelow(True)
ax.invert_yaxis()
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=8.5)

plt.savefig(os.path.join(OUT_DIR, "5_cvi_ranking.png"),
            dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved: 5_cvi_ranking.png")


# ─── Figure 6: summary panel ────────────────────────────────────────────────
print("Figure 6: summary panel...")

fig, axes = plt.subplots(2, 3, figsize=(18, 7.4), layout="constrained")

axes[0, 0].imshow(rgb_input, origin="upper", aspect="equal", interpolation="bilinear")
axes[0, 0].set_title("XRF false-color (R=Fe, G=Cu, B=Pb)",
                     fontsize=10.5, fontweight="bold")

axes[0, 1].imshow(seg_rgb, origin="upper", aspect="equal", interpolation="nearest")
axes[0, 1].set_title(f"SAM segmentation ({n_segments} regions)",
                     fontsize=10.5, fontweight="bold")

axes[0, 2].imshow(mat_rgb, origin="upper", aspect="equal", interpolation="nearest")
axes[0, 2].set_title("Dominant material", fontsize=10.5, fontweight="bold")

im = axes[1, 0].imshow(cvi, origin="upper", aspect="equal", cmap=RISK_CMAP,
                       interpolation="bilinear", vmin=0, vmax=1)
axes[1, 0].set_title("CVI — pixel level", fontsize=10.5, fontweight="bold")
_style_cb(plt.colorbar(im, ax=axes[1, 0], **_MAP_CB))

im = axes[1, 1].imshow(region_cvi_map, origin="upper", aspect="equal",
                       cmap=RISK_CMAP, interpolation="nearest", vmin=0, vmax=1)
axes[1, 1].set_title("Mean CVI per SAM region", fontsize=10.5, fontweight="bold")
_style_cb(plt.colorbar(im, ax=axes[1, 1], **_MAP_CB))

axes[1, 2].imshow(risk_rgb, origin="upper", aspect="equal", interpolation="nearest")
axes[1, 2].set_title("Dominant degradation mechanism",
                     fontsize=10.5, fontweight="bold")

for ax in axes.flatten():
    _blank(ax)

fig.suptitle(f"SAM + CVI pipeline — full analysis ({DATASET})\n"
             "Meta Segment Anything Model + Chemical Vulnerability Index",
             fontsize=13, fontweight="bold")
plt.savefig(os.path.join(OUT_DIR, "6_summary_panel.png"),
            dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved: 6_summary_panel.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  CONSERVATOR REPORT
# ═══════════════════════════════════════════════════════════════════════════════

print("\nWriting the conservator report...")

RECOMMENDATION = {
    "CRITICAL": "URGENT conservation assessment",
    "ELEVATED": "Priority assessment, plan consolidation",
    "MODERATE": "Preventive monitoring",
    "LOW":      "No immediate risk, routine monitoring",
}

report = [
    "=" * 75,
    "  CHEMICAL VULNERABILITY REPORT",
    "  Auto-generated: SAM segmentation + CVI analysis",
    f"  Dataset: {DATASET}",
    "=" * 75,
    "",
    "METHODOLOGY:",
    "  - Segmentation: Meta SAM (Segment Anything Model, ViT-B)",
    "  - Risk: Chemical Vulnerability Index (CVI) — 5 degradation rules",
    "  - Input: XRF maps of 5 elements (Ca, Ti, Fe, Cu, Pb) — Sn excluded",
    "",
    "STATISTICS:",
    f"  - Total pixels: {ROWS * COLS}",
    f"  - SAM segments: {n_segments}",
    f"  - Mean CVI: {cvi.mean():.3f}",
    f"  - Elevated risk (CVI >= 0.50): {np.mean(cvi >= 0.5) * 100:.1f}% of area",
    f"  - Critical risk (CVI >= 0.75): {np.mean(cvi >= 0.75) * 100:.1f}% of area",
    "",
    "=" * 75,
    "  TOP REGIONS BY RISK (conservation priorities)",
    "=" * 75,
]

for i, r in enumerate(region_reports[:10]):
    level = region_level(r["cvi_mean"])
    report += [
        "",
        f"  [{i + 1}] Region {r['id']}  —  {level}",
        f"      Area: {r['area_px']} px ({r['area_pct']:.1f}%)",
        f"      Material: {r['material']}",
        f"      CVI: {r['cvi_mean']:.3f} (max {r['cvi_max']:.3f})",
        f"      Risk: {r['dominant_risk']['id']} — {r['dominant_risk']['name']}",
        f"            {r['dominant_risk']['desc']}",
        f"      Recommendation: {RECOMMENDATION[level]}",
    ]

report += ["", "=" * 75, "  RISK RULES LEGEND", "=" * 75]
for p in RISK_RULES:
    report += [f"  {p['id']}: {p['name']} (w={p['w']})",
               f"      {p['desc']}", ""]

report_path = os.path.join(OUT_DIR, "conservation_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report))
print(f"  Saved: conservation_report.txt")


print(f"\n{'=' * 75}")
print("  SAM + CVI PIPELINE COMPLETE")
print(f"  Results: {os.path.abspath(OUT_DIR)}{os.sep}")
print("  Figures: 6 visualizations + conservation_report.txt")
print(f"{'=' * 75}")
