"""
make_submission_figures.py
===============================================================================
Publication-ready (submission) versions of the paper figures.

Style rules:
  - no titles/subtitles/stat boxes on the figures — descriptive text lives
    in submission/captions.txt instead
  - axis names, tick numbers, colorbars (numbers + short label) and legends
    are kept (English)
  - data annotations that the paper captions reference are kept: NMF
    emission-line labels and SAM highest-risk region labels
  - multi-panel figures carry small (a), (b), ... panel letters
  - minimal white space between panels

All figures: detector 10264, prova1 (as in the paper).

Run from the project root:
    python scripts/make_submission_figures.py [--no-sam]
"""

import argparse
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from scipy.ndimage import zoom as ndi_zoom, distance_transform_edt
from scipy.signal import find_peaks

# Reuse the analysis code from 02_vulnerability.py (module name starts with
# a digit, hence importlib).
vuln = importlib.import_module("02_vulnerability")

OUT_DIR = "submission"
os.makedirs(OUT_DIR, exist_ok=True)

ROWS, COLS = vuln.ROWS, vuln.COLS
ELEMENTS   = vuln.ELEMENTS
RISK_CMAP  = vuln.RISK_CMAP
ZONE_COLORS = vuln.ZONE_COLORS
ZONE_LABELS = vuln.ZONE_LABELS
RULE_COLORS = vuln.RULE_COLORS
RISK_RULES  = vuln.RISK_RULES
KNOWN_PEAKS = vuln.KNOWN_PEAKS

EL_LABEL = {"Ca": "Ca", "Ti": "Ti", "Fe": "Fe", "Cu": "Cu", "Pb_La": "Pb Lα"}

_MAP_CB = dict(fraction=0.046 * ROWS / COLS, pad=0.015)

SAM_CHECKPOINT = os.path.join("models", "sam_vit_b_01ec64.pth")


# ─── style helpers ───────────────────────────────────────────────────────────

def chip(ax, s):
    """Panel letter as a white-on-black chip (readable on any map)."""
    ax.text(0.014, 0.965, f"({s})", transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left", color="white",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="black",
                      alpha=0.6, edgecolor="none"))


def letter(ax, s):
    """Panel letter for white-background axes."""
    ax.text(0.02, 0.97, f"({s})", transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left", color="black")


def blank(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def style_cb(cb, label=None):
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_edgecolor("#CCCCCC")
    if label:
        cb.set_label(label, fontsize=9)


def tight(fig):
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.015,
                                hspace=0.015)


def save(fig, name, formats=("png",), dpi=300):
    for ext in formats:
        path = os.path.join(OUT_DIR, f"{name}.{ext}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.02,
                    facecolor="white")
        print(f"  Saved: {path}")
    plt.close(fig)


# ─── figures ─────────────────────────────────────────────────────────────────

def fig_element_maps(norm_maps):
    """Five P99-normalised element maps in one row (paper Fig. 2)."""
    fig, axes = plt.subplots(1, 5, figsize=(16, 2.1), layout="constrained")
    tight(fig)

    for j, el in enumerate(ELEMENTS):
        im = axes[j].imshow(norm_maps[el], origin="upper", aspect="equal",
                            cmap="inferno", interpolation="bilinear",
                            vmin=0, vmax=1)
        chip(axes[j], "abcde"[j])
        blank(axes[j])

    cb = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.008)
    style_cb(cb, "normalised intensity")
    save(fig, "fig_element_maps", formats=("pdf",))


def fig_nmf_components(nmf_res):
    """NMF endmembers + abundance maps (paper Fig. 3)."""
    K, H = nmf_res["K"], nmf_res["H"]
    energy, maps = nmf_res["energy"], nmf_res["maps"]
    colors = plt.cm.Dark2(np.linspace(0, 0.85, K))

    fig = plt.figure(figsize=(11, 1.85 * K), layout="constrained")
    tight(fig)
    gs = GridSpec(K, 2, figure=fig, width_ratios=[1.55, 1])

    for k in range(K):
        ax_sp = fig.add_subplot(gs[k, 0])
        h_plot = vuln.bridge_masked(H[k], energy)   # no gaps: straight bridges
        ax_sp.fill_between(energy, h_plot, alpha=0.25, color=colors[k])
        ax_sp.plot(energy, h_plot, color=colors[k], linewidth=1.3)
        ax_sp.set_xlim(1, 15)
        ax_sp.grid(True, alpha=0.15)
        ax_sp.set_ylabel("Intensity (CPS)", fontsize=8)
        ax_sp.tick_params(labelsize=7)
        letter(ax_sp, "abcdefgh"[k])
        if k == K - 1:
            ax_sp.set_xlabel("Energy (keV)", fontsize=9)
        else:
            ax_sp.set_xticklabels([])

        # emission-line labels — referenced by the paper caption, kept
        peaks_idx, _ = find_peaks(H[k], height=np.max(H[k]) * 0.1,
                                  distance=5, prominence=np.max(H[k]) * 0.05)
        ax_sp.set_ylim(top=np.max(H[k]) * 1.22)
        for pi in peaks_idx:
            kev = energy[pi]
            best_el, best_d = "", 999
            for el, el_kev in KNOWN_PEAKS.items():
                if abs(kev - el_kev) < best_d:
                    best_d = abs(kev - el_kev)
                    best_el = el
            if best_d < 0.4:
                ax_sp.annotate(best_el, xy=(kev, H[k][pi]), fontsize=6.5,
                               fontweight="bold", ha="center", va="bottom",
                               xytext=(0, 3), textcoords="offset points")

        ax_map = fig.add_subplot(gs[k, 1])
        m = nmf_res["maps"][:, :, k]          # already P99-normalised
        im = ax_map.imshow(np.clip(m, 0, 1), origin="upper", aspect="equal",
                           cmap="hot", interpolation="bilinear",
                           vmin=0, vmax=1)
        blank(ax_map)
        cb = plt.colorbar(im, ax=ax_map, **_MAP_CB)
        style_cb(cb)

    save(fig, "fig_nmf_components")


def fig_cvi_map(cvi):
    """Continuous CVI + four-zone classification (paper Fig. 4)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.4), layout="constrained")
    tight(fig)

    im = axes[0].imshow(cvi, origin="upper", aspect="equal", cmap=RISK_CMAP,
                        interpolation="bilinear", vmin=0, vmax=1)
    chip(axes[0], "a")
    blank(axes[0])
    style_cb(plt.colorbar(im, ax=axes[0], **_MAP_CB), "CVI")

    zones = np.digitize(cvi, [0.25, 0.50, 0.75])
    zone_cmap = LinearSegmentedColormap.from_list("zone", ZONE_COLORS, N=4)
    axes[1].imshow(zones, origin="upper", aspect="equal", cmap=zone_cmap,
                   interpolation="nearest", vmin=0, vmax=3)
    chip(axes[1], "b")
    blank(axes[1])
    axes[1].legend(handles=[mpatches.Patch(color=c, label=l)
                            for c, l in zip(ZONE_COLORS, ZONE_LABELS)],
                   loc="lower left", fontsize=8, frameon=True,
                   facecolor="white", framealpha=0.9, edgecolor="#CCCCCC")

    save(fig, "fig_cvi_map")


def fig_individual_risks(vuln_res):
    """Five per-rule risk maps in one row."""
    fig, axes = plt.subplots(1, 5, figsize=(16, 2.1), layout="constrained")
    tight(fig)

    for i, rule in enumerate(RISK_RULES):
        r = vuln_res["risk_maps"][rule["id"]]
        im = axes[i].imshow(r, origin="upper", aspect="equal", cmap=RISK_CMAP,
                            interpolation="bilinear", vmin=0, vmax=1)
        chip(axes[i], "abcde"[i])
        blank(axes[i])

    cb = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.008)
    style_cb(cb, "risk score")
    save(fig, "fig_individual_risks")


def fig_dominant_risk(vuln_res):
    """Dominant degradation mechanism per pixel (CVI > 0.2)."""
    dom, cvi = vuln_res["dominant"], vuln_res["cvi"]
    dom_masked = np.where(cvi > 0.2, dom, -1)

    rgb = np.full((ROWS, COLS, 3), 0.95)
    for i, c in enumerate(RULE_COLORS):
        rgb[dom_masked == i] = [int(c[j:j + 2], 16) / 255 for j in (1, 3, 5)]

    fig, ax = plt.subplots(figsize=(9, 5.1), layout="constrained")
    tight(fig)
    ax.imshow(rgb, origin="upper", aspect="equal", interpolation="nearest")
    blank(ax)

    short = {"R1": "R1: thermal mismatch (Ti/Ca)",
             "R2": "R2: Cu green pigment degradation",
             "R3": "R3: lead white darkening",
             "R4": "R4: trapped moisture (Ti/Cu)",
             "R5": "R5: Fe-catalyzed Pb oxidation"}
    patches = [mpatches.Patch(color=RULE_COLORS[i], label=short[r["id"]])
               for i, r in enumerate(RISK_RULES)]
    patches.append(mpatches.Patch(color="#f0f0f0", label="low risk (CVI ≤ 0.2)"))
    ax.legend(handles=patches, loc="upper center", ncol=3, fontsize=7.5,
              bbox_to_anchor=(0.5, -0.01), frameon=False)

    save(fig, "fig_dominant_risk")


def fig_validation(val_res, cvi_1, cvi_2):
    """Cross-scan robustness: maps, distributions, agreement, SSIM."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), layout="constrained")
    tight(fig)

    for j, (cvi, s) in enumerate(zip([cvi_1, cvi_2], "ab")):
        im = axes[0, j].imshow(cvi, origin="upper", aspect="equal",
                               cmap=RISK_CMAP, interpolation="bilinear",
                               vmin=0, vmax=1)
        chip(axes[0, j], s)
        blank(axes[0, j])
    style_cb(plt.colorbar(im, ax=axes[0, 1], **_MAP_CB), "CVI")

    diff = np.abs(cvi_1 - cvi_2)
    im = axes[0, 2].imshow(diff, origin="upper", aspect="equal", cmap="gray_r",
                           interpolation="bilinear", vmin=0, vmax=0.3)
    chip(axes[0, 2], "c")
    blank(axes[0, 2])
    style_cb(plt.colorbar(im, ax=axes[0, 2], **_MAP_CB), "|ΔCVI|")

    ax = axes[1, 0]
    bins = np.linspace(0, 1, 51)
    ax.hist(cvi_1.ravel(), bins=bins, alpha=0.65, color="#0072B2",
            label="prova1", density=True)
    ax.hist(cvi_2.ravel(), bins=bins, alpha=0.65, color="#E69F00",
            label="prova2", density=True)
    ax.set_xlabel("CVI", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, alpha=0.15)
    ax.tick_params(labelsize=7.5)
    letter(ax, "d")

    ax = axes[1, 1]
    idx = np.random.default_rng(42).choice(ROWS * COLS, size=2000, replace=False)
    ax.scatter(cvi_1.ravel()[idx], cvi_2.ravel()[idx],
               alpha=0.3, s=8, color="#0072B2", edgecolors="none")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1)
    ax.set_xlabel("CVI prova1", fontsize=9)
    ax.set_ylabel("CVI prova2", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    ax.tick_params(labelsize=7.5)
    letter(ax, "e")

    ax = axes[1, 2]
    im = ax.imshow(val_res["ssim_map"], origin="upper", aspect="equal",
                   cmap="RdYlGn", interpolation="bilinear", vmin=0, vmax=1)
    chip(ax, "f")
    blank(ax)
    style_cb(plt.colorbar(im, ax=ax, **_MAP_CB), "SSIM")

    save(fig, "fig_validation")


def fig_wasserstein_per_rule(val_res):
    """Per-rule Wasserstein distance between the two scans."""
    fig, ax = plt.subplots(figsize=(7.5, 3.6), layout="constrained")
    tight(fig)

    ids = [r["id"] for r in RISK_RULES]
    w1_vals = [val_res["w1_per_rule"][r["id"]] for r in RISK_RULES]

    ax.bar(ids, w1_vals, color=RULE_COLORS, width=0.62)
    ax.set_ylabel("Wasserstein distance W₁", fontsize=10)
    ax.set_xlabel("Degradation rule", fontsize=10)
    ax.grid(True, alpha=0.15, axis="y")
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)
    ax.set_ylim(0, max(0.06, max(w1_vals) * 1.25))
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0.05, color="#555555", linestyle="--", alpha=0.8, linewidth=1,
               label="stability threshold (W₁ = 0.05)")
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")

    save(fig, "fig_wasserstein_per_rule")


def run_sam(norm_maps):
    """SAM automatic segmentation (paper parameters). Returns segments,
    per-region info and the false-color input."""
    rgb_input = np.clip(np.stack([norm_maps["Fe"], norm_maps["Cu"],
                                  norm_maps["Pb_La"]], axis=2), 0, 1)

    SCALE = 8
    rgb_up = np.clip(np.stack([ndi_zoom(rgb_input[:, :, c], SCALE, order=3)
                               for c in range(3)], axis=2), 0, 1)
    sam_input = (rgb_up * 255).astype(np.uint8)

    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
    sam.eval()
    gen = SamAutomaticMaskGenerator(
        model=sam, points_per_side=32, pred_iou_thresh=0.86,
        stability_score_thresh=0.92, min_mask_region_area=100,
    )
    print("  Running SAM (CPU, a few minutes)...")
    masks = sorted(gen.generate(sam_input), key=lambda m: m["area"],
                   reverse=True)
    print(f"  SAM proposed {len(masks)} masks")

    segments = np.zeros((ROWS, COLS), dtype=int)
    painted_masks = {}
    valid_id = 0
    for md in masks:
        hr = md["segmentation"]
        lr = hr.reshape(ROWS, SCALE, COLS, SCALE).mean(axis=(1, 3)) > 0.5
        # Larger segments win overlaps; a mask must still paint >= 5 native
        # pixels of its own (paper: segments < 5 px are discarded) --
        # measuring the raw mask instead lets 1-2 px leftovers through.
        painted = (segments == 0) & lr
        if painted.sum() < 5:
            continue
        valid_id += 1
        segments[painted] = valid_id
        painted_masks[valid_id] = painted

    unassigned = segments == 0
    if unassigned.any() and painted_masks:
        min_dist = np.full((ROWS, COLS), np.inf)
        for sid, mask in painted_masks.items():
            dist = distance_transform_edt(~mask)
            closer = dist < min_dist
            segments[unassigned & closer] = sid
            min_dist[unassigned & closer] = dist[unassigned & closer]

    segments = absorb_fragments(segments, min_px=5)

    # Renumber the surviving segments consecutively and rebuild their info
    info = []
    for new_id, sid in enumerate(np.unique(segments), start=1):
        mask = segments == sid
        info.append({"id": new_id, "mask": mask, "area_px": int(mask.sum())})
    for seg in info:
        segments[seg["mask"]] = seg["id"]

    print(f"  Valid segments: {len(info)}")
    return segments, info, rgb_input


def absorb_fragments(segments, min_px=5):
    """Merge connected components smaller than min_px into their majority
    neighbour segment (removes speckle left by overlap resolution)."""
    from scipy.ndimage import label as cc_label, binary_dilation
    changed = True
    while changed:
        changed = False
        for sid in np.unique(segments):
            cc, n_cc = cc_label(segments == sid)
            for c in range(1, n_cc + 1):
                comp = cc == c
                if comp.sum() >= min_px:
                    continue
                ring = binary_dilation(comp) & ~comp
                neigh = segments[ring]
                neigh = neigh[neigh != sid]
                if neigh.size:
                    segments[comp] = np.bincount(neigh).argmax()
                    changed = True
    return segments


def fig_sam_segmentation(segments, info, cvi):
    """SAM segments + per-region mean CVI with top regions labelled
    (paper Fig. 5)."""
    rng = np.random.default_rng(42)
    seg_rgb = np.zeros((ROWS, COLS, 3))
    for seg in info:
        seg_rgb[segments == seg["id"]] = rng.random(3) * 0.7 + 0.3

    region_cvi = np.zeros((ROWS, COLS))
    stats = []
    for seg in info:
        mask = segments == seg["id"]
        mean_cvi = float(cvi[mask].mean())
        region_cvi[mask] = mean_cvi
        stats.append({"id": seg["id"], "cvi": mean_cvi,
                      "area_pct": mask.sum() / (ROWS * COLS) * 100,
                      "mask": mask})
    stats.sort(key=lambda s: s["cvi"], reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.4), layout="constrained")
    tight(fig)

    axes[0].imshow(seg_rgb, origin="upper", aspect="equal",
                   interpolation="nearest")
    chip(axes[0], "a")
    blank(axes[0])

    im = axes[1].imshow(region_cvi, origin="upper", aspect="equal",
                        cmap=RISK_CMAP, interpolation="nearest",
                        vmin=0, vmax=1)
    chip(axes[1], "b")
    blank(axes[1])
    style_cb(plt.colorbar(im, ax=axes[1], **_MAP_CB), "mean CVI")

    # highest-risk regions labelled — referenced by the paper caption, kept
    for s in [s for s in stats if s["area_pct"] >= 1][:6]:
        ys, xs = np.where(s["mask"])
        axes[1].text(xs.mean(), ys.mean(), f"R{s['id']}\n{s['cvi']:.2f}",
                     ha="center", va="center", fontsize=6.5,
                     fontweight="bold", color="white",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="black",
                               alpha=0.55, edgecolor="none"))

    save(fig, "fig_sam_segmentation", formats=("png", "pdf"))
    return len(info)


# ─── captions ────────────────────────────────────────────────────────────────

def write_captions(nmf_res, n_segments, val_res):
    comp = ", ".join(
        f"({'abcdefgh'[k]}) component {k + 1}: {nmf_res['names'][k]}"
        for k in range(nmf_res["K"])
    )
    rules_txt = "; ".join(
        f"({'abcde'[i]}) {r['id']}: {r['name_plain']} "
        f"({(r['el_a'] if r['el_a'] == r['el_b'] else r['el_a'] + '/' + r['el_b']).replace('_La', '')}, w={r['w']:.2f})"
        for i, r in enumerate(RISK_RULES)
    )
    text = f"""SUBMISSION FIGURES — PANEL KEY AND CAPTIONS
All panels: detector 10264, dataset prova1 (scan grid 120x60 px), unless
stated otherwise. Figures carry no titles; panel letters (a), (b), ...
are printed on the panels and explained here.

fig_mop.png / fig_mop.jpg
    Photograph of the mockup canvas painting (140 x 75 mm) used in the
    study. Prepared with commercially grounded canvas (titanium white),
    lead white, red and yellow ochre, and copper-based green pigments.
    (Image supplied separately by the authors — not generated here.)

fig_element_maps.pdf
    Spatial element maps for the five elements of interest, each
    normalised to its 99th percentile (shared colorbar).
    (a) Ca Ka   (b) Ti Ka   (c) Fe Ka   (d) Cu Ka   (e) Pb La

fig_nmf_components.png
    NMF blind decomposition (K={nmf_res['K']}) of the prova1 spectra
    (CPS-normalised, 1-15 keV; acquisition-artifact bands - the Hg La/Lb
    lines and the scatter tail above 12.95 keV - are zeroed before
    factorization and drawn as straight bridging segments).
    Left column: spectral endmembers H_k with automatically identified
    emission lines annotated on the peaks. Right column: corresponding
    spatial abundance maps W_k, each normalised to its 99th percentile.
    {comp}

fig_cvi_map.png
    Chemical Vulnerability Index.
    (a) Continuous CVI on the green-yellow-orange-red scale aligned with
        the four risk zones.
    (b) Four-zone classification for direct conservation prioritisation
        (legend on the panel): low < 0.25 <= moderate < 0.50 <=
        elevated < 0.75 <= critical.

fig_individual_risks.png
    Individual degradation-rule risk maps (shared colorbar):
    {rules_txt}

fig_dominant_risk.png
    Dominant degradation mechanism per pixel, shown only where
    CVI > 0.2 (light gray = low risk). Rule colors per the legend below
    the map.

fig_validation.png
    Cross-scan robustness validation, prova1 vs prova2.
    (a) CVI of prova1 (scan 1)
    (b) CVI of prova2 (scan 2, 7-day interval); shared colorbar
    (c) absolute per-pixel difference |CVI1 - CVI2|
    (d) CVI distributions of the two scans
        (W1 = {val_res['w1_cvi']:.4f})
    (e) pixel-wise agreement, 2000-pixel subsample
        (Pearson r = {val_res['pearson']:.4f})
    (f) local structural similarity map, 7x7 window
        (mean SSIM = {val_res['ssim']:.4f})

fig_wasserstein_per_rule.png
    Per-rule Wasserstein distance W1 between the prova1 and prova2 risk
    maps; the dashed line marks the acceptable-stability threshold
    (W1 = 0.05). Bars colored by rule identity (same colors as
    fig_dominant_risk).

fig_sam_segmentation.png / fig_sam_segmentation.pdf
    SAM segmentation ({n_segments} automatic segments).
    (a) Segments identified by SAM (random colors).
    (b) Per-region mean CVI; the highest-risk segments are labelled with
        region ID and mean CVI value.
"""
    path = os.path.join(OUT_DIR, "captions.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  Saved: {path}")


# ─── main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-sam", action="store_true",
                        help="skip fig_sam_segmentation")
    args = parser.parse_args()

    print("Loading element maps (detector 10264)...")
    maps_p1 = vuln.load_element_maps("prova1")
    maps_p2 = vuln.load_element_maps("prova2")

    print("Computing CVI...")
    vuln_p1 = vuln.phase2_vulnerability(maps_p1, "prova1")
    vuln_p2 = vuln.phase2_vulnerability(maps_p2, "prova2")
    val_res = vuln.phase3_validation(
        vuln_p1["cvi"], vuln_p2["cvi"],
        vuln_p1["risk_maps"], vuln_p2["risk_maps"])

    print("NMF decomposition...")
    D_p1 = vuln.load_spectra(
        "prova1", vuln.resolve_dataset_dir("aurora-antico1-prova1"))
    if D_p1 is None:
        sys.exit("ERROR: raw prova1 spectra not available for NMF.")
    nmf_res = vuln.phase1_nmf(D_p1, "prova1")

    print("\nRendering submission figures...")
    fig_element_maps(vuln_p1["norm"])
    fig_nmf_components(nmf_res)
    fig_cvi_map(vuln_p1["cvi"])
    fig_individual_risks(vuln_p1)
    fig_dominant_risk(vuln_p1)
    fig_validation(val_res, vuln_p1["cvi"], vuln_p2["cvi"])
    fig_wasserstein_per_rule(val_res)

    n_segments = 0
    if not args.no_sam:
        if not os.path.exists(SAM_CHECKPOINT):
            print(f"  WARNING: {SAM_CHECKPOINT} missing — "
                  "fig_sam_segmentation skipped.")
        else:
            segments, info, _ = run_sam(vuln_p1["norm"])
            n_segments = fig_sam_segmentation(segments, info, vuln_p1["cvi"])

    write_captions(nmf_res, n_segments, val_res)

    print(f"\nDone. Output: {os.path.abspath(OUT_DIR)}{os.sep}")
    print("NOTE: fig_mop.png / fig_mop.jpg are supplied separately "
          "(photograph of the mockup).")
