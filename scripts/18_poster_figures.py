"""
18_poster_figures.py
===============================================================================
Poster-grade versions of the figures POSTER_PLAN.md puts on the A0 poster.

WHY A SEPARATE SCRIPT
    The paper/talk figures (scripts 07, 11, 14, 17 and make_slides.py) are
    sized for a two-column page and a 16:9 slide: 11 pt labels on a 7-15 in
    canvas.  Dropped into a 237 mm poster column those labels come out at
    12-16 pt on the printed sheet, i.e. unreadable from 1.5 m.  This script
    re-renders the same content at TRUE POSTER SIZE, so one matplotlib point
    is one point on the printed A0.  Nothing here recomputes physics that the
    other scripts own: the fits are the imported model functions of
    07_geometry_fit.py refitted on the same CSV, the fusion maps go through
    14_fusion_showcase.py's own helpers, and every number is checked against
    the committed .txt reports before anything is written (see --no-verify).

    The original scripts are untouched, so the paper figures cannot drift.

GEOMETRY
    A0 portrait 841 x 1189 mm, 40 mm margins, 3 columns, 25 mm gutters
        column width = (841 - 2*40 - 2*25) / 3 = 237.0 mm exactly
    Every figure is emitted exactly one column wide, so it drops onto the
    poster at 100 % with no rescaling.  Override with --width-mm if the
    conference turns out to want a different sheet.

OUTPUT (results/poster_figs/)
    <name>.pdf   vector, fonts embedded as TrueType (PowerPoint / LaTeX)
    <name>.png   600 dpi raster at final size (Canva and other web editors)
    MANIFEST.txt exact placement size of every file, in mm and px

    The map panels are 60 x 120 measured pixels: 600 dpi is honest headroom
    for print, not extra information.  interpolation="nearest" is kept on
    purpose -- smoothing them would invent detail the scan does not have.

Input : results/registration/overlap_ratios.csv         (script 08)
        results/registration/positioning_sensitivity.csv (script 11)
        results/detector_diff/handoff2_ratio_curve.csv  (script 07)
        results/detector_diff/geometry_fit.txt          (script 07, checked)
        results/detector_diff/canvas_topography*.npy|txt (script 17)
        results/vulnerability_mapping/ablation_cube_*.npy
        xrf-denoise/data/processed/fused_*.npy, fused_heldout_px.json

Run from the project root:
    python scripts/18_poster_figures.py                  # everything
    python scripts/18_poster_figures.py --only hero      # one block
    python scripts/18_poster_figures.py --list
"""

import argparse
import csv
import importlib
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter   # noqa: E402

DIFF_DIR = os.path.join("results", "detector_diff")
REG_DIR = os.path.join("results", "registration")
OUT_DIR = os.path.join("results", "poster_figs")

MM_PER_IN = 25.4
DEFAULT_WIDTH_MM = 237.0        # one A0 column, see GEOMETRY above
DEFAULT_DPI = 600

BLUE = "#1f77b4"
ORANGE = "#d95f02"
GRAY = "#555555"

# Latin Modern Roman, the face the LaTeX poster sets its text in (lmodern),
# so the figures use the same one.  It is not a system font; register the
# OTF files straight from a TeX installation, or from a directory given on
# the command line.
LM_FAMILY = "Latin Modern Roman"
LM_CANDIDATE_DIRS = [
    os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "MiKTeX",
                 "fonts", "opentype", "public", "lm"),
    os.path.join("C:\\", "Program Files", "MiKTeX", "fonts", "opentype",
                 "public", "lm"),
    "/usr/share/texmf/fonts/opentype/public/lm",
    "/usr/share/texlive/texmf-dist/fonts/opentype/public/lm",
    os.path.join("poster", "fonts", "latin-modern"),
]


def register_latin_modern(font_dir=None):
    """Add the Latin Modern Roman 10 pt faces to matplotlib for this process.

    Returns the directory they came from, or None if none was found.  Only
    the 10 pt optical size is registered: every size shares the family name
    and matplotlib cannot tell them apart, so registering all of them makes
    the pick arbitrary.
    """
    import glob
    from matplotlib import font_manager as fm
    dirs = ([font_dir] if font_dir else []) + LM_CANDIDATE_DIRS
    for d in dirs:
        files = sorted(glob.glob(os.path.join(d, "lmroman10-*.otf")))
        if not files:
            continue
        for f in files:
            fm.fontManager.addfont(f)
        return d
    return None


def poster_rc(font):
    """rcParams for figures rendered at final printed size.

    Sizes are literal poster points: body text on the poster is 28-32 pt
    (POSTER_PLAN section 4), so axis labels at 24 pt and ticks at 20 pt sit
    just under it and clear the >= 20 pt floor for captions.
    """
    return {
        "font.family": font,
        # Computer Modern for mathtext: Latin Modern is its descendant, so a
        # $\psi_1$ or a $\theta$ looks like the poster's own math, and the
        # Greek that Latin Modern Roman lacks comes from the bundled cm fonts
        "mathtext.fontset": "cm",
        "font.size": 20,
        "axes.titlesize": 26,
        "axes.labelsize": 24,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 19,
        "figure.titlesize": 28,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.linewidth": 1.8,
        "axes.grid": False,
        "grid.linewidth": 1.2,
        "lines.linewidth": 3.2,
        "lines.markersize": 11,
        "errorbar.capsize": 6,
        "xtick.major.size": 8,
        "ytick.major.size": 8,
        "xtick.major.width": 1.8,
        "ytick.major.width": 1.8,
        "xtick.minor.size": 4,
        "ytick.minor.size": 4,
        # embed real (selectable, re-editable) TrueType instead of Type-3
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }


MANIFEST = []

# Layout knobs set from the command line.  The defaults reproduce the
# three-column set byte for byte; the two-column poster changes them (see
# poster/main.tex for the exact invocation).
WIDTH_OVERRIDE = {}             # block name -> width in mm, else --width-mm
HERO_PANELS = "stacked"         # "stacked" (one column) or "side" (two)
POSITIONING_ASPECT = 0.46       # canvas height / width before the crop


def block_width(name, width_mm):
    """Width this block is rendered at: an explicit override, else the
    column width."""
    return WIDTH_OVERRIDE.get(name, width_mm)


def _pdf_size_mm(path):
    """Width and height of a one-page PDF, or None if pypdf is not around."""
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            return None
    box = PdfReader(path).pages[0].mediabox
    return float(box.width) * 25.4 / 72.0, float(box.height) * 25.4 / 72.0


def render_to_width(draw, name, width_mm, dpi, h_ratio, tries=6):
    """Render ``draw(figsize) -> fig`` trimmed to its own ink, exactly
    ``width_mm`` wide.

    Saving with ``bbox_inches="tight"`` crops whatever white the layout
    engine left over -- which is what an aspect-equal image or a shrunken
    axes box produces, and what the poster cannot afford.  The emitted width
    is then the *drawing*, not the canvas, so solve for the canvas that trims
    to one column: the trim is nearly proportional, so a few passes land
    inside a fraction of a millimetre.  ``h_ratio`` only has to be generous;
    the crop takes back any excess height.

    The loop measures the PDF, not the PNG: the two backends measure text
    extents differently (a legend hanging off an axes edge came out 3 mm
    wider in vector than in raster), and the PDF is what the LaTeX poster
    places at the column width.  Falls back to the PNG without pypdf.
    """
    from PIL import Image
    os.makedirs(OUT_DIR, exist_ok=True)
    png = os.path.join(OUT_DIR, name + ".png")
    pdf = os.path.join(OUT_DIR, name + ".pdf")
    kw = dict(bbox_inches="tight", pad_inches=0.01)

    fig_w = width_mm / MM_PER_IN
    got = None
    for _ in range(tries):
        fig = draw((fig_w, fig_w * h_ratio))
        fig.savefig(pdf, **kw)
        plt.close(fig)
        got = _pdf_size_mm(pdf)
        if got is None:
            fig = draw((fig_w, fig_w * h_ratio))
            fig.savefig(png, dpi=dpi, **kw)
            plt.close(fig)
            with Image.open(png) as im:
                got = (im.size[0] / dpi * MM_PER_IN,
                       im.size[1] / dpi * MM_PER_IN)
        if abs(got[0] - width_mm) < 0.25:
            break
        fig_w *= width_mm / got[0]

    if abs(got[0] - width_mm) >= 0.25:
        print("  WARNING: %s did not converge to %.0f mm (got %.1f); "
              "something hangs past the axes and the crop is not "
              "proportional -- fix the figure, do not place this one"
              % (name, width_mm, got[0]))

    fig = draw((fig_w, fig_w * h_ratio))
    fig.savefig(png, dpi=dpi, **kw)
    plt.close(fig)
    with Image.open(png) as im:
        w_px, h_px = im.size

    MANIFEST.append({
        "name": name,
        "w_mm": got[0], "h_mm": got[1],          # the PDF, the placed file
        "w_px": int(w_px), "h_px": int(h_px), "dpi": dpi,
    })
    print("  %.1f x %.1f mm  (aspect %.3f)   png %d x %d px"
          % (got[0], got[1], got[1] / got[0], w_px, h_px))
    print("  saved: " + pdf)
    print("  saved: " + png)


def save(fig, name, dpi):
    """Write the vector and the raster copy, and record the placement size."""
    os.makedirs(OUT_DIR, exist_ok=True)
    w_in, h_in = fig.get_size_inches()
    paths = []
    for ext in ("pdf", "png"):
        p = os.path.join(OUT_DIR, name + "." + ext)
        fig.savefig(p, dpi=dpi)
        paths.append(p)
    plt.close(fig)
    MANIFEST.append({
        "name": name,
        "w_mm": w_in * MM_PER_IN,
        "h_mm": h_in * MM_PER_IN,
        "w_px": int(round(w_in * dpi)),
        "h_px": int(round(h_in * dpi)),
        "dpi": dpi,
    })
    for p in paths:
        print("  saved: " + p)


# ---------------------------------------------------------------------------
# report parsing -- the committed .txt files are the reference values that
# every recomputation below is checked against
# ---------------------------------------------------------------------------
def parse_geometry_fit_txt():
    path = os.path.join(DIFF_DIR, "geometry_fit.txt")
    keys = {
        "k": r"^k\s*=",
        "d_abs": r"^d_abs\s*=",
        "t_si1": r"^t_Si1\s*=",
        "s": r"^s \(lever arm\)\s*=",
        "ec": r"^Ec\s*=",
        "c": r"^c \(offset\)\s*=",
    }
    out = {}
    num = r"([-+]?[0-9]*\.?[0-9]+)"
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            for key, pat in keys.items():
                if re.match(pat, line):
                    m = re.search(pat[1:] + r"\s*" + num, line)
                    if m:
                        out[key] = float(m.group(1))
    missing = set(keys) - set(out)
    if missing:
        sys.exit("ERROR: could not parse %s from %s"
                 % (sorted(missing), path))
    return out


def parse_fusion_showcase_txt():
    """Held-out SNR/gain per (line, variant) from the script-14 report."""
    path = os.path.join(DIFF_DIR, "fusion_showcase.txt")
    out, in_held = {}, False
    with open(path) as f:
        for raw in f:
            line = raw.rstrip()
            if line.startswith("[heldout_px]"):
                in_held = True
                continue
            if line.startswith("[") and not line.startswith("[heldout_px]"):
                in_held = False
            if not in_held:
                continue
            parts = line.split()
            if len(parts) >= 3 and parts[1] in ("sum", "learned"):
                out[(parts[0], parts[1])] = float(parts[2])
    if not out:
        sys.exit("ERROR: no [heldout_px] rows parsed from " + path)
    return out


def parse_topography_txt():
    path = os.path.join(DIFF_DIR, "canvas_topography.txt")
    txt = open(path).read()
    r = re.search(r"cross-scan reproducibility of theta: r = ([0-9.]+)", txt)
    rms = re.search(r"RMS reproducible topographic signal: ([0-9.]+)", txt)
    if not (r and rms):
        sys.exit("ERROR: could not parse r / RMS from " + path)
    return float(r.group(1)), float(rms.group(1))


def read_overlap_rows():
    rows = []
    with open(os.path.join(REG_DIR, "overlap_ratios.csv"), newline="") as f:
        for row in csv.DictReader(f):
            if row["reliable"] == "True":
                rows.append(row)
    return rows


def check(label, got, want, tol):
    ok = abs(got - want) <= tol
    print("  [%s] %-34s got %.4f  ref %.4f  tol %.4f"
          % ("OK " if ok else "FAIL", label, got, want, tol))
    return ok


# ---------------------------------------------------------------------------
# block 1 -- the idea (geometry schematic)
# ---------------------------------------------------------------------------
SCHEMATIC_SCALE = 2.0           # 11 pt talk labels -> 22 pt on the poster

# The talk draws this square-ish (aspect 0.93).  At one A0 column that is
# 36 cm of height for six shapes, a fifth of the whole page.  The axes are
# aspect-equal, so the way to make it wide and short without distorting an
# angle is to change the drawn geometry: a shallower detector angle lowers
# the two heads, and a longer beam arrow spends the width that buys.
SCHEMATIC_WIDE = dict(det_angle_deg=38.0, beam_x0=-4.5,
                      xlim=(-4.9, 2.4), ylim=(-2.3, 2.3),
                      labels=("upper head", "lower head"),
                      psi_labels=(r"$\psi_1$", r"$\psi_2$"),
                      theta_label=r"$\theta$")
SCHEMATIC_ASPECT = 4.6 / 7.3


def build_schematic(width_mm, dpi, font, verify):
    print("")
    print("[block 1] idea schematic")
    sys.path.insert(0, os.path.join(ROOT, "presentation"))
    slides = importlib.import_module("make_slides")
    from PIL import Image

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "block1_idea_schematic.png")
    pdf = path[:-4] + ".pdf"

    # make_schematic saves with bbox_inches="tight", so the emitted width is
    # the drawing, not the canvas.  Solve for the canvas that trims to exactly
    # one column: the trim is nearly proportional, so a couple of passes get
    # inside a few tenths of a millimetre.
    width_mm = block_width("idea", width_mm)
    w_in = width_mm / MM_PER_IN
    fig_w = w_in
    with plt.rc_context(poster_rc(font)):
        for _ in range(5):
            slides.make_schematic(pdf, scale=SCHEMATIC_SCALE, dpi=dpi,
                                  figsize=(fig_w, fig_w * SCHEMATIC_ASPECT),
                                  **SCHEMATIC_WIDE)
            got = _pdf_size_mm(pdf)
            if got is None:
                slides.make_schematic(path, scale=SCHEMATIC_SCALE, dpi=dpi,
                                      figsize=(fig_w,
                                               fig_w * SCHEMATIC_ASPECT),
                                      **SCHEMATIC_WIDE)
                with Image.open(path) as im:
                    got = (im.size[0] / dpi * MM_PER_IN,
                           im.size[1] / dpi * MM_PER_IN)
            if abs(got[0] - width_mm) < 0.25:
                break
            fig_w *= width_mm / got[0]
        slides.make_schematic(path, scale=SCHEMATIC_SCALE, dpi=dpi,
                              figsize=(fig_w, fig_w * SCHEMATIC_ASPECT),
                              **SCHEMATIC_WIDE)
    with Image.open(path) as im:
        w_px, h_px = im.size

    MANIFEST.append({
        "name": "block1_idea_schematic",
        "w_mm": got[0], "h_mm": got[1],
        "w_px": w_px, "h_px": h_px, "dpi": dpi,
    })
    print("  %.1f x %.1f mm  (aspect %.3f)   png %d x %d px"
          % (got[0], got[1], got[1] / got[0], w_px, h_px))
    print("  saved: " + pdf)
    print("  saved: " + path)
    return True


# ---------------------------------------------------------------------------
# block 3 -- hero: the two-stage decomposition, panels stacked for a column
# ---------------------------------------------------------------------------
def build_hero(width_mm, dpi, font, verify):
    print("\n[block 3] hero -- two-stage decomposition")
    from scipy.optimize import curve_fit
    geo = importlib.import_module("07_geometry_fit")

    rows = read_overlap_rows()
    els = [r["element"] for r in rows]
    E = np.array([float(r["kev"]) for r in rows])
    rf = np.array([float(r["R_frontal_overlap"]) for r in rows])
    rf_err = np.array([float(r["sig_frontal_overlap"]) for r in rows])
    rr = np.array([float(r["R_ruotato"]) for r in rows])
    sr = np.array([float(r["sig_ruotato"]) for r in rows])
    delta = rr / rf - 1.0
    delta_err = (rr / rf) * np.hypot(sr / rr, rf_err / rf)

    # identical calls to 07_geometry_fit.py -- same models, same data, so the
    # poster cannot quote a different absorber than the paper
    p1, _ = curve_fit(geo.det_ratio, E, rf, p0=(0.9, 800.0, 300.0),
                      sigma=rf_err, absolute_sigma=True,
                      bounds=([0.1, 0.0, 50.0], [5.0, 3000.0, 1500.0]),
                      maxfev=20000)
    p2, _ = curve_fit(geo.tilt_shift, E, delta, p0=(0.3, 6.0, -0.02),
                      sigma=delta_err, absolute_sigma=True,
                      bounds=([-3.0, 1.0, -0.2], [3.0, 20.0, 0.2]),
                      maxfev=20000)

    ok = True
    if verify:
        ref = parse_geometry_fit_txt()
        ok &= check("stage1 k", p1[0], ref["k"], 5e-3)
        ok &= check("stage1 d_abs (um Be-eq)", p1[1], ref["d_abs"], 1.0)
        ok &= check("stage1 t_Si1 (um)", p1[2], ref["t_si1"], 1.0)
        ok &= check("stage2 s (deg/deg)", p2[0], ref["s"], 5e-3)
        ok &= check("stage2 Ec (keV)", p2[1], ref["ec"], 5e-2)
        ok &= check("stage2 c (offset)", p2[2], ref["c"], 5e-4)

    Ef = np.linspace(3.0, 16.0, 400)
    gp_mean, gp_sig = geo.gp_regress(E, delta, delta_err, Ef)[:2]

    # No legend and no fitted-parameter boxes: at one A0 column they cost more
    # height than the curves themselves, and every number they carried is
    # already a bullet next to the figure.  The caption names the three
    # elements instead.  The exported (model x GP residual) curve is dropped
    # for the same reason -- it overplots the model to within a line width.
    side = HERO_PANELS == "side"

    def draw(figsize):
        # One column stacks the two stages and shares the energy axis; a
        # two-column poster puts them side by side, which is 16 cm shorter
        # at 38 cm and lets the bullets sit next to a figure they can see.
        if side:
            fig, (axa, axb) = plt.subplots(1, 2, figsize=figsize,
                                           layout="constrained")
        else:
            fig, (axa, axb) = plt.subplots(2, 1, figsize=figsize,
                                           sharex=True, layout="constrained")

        # ---- stage 1 -----------------------------------------------------
        axa.errorbar(E, rf, yerr=rf_err, fmt="o", color=BLUE, zorder=4)
        axa.plot(Ef, geo.det_ratio(Ef, *p1), "-", color=ORANGE, zorder=3)
        for x, y, name in zip(E, rf, els):
            axa.annotate(name, (x, y), textcoords="offset points",
                         xytext=(8, 10), fontsize=20, color="0.2")
        axa.axhline(1.0, color="0.7", lw=1.4, ls="-", zorder=1)
        axa.set_yscale("log")
        axa.set_ylim(0.48, 8.5)
        # plain numbers at 0.5 / 1 / 2 / 5: a lone "10^0" tells nobody that
        # Ca sits at 5.8 and Pb L-gamma at 0.63
        axa.set_yticks([0.5, 1, 2, 5])
        axa.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
        axa.yaxis.set_minor_formatter(NullFormatter())
        if side:
            axa.set_xlabel("emission-line energy (keV)")
        else:
            axa.tick_params(labelbottom=False)
        axa.set_ylabel("R = upper / lower head", fontsize=21)
        axa.set_title("Stage 1: detector response")
        axa.grid(alpha=0.3)

        # ---- stage 2 -----------------------------------------------------
        axb.fill_between(Ef, 100 * (gp_mean - 2 * gp_sig),
                         100 * (gp_mean + 2 * gp_sig), color="0.86", zorder=1)
        axb.errorbar(E, 100 * delta, yerr=100 * delta_err, fmt="s",
                     color=BLUE, zorder=4)
        axb.plot(Ef, 100 * geo.tilt_shift(Ef, *p2), "-", color=ORANGE,
                 zorder=3)
        axb.axhline(0, color="0.4", lw=1.4, zorder=2)
        for x, y, name in zip(E, 100 * delta, els):
            axb.annotate(name, (x, y), textcoords="offset points",
                         xytext=(8, 10), fontsize=20, color="0.2")
        axb.set_xlabel("emission-line energy (keV)")
        axb.set_ylabel("tilt shift of R (%)", fontsize=21)
        axb.set_title("Stage 2: acquisition geometry")
        axb.grid(alpha=0.3)
        return fig

    with plt.rc_context(poster_rc(font)):
        render_to_width(draw, "block3_hero_geometry_fit",
                        block_width("hero", width_mm), dpi,
                        0.42 if side else 0.80)
    return ok


# ---------------------------------------------------------------------------
# block 4 -- learned fusion, Pb Ll block only (the 4x3 grid is too dense)
# ---------------------------------------------------------------------------
def build_fusion(width_mm, dpi, font, verify):
    print("\n[block 4] learned fusion (Pb Ll)")
    m14 = importlib.import_module("14_fusion_showcase")
    key, label = "PbLl", "Pb L$\\ell$"

    raw = {}
    for ds in m14.DATASETS:
        for det in m14.DETS:
            p = os.path.join(m14.CUBE_CACHE,
                             "ablation_cube_%s_%s.npy" % (ds, det))
            print("  [%s/%s] extracting line map..." % (ds, det))
            raw[(ds, det)] = m14.extract_line_maps(np.load(p), [key])

    fused = {}
    for ds in m14.DATASETS:
        p = os.path.join(m14.FUSED_DIR, "fused_%s.npy" % ds)
        if not os.path.exists(p):
            sys.exit("ERROR: %s missing -- run script 14 first." % p)
        print("  [%s/fused] extracting line map..." % ds)
        fused[ds] = m14.extract_line_maps(np.load(p).astype(np.float64), [key])

    # evaluation pixels: identical to 09_fusion.py / 14_fusion_showcase.py
    rc = np.add.outer(np.arange(m14.ROWS), np.arange(m14.COLS))
    mask_B = (rc % 2) == 1
    with open(os.path.join(m14.FUSED_DIR, "fused_heldout_px.json")) as f:
        rec = json.load(f)
    heldout = np.zeros(m14.ROWS * m14.COLS, dtype=bool)
    heldout[np.asarray(rec["val_indices"], dtype=int)] = True
    px = mask_B & heldout.reshape(m14.ROWS, m14.COLS)

    v_sum = {ds: raw[(ds, "10264")][key] + raw[(ds, "19511")][key]
             for ds in m14.DATASETS}
    v_lrn = {ds: fused[ds][key] for ds in m14.DATASETS}
    snr_sum = m14.snr(v_sum["prova1"], v_sum["prova2"], px)
    snr_lrn = m14.snr(v_lrn["prova1"], v_lrn["prova2"], px)
    gain = 100.0 * (snr_lrn / snr_sum - 1.0)
    cv = m14.cv_ratio_vs_sum(v_lrn, v_sum, px)

    ok = True
    if verify:
        ref = parse_fusion_showcase_txt()
        ok &= check("PbLl held-out SNR, summed", snr_sum, ref[(key, "sum")],
                    0.05)
        ok &= check("PbLl held-out SNR, learned", snr_lrn,
                    ref[(key, "learned")], 0.05)
    print("  gain %+.1f%%   spatial-contrast ratio %.3f" % (gain, cv))

    # display-only gain match (SNR itself is scale-invariant)
    g = (np.mean(v_sum["prova1"] + v_sum["prova2"])
         / np.mean(v_lrn["prova1"] + v_lrn["prova2"]))
    disp = {"sum": v_sum,
            "learned": {ds: g * v_lrn[ds] for ds in m14.DATASETS}}
    noise = {n: (m["prova1"] - m["prova2"]) / np.sqrt(2.0)
             for n, m in disp.items()}

    pool_sig = np.concatenate([disp[n][ds].ravel()
                               for n in ("sum", "learned")
                               for ds in m14.DATASETS])
    vmax = np.percentile(pool_sig, 99)
    nlim = np.percentile(np.concatenate(
        [np.abs(noise[n]).ravel() for n in ("sum", "learned")]), 99)

    # The six maps are the argument; everything else is scaffolding.
    #
    # Constrained layout cannot size a grid of aspect-equal images: it fixes
    # the cells first, the images then shrink inside them, and the surplus
    # comes back as white both between the rows and beside every map.  So the
    # geometry is solved instead of negotiated.  Margins are absolute inches
    # (the fonts are fixed poster points, they do not scale with the canvas),
    # the cell width follows from what is left, and the canvas height is
    # whatever makes a 2:1 image exactly fill its cell.
    WR = [1.0, 1.0, 0.05, 0.30, 1.0, 0.05]   # map map cbar spacer map cbar
    WSPACE, HSPACE = 0.10, 0.07
    IMG_ASPECT = float(m14.ROWS) / float(m14.COLS)

    def geometry(w_in):
        left = 0.69       # "learned / (N2N)", two rotated lines at 20 pt
        right = 0.49      # "-100" tick labels on the noise colour bar
        top = 0.40        # column titles at 19 pt, nothing above them
        bottom = 0.03
        units = sum(WR) + 5 * WSPACE * (sum(WR) / len(WR))
        cell = (w_in - left - right) / units
        img_h = IMG_ASPECT * cell
        h_in = (2.0 + HSPACE) * img_h + top + bottom
        return h_in, dict(left=left / w_in, right=1.0 - right / w_in,
                          top=1.0 - top / h_in, bottom=bottom / h_in)

    def draw(figsize):
        w_in = figsize[0]
        h_in, margins = geometry(w_in)
        fig = plt.figure(figsize=(w_in, h_in))
        gs = fig.add_gridspec(2, 6, width_ratios=WR, wspace=WSPACE,
                              hspace=HSPACE, **margins)
        titles = ["scan 1", "scan 2", "cross-scan noise"]
        ims = imn = None
        for row, name in enumerate(("sum", "learned")):
            axes_row = [fig.add_subplot(gs[row, c]) for c in (0, 1, 4)]
            for ax, ds in zip(axes_row[:2], m14.DATASETS):
                ims = ax.imshow(disp[name][ds], cmap="inferno", vmin=0,
                                vmax=vmax, interpolation="nearest")
            imn = axes_row[2].imshow(noise[name], cmap="RdBu_r", vmin=-nlim,
                                     vmax=nlim, interpolation="nearest")
            for ax in axes_row:
                ax.set_xticks([])
                ax.set_yticks([])
            if row == 0:
                for ax, t in zip(axes_row, titles):
                    ax.set_title(t, fontsize=19, pad=5)
            axes_row[0].set_ylabel("summed" if name == "sum"
                                   else "learned\n(N2N)", fontsize=20)
            txt = "SNR %.1f" % (snr_sum if name == "sum" else snr_lrn)
            if name == "learned":
                txt += "  (%+.0f%%)" % gain
            axes_row[2].text(
                0.97, 0.93, txt, transform=axes_row[2].transAxes,
                ha="right", va="top", fontsize=20,
                fontweight="bold" if name == "learned" else "normal",
                color="0.12",
                bbox=dict(facecolor="white", alpha=0.9, edgecolor="0.6",
                          boxstyle="round,pad=0.25"))
        cb = fig.colorbar(ims, cax=fig.add_subplot(gs[:, 2]))
        cb.ax.tick_params(labelsize=14)   # "net counts" lives in the caption
        cb = fig.colorbar(imn, cax=fig.add_subplot(gs[:, 5]))
        cb.ax.tick_params(labelsize=14)
        return fig

    with plt.rc_context(poster_rc(font)):
        render_to_width(draw, "block4_fusion_pbll",
                        block_width("fusion", width_mm), dpi, 0.34)
    return ok


# ---------------------------------------------------------------------------
# block 5 -- canvas topography, combined panel only
# ---------------------------------------------------------------------------
def build_topography(width_mm, dpi, font, verify):
    print("\n[block 5] canvas topography")
    comb = np.load(os.path.join(DIFF_DIR, "canvas_topography_combined.npy"))
    r_cross, rms_signal = parse_topography_txt()
    print("  cross-scan r = %.2f, RMS = %.1f deg (quoted in the poster text)"
          % (r_cross, rms_signal))
    lim = np.nanpercentile(np.abs(comb), 98)

    # The colour bar is an inset in axes coordinates, so it is exactly as tall
    # as the map: fig.colorbar() sizes itself to the axes *box*, which for an
    # aspect-equal image is taller than the image it describes.
    def draw(figsize):
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(comb, origin="upper", aspect="equal", cmap="RdBu_r",
                       vmin=-lim, vmax=lim, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        cax = ax.inset_axes([1.015, 0.0, 0.022, 1.0])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("local slope along tilt axis (deg)", fontsize=21)
        cb.ax.tick_params(labelsize=18)
        return fig

    with plt.rc_context(poster_rc(font)):
        render_to_width(draw, "block5_canvas_topography",
                        block_width("topography", width_mm), dpi, 0.62)
    return True


# ---------------------------------------------------------------------------
# block 6 -- positioning error budget (optional block)
# ---------------------------------------------------------------------------
def build_positioning(width_mm, dpi, font, verify):
    print("\n[block 6] positioning sensitivity")
    rel = []
    with open(os.path.join(REG_DIR, "positioning_sensitivity.csv"),
              newline="") as f:
        for row in csv.DictReader(f):
            if row["reliable"] == "True":
                rel.append(row)
    e = np.array([float(r["kev"]) for r in rel])
    order = np.argsort(e)
    labels = [rel[i]["element"] for i in order]
    x = np.arange(len(order))
    # repeatability floor, same definition as script 11
    base = np.array([float(rel[i]["baseline_pct_sum"]) for i in order])
    floor = np.sqrt(np.mean(base ** 2)) / 7.7

    # The title said what the caption says, so it goes; the legend loses its
    # box and spreads along one line, which is the only text the panel needs.
    def draw(figsize):
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        for v, marker, color, lab in (
                ("10264", "o", BLUE, "upper head"),
                ("19511", "s", ORANGE, "lower head"),
                ("sum", "D", GRAY, "summed map")):
            y = np.array([float(rel[i]["per_deg_" + v]) for i in order])
            yerr = np.array([float(rel[i]["per_deg_sig_" + v])
                             for i in order])
            ax.errorbar(x, y, yerr=yerr, fmt=marker + "-", color=color,
                        label=lab, lw=2.4, ms=10)
        ax.axhspan(-floor, floor, color="0.87", zorder=0,
                   label="repeatability floor")
        ax.axhline(0, color="0.4", lw=1.4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("emission line (increasing energy)")
        # shorter than the axes is tall, so the tight crop cannot clip it
        ax.set_ylabel("change per degree (%)")
        # Legend as a 2 x 2 block inside the axes, with headroom above the Ti
        # point.  One row of four was wider than the axes in Arial; a legend
        # that hangs past the axes edge makes the tight crop non-proportional
        # and the width loop cannot converge.
        ax.set_ylim(-0.36, 0.90)
        ax.legend(loc="upper right", ncol=2, frameon=False, fontsize=18,
                  columnspacing=1.2, handletextpad=0.5, borderaxespad=0.3)
        ax.grid(alpha=0.3)
        return fig

    with plt.rc_context(poster_rc(font)):
        render_to_width(draw, "block6_positioning_sensitivity",
                        block_width("positioning", width_mm), dpi,
                        POSITIONING_ASPECT)
    return True


# ---------------------------------------------------------------------------
# table -- per-line ratios as a booktabs fragment for the LaTeX poster
# ---------------------------------------------------------------------------
LATEX_LINE = {
    "Ca": r"Ca K$\alpha$", "Ti": r"Ti K$\alpha$", "Fe": r"Fe K$\alpha$",
    "Cu": r"Cu K$\alpha$", "PbLl": r"Pb L$\ell$", "PbLa": r"Pb L$\alpha$",
    "PbLb": r"Pb L$\beta$", "PbLg": r"Pb L$\gamma$",
}


def build_table(width_mm, dpi, font, verify):
    """Emit ratio_table.tex so the poster never hand-copies a number."""
    print("\n[table] per-line ratio table")
    rows = read_overlap_rows()
    out = [
        "% generated by scripts/18_poster_figures.py -- do not edit by hand",
        "% source: results/registration/overlap_ratios.csv (script 08)",
        r"% tabular* + \extracolsep{\fill}: set to the full column width, so the",
        r"% table reads as a block instead of a narrow island.",
        r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}} l r r r}",
        r"\toprule",
        r"\textbf{Line} & \textbf{keV} & $\mathbf{R}$ "
        r"& \textbf{tilt shift} \\",
        r"\midrule",
    ]
    for r in rows:
        el = r["element"]
        tilt = float(r["tilt_overlap_pct"])
        sig = float(r["significance_overlap_sigma"])
        out.append("%s & %.2f & %.3f & $%+.1f$\\,\\%% (%.0f$\\sigma$) \\\\"
                   % (LATEX_LINE.get(el, el), float(r["kev"]),
                      float(r["R_frontal_overlap"]), tilt, sig))
    out += [r"\bottomrule", r"\end{tabular*}"]

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "ratio_table.tex")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    print("  %d reliable lines, Ca %.2f -> Pb Lg %.3f"
          % (len(rows), float(rows[0]["R_frontal_overlap"]),
             float(rows[-1]["R_frontal_overlap"])))
    print("  saved: " + path)
    return True


BLOCKS = {
    "table": build_table,
    "idea": build_schematic,
    "hero": build_hero,
    "fusion": build_fusion,
    "topography": build_topography,
    "positioning": build_positioning,
}


def write_manifest(width_mm, dpi):
    path = os.path.join(OUT_DIR, "MANIFEST.txt")
    out = [
        "Poster figures -- placement sizes",
        "(generated by scripts/18_poster_figures.py)",
        "",
        "Target column width: %.1f mm   raster: %d dpi" % (width_mm, dpi),
        "Sizes are of the PDF (the file the LaTeX poster places); the PNG",
        "pixel counts are alongside.  Place at 100% -- do not rescale.",
        "If the poster is laid out at half scale (A1), place at half the",
        "width; the raster then prints at %d dpi effective." % (2 * dpi),
        "",
        "%-34s %9s %9s %9s %9s" % ("file", "w (mm)", "h (mm)", "w (px)",
                                   "h (px)"),
    ]
    for m in MANIFEST:
        out.append("%-34s %9.1f %9.1f %9d %9d"
                   % (m["name"], m["w_mm"], m["h_mm"], m["w_px"], m["h_px"]))
    out += [
        "",
        "PDF = vector, fonts embedded (PowerPoint, LaTeX, print shop).",
        "PNG = 600 dpi raster at the size above (Canva and web editors).",
        "",
        "The map panels (fusion, topography) hold 60 x 120 measured pixels.",
        "The dpi is print headroom, not resolution of the data; the blocky",
        "look is the scan grid and is left unsmoothed on purpose.",
    ]
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out[7:]))
    print("\n  saved: " + path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Render the A0 poster figures at true printed size.")
    ap.add_argument("--only", nargs="+", choices=sorted(BLOCKS),
                    help="build only these blocks (default: all)")
    ap.add_argument("--width-mm", type=float, default=DEFAULT_WIDTH_MM,
                    help="column width in mm (default %.0f)"
                         % DEFAULT_WIDTH_MM)
    ap.add_argument("--dpi", type=int, default=DEFAULT_DPI,
                    help="raster dpi (default %d)" % DEFAULT_DPI)
    ap.add_argument("--width-override", action="append", default=[],
                    metavar="BLOCK=MM",
                    help="render one block at a different width, e.g. "
                         "idea=210 for a schematic that sits beside its "
                         "text (repeatable)")
    ap.add_argument("--hero-panels", choices=("stacked", "side"),
                    default="stacked",
                    help="hero figure: stages stacked (one column, "
                         "default) or side by side (two-column poster)")
    ap.add_argument("--positioning-aspect", type=float,
                    default=POSITIONING_ASPECT,
                    help="canvas height/width of the positioning panel "
                         "(default %.2f; 0.34 makes a flat strip for a "
                         "wide column)" % POSITIONING_ASPECT)
    ap.add_argument("--font", default=LM_FAMILY,
                    help="figure font (default %s, the poster's own face; "
                         "registered from a TeX installation, see "
                         "--font-dir)" % LM_FAMILY)
    ap.add_argument("--font-dir", default=None,
                    help="directory holding lmroman10-*.otf, if the script "
                         "cannot find a TeX installation on its own")
    ap.add_argument("--out-dir", default=OUT_DIR,
                    help="where to write the figures (default %s); use a "
                         "second directory for a differently sized set "
                         "instead of overwriting the first" % OUT_DIR)
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the checks against the committed .txt reports")
    ap.add_argument("--list", action="store_true", help="list blocks and exit")
    args = ap.parse_args()

    if args.list:
        for k in sorted(BLOCKS):
            print(k)
        sys.exit(0)

    OUT_DIR = args.out_dir
    for item in args.width_override:
        name, _, mm = item.partition("=")
        if name not in BLOCKS or not mm:
            sys.exit("ERROR: --width-override expects BLOCK=MM with BLOCK "
                     "one of %s" % ", ".join(sorted(BLOCKS)))
        WIDTH_OVERRIDE[name] = float(mm)
    HERO_PANELS = args.hero_panels
    POSITIONING_ASPECT = args.positioning_aspect
    if args.font == LM_FAMILY:
        src = register_latin_modern(args.font_dir)
        if src is None:
            sys.exit("ERROR: Latin Modern Roman not found.  Install MiKTeX or "
                     "TeX Live, or pass --font-dir with the lmroman10-*.otf "
                     "files (GUST, lm2.004otf.zip).")
        print("  Latin Modern Roman from: %s" % src)
    names = args.only or ["table", "idea", "hero", "fusion", "topography",
                          "positioning"]
    print("=" * 70)
    print("  POSTER FIGURES -- %.0f mm column, %d dpi, font %s"
          % (args.width_mm, args.dpi, args.font))
    for name, mm in sorted(WIDTH_OVERRIDE.items()):
        print("  %s rendered at %.0f mm" % (name, mm))
    print("  hero panels: %s" % HERO_PANELS)
    print("  out: %s" % OUT_DIR)
    print("=" * 70)

    all_ok = True
    for n in names:
        all_ok &= bool(BLOCKS[n](args.width_mm, args.dpi, args.font,
                                 not args.no_verify))

    write_manifest(args.width_mm, args.dpi)

    if not all_ok:
        sys.exit("\nFAILED: a recomputed number disagrees with the committed"
                 " report -- the figures were written, but do NOT put them on"
                 " the poster until this is understood.")
    print("\nAll blocks rendered; every checked number matches the reports.")
