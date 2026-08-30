# A0 poster, Overleaf package

Upload the whole folder to a **blank** Overleaf project and Recompile.
`main.tex` is the main document, so Overleaf picks it up by itself.
Compiler: **pdfLaTeX**.

| file | what it is |
|---|---|
| `main.tex` | the poster: 7 blocks, 5 figures, two columns, sized to **one page** |
| `Hylangtechposter.cls` | stand-in class (see below) |
| `figures/*.pdf` | vector figures, rendered at **380 mm**, exactly one column (the schematic at 210 mm, beside its paragraph) |
| `figures/MANIFEST.txt` | exact size of each figure |
| `figures/ratio_table.tex` | per-line ratio table, generated; currently cut (the hero figure plots the same numbers), kept commented at the bottom of `main.tex` |

## Paper size

The output is **841 x 1189 mm, A0 portrait**, pinned in the class.

This needed fixing: `a0poster` is a dvips-era class that only emits
`\special{papersize=...}`. pdfLaTeX ignores that and ships whatever
`\pdfpagewidth`/`\pdfpageheight` say, A4 by default, so an uncorrected
build pours an A0 text block onto A4 and spills over many pages. The class
sets the pdf page registers explicitly. (a0poster's own `a0` is also
83.96 x 118.82 cm, marginally under ISO; the class pins the exact size.)

Check it after compiling: the PDF should report 841 x 1189 mm, and it should
be one page.

## Layout

Two columns of 38 cm with a 3 cm gutter, one `\columnbreak` after the
topography figure, so every block sits where `main.tex` says it does instead
of where `multicols` happens to balance it:

| column | blocks | est. height |
|---|---|---|
| 1 | idea (paragraph beside the schematic), two-stage decomposition (hero figure, panels side by side), canvas topography | ends at ~105 cm |
| 2 | instrument and data, learned fusion, positioning (optional), take-home box, acknowledgements | ends at ~105 cm |

Measured on the compiled PDF (pdfLaTeX, MiKTeX 26.5): both columns end
about 105 cm down the 118.9 cm sheet, so each has roughly 11 cm of spare
height for text edits. (The pre-compile arithmetic had put column 2 at
93 of 92 cm; it was pessimistic by about 15 %.) Body text is `\large` (29.9 pt on a0poster),
the size `POSTER_PLAN.md` section 4 asks for; block headlines are `\LARGE`
(43 pt) assertions, so a visitor who reads only the headlines gets the whole
story.

Three things make two wide columns hold column-width figures (a first
two-column attempt with the three-column figure set ran ~105 cm per column):

- the schematic is rendered at 21 cm and sits **beside** its paragraph in a
  pair of minipages instead of under it (24 cm tall at full width);
- the hero figure puts stage 1 and stage 2 **side by side** instead of
  stacked (`--hero-panels side`), 16 cm shorter at 38 cm;
- the positioning panel is a flat strip (`--positioning-aspect 0.34`).

**If a column still spills**, in this order:

1. delete the block marked `OPTIONAL BLOCK` in column 2
   (positioning error budget, ~22 cm). `POSTER_PLAN.md` ranks it last and calls it filler.
2. re-render the figures narrower and place them at a fixed width:

```bash
python scripts/18_poster_figures.py --width-mm 340 --width-override idea=210 \
    --hero-panels side --positioning-aspect 0.34 --out-dir poster/figures
rm poster/figures/*.png            # LaTeX only needs the PDFs
# then use width=34cm instead of width=\columnwidth in main.tex
```

The figures are rendered at **true printed size**, one matplotlib point is
one point on the sheet, which is why `width=\columnwidth` (38 cm) is a scale
factor of 1. Scaling a wider render down with `width=0.8\linewidth` would
shrink the axis labels below the 20 pt floor. The exact command that
produced the current set is in the header comment of `main.tex`.

## Typeface

Latin Modern everywhere. The class loads `lmodern` for the text; the figures
are drawn in **Latin Modern Roman** as well, which `scripts/18_poster_figures.py`
registers with matplotlib straight from the MiKTeX tree
(`fonts/opentype/public/lm/lmroman10-*.otf`), so nothing has to be installed
as a system font. On a machine without a TeX installation pass
`--font-dir` with those four files (GUST, `lm2.004otf.zip`). Greek and
subscripts in the figures come from matplotlib's bundled Computer Modern
(`mathtext.fontset = cm`), Latin Modern's parent, so they match the poster's
own math. Every figure PDF embeds `LMRoman10-Regular` (and `-Bold` where a
label is bold); check with any PDF font inspector.

## What the figures carry, and what the text does

No figure has a title: the caption says what the panel is, and every number
the old titles and legends carried lives in the bullets next to the figure.
The hero figure keeps its two panel labels (stage 1 / stage 2) because they
tell the panels apart. The heads are named by position, **upper head** and
**lower head**, matching the schematic; the serial numbers are in the paper.

## Colour

One accent, `posterblue` (RGB 31,56,100), for the title, section heads,
caption labels, header rules and the take-home box. Everything else is
black.

## Logos

The originals live in this folder (`etf.png`, `vinca.png`,
`arsmensurae.png`, `palata.jpg`); `figures/logo_*.png` are the same files
trimmed to their ink (the Vinca and Palace files are white marks on navy
and keep that ground as a badge). Placement, in `main.tex`:

- header, between the two rules: ETF (5 cm), Vinca (5 cm), Ars Mensurae
  (2.4 cm), each raised to its vertical centre so the row aligns on the
  middle; IDArtScience has no logo;
- bottom of column 2: the Palace of Science beside the acknowledgement
  sentence that names it (5.2 cm wide, minipage pair).

Resolution at print: ETF 264 px at 5 cm (~135 dpi) and Vinca 217 px at
5 cm (~110 dpi) are acceptable for viewing distance; the Palace at 5.2 cm
is ~190 dpi. **Ars Mensurae is 269 x 41 px**, ~45 dpi even at 2.4 cm, and
will look soft up close; a vector or a wider PNG from them would fix it,
drop the file in as `figures/logo_arsmensurae.png` (any size, the height
is set in `main.tex`).

## The class file

The original `Hylangtechposter.cls` belongs to the
[HYlangtech\_postertemplate](https://www.overleaf.com/latex/templates/hylangtech-postertemplate/hnrppvmhxnwm)
Overleaf template and is not redistributable from this repository, which is
why the first compile failed with `File 'Hylangtechposter.cls' not found`.

The one in this folder is a **stand-in** matching the interface `main.tex`
uses (`[portrait]`, `\printheader`, the a0poster font ladder, the packages).
It compiles on its own but does not look identical to the original.

**For the original design:** open the template link with *Open as Template*,
upload `main.tex` and `figures/`, and **delete our `Hylangtechposter.cls`**.
`main.tex` needs no edits either way, but then replace the header logos,
which are the FoTran / EU / ERC marks of ERC grant 771113 at the University
of Helsinki, not our project. The original template is three columns wide
by default; the two-column geometry lives in our class, so with the original
class the columns follow the template again.

## Compiling locally

```
cd poster
pdflatex -interaction=nonstopmode main.tex
```

Needs `a0poster`, `lmodern`, `multicol`, `caption`, `booktabs`, `xcolor`,
`graphicx`, `amsmath`. On MiKTeX let it install missing packages on the fly
(`initexmf --set-config-value=[MPM]AutoInstall=1`) or install them first
with `miktex packages install a0poster caption booktabs`.

## Verified

Compiled locally with pdfLaTeX (MiKTeX 26.5, 2026-08-24): one page,
841 x 1189 mm, every embedded font is Latin Modern (`LMRoman*`, `LMMath*`)
apart from the Computer Modern Greek inside the figures. The only log
warning is an underfull box from the forced line break in the title.
`main.pdf` in this folder is that build.

## Still open

`POSTER_PLAN.md` section 5 has the conference's size and orientation
unconfirmed; `[portrait]` and A0 are assumptions.
