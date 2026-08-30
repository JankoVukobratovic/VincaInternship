%==============================================================================
%  Hylangtechposter.cls -- STAND-IN
%
%  WHAT THIS IS
%  The poster was drafted against the HYlangtech_postertemplate Overleaf
%  template, whose class file is not redistributable from our repository.
%  This file provides the same interface -- the [portrait] option,
%  \printheader, the a0poster font ladder up to \VERYHuge, and the packages
%  main.tex uses -- so the project compiles on its own.
%
%  It is NOT the original class, and the result will not look identical.
%  If you want the exact original design, open the template on Overleaf
%    https://www.overleaf.com/latex/templates/hylangtech-postertemplate/hnrppvmhxnwm
%  and delete this file from the project; main.tex then compiles against the
%  real class unchanged.
%
%  Deliberate difference: \printheader here draws no institution text (the
%  affiliations already sit under the authors) and takes OUR logos through
%  \headerlogos.  The original prints the FoTran / EU / ERC logos of ERC grant
%  771113 at the University of Helsinki, which is not our project and must
%  not appear on our poster.
%==============================================================================
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{Hylangtechposter}[2026/08/24 Stand-in A0 poster class]

% ---- orientation -------------------------------------------------------------
\newif\ifHYT@portrait \HYT@portraittrue
\DeclareOption{portrait}{\HYT@portraittrue}
\DeclareOption{landscape}{\HYT@portraitfalse}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{a0poster}}
\ProcessOptions\relax

% a0poster gives true-size A0 typesetting: one pt in the source is one pt on
% the printed sheet, which is what the figures in figures/ were rendered for.
\ifHYT@portrait
  \LoadClass[a0,portrait]{a0poster}
\else
  \LoadClass[a0,landscape]{a0poster}
\fi

% ---- paper size ---------------------------------------------------------------
% a0poster is a dvips-era class: it sets \paperwidth/\paperheight and emits
% \special{papersize=...}, which is an instruction to dvips.  pdfLaTeX ignores
% that special entirely and ships whatever \pdfpagewidth/\pdfpageheight say --
% by default A4 or letter.  The result is a correctly typeset A0 text block
% poured onto an A4 sheet, i.e. a poster spread over many pages.  Setting the
% pdf page registers is what actually makes the output A0.
%
% a0poster's own "a0" is 83.96 x 118.82 cm, marginally under the ISO size, so
% pin the exact A0 here and keep the text block consistent with it.
\ifHYT@portrait
  \setlength{\paperwidth}{841mm}
  \setlength{\paperheight}{1189mm}
\else
  \setlength{\paperwidth}{1189mm}
  \setlength{\paperheight}{841mm}
\fi
\@ifundefined{pdfpagewidth}{}{%
  \setlength{\pdfpagewidth}{\paperwidth}%
  \setlength{\pdfpageheight}{\paperheight}%
}

% ---- packages main.tex relies on ---------------------------------------------
\RequirePackage[T1]{fontenc}
\RequirePackage{lmodern}          % scalable outlines, so the huge sizes stay sharp
\RequirePackage{xcolor}
\RequirePackage{graphicx}
\RequirePackage{multicol}
\RequirePackage{booktabs}
\RequirePackage{caption}          % provides \captionof
\RequirePackage{amsmath}

% ---- colour -------------------------------------------------------------------
% One accent, a dark navy, for every coloured element: title, section heads,
% caption labels, the header rules and the take-home paragraph.  It is the
% same #1F3864 the pptx layout uses, so the two builds match, and it is dark
% enough to read as text rather than as decoration.
\definecolor{posterblue}{RGB}{31,56,100}

% ---- page ---------------------------------------------------------------------
% A0 portrait is 841 x 1189 mm.  Text 79 cm wide with one 3 cm gutter makes
% each of the two columns exactly 38 cm.  Figures go in at \columnwidth,
% which must equal the width they were rendered at (figures/MANIFEST.txt):
% they are true-size, so that scale factor is 1 and the axis labels stay at
% the point size they were drawn at.
\setlength{\textwidth}{79cm}
\setlength{\columnsep}{3cm}
\setlength{\columnseprule}{0pt}
\setlength{\textheight}{114cm}
\setlength{\oddsidemargin}{-1in}
\addtolength{\oddsidemargin}{2.55cm}
\setlength{\evensidemargin}{\oddsidemargin}
\setlength{\topmargin}{-1in}
\addtolength{\topmargin}{2.0cm}
\setlength{\headheight}{0pt}
\setlength{\headsep}{0pt}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.7ex plus 0.3ex minus 0.2ex}
\pagestyle{empty}

% ---- section headings ---------------------------------------------------------
% Assertion headlines, per POSTER_PLAN.md section 3: large, navy, unnumbered.
\setcounter{secnumdepth}{0}
\renewcommand{\section}{\@startsection{section}{1}{0pt}%
  {1.5ex plus .6ex minus .2ex}%
  {0.7ex plus .2ex}%
  {\color{posterblue}\bfseries\LARGE}}
\renewcommand{\subsection}{\@startsection{subsection}{2}{0pt}%
  {1.6ex plus 1ex minus .2ex}%
  {0.8ex plus .2ex}%
  {\color{posterblue}\bfseries\Large}}

\captionsetup{labelfont={bf,color=posterblue},textfont={},skip=0.6ex}

% ---- header -------------------------------------------------------------------
% Redefine \headerlogos in the preamble to drop real logo images in, e.g.
%   \renewcommand{\headerlogos}{\includegraphics[height=6cm]{etf.pdf}\hfill
%                               \includegraphics[height=6cm]{vinca.pdf}}
% With it empty the header is just the two rules.
\newcommand{\headerlogos}{}

\newcommand{\printheader}{%
  \noindent\textcolor{posterblue}{\rule{\textwidth}{10pt}}\\[0.7cm]
  \ifx\headerlogos\@empty\else
    \noindent\makebox[\textwidth]{\headerlogos}\\[0.6cm]
  \fi
  \noindent\textcolor{posterblue}{\rule{\textwidth}{4pt}}\\[1.2cm]
}

\endinput
