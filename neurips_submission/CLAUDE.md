# neurips_submission - working notes for instant context reload
(written 2026-08-28 03:20 by Claude at Dimitrije's request; keep this file
updated after every step; it is the single source of truth for the state
of the Sim2Science submission)

## 0. Hard facts
- Venue: Sim2Science: ML with Imperfect Scientific Models, NeurIPS 2026
  workshop, Paris, Dec 12/13. Site https://www.sim2science.com (cfp.html),
  OpenReview venue NeurIPS.cc/2026/Workshop/Sim2Sci.
- DEADLINE EXTENDED: **Sep 2, 2026, 23:59 AoE** (Aug 29 struck through on the
  site). Reviewing Sep 8 to Sep 23, notification Sep 29, camera-ready TBD.
- Format: NeurIPS 2026 LaTeX, `\usepackage[dblblindworkshop]{neurips_2026}`,
  `\workshoptitle{Sim2Science}`; 5 pages excl. references (Tiny Papers = 2
  pages, we target 5); double-blind; reproducibility checklist mandatory
  (desk reject without it); appendix unlimited; non-archival.
- RECIPROCAL REVIEWING: one co-author with "sufficient expertise and
  publications" is nominated at submission and reviews 2 papers Sep 8-23;
  no-show or bad reviews = desk reject of OUR paper. Must be a Vinca mentor,
  not the student. NOT YET AGREED with the mentors (open item).
- Team: Dimitrije (student, first author) now owns WP1, WP2, WP3 (teammates
  delivered nothing by Aug 28). Chat with Dimitrije in Serbian; all repo
  content English; console is cp1250: ASCII-only prints; `encoding="utf-8"`
  on every text file open. NEVER add a Co-Authored-By Claude trailer to
  commits. No em dashes in prose/figures. Dark navy accent #1f2a44, grey
  #8c8c8c, orange #c8641e.
- Instrument (mentor Stefano, email 2026-08-28, see INSTRUMENT.md): two
  Amptek FAST SDD; 19511 horizontal 70 mm^2; 10264 at 40 deg from
  horizontal 25 mm^2; both 12.5 um Be window; tube 37 kV, 40 uA. Maps used
  everywhere = detector-summed net counts. Do not infer distances/solid
  angles.

## 1. Repo layout (repo root C:/everything/projekti/VincaInternship)
- `neurips-restore/` FROZEN validated code: src/forward_model.py (measured
  warp, per-line tilt gains, noise Var = k*counts, k 4.2-9.0), datagen.py
  (forward_sharp = cubic sampling = v2 training simulator), model.py
  (RestorationUNet, 469k params, residual, zero-init head), eval.py
  (score_pair: r, ssim, bias_pct, cv_ratio). experiments/checkpoint.pt =
  MVP nominal net. results/*.txt = MVP numbers.
- `neurips_submission/` the package. Run everything from the REPO ROOT.
  - config.py: TRAIN/QUICK_TRAIN, JITTER (measured sigmas, sources in the
    comment block), ENSEMBLE_N=12, COVERAGE_Z, WP1_CASES, WP1_HARSH_CASE,
    DEFECT_LADDERS (18 rungs), DEFECT_X, WP2_TEST, GRID (WP3), FIG_LINES =
    Ca, Cu, PbLa.
  - common/: core.py (bootstrap), perturb.py (SimKnobs: noise_k_scale,
    gain_scale, angle_bias_deg, blur_mode, warp_shift_px, warp_rot_deg,
    gain_pct_offset(8), fresh_frac; forward_perturbed; sample; jittered),
    training.py (train_net), restore.py (degrade/apply_network/
    score_candidates, load_mvp_net), classical.py (nearest, biharmonic,
    telea, ns fills in the tilted frame), io_utils.py (CSV convention).
  - main.py: `--stage smoke|verify|status|wp1|wp2|wp3|figures|all [--quick]`.
    verify MUST stay green after any change to perturb/training/classical.
  - wp1_uq_ensemble/exp_ensemble_uq.py, exp_adaptive_scan.py  (DONE, run)
  - wp2_simulator_audit/exp_defect_tolerance.py (DONE, run, + crosstab),
    exp_diagnostics.py (DONE, run)
  - wp3_degradation_benchmark/exp_degradation_grid.py (DONE, run)
  - results/*.csv (commit), results/wp*_summary.txt, results/wp*_paper_section.md
    (paper text drafts with all numbers), results/wp1_ensemble/ and
    wp2_rungs/ (checkpoints, gitignored), figures/*.png+pdf.
  - INSTRUMENT.md, README.md (status sections per WP), this CLAUDE.md.
- NOTHING IS COMMITTED YET for Aug 28 work (git status shows all modified).
  Commit when Dimitrije says so; plain message, no AI attribution.

## 2. What is DONE and the exact numbers to quote
WP1 (results/wp1_summary.txt, wp1_adaptive_summary.txt, wp1_paper_section.md):
- 12 jitter + 12 control members (CPU, ~10 min each, 4 parallel shells).
- Spread jitter/control 2.17 (sim footprint) / 2.80 (real); simulator share
  of jitter variance 62 % / 86 %; sigma_sim grows with angle 104 -> 148
  counts (8 -> 25 deg).
- Coverage real footprint (band total = spread + propagated noise + k*truth):
  jitter 0.605/0.866/0.953 at z=1/2/3, control 0.530/0.802/0.907.
  Sim footprint: jitter total 0.635/0.876 (z=1/2), upper bound total_ref
  0.850/0.969; spread-only jitter 0.520/0.753, control 0.252/0.440.
- Hole (14x20): z=2 coverage 0.49 jitter / 0.60 control, rms err 940 vs
  spread 320-410, Spearman ~0.1, jitter NARROWER than control (0.85).
- Accuracy real: jitter_mean r >= det on 5/8 lines (rest within 0.002),
  mean r 0.9527 vs det 0.9526, cv 0.979 vs 0.947; MVP single and control
  mean lose on 8/8 (0.9485, 0.9496). Sim footprint jitter 0.9676 vs
  control 0.9706; hole 0.259 vs 0.331.
- Adaptive scan: r>=0.90 at 79.4 % measured (adaptive) vs 84.9 random /
  84.5 raster / 71.9 oracle; r>=0.95: 88.1 / 93.0 / 93.1 / 82.0.
WP2 diagnostics (results/wp2_diag_summary.txt, wp2_paper_section.md):
- Null = jittered-within-calibration vs nominal pairs (24). A first null
  with noise-only pairs flagged everything on the real scan (z=60 for a
  0.3 deg rotation): DO NOT go back to it.
- Pre-registered rule, per angle (7.7 and 20): blur 5/5, shift 0.5/1/2 px
  15/15, rot 1/2 deg 10/10, noise x0.25/x2/x4 15/15, noise x0.5 0/5 (1/5
  at 20 deg), gain family 0/40, false alarms 2/24. Grouped 67/114 (68 at
  20 deg); LOO nearest-centroid 73/114.
- Post-hoc gain_proj (projection on g-1): gain x0 and x2 recovered 10/10,
  x0.5/x1.5 and angle +-2/+-5 still 0/30. Post-hoc grouped 77/114 (78).
- REAL scan: vs v1 bilinear -> "blur" (hf real/sim 1.7-2.1, mf z 5.1, cv z
  14.9); vs validated forward() -> "blur"; vs v2 cubic -> "noise_k" (hf
  1.18, k_ratio 1.19, mf z 0.6, dy -0.03 px, rot 0.27 deg, gain_proj 0.03).
WP3 (results/wp3_regime_summary.txt, wp3_paper_section.md):
- 180 cases (4 angles x 5 holes x 3 doses x 3 seeds) + 3 sharp + real.
- Hole, dose 1: biharmonic beats net 16/16 (20 deg 14x20: 0.467 vs 0.415;
  6x8: 0.529 vs 0.359); ens_jitter wins only the 10x14 column (+0.005 to
  +0.015); hybrid biharmonic+net wins 16/16 by 0.005-0.03.
- Dose 0.25, 14x20 hole: net 0.030, ens -0.007, biharmonic 0.455.
- Footprint: learned beats classical everywhere (0x0: net 0.985 vs det
  0.977; 14x20: ens 0.952, net 0.948, best classical 0.937, det 0.781).
- Hole r is angle-independent (measured warp does not scale with angle).
- Sharp vs validated: footprint r moves <= 0.007, hole <= 0.014, net cv
  0.944 -> 0.970; no ranking changes.

## 2b. WP2 tolerance + crosstab RESULTS (done ~06:45, results/wp2_tolerance_summary.txt)
- Seed band (12 nominal WP1 controls): delta r sim fp [+0.082, +0.090],
  real [-0.0052, -0.0021]; HOLE band [-0.03, +0.27] = seed-dominated,
  hole excluded from tolerance claims.
- noise x0.25-x4: harmless (inside band sim, within 0.005 real).
  gain x0-x2, angle +-2/5 deg: harmless. GEOMETRY is the cliff:
  shift 0.5 px = -0.010 real (out of band), 1 px = net WORSE than physics
  (-0.057 sim, -0.086 real), 2 px catastrophic (-0.23/-0.27); rot 1 deg
  -0.019, 2 deg -0.047 + cv 0.963. BLUR: invisible on the sim testbed
  (+0.089, testbed shares the bilinear acquisition) but on REAL leaves the
  band (-0.0072) with cv overshoot 1.013 = the v1 signature.
- Crosstab (in the same txt + wp2_paper_section.md): every material harm
  (>0.005 exceedance) is VISIBLE to the pre-registered rule; invisible
  rungs are harmless; single formal exception gain_x1.5 with exceedance
  0.0002 = band resolution. figures/wp2_tolerance_curves.png done.

## 2c. WP4 ABC RESULTS (done ~06:50, results/wp4_abc_draws.csv, log wp4_abc_full.log)
- 3000 draws, 54 s; accept 5 % (sens 2/10 %): blur_bilinear prior 0.51 ->
  POSTERIOR 0.00 at all levels (the loop rejects v1 by itself);
  noise_k_scale 1.07 -> 1.43 +- 0.35; warp_rot 0 -> +0.15 +- 0.17 deg;
  warp_dy +0.075 +- 0.06 px; gain offsets mildly positive Ca/Ti.
- PPC: posterior 0/12 "ok", ALL verdicts noise_k, but decomposition
  (results/wp4_ppc.csv, A per stat): posterior vs nominal: hf 1.9 vs 3.0
  (below thr 2.05), k 2.45 vs 3.38, cv stays 4.6 -> the RESIDUAL misfit is
  PER-LINE noise (real/sim var ratios per line: Ca 1.91, Ti 1.65, Fe 0.94,
  Cu 1.24, PbLl 0.89, PbLa 4.55, PbLb 2.86, PbLg 0.88, from MVP-2) which a
  GLOBAL k knob cannot express -> "the PPC names the missing knob
  (per-line noise scales)" = the closing sentence of WP4.
- Posterior ensemble training LAUNCHED (~06:55, 4 shells x 3 members,
  logs scratchpad/wp4_train_*.log, ckpts results/wp4_posterior/post_XX.pt).

## 2d. WP4 ENSEMBLE RESULTS (done ~08:0x, results/wp4_summary.txt)
- Posterior ensemble = BEST on real: r 0.9539 (det 0.9526, prior 0.9527,
  control 0.9496, mvp 0.9485); >= det on 7/8 lines (Cu -0.0003); spread 77
  vs prior 106 (27 % narrower); cov z=2/3: 0.854/0.942 vs prior 0.866/0.953
  vs control 0.802/0.907; |bias| 1.07 %; err rms 223 (best). Sim cases:
  posterior between control and prior (correct). wp4_paper_section.md done,
  figure wp4_prior_posterior done. EXPERIMENTS COMPLETE, verify green,
  status: all 11 CSVs present.

## 2e. PRE-PAPER HARDENING (agreed ~12:30 "kreni, uradi 1 2 i 3 paralelno")
1. ROUND 2 of the loop DONE: SimKnobs got `noise_k_line_scale` (per-line k
   multipliers; drawn LAST in jittered() and only if spec has
   noise_k_line_log_sd, so the rng stream of old specs is bit-identical -
   VERIFIED against WP1 jitter_00). `--abc2` runs ABC with prior
   ROUND2_SPEC (log-sd 0.6), writes wp4_abc_draws_r2/_marginals_r2/_ppc_r2
   (never touches members.json). RESULT: marginals move right (nkeff PbLa/
   PbLb 1.9-2.2, Cu 2.2; bilinear still 0.000) but d_min only 2.16->2.02
   vs NULL FLOOR 0.95 +- 0.24 (computed; nominal cubic 2.72), PPC still
   0/12 ok -> residual = unmodelled structure (flat-field, scatter, drift)
   + rejection-ABC limits in 22 dims. wp4_paper_section.md has the
   round-2 paragraph and updated three-sentences.
2. NLL proper score DONE (results/wp4_nll_summary.txt): REAL fp total:
   PRIOR best 6.449, posterior 6.584, control 7.270 (the posterior's
   narrower band does NOT win the proper score; the gap sits on Ca/PbLa/
   PbLb = the unmodelled per-line noise lines, coherent with the PPC).
   Hole: control 11.0 << posterior 15.0 / prior 15.5 (NLL punishes the
   confidently-wrong fills). wp4_paper_section.md updated honestly
   (posterior = "most accurate", prior = marginally better proper score).
3. Extra seeds DONE (n=4 per borderline rung, in the tolerance summary):
   shift_0.5px REAL all negative [-0.0130, -0.0045], sim fp [+0.050,
   +0.058] vs band [+0.082, +0.090]; blur REAL out of band for ALL 4
   seeds [-0.0116, -0.0055]. wp2_paper_section.md updated with ranges +
   new caveat 4. HARDENING COMPLETE; only the paper (item C) remains.

## 3. What REMAINS (in order)
A. WP2 tolerance evaluation (DONE, see 2b):
   1. wait until `ls neurips_submission/results/wp2_rungs/*.pt | wc -l` = 18
      (training logs in the session scratchpad wp2_train_*.log; 4 shells
      `--train-only --rungs 1-5|6-10|11-14|15-18 --threads 3`).
   2. `python neurips_submission/main.py --stage wp2` (runs tolerance eval
      with the WP1 control members as the seed band, then diagnostics
      again - diagnostics is fast, fine) OR
      `python neurips_submission/wp2_simulator_audit/exp_defect_tolerance.py`
      then `--figures`. Outputs results/wp2_defect_tolerance.csv,
      wp2_tolerance_summary.txt, figures/wp2_tolerance_curves.png.
   3. Fill the "[TO FILL]" tolerance paragraph in results/wp2_paper_section.md:
      per family the rung where delta r leaves the 12-seed band / drops
      below det, sim footprint, hole, real; blur rung = the real-scan anchor.
   4. Cross-table visibility x damage: diagnostics verdicts (wp2_diag_confusion
      .csv, pre-registered) x tolerance delta r per rung -> four quadrants
      (visible+harmful, invisible+harmless, invisible+harmful = the
      important row, visible+harmless). Add as `crosstab()` in
      exp_defect_tolerance.py or a small script; put in the WP2 text.
B. WP4 closed loop (agreed with Dimitrije 03:10, "da, hajde"): new folder
   `wp4_closed_loop/exp_simulator_posterior.py`. Spec:
   1. ABC over SimKnobs from ONE real scan. Prior: config.JITTER draws via
      perturb.jittered PLUS blur_mode in {cubic, bilinear} with p = 0.5.
      For each of N = 3000 draws theta: S = perturb.forward_perturbed(
      prova1, 7.7, rng, theta); summary = exp_diagnostics.battery(ruotato,
      S, 7.7) (9 stats x 8 lines incl. gain_proj). Whitening: Null built
      as in the diagnostics real verdict (pairs null_pair(prova2, prova1,
      7.7, seed), 24 pairs) -> distance d = RMS over (stat, line) of
      (value - mu_null)/sd_null. Accept the closest 5 % (report 2 % and
      10 % sensitivity). Save results/wp4_abc_draws.csv (all draws, knobs,
      d, accepted flag) and marginals (mean/sd/quantiles per knob, prior
      vs posterior, P(blur = cubic)). Expected from diagnostics: noise_k
      ~1.2, blur cubic, registration inside sigma; this is a prediction to
      CHECK, not to assume.
   2. Posterior predictive check: for 12 posterior draws and 12 prior
      draws run exp_diagnostics.identify(battery(ruotato, S), null2,
      posthoc=True); count "ok" verdicts. Posterior should be mostly ok.
   3. Posterior ensemble: 12 nets, knobs = 12 accepted draws sampled
      without replacement (seeded), seeds = WP1 jitter seeds
      (config.BASE_SEED + 100 i), training.train_net with config.TRAIN,
      cache results/wp4_posterior/post_XX.pt + json (gitignore it), same
      `--train-only --members a-b --threads 3` CLI as WP1 for 4 shells.
   4. Evaluation on the REAL scan (the point) + the 8 validated dose-1
      sim cases of config.WP1_CASES: three ensembles control (WP1
      results/wp1_ensemble/control_XX.pt), prior (jitter_XX.pt), posterior.
      Reuse exp_ensemble_uq.ensemble_predict, aleatoric_sigma,
      coverage_rows (bands ens/total_noref/total as in WP1 real anchor),
      accuracy_rows; add spread rms and Spearman(sigma, |err|).
      Output results/wp4_posterior_ensemble.csv, wp4_summary.txt,
      figures/wp4_prior_posterior.png (knob marginals prior vs posterior +
      real-scan spread/coverage/r for the three ensembles).
   5. Claim to test (can be lost): posterior spread < prior spread on the
      real scan while coverage stays at or above prior's and r does not
      drop. Report whatever comes out.
   6. Paper text results/wp4_paper_section.md.
C. Paper: DONE (~14:30). paper/main.tex + filled checklist.tex compiled to
   paper/main.pdf: 15 pages total, MAIN TEXT ENDS ON PAGE 5 exactly
   (references from page 6, appendix A+B, checklist last; dblblindworkshop
   + \workshoptitle{Sim2Science}; anonymized: scans renamed F1/F2/T, no
   lab names; wp4 figure panel retitled "r vs F2"; no em dashes; 12 real
   references, thebibliography, no bibtex needed; pdflatex x2 builds
   clean). Style source: Formatting_Instructions_For_NeurIPS_2026/ (was in
   the repo, no download needed). REMAINS: Dimitrije's read-through, then
   any number he wants double-checked; camera-ready de-anonymization later. Structure agreed: setup (INSTRUMENT.md + MVP
   simulator) -> burned by blur (v1/v2) -> WP1 ensemble UQ -> WP2 audit
   (tolerance + blind diagnostics) -> WP4 loop -> WP3 regime map -> real
   anchor everywhere -> limitations (one painting, one instrument, sigmas
   are lower bounds). Figures: wp1_calibration, wp1_spread_maps,
   wp2_diag_confusion, wp2_tolerance_curves, wp3_regime_map, wp4 figure;
   adaptive scan = one outlook paragraph.
D. Checklist, anonymization, reviewer nomination (mentor), commit, submit.

## 5. REVIEW ROUND (2026-08-28, from neurips_submission/whaswrong.txt)
Dimitrije reviewed the paper; all points addressed and pushed:
- ALL figures regenerated with print-sized canvases (fonts readable at
  linewidth); WP4 got contrasting grey-vs-navy marginals + visible orange
  physics curve; WP3 legend moved outside; WP2 tolerance shows 2 rows
  (hole row dropped, seed-dominated); log axis now plain numbers;
  "(up = better)" on accuracy axes; main text uses compact
  wp3_regime_heat, full wp3_regime_map in the appendix.
- Text: numeric citations (PassOptionsToPackage numbers), Sec 2 title
  sobered, F1/F2/T stated to be the SAME painting, coverage honesty
  (3-sigma tail failure 0.047 vs 0.003 stated, "near-calibrated" removed),
  blur lesson promoted to contribution No. 1, "2/10%" rephrased, rms
  explained, tone sweep. Line numbers = submission mode (gone in final).
- PITFALL FOUND: an earlier heredoc script had rfind()==-1 and silently
  DUPLICATED main.tex (LaTeX ignored the second copy after end{document});
  truncated back to one copy. Bash-heredoc python with backslashes is
  UNRELIABLE here (wrapper eats double backslashes: \b -> backspace);
  always use Write-a-script-file or the Edit tool for such patches.
- Main text ends EXACTLY on page 5; References from page 6; 16 pages
  total; verify green; no em dashes.

## 4. Commands and pitfalls
- Full WP1 rerun is NOT needed (cached members + CSVs). `--quick` runs use
  separate member dirs (wp1_ensemble_quick, wp2_rungs_quick) - never mix.
- Training on this 12-core CPU: ~10 min per net alone, ~18 min with 4
  shells x 3 threads plus other load. Colab not needed (data-gen bound).
- Bash heredocs with `'''` python strings sometimes broke; write patch
  scripts to the scratchpad with the Write tool and run them.
- io_utils.write_rows rebuilds the header as the union of keys; appending
  from parallel processes races - write per-member json, rebuild CSV.
- Figures must be built from CSV/npz only (restartable rule).
- honest-evaluation rules: prova2 and ruotato never in training; every
  table has the real anchor row; matching-not-beating = negative result.
