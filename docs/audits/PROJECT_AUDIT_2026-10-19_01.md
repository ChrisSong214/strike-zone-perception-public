# Project audit — October 19, 2026

**Verdict: the project is not yet reproducible as a validated end-to-end research pipeline.** The numerical components and existing unit tests are useful, but the current pipeline has two execution blockers, inconsistent inference preprocessing, incompatible saved results, and validation claims stronger than the implemented experiments.

This is an audit of the current local project, anchored to commit `54ca0d3`. Findings concern the current source and artifacts, including ignored local data; they are not a repetition of all historical findings. No source fixes, model retraining, leaderboard regeneration, production database writes, or cluster jobs were performed. A new audit report is the only intended project deliverable. Existing user changes were preserved.

P1 means a pipeline blocker or an issue that must be resolved before relying on the affected research result. P2 means a material correctness, integration, or methodological issue. P3 means a smaller conditional defect.

## Verification and coverage

| Check | Fresh result | Interpretation |
|---|---|---|
| Python parsing | 66 source/test files parsed | Syntax is valid; runtime correctness is separate. |
| Full test suite | **49 passed, 2 warnings**, 6.49 seconds | Run in isolated Python 3.11 with `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `MPLBACKEND=Agg`. |
| Statement coverage | **796 / 5,614 statements, 14.18%** across `src`, `scripts`, and `analysis` | Training, feature-building, exporting, and most analytical entrypoints are not exercised. |
| Wheel build | **Succeeded** | Built `strike_zone_perception-0.1.0-py3-none-any.whl`; this does not certify pipeline execution. |
| Dependency advisory scan | **No known vulnerabilities in 163 resolved packages** | Applies to the newly installed audit environment, not a pinned production environment or a complete security certification. |
| Credential-pattern scan | No matches in 106 scanned text files | Limited private-key/common-token pattern scan; not proof that every secret is absent. |
| Notebook inspection | All five inspected; notebook 05 contains a saved `KeyError` | Notebooks were not executed because their cells can write data or call external services. |
| PDF inspection | All 25 presentation pages and all 10 reference-PDF pages extracted, rendered, and visually inspected | The presentation contains outdated scientific labels and results, detailed below. |
| Local data inspection | Read-only SQLite queries, Parquet schemas, and memory-mapped trajectory/checkpoint inspection | No full-season inference or pipeline regeneration. |

The repository's `.venv` failed during test collection because macOS refused a NumPy binary. The first unrestricted isolated run then terminated in native XGBoost code. The complete suite passed after limiting native thread counts. These environmental failures are recorded separately from source defects. The audit environment lives at `/private/tmp/sagb-audit-20261019-env`; resolved versions are not a production lockfile.

## P1 findings

### A01 — Swing-model training crashes before computing or saving ratings

**Evidence:** [scripts/train_swing_model.py:172](/Users/chrissong/sp26_sagb_dj/scripts/train_swing_model.py:172), [scripts/train_swing_model.py:227](/Users/chrissong/sp26_sagb_dj/scripts/train_swing_model.py:227).

The import inside `main()` makes `get_db_connection` local throughout that function. Its earlier call therefore raises `UnboundLocalError`. This occurs after model and prediction-file writes, but before Bayesian fitting and rating persistence, leaving a partially updated run.

**Verification:** executed the actual `main()` with tiny input data and intercepted training, filesystem writes, and Parquet output. It raised `UnboundLocalError: cannot access local variable 'get_db_connection' where it is not associated with a value` at the expected call.

**Repair:** remove the redundant local import and add a small entrypoint integration check that reaches rating persistence against a temporary database.

### A02 — Existing trajectory caches cannot be used by the current feature builder

**Evidence:** [scripts/build_swing_features.py:180](/Users/chrissong/sp26_sagb_dj/scripts/build_swing_features.py:180), [scripts/build_training_data.py:191](/Users/chrissong/sp26_sagb_dj/scripts/build_training_data.py:191).

The cached path requires `game_pk`, `at_bat_number`, and `pitch_number` labels. The actual local `trajectories_2025.pt` contains only `plate_x`, `plate_z`, `sz_top`, `sz_bot`, `is_swing`, and `batter_id`. Consequently, the default cached path would finish expensive whole-season inference and then fail at `labels['game_pk']`. The current builder can create the new identifiers, but existing files have not been migrated.

**Repair:** validate cache schema before inference; rebuild incompatible caches or use an explicit safe fallback. Add a cache-format version and input fingerprint rather than treating file existence as compatibility.

### A03 — Cached inference normalizes padding differently from training and online scoring

**Evidence:** [scripts/build_swing_features.py:169](/Users/chrissong/sp26_sagb_dj/scripts/build_swing_features.py:169), [src/dataset.py:153](/Users/chrissong/sp26_sagb_dj/src/dataset.py:153), [src/lstm_scoring.py:123](/Users/chrissong/sp26_sagb_dj/src/lstm_scoring.py:123).

Cached inference normalizes every timestep, converting zero padding into nonzero feature vectors. Training and online scoring normalize only valid timesteps. The LSTM processes the changed prefix, so its hidden state and decision distribution differ.

**Verification:** with the actual checkpoint and first 128 stored 2025 pitches, 127 were padded. Cached versus online decision means differed by up to **0.1028 ft horizontally and 0.2317 ft vertically** (1.23 and 2.78 inches). Mean absolute differences were 0.0257 and 0.0608 ft. Maximum correlation-parameter difference was 0.1415. This is a preprocessing comparison, not held-out accuracy evidence.

**Repair:** share a single valid-timestep normalization routine across training and both inference routes. Require numerical agreement on identical sequences before rebuilding features.

### A04 — The upstream “holdout” audit does not hold data out of model training

**Evidence:** [analysis/full_pipeline_holdout_audit.py:64](/Users/chrissong/sp26_sagb_dj/analysis/full_pipeline_holdout_audit.py:64), [analysis/full_pipeline_holdout_audit.py:156](/Users/chrissong/sp26_sagb_dj/analysis/full_pipeline_holdout_audit.py:156), [README.md:153](/Users/chrissong/sp26_sagb_dj/README.md:153).

Both temporal and group checks evaluate already computed `lstm_p_strike`. Group loops discard the training indices and partition predictions from one existing model. There is no fold-specific upstream fit or preprocessing. Current LSTM training still splits randomly by pitch at [scripts/train_model.py:240](/Users/chrissong/sp26_sagb_dj/scripts/train_model.py:240).

These calculations can describe calibration differences across subsets. They cannot establish unseen-pitcher/batter generalization, future-season performance, or “zero overfitting.” A native synthetic group check returned perfect AUC without fitting any model, confirming the diagnostic-only behavior. Proper evaluation must exclude test observations during fitting; see the [scikit-learn evaluation guidance](https://scikit-learn.org/stable/modules/cross_validation.html).

**Repair:** label existing results as subgroup diagnostics. For actual holdouts, refit the upstream model and normalization while excluding each test group/time interval, then regenerate all downstream predictions.

### A05 — Saved results disagree and do not match the current estimators

**Evidence:** [scripts/export_leaderboard.py:46](/Users/chrissong/sp26_sagb_dj/scripts/export_leaderboard.py:46), [data/leaderboards/2025_dva_leaderboard.csv:1](/Users/chrissong/sp26_sagb_dj/data/leaderboards/2025_dva_leaderboard.csv:1), [scripts/run_commit_horizon.py:347](/Users/chrissong/sp26_sagb_dj/scripts/run_commit_horizon.py:347), [scripts/run_uncertainty_sensitivity.py:214](/Users/chrissong/sp26_sagb_dj/scripts/run_uncertainty_sensitivity.py:214).

| Artifact comparison | Observed mismatch |
|---|---|
| SQLite versus 2025 DVA CSV | Same 532 batter-season keys and identical pitch counts; **524 ranks**, 521 ratings, and all 532 actual/tendency DVA values differ. Maximum rank difference: 189. |
| Example: Andrew McCutchen | Database rank 16 / actual DVA 0.063880; CSV rank 9 / actual DVA 0.051706. |
| Current DVA exporter | Reads the database directly and would overwrite the canonical CSV with this different result set. |
| Current DVA estimator versus CSV | Current `score_batters` always emits `ev_dva_eb`, `alignment_dva_eb`, and shrinkage diagnostics. The saved CSV lacks them. Its rating is not demonstrated to be the current EV-DVA ranking. |
| Horizon leaderboard | Its 421 rows lack `info_horizon_tau`, `info_horizon_se`, and `info_horizon_ms`, which the current smooth estimator always writes. |
| Uncertainty leaderboard | Its 352 rows lack `beta_edge_i` and `beta_perp_i`, which the current controlled estimator always writes. |
| Baseline provenance | No `oof_predictions_2025.parquet` exists locally; the 724,002-row feature file also lacks OOF predictions. |

The numerical examples in README/presentation findings therefore cannot be attributed to the revised methods merely because the source now implements them. In particular, the saved Stephenson coefficient 1.9821 is presented alongside claims about newly introduced controls.

**Repair:** select an authoritative run, record code/checkpoint/input hashes and metric definitions, rebuild the dependent artifacts together, and compare database/CSV/figures before publication. Preserve old outputs under explicit historical labels.

### A06 — The uncertainty script uses an invalid significance test and always declares significance

**Evidence:** [scripts/run_uncertainty_sensitivity.py:289](/Users/chrissong/sp26_sagb_dj/scripts/run_uncertainty_sensitivity.py:289), [README.md:125](/Users/chrissong/sp26_sagb_dj/README.md:125).

The script treats a sum of cross-validated predictive log-likelihood differences between penalized models as a classical chi-square likelihood-ratio statistic, using `len(eval_sample) * 2` degrees of freedom. These are not maximized nested likelihoods on the same fitting sample, so that reference distribution does not establish the printed p-value. The significance sentence is unconditional, including when the computed evidence is nonsignificant or negative. Classical LR comparisons concern restricted versus unrestricted fitted likelihoods; see the [statsmodels LR-test contract](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.compare_lr_test.html).

**Repair:** retain paired held-out losses and estimate uncertainty using a design that respects repeated games/batters. Report effect size, interval, and actual test result; condition the conclusion on those results.

### A07 — Several downstream analyses bypass the strict OOF prediction loader

**Evidence:** [scripts/run_mirage_decomposition.py:164](/Users/chrissong/sp26_sagb_dj/scripts/run_mirage_decomposition.py:164), [scripts/run_uncertainty_sensitivity.py:145](/Users/chrissong/sp26_sagb_dj/scripts/run_uncertainty_sensitivity.py:145), [analysis/stabilization_analysis.py:55](/Users/chrissong/sp26_sagb_dj/analysis/stabilization_analysis.py:55), [analysis/full_pipeline_holdout_audit.py:90](/Users/chrissong/sp26_sagb_dj/analysis/full_pipeline_holdout_audit.py:90).

These paths load a season model and predict on the analyzed data rather than joining verified batter-OOF predictions. The trainer's saved standalone model fits 80% of batters and uses the remaining 20% for early stopping; predicting all rows therefore mixes training and validation treatment. The temporal audit's missing-model fallback explicitly uses the second half for early stopping, which also disqualifies it as an untouched temporal test.

**Repair:** route retrospective residual analyses through the canonical strict OOF loader, using a common prediction-column convention. Use independently trained temporal models for future-period validation. Any exploratory saved-model fallback should be explicit and labeled.

### A08 — The acceleration audit cannot demonstrate prefix-only information availability

**Evidence:** [analysis/full_pipeline_holdout_audit.py:249](/Users/chrissong/sp26_sagb_dj/analysis/full_pipeline_holdout_audit.py:249), [analysis/full_pipeline_holdout_audit.py:273](/Users/chrissong/sp26_sagb_dj/analysis/full_pipeline_holdout_audit.py:273), [analysis/full_pipeline_holdout_audit.py:297](/Users/chrissong/sp26_sagb_dj/analysis/full_pipeline_holdout_audit.py:297).

The audit constructs early positions/velocities from the supplied fitted accelerations, then differences those constructed velocities to recover the same accelerations. It also uses full reconstructed flight duration to set the cutoff and remaining time. Near-zero reconstruction error is algebraically expected and does not show that independent early observations recover those inputs without retrospective information.

The proxy framing is a useful qualification, but calling this experiment a demonstrated “leak-free benchmark” or a calibrated Bayesian upper bound goes beyond its evidence.

**Repair:** describe this as a deterministic reconstruction/sensitivity check. Establish prefix validity with independently measured early samples or a fit performed only on genuinely truncated observations; compare against a simple analytical baseline. Keep biological-perception interpretation explicitly exploratory.

## P2 findings

### A09 — Mirage truth and strike probability use different horizontal zones

**Evidence:** [scripts/build_swing_features.py:143](/Users/chrissong/sp26_sagb_dj/scripts/build_swing_features.py:143), [scripts/build_swing_features.py:237](/Users/chrissong/sp26_sagb_dj/scripts/build_swing_features.py:237).

`true_strike` uses half-width 0.833 ft, while `in_zone` and the probability integral use 17/24 ≈ 0.7083 ft. Actual `derive_features` execution at `(x=.75, z=2.5)` gives `in_zone=0` and `true_strike=1`. Subtracting the wider-zone truth from the narrower-zone probability can manufacture large mirage deltas even for accurate location predictions.

**Repair:** use one explicit geometry for truth and probability, including any ball-overlap allowance. If retaining alternate zone definitions, name them separately and never subtract across them as though they were identical.

### A10 — Boundary-perpendicular uncertainty selects the wrong direction outside the zone

**Evidence:** [src/lstm_scoring.py:359](/Users/chrissong/sp26_sagb_dj/src/lstm_scoring.py:359), [scripts/run_uncertainty_sensitivity.py:138](/Users/chrissong/sp26_sagb_dj/scripts/run_uncertainty_sensitivity.py:138).

The sigma-selection calculation measures distance to infinite supporting lines, while the distance calculation uses the bounded rectangle. Native execution at mean `(2,3.49)`, sigmas `(.1,.4)`, and zone `x±.7083, z[1.5,3.5]` returns edge distance 1.29167 ft and sigma 0.4. The nearest boundary point lies directly left, so its normal sigma is 0.1.

**Repair:** determine the nearest boundary point before selecting the normal. Outside-corner projections require a defined corner convention and, for a displacement-normal definition, the full covariance in `nᵀΣn`.

### A11 — The kinematic audit mixes release-position and 50-foot velocity reference frames

**Evidence:** [analysis/full_pipeline_holdout_audit.py:240](/Users/chrissong/sp26_sagb_dj/analysis/full_pipeline_holdout_audit.py:240), [analysis/full_pipeline_holdout_audit.py:249](/Users/chrissong/sp26_sagb_dj/analysis/full_pipeline_holdout_audit.py:249), [src/physics.py:57](/Users/chrissong/sp26_sagb_dj/src/physics.py:57).

The audit treats Statcast `vx0/vy0/vz0` as velocities at release, although they are defined at y=50 ft in the [official Statcast documentation](https://baseballsavant.mlb.com/csv-docs). The core trajectory implementation already handles this reference conversion; the audit reimplements it incorrectly.

**Verification:** on 1,000 local 2025 rows, extending the audit's equations to the endpoint produced mean absolute errors of **1.43 inches horizontally and 4.52 inches vertically**, versus **0.040 / 0.040 inches** using the existing core physics. Its inferred flight duration averaged 3.14 ms longer. This comparison diagnoses the reference-frame error; it is not a new held-out model evaluation.

**Repair:** reuse the core trajectory/time functions, handle zero acceleration and invalid inputs, and regenerate sensitivity results. Also distinguish squared Pearson correlation from predictive `r2_score`; the former can remain high under systematic bias.

### A12 — Dual-baseline support is not connected to the documented training pipeline

**Evidence:** [scripts/train_swing_model.py:78](/Users/chrissong/sp26_sagb_dj/scripts/train_swing_model.py:78), [src/swing_model.py:475](/Users/chrissong/sp26_sagb_dj/src/swing_model.py:475), [src/swing_model.py:577](/Users/chrissong/sp26_sagb_dj/src/swing_model.py:577), [README.md:28](/Users/chrissong/sp26_sagb_dj/README.md:28).

The library defines a decision-time feature mode and optional perceptual model, but the trainer constructs only the default final-location model, offers no mode argument, and generates OOF predictions only for that model. `score_batters` does not supply the optional second model. No production caller computes `deception_swing_oe` from two independently cross-fitted baselines.

Furthermore, the decision-time feature list includes retrospective `pfx_x/pfx_z` at [src/swing_model.py:46](/Users/chrissong/sp26_sagb_dj/src/swing_model.py:46), without an independently verified cutoff-limited estimate. Removing final coordinates alone does not establish decision-time availability.

**Repair:** either label dual baselines as library/prototype support or wire both through training, OOF persistence, scoring, and artifact provenance. Define the early-information contract and exclude or reconstruct retrospective movement features accordingly. A difference between predictive baselines should not be presented as a causal measure of induced swings without additional identification.

### A13 — Nondefault audit years still use 2025 data, cutoff, model, and output name

**Evidence:** [analysis/full_pipeline_holdout_audit.py:64](/Users/chrissong/sp26_sagb_dj/analysis/full_pipeline_holdout_audit.py:64), [analysis/full_pipeline_holdout_audit.py:91](/Users/chrissong/sp26_sagb_dj/analysis/full_pipeline_holdout_audit.py:91), [analysis/full_pipeline_holdout_audit.py:229](/Users/chrissong/sp26_sagb_dj/analysis/full_pipeline_holdout_audit.py:229), [analysis/full_pipeline_holdout_audit.py:344](/Users/chrissong/sp26_sagb_dj/analysis/full_pipeline_holdout_audit.py:344).

The public entrypoint accepts `--year`, but temporal splitting uses literal July 1, 2025, the league model path and kinematic SQL hardcode 2025, and all years overwrite one summary filename. Supplying a prior season's features leaves the second temporal half empty and metric evaluation fails.

**Repair:** pass the season through every helper, derive the cutoff/model/query/output consistently, and validate that both subsets contain sufficient rows and classes.

### A14 — The documented information-horizon formula differs from the estimator

**Evidence:** [README.md:138](/Users/chrissong/sp26_sagb_dj/README.md:138), [docs/presentation/PRESENTATION_FINDINGS_2020_2025.md:197](/Users/chrissong/sp26_sagb_dj/docs/presentation/PRESENTATION_FINDINGS_2020_2025.md:197), [scripts/run_commit_horizon.py:267](/Users/chrissong/sp26_sagb_dj/scripts/run_commit_horizon.py:267).

The code computes a weighted average of absolute horizon midpoints. The displayed formula adds `tau_0` again. With a starting fraction of 0.25, this shifts the documented result by 0.25 and can place it outside the advertised grid.

**Repair:** remove the extra offset, or redefine the midpoint variables as offsets and update code/artifacts consistently. Do not attach newly claimed standard errors to the older grid-search artifact in A05.

### A15 — The presentation PDF still contains the old synthetic figure and metric definition

**Evidence:** [SAGB Conference Presentation - Chris Song.pdf](</Users/chrissong/sp26_sagb_dj/docs/presentation/SAGB Conference Presentation - Chris Song.pdf>), pages 19–24; [analysis/generate_cloud.py:51](/Users/chrissong/sp26_sagb_dj/analysis/generate_cloud.py:51); [src/swing_model.py:617](/Users/chrissong/sp26_sagb_dj/src/swing_model.py:617).

All 25 pages were freshly rendered and inspected. Page 19 still calls the manually specified schematic an LSTM output and labels probabilities approximately 48% and 94%. Integrating the distributions used by the current source gives **50.77% and 99.46%**. The current PNG generator correctly labels its output as schematic, but that repair has not propagated into the PDF.

Page 20 describes strike-alignment DVA, whereas current ratings use empirical-Bayes EV-DVA. Pages 22–24 show historical leaderboard values; for example PDF Tucker rating 127.4 differs from the current CSV's 128.7. The PDF is not documentation of the latest pipeline.

**Repair:** clearly label the existing deck as a historical presentation or regenerate figures, formulae, and results from one authoritative run. Distinguish schematic examples from measured model predictions. Page 14 also has a visible text collision beneath the title.

### A16 — Incremental downloads skip the remainder of partially downloaded days

**Evidence:** [src/data_handler.py:180](/Users/chrissong/sp26_sagb_dj/src/data_handler.py:180).

Incremental refresh starts at `max(saved_game_date) + 1 day`. A refresh during a day's games therefore records that day as complete; subsequent incremental refreshes never request its remaining pitches or later corrections.

**Repair:** re-fetch an overlapping date window and replace/deduplicate by stable pitch key. Test two refreshes where the second response adds pitches to the already cached last date.

### A17 — Player metadata refresh omits shard zero

**Evidence:** [src/data_handler.py:116](/Users/chrissong/sp26_sagb_dj/src/data_handler.py:116).

The Chadwick shard list contains `1…9,a…f`, omitting `people-0.csv`. A full metadata rebuild consequently excludes that shard's players and identifiers. The upstream file exists at the same naming pattern as the included shards.

**Repair:** enumerate all 16 hexadecimal shards and verify expected shard coverage before replacing cached metadata.

### A18 — The classical eye-scoring track discards covariance

**Evidence:** [src/scoring.py:44](/Users/chrissong/sp26_sagb_dj/src/scoring.py:44), [src/scoring.py:69](/Users/chrissong/sp26_sagb_dj/src/scoring.py:69), [scripts/backfill_physical_metrics.py:91](/Users/chrissong/sp26_sagb_dj/scripts/backfill_physical_metrics.py:91).

Both scalar and batch scorers multiply marginal probabilities; the persisted physical metrics contain only marginal sigmas. This conflicts with the repository's correlated-integration convention. A corner-centered Gaussian with marginal sigma 0.2 and rho 0.8 has correlated strike probability about 0.397584, while the classical scorer returns 0.25.

**Repair:** retain cross-axis covariance and use the same correlated rectangle integral, or explicitly scope this track as an independence approximation and amend its claims/conventions. The separate LSTM integration is already correlated. Physical metric/rating tables are currently empty, so this is a demonstrated modeling-contract defect, not a claim about corrupted saved physical rankings.

### A19 — The presentation suite exits successfully when analyses fail

**Evidence:** [analysis/run_presentation_suite.py:50](/Users/chrissong/sp26_sagb_dj/analysis/run_presentation_suite.py:50), [analysis/run_presentation_suite.py:78](/Users/chrissong/sp26_sagb_dj/analysis/run_presentation_suite.py:78).

Failures are caught and printed, but no failure result or nonzero exit status reaches the caller. The suite also lists every preexisting figure/CSV as generated, regardless of which analyses succeeded. This can let automation continue with stale outputs.

**Verification:** invoked the real suite with one intercepted child process that raises `CalledProcessError`. It printed `FAILED` but returned `None`; the entrypoint does not convert failure into a nonzero exit code.

**Repair:** aggregate failures and exit nonzero. Track outputs from the current run and report prior artifacts separately.

## P3 findings

### A20 — Exported player names are not repaired row by row

**Evidence:** [scripts/export_leaderboard.py:69](/Users/chrissong/sp26_sagb_dj/scripts/export_leaderboard.py:69).

The lookup runs only when the entire name series is null. Partial nulls remain unresolved; empty/whitespace strings also count toward the number of existing entries. Current published CSV names are populated, so this is a conditional defect rather than an observed blank-name export.

**Repair:** identify missing/blank names per row, coalesce stripped fields, and fetch only unresolved identifiers.

### A21 — Notebook 05 retains an incompatible rating schema

**Evidence:** [notebooks/05_eye_rating_leaderboard.ipynb](/Users/chrissong/sp26_sagb_dj/notebooks/05_eye_rating_leaderboard.ipynb), cells 3, 5–7.

The notebook queries `eye_ratings`, then requires `eye_sigma_ft`. Its saved output is an LSTM-shaped table containing `mean_sigma_x/mean_sigma_z`, followed by the saved `KeyError` in cell 5. The live legacy/physical/LSTM rating tables are empty. The notebook therefore does not provide a working current leaderboard demonstration.

**Repair:** select a single explicitly named modeling track and matching schema; explain the required generation step and regenerate saved outputs. Keep this distinct from the primary EV-DVA leaderboard.

## Additional limitations and confirmed improvements

These are not additional counted defects:

- The fixed run-value table in `src/swing_model.py:347` claims derivation from more than 3.8 million pitches, and `compute_dva` describes OOF empirical swing values, but no fitting/export procedure or fold-specific run-value provenance is supplied. Treat these as fixed supplied utility assumptions until their derivation is reproducible; their numerical accuracy was not certified by this audit.
- Learned Gaussian sigma is conditional residual uncertainty, not an identified human sensory-acuity measurement. An observational swing policy or information horizon also does not directly identify neurological commitment time. Existing proxy language improves the framing but does not replace held-out calibration or external validation.
- The all-years dataset path loads every season's tensors before allocating destination arrays. Given approximately 13 GiB of trajectory storage, all-years training remains a memory-planning concern on an 18 GiB laptop. No large allocation/OOM experiment was induced.
- Native dependency versions are unconstrained above their minimums; there is no committed lockfile or CI workflow. The successful isolated run establishes one working environment, not all Python/package combinations advertised by the metadata.
- A preexisting graphify graph was absent; findings rely on direct source, numerical, schema, database, and visual evidence. No graph rebuild was required for this audit.
- Earlier repairs are present: the primary LSTM integral incorporates rho; normalization statistics use training indices; outer batter folds use an inner stopping split; long flights are resampled; physical movement noise remains in feet; posterior fusion uses full covariance; backfill seeds are pitch-specific and failed rows are retryable; rating writers preserve seasons and migrate columns. These repairs do not resolve the integration/artifact problems above.

## Recommended repair order and acceptance gates

1. **Restore runnable pipeline integration:** A01–A03. Require a tiny real feature/training/scoring/export run using a temporary database and versioned cache, with cached/online prediction agreement.
2. **Choose consistent metric geometry and controls:** A09–A12, A18. Establish the intended observation contract and complete the dual-baseline implementation only if its comparison remains in the research claims.
3. **Replace unsupported evaluation:** A04, A06–A08, A11, A13. Train genuine upstream group/temporal holdouts, use strict downstream OOF baselines, and report uncertainty for paired held-out losses.
4. **Regenerate authoritative artifacts:** A05, A14–A15, A21. Record source/input/checkpoint hashes, cutoff, geometry, utility-table version, fold identities, and metric definition. Verify database/CSV/figures/deck all reference the same run.
5. **Harden routine execution:** A16–A17, A19–A20. Add overlapping refresh, full metadata coverage, failure exit status, and row-level name handling. Add CI with one documented environment and the small pipeline integration check.

**Completion gate:** existing unit tests passing is necessary but insufficient. Close the pipeline blockers with executed integration evidence, then regenerate and independently validate the results before describing the project as a validated perception/decision-rating pipeline.
