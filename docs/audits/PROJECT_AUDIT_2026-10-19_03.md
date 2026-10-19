# Project audit — October 19, 2026, third pass

**Verdict: the earlier integration repairs now work on the checked paths, and the regenerated 2025 database and CSV agree. However, those regenerated results still use an incompatible legacy feature file. Other remaining risks concern mixed metric definitions across seasons, conclusions contradicted by the regenerated data, and new API/consumer defects.**

Scope: commit `d2f643f47f6f547f35a5c43ea5a000cb669161b3` plus the uncommitted working-copy changes present at the start of this audit, including three new audit-cycle test files. Compared against the [second audit](/Users/chrissong/sp26_sagb_dj/docs/audits/PROJECT_AUDIT_2026-10-19_02.md). Existing edits were preserved; this audit did not repair code or regenerate production data.

**Nine current findings: four P1, four P2, and one P3.** P1 concerns central result validity or publication accuracy; P2 concerns a material numerical or workflow defect; P3 is a smaller API regression. The numerical utility in T07 currently has no production caller, and the malformed predictions in T06 are a fixture, not an observed defect in the current OOF artifact.

## Fresh verification

| Check | Result | Scope and limitations |
|---|---|---|
| Full test suite | **81 passed, 2 warnings**, 6.05 seconds | Isolated Python 3.11, native thread counts limited to one. |
| Statement coverage | **1,207 / 5,976 statements, 20.20%** | An increase from the second audit's 15.24%; passing tests still do not certify the unexercised entrypoints. |
| Python parsing/static checks | **71 files parsed**, no syntax errors; no undefined-name/before-assignment findings | Does not detect the semantic/API defects below. |
| Package build | **Wheel built successfully** | Packaging check, not an end-to-end research run. |
| 2025 database/CSV parity | Same **532 batter-season rows**, zero mismatches across checked rank, rating, DVA, EV/EB, alignment, and perceptual/deception fields | Earlier numerical inconsistency is repaired. Current CSV has 26 columns with no nulls. |
| Current OOF artifact | **724,002 rows**, five folds, complete objective/perceptual predictions, no duplicate pitch keys, one fold per batter, correct season | Pitch keys and batter identities exactly cover the stored feature file. No independently authenticated training manifest is available. |
| Regenerated diagnostic schemas | Horizon includes `info_horizon_tau/se/ms`; uncertainty includes `beta_edge_i/beta_perp_i` | Prior missing-schema findings are repaired. Interpretation must still follow actual values. |
| Raw-pitch spot check | Three real raw-pitch LSTM predictions matched saved feature means/sigmas/correlations to floating-point precision | Small consistency check, not held-out calibration. |
| Physical integration repair | Actual miniature backfill migrated a legacy temporary database, retried a row lacking rho, preserved correlation through loading, and handled invalid physics without aborting | Current production physical tables are still empty. Their schema migration occurs when backfill runs. |
| Notebook 05 | Current cells 3–7 executed against read-only live data and loaded 532 rows with an explicit DVA fallback notice | Missing eye sigmas are expected because physical ratings are absent; this is not a measured eye-rating result. |
| Kinematic sensitivity | Read-only current 50,000-row function executed | No full temporal pipeline, training, or output regeneration. |
| New reproductions | Strict-loader malformed perceptual predictions, absent plotting examples, covariance distance, scalar/vector broadcasting, and batter naming checked | Temporary fixtures or intercepted persistence only. |

Logs, coverage, source fingerprints, and numerical summaries are under `/private/tmp/sagb-audit03-20261019-*`. No full-season model training, LSTM inference, production backfill, or write-enabled presentation suite was run. The unchanged historical PDF was not rerendered; the current README's leaderboard and case-study PNGs were visually inspected.

## Findings

### T01 — P1: the regenerated pipeline still trains and scores on legacy feature geometry

**Evidence:** [scripts/build_swing_features.py:108](/Users/chrissong/sp26_sagb_dj/scripts/build_swing_features.py:108), [scripts/build_swing_features.py:120](/Users/chrissong/sp26_sagb_dj/scripts/build_swing_features.py:120), [scripts/train_swing_model.py:60](/Users/chrissong/sp26_sagb_dj/scripts/train_swing_model.py:60), local `data/training/swing_features_2025.parquet`.

The 2025 model, OOF predictions, and leaderboards were regenerated, but the feature file they consume still implements earlier definitions. Current source uses rectangular Chebyshev distance and derives `in_zone` from that rectangle. The saved 724,002-row feature file instead contains Euclidean distance and the categorical Statcast zone indicator.

**Executed full-column comparison:**

| Saved feature versus current source definition | Mismatching rows |
|---|---:|
| `in_zone` versus current normalized Chebyshev feature-builder labels | **65,909** |
| `in_zone` versus direct physical rectangular bounds | **65,067** |
| `dist_from_center` versus `max(abs(loc_x_norm),abs(loc_z_norm))` | **718,198** |
| `loc_bin` versus current feature-builder bins | **103,740** |
| Saved distance versus Euclidean `hypot(loc_x_norm,loc_z_norm)` | **0** |
| Saved `in_zone` versus numeric Statcast zone categories 1–9 | **0** |

A concrete row has `plate_x=-.77`, vertical location inside its zone, and saved `in_zone=1`; the current horizontal bound is approximately `.7083` ft, so current source assigns zero. The two rectangular label counts differ because direct and normalized comparisons handle floating-point boundary values differently. Tucker's current exported chase rate `.1746575342` matches the stored flags, whereas current feature-builder labels give `.2310984308`.

This affects behavioral baseline features, Bayesian context bins, chase/zone rates, and uncertainty cohort selection. Mirage reconstructs its own rectangular truth labels, so those labels should not be described as still stale; its baseline nevertheless inherits the old feature geometry. Agreement between OOF keys, database rows, and CSV values cannot detect a shared stale upstream representation. The three matching LSTM spot checks establish consistency of model parameters, not consistency of derived zone features.

**Repair:** rebuild and version the feature file from current source, then retrain both baselines, regenerate OOF predictions, ratings, and dependent analyses together. Require a feature-definition/version fingerprint before training; file existence and matching pitch keys are insufficient. Verify derived columns against their defining equations before accepting the regenerated run.

### T02 — P1: multi-season analyses compare different metrics under the same column names

**Evidence:** [analysis/common_loader.py:74](/Users/chrissong/sp26_sagb_dj/analysis/common_loader.py:74), [analysis/common_loader.py:116](/Users/chrissong/sp26_sagb_dj/analysis/common_loader.py:116), [analysis/yoy_reliability.py:25](/Users/chrissong/sp26_sagb_dj/analysis/yoy_reliability.py:25), [analysis/breakout_detection.py:39](/Users/chrissong/sp26_sagb_dj/analysis/breakout_detection.py:39).

The regenerated 2025 leaderboard uses EV-DVA, while 2020–2024 retain the older strike-alignment blend. The common loader concatenates them without a metric-definition discriminator. Standardizing both to a 100/10 rating scale does not make their underlying estimands identical.

**Executed artifact check:**

| Seasons | Definition verified from every CSV row |
|---|---|
| 2020–2024 | `dva_mean = 0.5 * (tendency_dva + actual_dva)`; maximum numerical error about `1e-16`. |
| 2025 | `dva_mean = ev_dva_eb` exactly; difference from the old blend reaches `0.0853024`. |

Year-over-year reliability treats the 2024→2025 definition change as same-metric persistence. Breakout analysis subtracts the two ratings and interprets the result as a player change. The new schema README documents different columns but does not prevent these comparisons.

There is an important distinction for predictive validity: [analysis/predictive_validity.py:48](/Users/chrissong/sp26_sagb_dj/analysis/predictive_validity.py:48) uses the earlier season's rating to predict the later season's outcome. Its latest predictor season is 2024. Those forecasting results therefore evaluate the **old blend**, not the new production EV-DVA; the issue is unsupported attribution, not 2025 predictor rows entering those folds.

**Repair:** regenerate all analyzed seasons under a common metric, or explicitly separate analyses by metric version. Persist `metric_definition` and validate it before comparing or pooling years. Report the existing forecast experiment as validation of the historical metric until EV-DVA is actually evaluated.

### T03 — P1: the regenerated horizon data do not support the claimed biomechanical confirmation

**Evidence:** [README.md:141](/Users/chrissong/sp26_sagb_dj/README.md:141), [docs/presentation/PRESENTATION_FINDINGS_2020_2025.md:210](/Users/chrissong/sp26_sagb_dj/docs/presentation/PRESENTATION_FINDINGS_2020_2025.md:210), [scripts/run_commit_horizon.py:412](/Users/chrissong/sp26_sagb_dj/scripts/run_commit_horizon.py:412).

Documentation says bat tracking confirms that mechanics constrain the inferred horizon. The script prints mechanistic interpretation unconditionally after computing correlations. The regenerated export provides no detectable linear association with either named mechanical variable.

**Recomputed from all 421 complete exported rows:**

| Association with `info_horizon_tau` | Pearson r | p-value |
|---|---:|---:|
| Mean bat speed | −0.01814 | 0.71057 |
| Mean swing length | 0.00037 | 0.99398 |

These null associations do not establish absence of every possible relationship; they do not substantiate the claimed confirmation. The published population counts are also stale: the current `≤.32 / (.32,.42] / >.42` buckets contain **8 / 351 / 62** hitters, versus **110 / 137 / 174** in the findings document. Current examples include Betts `.393±.024`, Stanton `.332±.087`, and Freeman `.420±.016`, rather than the older displayed horizons.

**Repair:** report the measured associations and their limitations; remove unconditional confirmation/mechanistic language. Update population and player summaries from the same regenerated artifact. Any stronger biomechanical claim needs a separately specified validation experiment.

### T04 — P1: README figures and research summaries still describe superseded results

**Evidence:** [README.md:42](/Users/chrissong/sp26_sagb_dj/README.md:42), [README.md:85](/Users/chrissong/sp26_sagb_dj/README.md:85), [assets/dva_top10.png](/Users/chrissong/sp26_sagb_dj/assets/dva_top10.png), [assets/stowers_vs_kwan.png](/Users/chrissong/sp26_sagb_dj/assets/stowers_vs_kwan.png), [docs/presentation/PRESENTATION_FINDINGS_2020_2025.md:180](/Users/chrissong/sp26_sagb_dj/docs/presentation/PRESENTATION_FINDINGS_2020_2025.md:180).

Database/CSV parity is repaired, but not every consumer was regenerated. Fresh visual inspection found the README top-ten image still puts Tucker first at 128.7; the current CSV puts **Leo Rivas first at 129.7 and Tucker third at 125.5**. Both qualify above 500 pitches, so this is not a minimum-sample explanation.

The README case-study image still shows Stowers **114.8** versus Kwan **97.1**, supporting its “passivity penalty” story. Current EV-DVA ratings are **Stowers 103.9 / rank 196** and **Kwan 108.1 / rank 104**. The direction reverses. The historical comparison may be retained as a clearly identified historical metric, but cannot describe the current EV-DVA leaderboard.

Additional current-versus-published discrepancies:

| Item | Published | Current artifact/computation |
|---|---|---|
| Stephenson uncertainty coefficient | `+1.98`, `p<.001` | `beta_U_i=.9097`, SE `.5180`. |
| Soderstrom uncertainty coefficient | `+1.77`, `p<.001` | `1.2262`, SE `.4617`. |
| Acuña uncertainty coefficient | `−2.06`, `p<.001` | `−1.0515`, SE `.4495`. |
| Linear tangent horizontal R² | `.8237` | Current 50,000-row function and summary CSV: `.8154875`. |
| Linear tangent vertical R² | `.2010` | Current function and summary CSV: `.1566840`. |

The coefficient values alone establish that these are different results; this audit does not endorse ordinary Wald inference for the penalized fits. The corresponding prescriptive scouting recommendations should not retain the old significance claims. The current ballistic calculation returns vertical R² `.8667218` and MAE `3.64515` inches, but those ballistic outputs are omitted from the saved summary rows, weakening the documentation's provenance trail.

**Repair:** regenerate every current-facing figure and summary from the chosen run, or explicitly label the retained historical metric. Add a publication consistency check that verifies depicted leaders, player comparisons, reported coefficients, and summary metrics against their actual source artifacts.

### T05 — P2: new LSTM leaderboard names identify the opposing pitcher

**Evidence:** [scripts/run_eye_rating_lstm.py:43](/Users/chrissong/sp26_sagb_dj/scripts/run_eye_rating_lstm.py:43), [scripts/run_eye_rating_lstm.py:118](/Users/chrissong/sp26_sagb_dj/scripts/run_eye_rating_lstm.py:118), [scripts/run_eye_rating_lstm.py:135](/Users/chrissong/sp26_sagb_dj/scripts/run_eye_rating_lstm.py:135).

The new naming path copies the first Statcast `player_name` in a batter's pitch group into that batter's rating. In these records the field identifies the pitcher.

**Live identity check:** a real row pairs batter `672386` (Alejandro Kirk) with pitcher `808967` and `player_name='Yamamoto, Yoshinobu'`. Local player metadata independently maps `808967` to Yoshinobu Yamamoto. The same batter's rows contain other opposing pitchers' names.

**Executed consumer check:** the actual leaderboard routine with a scorer fixture and save recorder persisted `batter_id=672386, player_name='Yamamoto, Yoshinobu'`. This demonstrates the bad name reaching persistence, not an already observed corrupted production LSTM leaderboard; that production table remains empty.

**Repair:** resolve names by batter identifier using the verified batter lookup. If unavailable, retain the identifier or an explicit unresolved label. Test identity correctness rather than propagation of an arbitrary `player_name` string.

### T06 — P2: strict OOF loading does not validate perceptual prediction values

**Evidence:** [src/swing_model.py:833](/Users/chrissong/sp26_sagb_dj/src/swing_model.py:833), [src/swing_model.py:943](/Users/chrissong/sp26_sagb_dj/src/swing_model.py:943).

The loader checks objective probabilities for finiteness and `[0,1]` bounds, then copies the optional perceptual prediction column without equivalent checks. Downstream scoring now correctly preserves those supplied perceptual predictions, making their validation important.

**Executed reproduction:** an otherwise valid two-batter/two-fold temporary artifact with objective probabilities `.2` and perceptual values `[inf, 2]` passed `require_oof=True` and was announced as verified. Actual scoring can then produce infinite deception and invalid perceptual EV values.

The current real OOF file passed finite/bounds checks; this finding concerns the strict loader's contract under malformed input.

**Repair:** validate every supplied prediction column, including perceptual values and coverage, before accepting the artifact. Reject invalid columns under strict mode and preserve explicit provenance for any permitted exploratory fallback.

### T07 — P2: the covariance distance helper does not compute its documented nearest-boundary distance

**Evidence:** [src/lstm_scoring.py:333](/Users/chrissong/sp26_sagb_dj/src/lstm_scoring.py:333), [src/lstm_scoring.py:373](/Users/chrissong/sp26_sagb_dj/src/lstm_scoring.py:373).

The Mahalanobis helper inflates inside marginal distances by `1/sqrt(1-rho²)` and uses Euclidean coordinate clipping outside. The nearest point in the covariance metric can move along an edge; neither shortcut minimizes the documented quadratic form over the bounded rectangle.

**Native comparison against independent bounded minimization of all four edge quadratics:** zone `x±17/24,z[1.5,3.5]`, sigmas `.1/.1`, rho `.8`:

| Mean | Function result | Minimum covariance distance |
|---|---:|---:|
| `(0,2.5)` | 11.80556 | 7.08333 |
| `(1,2.5)` | 4.86111 | 2.91667 |

No production caller was found; this does not establish an error in the current primary leaderboard. A newly added regression test asserts the same incorrect inside scaling, so test agreement does not validate the mathematics.

**Repair:** minimize the quadratic along each bounded edge and choose the minimum, or rename/document a deliberately different quantity. Use an independent analytical/optimization reference in the test.

### T08 — P2: a valid uncertainty cohort crashes if its featured plotting examples are absent

**Evidence:** [scripts/run_uncertainty_sensitivity.py:415](/Users/chrissong/sp26_sagb_dj/scripts/run_uncertainty_sensitivity.py:415), [scripts/run_uncertainty_sensitivity.py:459](/Users/chrissong/sp26_sagb_dj/scripts/run_uncertainty_sensitivity.py:459).

The new guard handles an entirely empty leaderboard, but the plot still indexes Betts and Ohtani rows unconditionally. A different season or qualification threshold can produce a valid nonempty cohort without one or both players. The main analysis saves its CSV before calling this plot, so failure can leave a partial run.

**Executed reproduction:** a nonempty one-other-hitter cohort raised `IndexError: index 0 is out of bounds for axis 0 with size 0`. Figure writes were intercepted.

**Repair:** check each selected player, choose available examples, or skip an unavailable comparison with an explicit message. Verify a nonempty cohort that excludes every preferred example.

### T09 — P3: scalar means combined with vector uncertainty now fail

**Evidence:** [src/lstm_scoring.py:292](/Users/chrissong/sp26_sagb_dj/src/lstm_scoring.py:292), [src/lstm_scoring.py:317](/Users/chrissong/sp26_sagb_dj/src/lstm_scoring.py:317).

Scalar-return detection considers only `mu_x/mu_z`, although uncertainty, correlation, or zone bounds can broadcast the result into a vector. It then calls `.item()` on that vector.

**Executed reproduction:** `compute_p_strike(0., 2.5, np.array([.1,.2]), .1, 3.5, 1.5, rho=.2)` raises `ValueError: can only convert an array of size 1 to a Python scalar`. The distance helpers use the same scalar-detection pattern. Ordinary batch scoring with vector means is unaffected.

**Repair:** determine scalar return from the fully broadcast result shape. Test mixed scalar/vector means, sigmas, correlation, and bounds in addition to the all-scalar fixtures.

## Verification of the previous ten findings

| Previous item | Current status |
|---|---|
| R01 uncertainty `rho` wrapper crash | **Resolved:** wrapper accepts/forwards rho; current tests and native call pass. |
| R02 perceptual OOF overwrite | **Resolved for valid supplied predictions:** equal objective/perceptual OOF values remain equal with standalone model calls forbidden. T06 identifies a separate input-validation gap. |
| R03 OOF column mismatch | **Resolved:** verified predictions populate all consumer aliases and optional perceptual values. Consumers still permit `require_oof=False`; absence/corruption can therefore fall back to a saved model unless policy is tightened. |
| R04 physical return arity | **Resolved:** empty/nonfinite results return four values; actual miniature backfill handles failure without aborting. |
| R05 2025 numerical/schema mismatch | **Numerical and schema repair verified:** database/CSV parity, OOF coverage, model modes, and diagnostic columns now pass. Upstream feature reproducibility and broader cross-season/publication consistency remain open as T01–T04. Unified code/input/checkpoint hashes are still absent. |
| R06 upstream holdout framing | **Partially addressed:** functions, subgroup CSV categories, and findings now identify fixed-model subgroup stability. Remaining “cross-validation,” module-header holdout wording, and leak-free proxy claims should be reconciled. No new upstream exclusion/refitting experiment has been demonstrated. |
| R07 physical covariance integration | **Resolved for migrated workflow:** backfill migrates the old schema and retries missing-rho rows; loader retains rho. Existing empty production physical schema still requires running backfill before its leaderboard query. |
| R08 additive covariance | **Resolved:** previous nonzero-visual-noise fixture now returns `.315494669`, matching the intended covariance model. |
| R09 kinematic computation/reporting | **Computation repaired; publication trail partial:** current 50,000-row function agrees with the saved linear summary. README/findings still differ and ballistic metrics are not exported; see T04. |
| R10 no-qualified-batter crash | **Resolved:** empty leaderboard has a stable schema and temporal evaluation checks it before merging. |

Notebook 05's earlier schema/runtime failure is also resolved for current data through an explicit DVA fallback. That successful execution should not be described as validation of a populated physical/LSTM eye-rating table.

## Recommended repair order

1. Rebuild/version the stale features and dependent current-season artifacts first (T01). Resolve cross-season metric comparability and update biomechanical/publication conclusions from the measured artifacts (T02–T04). These affect the project's central research claims.
2. Correct batter identity and validate all accepted OOF columns (T05–T06).
3. Fix the plotting cohort guard and broadcasting contract; correct the unused covariance-distance utility before adopting it in analyses (T07–T09).
4. Preserve the runtime repairs already verified. Add consumer-level tests for identity, metric version, complete prediction validation, and current-facing artifact consistency instead of adding only formula-mirroring or all-scalar tests.

The project has progressed from broken integration toward a runnable current-season pipeline. The next gate is consistent scientific interpretation and versioned evidence across consumers, not another claim that passing unit tests certifies every research result.
