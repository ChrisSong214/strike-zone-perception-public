# Verification of the latest repairs

Reviewed October 18, 2026 against the [original audit](/Users/chrissong/sp26_sagb_dj/PROJECT_AUDIT_2026-10-18.md) and [second audit](/Users/chrissong/sp26_sagb_dj/PROJECT_AUDIT_2026-10-18_RESCAN.md).

**Verdict: the claim that all issues are resolved is not supported.** Many repairs are substantive and correct, but there are incomplete integrations, unchanged results, remaining evaluation gaps, and new regressions. The most urgent new problem is a ratings save operation that deletes existing results before failing against the current database schema.

Of the original 24 findings, **13 specific defects are resolved in source, 7 are partially addressed, and 4 remain open**. These categories do not count the additional findings below, and “resolved in source” does not mean the complete native runtime or regenerated outputs were validated. Nineteen of the 49 Python files changed since the previous source snapshot; all 49 currently parse.

## Highest-priority new regression

### V1 — P1: Saving new ratings would delete the existing season and then fail

[scripts/train_swing_model.py:230](/Users/chrissong/sp26_sagb_dj/scripts/train_swing_model.py:230) detects the newly added `season` column, deletes rows for that season, commits, and only then appends new ratings. The shared helper repeats this sequence at [src/db_manager.py:100](/Users/chrissong/sp26_sagb_dj/src/db_manager.py:100).

The actual `swing_ratings` table still contains the legacy columns, plus `season`. It lacks `dva_mean`, `restricted_chase_rate`, `swing_rate`, other diagnostics, `name`, and `updated_at`, which the new writer supplies. Therefore the append fails after the deletion has become permanent.

**Reproduction:** copied the current schema and all **532 rows** to an in-memory database, then executed the extracted current save helper with a current-format result. It raised `table swing_ratings has no column named dva_mean`; the in-memory table contained **0 rows afterward**. The trainer's own deletion/append sequence independently reproduced the same failure. The project database was not modified and still contains all 532 rows.

**Repair:** validate and migrate the complete schema before mutation, and make replacement atomic so a failed insertion preserves the prior season. This must be fixed before rerunning a pipeline that saves ratings.

## Other new or incomplete problems

### V2 — P2: The new prediction-file path can silently supply incorrect baselines

Prediction persistence is now implemented, and four standalone analyses prefer the saved file. That is a real improvement. However, [compare_bregman_ramirez.py:25](/Users/chrissong/sp26_sagb_dj/compare_bregman_ramirez.py:25) and the equivalent code in `stowers_kwan_full.py`, `compare_stowers_kwan_40.py`, and `find_elite_decisions.py` replace missing joined predictions with **0.5**. Two pitch rows joined to a one-row prediction artifact produce `[0.9, 0.5]` without an error. Missing coverage therefore changes DVA silently. Duplicate keys are also not rejected and can multiply pitch rows.

There are two further gaps:

- No `oof_predictions_2025.parquet` or `swing_pitch_predictions` table currently exists. The analyses still use the older saved-model fallback when the file is absent, so they do not yet reproduce the repaired out-of-fold leaderboard calculation.
- [scripts/train_swing_model.py:152](/Users/chrissong/sp26_sagb_dj/scripts/train_swing_model.py:152) saves non-OOF predictions with fold zero into the same `oof_predictions_YEAR.parquet` path when OOF is disabled. Consumers discard the fold column and announce the file as out-of-fold anyway.

Require a one-to-one, complete pitch-key match, valid fold provenance, and matching feature/model run identity. Distinguish an explicitly requested exploratory fallback from a reproducible retrospective rating.

### V3 — P2: Season-preserving writers are paired with season-blind readers

The writer now attempts to retain other seasons, but [get_final_leaderboard.py:15](/Users/chrissong/sp26_sagb_dj/get_final_leaderboard.py:15) takes the top 15 across every season. [generate_leaderboard_image.py:21](/Users/chrissong/sp26_sagb_dj/generate_leaderboard_image.py:21) selects every season and assigns new ranks. The comparison scripts load 2025 pitch features, query ratings without a season filter, and select the first row for each batter; see [compare_bregman_ramirez.py:13](/Users/chrissong/sp26_sagb_dj/compare_bregman_ramirez.py:13) and line 56.

Once a second season is present, these readers can combine one year's pitch analysis with another year's rating or render a mixed-year leaderboard. The current database contains only 2025, so this is a verified source-level incompatibility rather than an observed mixed-year output. Add explicit season and run selection to every consumer.

The physical and LSTM eye-rating scripts also still standardize and rank requested years together, so adding another year changes a season's rating scale.

### V4 — P2: Resume treats failed simulations as completed

[scripts/backfill_physical_metrics.py:106](/Users/chrissong/sp26_sagb_dj/scripts/backfill_physical_metrics.py:106) loads every stored pitch key as completed. Lines 132–134 skip those keys. Failed clouds return three NaNs, but lines 149–160 still save them and add their keys to the completed set.

An in-memory test of the actual upsert stores those NaNs as SQL NULLs; the exact resume query then includes the failed pitch. A later run skips it even if its input data has been repaired. Track success/failure explicitly and retry failed rows by default, or document an explicit retry policy.

### V5 — P2: Restarting a seeded backfill changes the remaining simulations

[scripts/backfill_physical_metrics.py:84](/Users/chrissong/sp26_sagb_dj/scripts/backfill_physical_metrics.py:84) resets one global random stream. Resume skips previously completed pitches without consuming their draws. Consequently the remaining pitches receive different random samples than in an uninterrupted run.

In a controlled two-record reproduction using the same real pitch inputs and seed 42, the second record's `(strike probability, sigma_x, sigma_z)` was approximately `(0.64, 0.3931, 0.2818)` uninterrupted and `(0.68, 0.2932, 0.2674)` after restarting with the first record completed. This is a reproducibility defect, not evidence of distributional bias. Derive randomness from the run seed and stable pitch key, and persist the simulation/covariance version.

### V6 — P2, modeling limitation: Physical/perceptual fusion still ignores cross-axis covariance

[src/perception.py:110](/Users/chrissong/sp26_sagb_dj/src/perception.py:110) updates the two marginal variances independently and rescales each particle coordinate. This fixes the original scalar-variance error, but it does not yield the full Gaussian posterior for a correlated physical cloud.

Using deterministic particles with prior covariance `[[1, 0.9], [0.9, 1]]` and unit independent visual noise, executing the current function gives covariance `[[0.5, 0.45], [0.45, 0.5]]`. The full Gaussian update `(P^-1 + I)^-1` gives approximately `[[0.37304, 0.28213], [0.28213, 0.37304]]`.

Use a full covariance update and matching particle transformation, or explicitly identify this as a marginal approximation and validate its consequences. The corrected correlated strike integral in the separate LSTM path does not resolve this fusion approximation.

### V7 — P2: Memory use is reduced in one stage but remains unbounded elsewhere

The normalization-statistics calculation is now chunked: the specific large training-subset copies from the second audit are gone. However, [src/dataset.py:45](/Users/chrissong/sp26_sagb_dj/src/dataset.py:45) loads every year's sequences into a list, and line 61 concatenates them while the source tensors remain alive. Saved tensor storage totals **13.06 GiB**, implying at least **26.12 GiB** of simultaneous sequence allocations during concatenation, before other data, on this 18 GiB machine. This is an allocation analysis; no large allocation or out-of-memory failure was induced.

Inference has a related issue: [scripts/build_swing_features.py:144](/Users/chrissong/sp26_sagb_dj/scripts/build_swing_features.py:144) accepts `batch_size` but passes the whole frame to `batch_predict`. A focused test requesting batches of two for seven rows observed one seven-row call. [src/lstm_scoring.py:181](/Users/chrissong/sp26_sagb_dj/src/lstm_scoring.py:181) constructs all device tensors and concatenates them before chunking model execution. Forward-pass chunks therefore do not bound input preparation memory. The current 2025 feature query selects 724,002 pitches.

Stream or memory-map training inputs and bound sequence construction as well as inference. The normalization comment claiming less than 150 MB is also too small for the default chunk's simultaneous float32/float64 intermediates.

### V8 — Definition mismatch: two different strike-zone labels coexist

[scripts/build_swing_features.py:94](/Users/chrissong/sp26_sagb_dj/scripts/build_swing_features.py:94) derives `in_zone` from the stored Statcast zone category, whereas lines 99–116 derive location bins from a 17-inch rectangular center-coordinate boundary. These are not equivalent in the current data: **65,067 of 724,002 eligible 2025 pitches** have an inside Statcast zone category but lie outside that rectangle. One example has `zone=1`, `plate_x=-0.77`, inside vertical bounds, but outside the rectangle's ±0.7083 ft horizontal bounds.

The repaired chase statistic uses the categorical definition; the probability integral and context bins use the rectangle. Neither definition alone is established as wrong by this comparison, but presenting them as the same zone is inconsistent. Choose and document the intended convention, including any ball-edge allowance, and label comparisons accordingly.

## Repairs that are confirmed

- **Outer-fold isolation:** default OOF training now selects stopping rounds using an inner batter split. Outer-fold labels are used only for evaluation/prediction.
- **Best iteration:** `LeagueSwingModel.predict` now supplies the selected iteration range and saves its metadata. A fake-booster test with best iteration 2 received `(0, 3)`. Native XGBoost execution was not attempted.
- **Actual-component shrinkage:** the new denominator adds a prior weight of 15. The former one-context reproduction now yields actual DVA approximately **0.001073 at n=1** and **0.460299 at n=200**, instead of 0.5 at both sample sizes.
- **Full-flight timing:** long trajectories are resampled across their complete duration. The earlier 115-sample pitch now selects 230 ms, approximately **40.67%** of its flight, instead of 48.63%. The remaining difference from 40% is sample rounding.
- **Particle alignment and invalid-state handling:** distance interpolation puts posterior paths on common planes. In a focused 100-path reproduction, all posterior paths ended at y=1.417 despite unequal original lengths. Missing release z is rejected; the release-y fallback with valid extension still works.
- **Resume/upsert mechanics:** inserting the same pitch twice updates one row without a primary-key collision. Failure/reproducibility limitations are V4–V5.
- **Figure and export guards:** the Gaussian distributions are now constructed before use, the figure title says schematic, and the regenerated PNG shows 50.8%/99.5%. The exporter rejects a missing season column and filters an explicit requested season.
- **Definitions/documentation:** standard outside-zone chase and restricted far-outside chase are separate source fields; the guidance now distinguishes the three modeling tracks and identifies LSTM sigma as residual uncertainty.

## Saved artifacts and remaining scientific claims

The **saved leaderboard remains stale**. Tucker still has the earlier chase rate `0.1061269146608315` and rating 127.4. The feature Parquet remains dated April 7; no new OOF prediction artifact exists. Adding a season column and re-exporting the CSV did not recompute these results.

The physical metrics, physical eye ratings, LSTM eye ratings, and legacy eye ratings tables now exist, but **each contains zero rows**. The former missing-table failure is partially repaired; it is not evidence of an operational populated pipeline. The two LSTM checkpoints remain byte-identical and identify 2025 training. There is still no completed saved all-years comparison.

The presentation PDF is byte-identical to the previous audit: SHA-256 `ca1f6cf6adf3b6a769f06c6050fe5478608f182842499d30f04242823d1c3b4f`. It therefore retains the older figure and incomplete final-score explanation. Updating the standalone PNG did not update the deck. Notebook 03 still overlays mirrored and unmirrored paths and conflicting plate planes; notebook 02 still selects a pitcher independently of its hardcoded title.

The interpretation gaps remain material. A documentation caveat does not validate human perception from retrospective fitted physics. Decision-time coordinate error metrics do not establish interval coverage or strike-probability calibration. The LSTM training split remains random by pitch ([scripts/train_model.py:240](/Users/chrissong/sp26_sagb_dj/scripts/train_model.py:240)); newly saved game IDs are discarded by the dataset's label selection. DVA still rewards strike-probability agreement using a universal 0.5 threshold, while some descriptions call decisions optimal or correct without count-specific baseball utility.

## Disposition of every original finding

| Original ID | Status | Verification / remaining limitation |
|---|---|---|
| 1 — Builder syntax | Resolved | Builder and all other Python files parse. |
| 2 — Human-perception interpretation | Partial | Guidance caveat added; model validation and contradictory source claims remain. |
| 3 — Correlation omitted from strike probability | Resolved | Correlated integration and callers remain in source; native SciPy integration not rerun. |
| 4 — Unequal leaderboard baseline evaluation | Resolved | Default leaderboard generation uses untouched outer folds; downstream artifact problems are V2. |
| 5 — Stale geometry/results | Open | Existing features/ratings remain unreconstructed. |
| 6 — Synthetic figure representation | Partial | Source and standalone PNG corrected; presentation unchanged. |
| 7 — Movement units | Resolved | Perturbations remain in feet. |
| 8 — Scalar posterior variance / conflicting tracks | Resolved | Original scalar calculation and track descriptions corrected; additional covariance limitation is V6. |
| 9 — Path alignment | Resolved | Common distance-plane interpolation passes focused reproduction. |
| 10 — Missing release-y behavior | Resolved | Fallback works; additional finite-state checks now reject reproduced invalid cases. |
| 11 — Empty cloud as certainty | Resolved | Failure returns NaNs rather than zero uncertainty. |
| 12 — Backfill restart | Partial | Upsert/resume added; failed-row retry and deterministic restart remain broken. |
| 13 — Preprocessing/split | Partial | Training-only chunked scaling works in source; grouped evaluation absent and load-time capacity issue remains. |
| 14 — Decision-time validation | Partial | Coordinate metrics added; calibration and regenerated evidence missing. |
| 15 — Truncated-flight timing | Resolved | Complete-flight resampling replaces suffix truncation. |
| 16 — Confidence cancellation | Resolved | Actual-component prior denominator provides shrinkage. |
| 17 — Restricted chase mislabeled | Resolved | Separate standard/restricted fields now exist; saved results remain stale and zone convention needs clarification. |
| 18 — Utility interpreted as optimal baseball value | Open | Strike-agreement utility and overstatements persist. |
| 19 — Database/notebook schema | Partial | Separate method tables exist but are empty; new ratings schema is incompatible with the writer. |
| 20 — Season/provenance | Partial | Season handling added incompletely; V1–V3, missing run identity, overwritten prediction history/default model. |
| 21 — Core dependency declarations | Resolved | Required packages declared; installation/runtime compatibility not validated. |
| 22 — Presentation formula | Open | PDF unchanged; final blend/aggregation remains incompletely documented. |
| 23 — Notebook plots/title | Open | Relevant notebook cells remain inconsistent. |
| 24 — Obsolete combined-pipeline guidance | Resolved | Three tracks and real workflows documented. Calling the current state production-ready remains stronger than the evidence. |

For the second report's R1–R12: **R1, R2, R3, R5, R6's specific normalization-copy defect, R9, R10, and R11 are resolved in source; R4, R8, and R12 are partial; R7 remains open.** Related broader or newly exposed failures are detailed above. These IDs overlap the original findings and should not be added to the original count.

## Verification scope

- Parsed 49 Python files; ran the existing three physics boundary unit tests successfully in the previously working bundled runtime.
- Used isolated current-source functions, deterministic numerical examples, fake runtime interfaces where stated, read-only SQLite queries, in-memory database copies, source inspection, and saved-artifact hashes/metadata.
- Tests do not cover the full pipeline. Existing boundary tests even expect rejection of zero-acceleration motion; their passing result does not validate the general physics solver. No new regression tests accompany these repairs in the source snapshot.
- Did not invoke the project's blocked `.venv`, import new native numerical packages, execute full model training/inference, rebuild features, backfill the database, or regenerate presentation outputs.
- Source hashes were unchanged during review. Source, live database, existing reports, checkpoints, notebooks, and existing outputs were preserved. This report is the new project deliverable; imports/tests may create caches.

**Recommended next step:** repair the atomic schema migration first, then enforce prediction/season/run consistency and finish failed-row handling. After those changes, regenerate the dependent artifacts and run an end-to-end verification. Scientific interpretation and calibration require explicit evidence beyond successful execution.
