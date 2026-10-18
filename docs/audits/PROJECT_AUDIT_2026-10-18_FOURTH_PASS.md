# Fourth audit: verification of the latest fixes

Reviewed October 18, 2026, against the [previous verification report](/Users/chrissong/sp26_sagb_dj/PROJECT_AUDIT_2026-10-18_GEMINI_VERIFICATION.md) and its V1–V8 findings.

**Verdict: several important fixes are confirmed, but “everything is fixed” remains incorrect.** The previous swing-ratings deletion failure is repaired. Failed-simulation retry, deterministic restart, correlated fusion, feature batching, and zone consistency have also improved. New regressions remain in the eye-rating pipelines and CSV export, while prediction provenance and saved-output validity remain incomplete.

Sixteen Python files changed since the previous audit. All 49 Python files parse, and the three existing physics boundary tests pass. These checks do not establish an end-to-end pipeline pass.

## Highest-priority findings

### F1 — P1: The installed pandas version causes eye-rating saves to lose season history

Both eye-rating scripts now standardize each season with `groupby('season', group_keys=False).apply(standardize_season)`, followed by `reset_index(drop=True)`: [scripts/run_eye_rating.py:149](/Users/chrissong/sp26_sagb_dj/scripts/run_eye_rating.py:149) and [scripts/run_eye_rating_lstm.py:144](/Users/chrissong/sp26_sagb_dj/scripts/run_eye_rating_lstm.py:144).

The project's installed package metadata identifies **pandas 3.0.0**. Its local Python source excludes grouping columns from the frames passed to `apply`. This behavior is also documented in the [official pandas 3.0 release notes](https://pandas.pydata.org/docs/whatsnew/v3.0.0.html). The new standardization functions return copies of those frames without restoring `season`; `group_keys=False` and the subsequent index reset do not restore it either.

The resulting frame reaches [src/db_manager.py:127](/Users/chrissong/sp26_sagb_dj/src/db_manager.py:127) without a season column. The helper then follows line 133's whole-table `if_exists='replace'` branch, deleting other seasons and removing the season column from the new table.

**Reproduction:** executed the current standardization functions using the equivalent `include_groups=False` behavior in the working pandas 2.2.3 runtime, then passed their output to the actual current save helper against an in-memory database. A pre-existing 2024 row disappeared when saving 2025, and the replacement table had no `season` column. Both physical and LSTM paths reproduced this. The installed pandas 3.0 native environment was not imported.

This is a separate failure from the former 532-row swing-ratings bug. That earlier bug is fixed; this new issue affects the physical/LSTM eye-rating writers. Their current project tables are empty, so this review did not observe existing eye-rating history being lost.

**Repair:** explicitly preserve season through aggregation, preferably using group transforms or assigning the group key deliberately. Require a valid season column in a season-preserving writer instead of silently replacing the entire table when it is missing.

### F2 — P2: Both eye-rating scripts crash after saving

The new `standardize_season` functions define `mu` and `sigma` locally, but the outer functions still use those names in their summary prints: [scripts/run_eye_rating.py:157](/Users/chrissong/sp26_sagb_dj/scripts/run_eye_rating.py:157) and [scripts/run_eye_rating_lstm.py:152](/Users/chrissong/sp26_sagb_dj/scripts/run_eye_rating_lstm.py:152).

Executing the current aggregation/summary segments with toy data and stubbed writers raises **`NameError: name 'mu' is not defined`** in both cases. The error occurs after saving, so the scripts can modify outputs and then report failure without printing the leaderboard or returning normally. Print explicit per-season summaries or remove the invalid aggregate references.

### F3 — P2: The “verified OOF” loader does not verify OOF provenance

[src/swing_model.py:537](/Users/chrissong/sp26_sagb_dj/src/swing_model.py:537) checks prediction-key uniqueness and coverage, but never checks `is_oof` or `oof_fold`. A complete two-row artifact explicitly containing **`is_oof=False` and `oof_fold=0`** was accepted even with **`require_oof=True`**, and the helper printed “Loaded verified out-of-fold baseline predictions.”

The latest trainer correctly separates new in-sample and OOF filenames, but the loader still accepts stale, mislabeled, or old-format non-OOF files. Validate provenance fields, fold assignments, prediction values, and feature/model run identity before calling an artifact verified.

The silent 0.5 fill from the previous version is gone. Missing coverage and duplicate prediction keys now raise in strict mode. However, all four standalone consumers still use the permissive default, allowing absent or incomplete prediction artifacts to switch to a saved-model baseline. No OOF artifact exists in the current project, so this fallback remains the active path.

### F4 — P2: Model fallback prefers an unrelated default over the requested season

[src/swing_model.py:558](/Users/chrissong/sp26_sagb_dj/src/swing_model.py:558) tries `swing_model.xgb` before `swing_model_YEAR.xgb`. The training script overwrites the default alias after each season's fit.

In an isolated test requesting **2024**, with both default and 2024-specific models available, the actual helper chose the unversioned default. After another season is trained, analyses can therefore use that season's baseline despite having the correct season-specific model available. Prefer the requested season's model and verify its metadata; require an explicit choice for a generic fallback.

### F5 — P2: The repaired schema migration creates duplicate player-name columns in CSV exports

The database migration preserves legacy `player_name` and adds the current trainer's `name` column. Newly saved rows contain NULL in the old column and a populated value in `name`. [export_leaderboard.py:61](/Users/chrissong/sp26_sagb_dj/export_leaderboard.py:61) then renames `name` to `player_name` without coalescing or removing the existing column.

**Reproduction:** used the live table schema in memory, saved a current-format row through the new helper, and executed the export transformation into an in-memory CSV. The header contained **two `player_name` columns**. On readback, `player_name` was blank and the real value appeared under `player_name.1`.

Normalize to one canonical name field before export, handling existing-but-empty legacy columns explicitly.

### F6 — P2: One surviving physical trajectory produces a NaN posterior

The full covariance repair works for a sufficiently populated cloud. With only one valid trajectory, however, [src/perception.py:113](/Users/chrissong/sp26_sagb_dj/src/perception.py:113) calls sample covariance with insufficient observations. It returns a 2×2 NaN matrix. The following shape check accepts it, and the matrix operations propagate nonfinite values.

Executing the actual function with one finite two-point trajectory returned `x=[NaN, NaN]`, `z=[NaN, NaN]`, and finite y, accompanied by numerical warnings. It did not return an explicit failure. Check sample count and covariance finiteness before fusion and define the insufficient-particle behavior.

### F7 — P2: Season selection is still incomplete and one new option mislabels figures

The main viewer, image query, and three targeted comparison scripts now filter seasons. But [generate_leaderboard_image.py:103](/Users/chrissong/sp26_sagb_dj/generate_leaderboard_image.py:103) still renders **“2025 DVA Leaders”** for every `--year` value and writes the same output filenames. If 2024 data is available, `--year 2024` produces a 2025-labeled figure containing 2024 results.

Season-blind queries also remain in [get_t30_b30.py:17](/Users/chrissong/sp26_sagb_dj/get_t30_b30.py:17), [compare_call.py:14](/Users/chrissong/sp26_sagb_dj/compare_call.py:14), [find_twins.py:16](/Users/chrissong/sp26_sagb_dj/find_twins.py:16), [find_best_example.py:23](/Users/chrissong/sp26_sagb_dj/find_best_example.py:23), [find_example.py:7](/Users/chrissong/sp26_sagb_dj/find_example.py:7), and notebook 05. The main viewer and image generator still omit their season filter if the column is missing, rather than rejecting unknown-season data.

Complete season selection across readers, titles, and filenames. The current database contains only 2025 swing ratings, so mixed-season output remains a conditional source defect rather than an observed current leaderboard.

## Verification of the previous V1–V8 findings

| Prior finding | Current disposition | Evidence |
|---|---|---|
| V1 — Delete existing ratings before failed append | **Resolved for the reported failure** | Current helper migrates missing columns before replacement. A successful save preserved a separate season. A forced SQLite insert failure preserved all 532 prior 2025 rows plus a synthetic 2024 row. Trainer now calls the shared helper. F1 is a separate missing-season path. |
| V2 — Incorrect/incomplete prediction-file handling | **Partial** | Silent 0.5 fills removed; duplicate keys and strict-mode missing coverage rejected; new non-OOF files have separate names. Provenance is not checked, permissive fallback remains, and no current OOF file exists; F3–F4. |
| V3 — Season-blind consumers | **Partial** | Targeted readers improved, but several remain unfiltered and figure titles are hardcoded. New eye-rating grouping has F1–F2 failures. |
| V4 — Failed simulations treated as completed | **Resolved for the reproduced failure** | Resume requires all three metrics to be non-NULL. Actual upsert/query test excludes failed rows and retains successful rows. |
| V5 — Different random results after restart | **Resolved for identical run settings** | Per-pitch seed and explicit RNG reproduce exactly the same metrics despite unrelated global RNG draws. Configuration/version identity is still not persisted. |
| V6 — Ignored cross-axis covariance | **Resolved for an adequate cloud** | Full matrix update and particle transform match the correlated Gaussian example within 1.54×10⁻⁷, attributable to regularization. New one-particle case fails; F6. |
| V7 — Memory use | **Improved; full-scale feasibility unverified** | Feature inference now genuinely chunks preparation: seven rows with chunk size two produce calls `[2,2,2,1]`. Single-year dataset avoids concatenation. Multi-year loading still loads all source years before allocating/copying the destination; no full-scale memory benchmark was run. |
| V8 — Conflicting zone labels | **Resolved in feature source** | `in_zone` and context bins now use the same rectangle; the original category is retained separately. Previously contradictory example pitches now classify consistently. Saved features remain old. |

The new multi-year loader frees source tensors incrementally while copying into preallocated output. The previous report's **26.12 GiB resident-memory peak should not be carried forward as an established fact for this implementation**: destination pages may be committed lazily. Current source data alone contains approximately 13.06 GiB of sequence storage. The loader still scales with the total dataset, but this review did not establish an out-of-memory failure.

Retained repairs from earlier passes include the builder syntax fix, movement units, release-y fallback, finite-state checks, full-flight resampling, outer/inner batter split separation, best-iteration prediction, actual-component shrinkage, corrected schematic PNG, and explicit export season guard.

## Original unresolved findings that remain

The earlier audit's remaining issues are not all executable defects that can be closed by changing a few lines:

- **Human-perception interpretation (original 2):** documentation now qualifies sigma, but the input physics and endpoint likelihood remain unchanged. Perceptual validity is still unestablished, and some source descriptions retain stronger claims.
- **Saved results (5):** the current database still contains 532 old 2025 swing ratings. Tucker remains at rating 127.4 and chase rate `0.1061269146608315`. The feature file remains the April 7 artifact; no persisted OOF predictions exist. Re-exporting the same table did not recompute ratings.
- **Presentation (6 and 22):** the standalone PNG is corrected, but the PDF is unchanged. It retains the older figure and incomplete final-score explanation.
- **Evaluation (13 and 14):** the dataset now preserves game IDs, but the trainer still randomly splits individual pitches. There are no decision-time interval-coverage or strike-probability calibration results. Saved checkpoints remain unchanged.
- **Metric interpretation (18):** DVA still measures baseline-adjusted strike-probability agreement. It is not a count-specific baseball utility model, despite remaining “optimal”/“correct” descriptions.
- **Schemas and run identity (19 and 20):** the four physical/LSTM metrics and rating tables remain empty. Feature, model, and prediction artifacts still lack sufficient shared run identity. The prediction DB writer still replaces the entire prediction table. Backfill success is keyed only by pitch, so changing seed or covariance while resuming can mix configurations.
- **Notebook consistency (23):** notebook 03 still combines mirrored and unmirrored paths and conflicting plate planes; notebook 02 retains its selection/title mismatch. Notebook 05 retains stale saved output and a season-blind query.

The original backfill collision/restart mechanics are now repaired, although simulation configuration provenance remains incomplete. Other previously closed findings were inspected for retention, without claiming a new full native-runtime verification.

## Saved-artifact evidence

- `swing_features_2025.parquet` SHA-256: `c42c4d229bc041bf13107c2f14161c864c36c2d4dbdb1d7654b920676314bc72`; still April 7.
- Presentation SHA-256: `ca1f6cf6adf3b6a769f06c6050fe5478608f182842499d30f04242823d1c3b4f`; unchanged from earlier reviews.
- Both LSTM checkpoints remain byte-identical: `4c853b437096eba20236df09ac28a7f5255c59ee42cc8b2beaa0bcc637a30e95`. No completed all-years comparison artifact was found.
- `physical_pitch_metrics`, `physical_eye_ratings`, `lstm_eye_ratings`, and `eye_ratings` each contain zero rows. All 532 swing-rating rows remain intact in the live database.

## Scope and limits

This was a read-only code and artifact audit with isolated reproductions. Tests used the working bundled Python runtime, NumPy/Pandas, extracted current functions, fake model/scorer interfaces where stated, and in-memory SQLite copies. Pandas 3 behavior was established from installed metadata/source and official documentation, then reproduced using the equivalent supported mode in pandas 2.2.3.

No project `.venv` native libraries were loaded, and no security settings were changed. No training, full model inference, live backfill, feature rebuild, or presentation regeneration was performed. All 49 source files parsed and all three existing boundary unit tests passed; those tests do not cover the newly identified failures. Source hashes remained unchanged throughout this audit. The project database and existing outputs were preserved.

**Repair priority:** preserve season through eye-rating aggregation and reject seasonless writes; fix the post-save NameErrors; enforce baseline provenance and correct season-model selection; normalize the exported name field; handle insufficient particle clouds; finish season-aware readers. Then rebuild and verify the dependent artifacts. Calibration and perception claims still require measured evidence.
