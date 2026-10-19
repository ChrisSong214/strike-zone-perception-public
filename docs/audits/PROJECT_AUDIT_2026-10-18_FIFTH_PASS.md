# Fifth audit — verification after the latest repairs

Reviewed October 18, 2026. Baseline: [fourth audit](/Users/chrissong/sp26_sagb_dj/PROJECT_AUDIT_2026-10-18_FOURTH_PASS.md). This report assesses the current files, including 17 Python files changed since that audit.

**Verdict: several important repairs work, but the project is not fully fixed.** Season-preserving eye-rating saves, summary printing, season-specific model preference, and singleton trajectory handling are repaired. Two new output regressions were introduced. Prediction provenance remains incomplete, and an additional physical-scoring qualification error was reproduced. Existing saved results have not been regenerated to establish that the repaired pipeline works end to end.

The overall research workflow is coherent: prepare trajectories, estimate location uncertainty, derive strike probabilities, and compare decisions with a baseline. That coherence does not establish that fitted uncertainty measures human eyesight, that strike agreement is optimal baseball decision-making, or that current published outputs were produced by the current code.

## Current actionable findings

### A1 — P2: CSV export crashes on the newly migrated name schema

At [export_leaderboard.py:51](/Users/chrissong/sp26_sagb_dj/export_leaderboard.py:51), the new name-coalescing branch uses `np.nan`, but the module never imports NumPy. The undefined name also occurs at lines 52 and 58.

The shared ratings writer preserves the legacy `player_name` column and adds the trainer's `name` column. Consequently, a newly saved row can have `player_name=NULL` and `name="Kyle Tucker"`. Executing the actual export transformation on that shape raises:

```text
NameError: name 'np' is not defined
```

The exception happens before CSV writing. The current legacy table has only `player_name`, so exporting its old rows avoids this branch and does not verify the repaired migration/export workflow.

**Repair:** import the dependency or use a pandas-native missing value. Verify a migrated frame containing both columns produces exactly one populated `player_name` CSV column. The previous duplicate-column defect is addressed structurally, but its replacement currently fails at runtime.

### A2 — P2: Leaderboard image generation exits without creating images

The year-aware filenames and both calls to `make_table` were indented inside `make_table` itself: [generate_leaderboard_image.py:104](/Users/chrissong/sp26_sagb_dj/generate_leaderboard_image.py:104). No top-level call remains.

Executing the current module with safe database access and a drawing recorder read the current leaderboard, completed normally, and recorded **zero image-save calls**. A separate in-memory fixture containing 20 valid 2024 rows also produced zero drawing calls. These checks replaced plotting imports and made no image writes.

Calling `make_table` manually would reach the unconditional self-call at line 107 after saving its first figure, causing recursion instead of producing the intended pair of images. The existing image files therefore cannot demonstrate that the updated script succeeds.

**Repair:** move lines 104–108 outside the function. Confirm one normal invocation saves exactly two images, with the selected year in their titles and appropriate filenames.

### A3 — P2: Strict out-of-fold validation still accepts incomplete provenance

The loader now rejects explicit false flags, invalid probabilities, duplicate pitch keys, and incomplete coverage. However, [src/swing_model.py:557](/Users/chrissong/sp26_sagb_dj/src/swing_model.py:557) uses `.all()` without rejecting missing flags, and [line 562](/Users/chrissong/sp26_sagb_dj/src/swing_model.py:562) checks fold count only when the fold column happens to exist.

Actual-function tests with `require_oof=True` accepted every case below and printed “Loaded verified out-of-fold baseline predictions”:

| Invalid or unverifiable artifact | Observed result |
|---|---|
| `is_oof=True`, missing `oof_fold` column | Accepted |
| `is_oof=True`, fold IDs `[0, 1]` | Accepted; current trainer reserves zero for the in-sample path |
| `is_oof=[True, None]`, fold IDs `[1, 2]` | Accepted; missing flag skipped by `.all()` |
| Two pitches belonging to the same batter assigned different outer folds | Accepted despite the batter-grouped training contract |

These fixtures demonstrate missing loader checks; they do not prove that the current trainer emits these malformed files. The trainer's repaired outer/inner split remains present. No current OOF prediction artifact exists to validate against it.

All four standalone baseline consumers still use the permissive default and can fall back to a saved booster. Such inference does not become out-of-fold evaluation merely because it returns valid probabilities.

**Repair:** require complete boolean flags and valid fold assignments consistent with the training contract; verify batter grouping; persist and match feature/model/run identity. Require OOF predictions where evaluation claims depend on held-out predictions. Field checks alone cannot prove how a model was trained.

### A4 — P2: The generic model's season check reads the wrong metadata format

Requested-season model preference is now correct. The remaining fallback check is incompatible with the model writer:

- [src/swing_model.py:123](/Users/chrissong/sp26_sagb_dj/src/swing_model.py:123) writes `.meta` as binary pickle, containing feature names, importances, and best iteration.
- [src/swing_model.py:593](/Users/chrissong/sp26_sagb_dj/src/swing_model.py:593) opens that file as text and attempts JSON decoding.
- Parsing errors are silently swallowed at lines 601–602. The writer also supplies no `year` field.

A focused fallback test requesting 2024 with pickle metadata containing `year=2023` selected the generic model without issuing the intended mismatch warning. With both the generic and 2024-specific models available, a separate test correctly selected the 2024 model.

**Repair:** use one documented metadata format, persist training season and run identity, and explicitly handle absent or incompatible provenance. The old ordering bug is closed; generic fallback still does not establish season compatibility.

### A5 — P2: Physical-rating qualification counts observations that fitting discards

[scripts/run_eye_rating.py:93](/Users/chrissong/sp26_sagb_dj/scripts/run_eye_rating.py:93) qualifies batters using all joined rows. [src/scoring.py:95](/Users/chrissong/sp26_sagb_dj/src/scoring.py:95) then drops rows with missing required values. The pipeline fits without checking the surviving count and reports the original unfiltered count as `n_pitches` at [line 117](/Users/chrissong/sp26_sagb_dj/scripts/run_eye_rating.py:117).

This is reachable because failed simulations are deliberately stored with NULL metrics, and the SQL join does not exclude them.

Actual-source fixtures containing 200 joined rows and respectively **0, 1, or 199 usable rows** qualified in all three cases. The scorer retained only those 0, 1, or 199 rows, while the result would report 200. For zero usable rows, the actual loss function returned `-0.0` at both sigma bounds: an empty, flat objective. `fit_sigma` has no empty-data guard.

The optimizer itself was not run in this reproduction; its SciPy interface was excluded to avoid loading additional native libraries. The demonstrated failure is that qualification passes and fitting is invoked with no observations. Any returned estimate would have no supporting likelihood data.

**Repair:** filter usable finite observations before qualification, enforce `min_pitches` on the observations actually fitted, reject empty fits, and report the fitted count. The physical-rating tables are currently empty, so this is a reproduced workflow defect, not evidence of an already published bad physical rating.

## Verification of fourth-pass findings

| Prior finding | Current disposition | Verification |
|---|---|---|
| F1 — Eye-rating aggregation loses season and replaces history | **Resolved for the reported failure** | Both scripts explicitly restore season. Actual aggregation followed by the actual writer preserved a 2024 row while adding 2025 rows in isolated databases. Missing/NULL season is rejected before connecting. |
| F2 — Post-save summary NameErrors | **Resolved** | Both current aggregation/summary segments execute with two-season fixtures and use valid per-season summary variables. |
| F3 — Unverified OOF provenance | **Partial** | Explicit false flags and invalid probabilities are now rejected; incomplete flags/folds/grouping remain accepted. See A3. |
| F4 — Generic model preferred over requested season | **Ordering defect resolved; provenance partial** | Seasonal model wins when both exist. New generic metadata check is incompatible with writer format. See A4. |
| F5 — Duplicate exported player-name columns | **Blocked by replacement regression** | Coalescing replaces the duplicate rename, but its missing import crashes on the migrated schema. See A1. |
| F6 — Singleton cloud yields NaN posterior | **Resolved for the reported failure** | Empty, singleton, two-particle, and correlated-cloud tests return finite paths or explicit empty output without warnings. |
| F7 — Incomplete season selection | **Standalone queries repaired; overall partial** | Previously listed scripts now filter the selected year and reject unknown-season schemas. Image title/filename expressions are fixed but unreachable because of A2. Notebook 05 remains season-blind. |

The singleton posterior now uses a small fallback prior covariance. A singleton cannot represent nonzero ensemble covariance; this remains a limitation, not a recurrence of the NaN failure. With four correlated particles, the full posterior covariance still matches the reference Gaussian calculation within **1.54 × 10⁻⁷**, attributable to regularization.

Retained trajectory checks also passed: full-flight resampling, release-y fallback, rejection of missing release-z, empty-cloud failure metrics, and deterministic per-pitch simulation. The sampled real trajectory selects approximately **40.6688%** of full flight, and differing physical path lengths align to the observed trajectory length and plate plane. Prior swing split, best-iteration, shrinkage, and zone-consistency repairs remain in source.

## Remaining project-level limitations

These are carried forward separately from the five executable/validation findings above:

- **Saved outputs remain stale.** The database still contains 532 legacy 2025 swing ratings. Tucker remains rank 1, rating 127.4, chase rate `0.1061269146608315`. Re-exporting this table does not rerun feature extraction or scoring. No OOF prediction file or prediction table is present.
- **Physical/LSTM results are not established.** `physical_pitch_metrics`, `physical_eye_ratings`, `lstm_eye_ratings`, and `eye_ratings` each remain empty. Both saved LSTM checkpoints are unchanged and byte-identical.
- **Evaluation remains incomplete.** [scripts/train_model.py:236](/Users/chrissong/sp26_sagb_dj/scripts/train_model.py:236) splits individual pitch indices, rather than holding out whole games. Its evaluation reports coordinate errors and average uncertainty without decision-time interval coverage or strike-probability calibration. The preserved game IDs have not become a grouped validation policy.
- **Scientific interpretation remains narrower than some descriptions.** Retrospective physical trajectories and a swing-decision likelihood do not establish human sensory uncertainty. DVA measures baseline-adjusted strike-probability agreement, without a count-specific baseball utility model. Documentation caveats help, but do not constitute validation of eyesight or optimal-decision claims.
- **Run provenance is incomplete.** Features, predictions, checkpoints, and ratings lack a shared verified run identity. Backfill resume keys do not encode changed simulation settings. [scripts/train_swing_model.py:173](/Users/chrissong/sp26_sagb_dj/scripts/train_swing_model.py:173) still replaces the entire prediction table, so generating another season can remove prior-season database predictions.
- **Presentation and notebooks remain inconsistent.** The presentation PDF is unchanged from the prior audited version; the corrected standalone schematic has not updated that PDF. Notebook 02's selection/title mismatch, notebook 03's mixed coordinate conventions, and notebook 05's stale output remain. Notebook 05 still queries all eye-rating seasons while labeling results as 2025 at [line 187](/Users/chrissong/sp26_sagb_dj/notebooks/05_eye_rating_leaderboard.ipynb:187).

### Artifact fingerprints rechecked

| Artifact | Current SHA-256 | Status |
|---|---|---|
| `data/training/swing_features_2025.parquet` | `c42c4d229bc041bf13107c2f14161c864c36c2d4dbdb1d7654b920676314bc72` | Unchanged April 7 artifact |
| `SAGB Conference Presentation - Chris Song.pdf` | `ca1f6cf6adf3b6a769f06c6050fe5478608f182842499d30f04242823d1c3b4f` | Unchanged |
| `data/models/best_model.pt` and `model_2025_only.pt` | `4c853b437096eba20236df09ac28a7f5255c59ee42cc8b2beaa0bcc637a30e95` | Both unchanged and identical |

## Verification scope and limits

- All **49 Python files parse**. The **three existing physics boundary unit tests pass** on this pass.
- Focused checks used actual current source, extracted functions where necessary, isolated/in-memory SQLite, bundled NumPy/Pandas, fake model/plotting interfaces, and a pure-Python normal-CDF stand-in for the scoring fixture. Passing these checks is not a full native-runtime or end-to-end test.
- No project `.venv` native libraries were loaded. No macOS security settings were changed. No training, backfill, feature rebuild, or full model inference was launched.
- The audit preserved project source, database contents, and existing outputs. Only this new report was added. The 49 source hashes were checked against the starting snapshot.
- PDF/notebook concerns were retained from prior inspection and checked against unchanged artifacts; this pass did not rerender the entire presentation.

**Recommended order:** repair CSV and image generation; enforce physical-fit qualification; finish OOF and model provenance checks; then regenerate dependent features, predictions, ratings, figures, and presentation results from a recorded configuration. Evaluate calibration and grouped generalization before strengthening scientific claims.
