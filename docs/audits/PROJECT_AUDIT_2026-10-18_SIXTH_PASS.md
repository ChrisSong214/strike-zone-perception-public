# Sixth audit — latest fixes and repository reorganization

Reviewed October 18, 2026 against the [fifth audit](/Users/chrissong/sp26_sagb_dj/docs/audits/PROJECT_AUDIT_2026-10-18_FIFTH_PASS.md) and its source fingerprints. The project now has a Git repository; this audit includes the working-tree changes beyond commit `b5e1945`, not just committed code.

**Verdict: the five specific fifth-pass failures have been addressed, but the project is not yet verified end to end.** The CSV name transformation, image-generation calls, four reported OOF validation cases, metadata format compatibility, and physical-scoring qualification all pass focused checks. Three additional workflow defects remain, along with incomplete artifact provenance and scientific validation.

## Current findings

### S1 — P2: LSTM ratings still admit batters below the usable-pitch minimum

The physical scorer now enforces its minimum correctly. The analogous LSTM path still qualifies on input rows at [scripts/run_eye_rating_lstm.py:98](/Users/chrissong/sp26_sagb_dj/scripts/run_eye_rating_lstm.py:98), then appends any non-None scoring result at [line 117](/Users/chrissong/sp26_sagb_dj/scripts/run_eye_rating_lstm.py:117), without checking the result's `n_pitches` against `min_pitches`.

The new SQL and finite-value filters help, but finite inputs do not guarantee a valid trajectory. [src/lstm_scoring.py:181](/Users/chrissong/sp26_sagb_dj/src/lstm_scoring.py:181) excludes pitches whose trajectory preparation fails. `score_batter` reports the surviving count and failed count, so the caller has the information needed to reject an undersized fit.

**Reproduction:** constructed 200 rows with finite required numeric fields. Of these, 199 had zero y acceleration and were rejected by the actual trajectory function; one produced a valid trajectory. Executed the actual `score_batter`, strike-probability calculation, and leaderboard function, supplying a finite model-output fixture for that one surviving trajectory and recording the would-be database save:

```text
REAL_TRAJECTORY_SUCCESS 1 of 200
ACTUAL_SCORE_BATTER 1 used 199 failed
SAVED_BELOW_MINIMUM [{'batter_id': 100, 'season': 2025,
                     'n_pitches': 1, 'eye_rating': 100.0}]
```

This is an isolated failure-path test, not evidence that 199 such failures occurred for a real batter. No LSTM training/inference or live database write was performed. It establishes that the advertised minimum of 200 usable observations is not enforced after scoring.

**Repair:** require `result['n_pitches'] >= min_pitches` before adding a rating, and report exclusions and failed counts. Keep the existing prefilter as an efficiency improvement.

### S2 — P2: The documented first pipeline command is invalid

[README.md:198](/Users/chrissong/sp26_sagb_dj/README.md:198) instructs users to run:

```text
python3 scripts/build_training_data.py --years 2025
```

But [scripts/build_training_data.py:210](/Users/chrissong/sp26_sagb_dj/scripts/build_training_data.py:210) converts every positional argument to an integer; it has no `--years` option. Executing that exact argument-parsing block with the documented arguments raises:

```text
ValueError: invalid literal for int() with base 10: '--years'
```

The supported invocation is `python3 scripts/build_training_data.py 2025`. Either correct the README or implement the documented option. This reproduction stops at parsing and does not import PyTorch or build data.

The fresh-checkout workflow also needs a database preparation step: this builder reads the `statcast` SQL table, while the database is Git-ignored. The displayed sequence does not explain how local raw files become that table. Existing local data masks this setup prerequisite.

### S3 — P2: Default CSV export does not update the advertised leaderboard file

[scripts/export_leaderboard.py:31](/Users/chrissong/sp26_sagb_dj/scripts/export_leaderboard.py:31) creates `data/leaderboards`, but line 33 does not use that directory. The default destination is the bare filename `2025_dva_leaderboard.csv`, relative to the caller's current directory.

The README links to `data/leaderboards/2025_dva_leaderboard.csv`. Running its export command from the repository root therefore writes a different file and leaves the advertised/tracked leaderboard untouched.

**Reproduction:** executed the actual exporter against an isolated migrated-schema database and intercepted `to_csv`. Name coalescing succeeded, but the recorded destination was:

```text
2025_dva_leaderboard.csv
```

**Repair:** join the default filename to `default_dir`; retain explicit `--output` behavior. Verify the exported destination matches the file linked in the README.

## Fifth-pass repair verification

| Fifth-pass finding | Current result | Evidence from this pass |
|---|---|---|
| A1 — Missing NumPy import crashes migrated-name export | **Resolved for the reported failure** | NumPy is imported. Actual exporter coalesced `[NULL, 'Existing']` plus `['Current', 'Other']` into one `player_name` column containing `['Current', 'Existing']`. No network lookup or CSV write was needed. S3 is a separate destination problem. |
| A2 — Image calls accidentally nested inside themselves | **Resolved for the reported failure** | Current module invoked exactly two save calls against a plotting recorder for a 20-row 2024 fixture. Destinations were `assets/dva_top10_2024.png` and `assets/dva_bottom10_2024.png`; both titles used 2024. This verifies control flow and naming, not rendered image appearance. |
| A3 — Four malformed OOF fixtures accepted | **All four reported cases now rejected; broader provenance remains incomplete** | Missing fold column, fold zero, null OOF flag, and a batter split across artifact folds each raise `ValueError` in strict mode. A valid two-batter/two-fold fixture succeeds. |
| A4 — JSON reader cannot read pickle metadata | **Resolved for the reported format mismatch** | Model writer now stores JSON with season; trainer supplies the year. Fallback reader supports JSON and legacy pickle. Both formats emitted the expected 2023-versus-2024 warning in isolated tests. |
| A5 — Physical scorer qualifies unusable rows | **Resolved for the reported failure** | Actual physical leaderboard invoked the optimizer zero times for 0, 1, or 199 usable observations out of 200, and once for 200 usable observations. The saved count was 200. Direct empty-fit attempts raise. The optimizer was stubbed; this checks eligibility and reporting, not optimization accuracy. |

The older season-preserving ratings writer and per-season summary repairs remain in source. The prediction-table writer now deletes/appends the selected season when a season column exists, instead of replacing all seasons. Its legacy seasonless-table branch still explicitly drops that table; that is a migration limitation rather than preservation of unknown historical seasons. No live prediction table currently exists to migrate.

## Remaining provenance and interpretation gaps

### OOF validation is stronger, but still not complete run verification

The new grouping check at [src/swing_model.py:606](/Users/chrissong/sp26_sagb_dj/src/swing_model.py:606) trusts artifact batter IDs when present; it does not reconcile them with the requested input rows. An isolated artifact assigning two pitch keys to different batters/folds was accepted when the input assigned both keys to the same batter. The loader also accepted a file named for 2025 whose `season` column said 2024.

These are inconsistent-artifact fixtures, not claims that the repaired trainer creates them. They show that “verified” currently means selected field checks and pitch-key coverage, rather than verified agreement with source data and model history. Feature/checkpoint/cutoff/run identity is still not persisted and matched across artifacts. Four standalone consumers still allow saved-model fallback, which is explicitly announced but is not OOF evaluation. No current OOF file exists.

Generic model fallback now warns about a known mismatched season or a legacy missing year. It still proceeds. The reported format defect is fixed; the warning should not be interpreted as proof of compatibility.

### Saved results have not established the revised pipeline

Read-only database and file checks found:

- **532 swing-rating rows**, still all legacy 2025 results. Tucker remains at rating **127.4**, chase rate **0.1061269146608315**.
- **Zero rows** in `physical_pitch_metrics`, `physical_eye_ratings`, `lstm_eye_ratings`, and `eye_ratings`.
- No persisted prediction files under `data/training` and no `swing_pitch_predictions` table.
- The feature Parquet, best LSTM checkpoint, and presentation PDF retain their previous hashes. The old duplicate `model_2025_only.pt` is no longer present; no replacement completed comparison result was found.

| Artifact | SHA-256 |
|---|---|
| `data/training/swing_features_2025.parquet` | `c42c4d229bc041bf13107c2f14161c864c36c2d4dbdb1d7654b920676314bc72` |
| `data/models/best_model.pt` | `4c853b437096eba20236df09ac28a7f5255c59ee42cc8b2beaa0bcc637a30e95` |
| `docs/presentation/SAGB Conference Presentation - Chris Song.pdf` | `ca1f6cf6adf3b6a769f06c6050fe5478608f182842499d30f04242823d1c3b4f` |
| `data/leaderboards/2025_dva_leaderboard.csv` | `54fc30aceb987fe45d8e2d7e9e83cf0925e4b001debc547eb513f9e2d9b9f0af` |

### Scientific and notebook consistency

The overall architecture remains a coherent research prototype. Its validation does not yet support stronger claims about human perception or optimal baseball choices. The new README says the model uses information actually available to the batter and describes estimated location in the baseline, while `XGBOOST_FEATURES` includes measured final `plate_x`, `plate_z`, and derived final-zone values. The perception inputs remain retrospectively reconstructed physics. Those are legitimate modeling choices for some analyses, but they need to be described accurately and do not establish a human-information constraint.

The LSTM trainer still randomly partitions individual pitches. Decision-time interval coverage, strike-probability calibration, and held-out-game results remain unestablished. DVA still scores baseline-adjusted strike agreement rather than count-specific baseball utility. The CLAUDE.md caveat about residual uncertainty is appropriate; the stronger README language should be aligned with it.

Notebook repairs made progress: notebook 02 now derives the selected pitcher's name; notebook 03 moves the plate/zone to the correct front-plane convention; notebook 05 filters 2025 when season metadata exists. Notebook 05 still falls back to unfiltered data when the season column is missing and labels it 2025. The three notebooks' saved outputs are unchanged relative to the committed pre-repair versions, so source edits have not refreshed their displayed results. The presentation PDF is likewise unchanged.

## Verification scope

- **53 physical Python files**, plus two root-level compatibility symlinks, parse successfully. Their 55 path fingerprints remained unchanged during this audit. The physical file count includes five scratch scripts.
- **Three existing physics tests passed through unittest.** Three existing perception/uncertainty test functions also passed when executed directly with their unused pytest imports omitted.
- Additional isolated checks covered export schema/naming, image invocation/naming, OOF rejection/acceptance, both metadata formats, physical eligibility, LSTM post-scoring eligibility, and the documented builder parser.
- The attempted pytest invocation stopped because the bundled runtime has **no pytest installed**. The full suite was not run. There are currently **16 test functions/methods**, so the README's “9 passed” statement is also stale; this audit does not replace it with an unmeasured full-suite pass.
- Checks used the bundled Python runtime, NumPy/Pandas, source extraction, in-memory SQLite, and explicitly identified model/optimizer/plotting stand-ins. No project `.venv`, PyTorch, XGBoost, SciPy, or Parquet native inference/reading dependencies were loaded for these checks.
- No source fixes, training, backfill, feature regeneration, live database edits, or asset overwrites were performed. This report is the deliverable. The PDF was fingerprinted, not rerendered in this pass.

**Next repair order:** enforce the LSTM's usable-pitch threshold; correct the documented builder command and database setup prerequisite; fix the default CSV destination; finish artifact identity checks. Then rebuild and verify the results before presenting the repository as an end-to-end validated pipeline.
