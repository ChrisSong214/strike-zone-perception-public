# Seventh audit — closeout verification

Reviewed October 18, 2026 against the [sixth report](/Users/chrissong/sp26_sagb_dj/docs/audits/PROJECT_AUDIT_2026-10-18_SIXTH_PASS.md). Scope: the latest working-tree changes, reported failure cases, retained repairs, setup instructions, and current saved artifacts.

**Verdict: all three sixth-pass code/workflow findings are repaired in focused execution. The two additional OOF mismatch examples are also rejected now. No additional runtime regression was found in the changed paths reviewed. One fresh-install setup instruction remains incorrect.**

This supports closing the specific repair items below. It does not establish an end-to-end pipeline pass or validate the old saved research results. The next useful verification stage is a controlled integration run, rather than another unrestricted static audit.

## Confirmed repairs

| Item | Fresh verification | Result |
|---|---|---|
| S1 — LSTM admits fewer than 200 usable pitches | Executed the actual leaderboard function with 0, 1, 199, and 200 surviving observations. Only the 200-observation result reached the save recorder. | **Resolved** |
| S1 — Original one-survivor reproduction | Used the actual trajectory function on 200 finite-input rows: 199 failed trajectory generation and one succeeded. Actual scoring/aggregation, with a model-output fixture, now excludes the batter and performs no save. | **Resolved** |
| S2 — Documented `--years` command crashes | Executed the current CLI entrypoint with a builder recorder for default arguments, `--years 2025`, positional `2024 2025`, and `--years 2024 2025`. Each dispatched exactly the requested seasons. | **Resolved** |
| S3 — Default export misses the advertised file | Executed the exporter against an in-memory migrated-schema database and intercepted `to_csv`. Destination is now the absolute `data/leaderboards/2025_dva_leaderboard.csv` path. | **Resolved** |
| OOF artifact/input batter IDs disagree | Actual strict loader rejects the previously accepted mismatched-identity fixture. | **Resolved for the reported case** |
| OOF file contains the wrong season | Actual strict loader rejects a 2025-named fixture containing 2024 season values. | **Resolved for the reported case** |

Retained repairs also passed focused checks:

- Migrated `name`/`player_name` fields produce one populated export column.
- Image generation invokes exactly two saves with 2024-specific titles and asset filenames; plotting was replaced with a recorder, so this does not certify rendered appearance.
- Missing fold IDs, fold zero, null OOF flags, same-batter fold splits, duplicate pitch keys, and nonfinite probabilities are rejected. A valid two-batter/two-fold fixture is accepted.
- Both JSON and legacy pickle model metadata produce the expected year-mismatch warning.
- Physical scoring rejects 0, 1, and 199 usable observations at a minimum of 200, accepts 200, and rejects direct empty-fit attempts. The optimizer was stubbed; these checks establish qualification and reporting.

## Remaining actionable issue

### R1 — P2: The new database-setup command does not set up a database

[README.md:201](/Users/chrissong/sp26_sagb_dj/README.md:201) describes the following as a download/setup step before building training tensors:

```text
python3 -c "from src.data_handler import StatcastDataHandler; StatcastDataHandler().fetch_statcast(2025)"
```

[src/data_handler.py:233](/Users/chrissong/sp26_sagb_dj/src/data_handler.py:233) implements `fetch_statcast` as a local Feather-file reader. It does not download data and does not write the `statcast` SQLite table. Even with cached Feather files, the returned DataFrame is discarded by the documented command.

**Reproduction:** executed the actual reader method against an empty temporary cache. It returned zero rows and created no files. Then executed the builder's actual SQL-loading function against an empty in-memory database; it failed with `no such table: statcast`. Neither check performed network requests or touched the project database.

**Required correction:** document the real download/cache operation, `update_local_sc(just_current=False, specific_year=2025)`, followed by the separate initial database-population operation, `sync_feather_to_sql([2025])`. The latter replaces the existing `statcast` table on its first write, so these instructions must distinguish initializing a new database from updating one with existing seasons or edits. Alternatively, the README's existing-database route works if that database already contains the required table and schema.

This is a fresh-setup documentation defect, not a recurrence of the repaired `--years` parser failure. No download or synchronization was run during this audit.

## Test coverage and evidence limits

The new [qualification test](/Users/chrissong/sp26_sagb_dj/tests/test_lstm_scoring.py:46) duplicates a miniature filtering loop rather than calling production `run_leaderboard`. It would still pass if the actual guard were removed. This does not invalidate the repair: this audit separately exercised the real function. For durable regression protection, replace the duplicated loop with a test that calls the real function using a stub scorer and save recorder.

There are now **17 test functions/methods**, matching the README's count. The README's full “17 passed” result was **not independently reproduced here**: the bundled safe runtime has no pytest installed, and the project environment associated with macOS alerts was not used.

Fresh checks completed:

- All **53 physical Python files** and two compatibility symlink paths parse. Their 55 path fingerprints remained unchanged during the audit.
- **Three existing physics tests passed through unittest.**
- **Three existing perception/uncertainty test functions passed directly**, omitting their unused pytest imports.
- Isolated integration checks exercised the current source for the repair matrix above. The builder's expensive work, model predictions, optimizer, plotting, and output writes were replaced with explicit recorders/fixtures where stated.

These results are stronger than syntax inspection, but remain narrower than a complete native-runtime test suite or full pipeline execution.

## Saved outputs and scientific claims

Current read-only checks still show **532 swing ratings**, no persisted OOF prediction files, and zero rows in each of `physical_pitch_metrics`, `physical_eye_ratings`, `lstm_eye_ratings`, and `eye_ratings`.

The following artifacts remain byte-for-byte unchanged from the earlier audited versions:

| Artifact | SHA-256 |
|---|---|
| `data/training/swing_features_2025.parquet` | `c42c4d229bc041bf13107c2f14161c864c36c2d4dbdb1d7654b920676314bc72` |
| `data/models/best_model.pt` | `4c853b437096eba20236df09ac28a7f5255c59ee42cc8b2beaa0bcc637a30e95` |
| `docs/presentation/SAGB Conference Presentation - Chris Song.pdf` | `ca1f6cf6adf3b6a769f06c6050fe5478608f182842499d30f04242823d1c3b4f` |

The README now accurately states that the baseline uses measured final location and that DVA is baseline-adjusted strike-probability agreement, rather than a full run-expectancy utility model. That is a meaningful documentation improvement. It does not supply the previously missing held-out-game evaluation, decision-time uncertainty calibration, human-perception validation, or shared feature/model/run identity. Those remain research and reproducibility tasks, separate from whether these specific code fixes work.

## Closeout recommendation

Close S1–S3 and the two demonstrated OOF mismatch cases. Correct R1 and strengthen the production-path regression test. Then run the complete suite and a controlled pipeline integration check in a known-working environment, recording the configuration and generated artifact identities. Regenerate dependent results before claiming the saved leaderboard or presentation validates the repaired code.

No project source, existing outputs, or live database contents were changed by this audit. Only this report was added. No training, backfill, downloads, database synchronization, or project `.venv` execution was performed.
