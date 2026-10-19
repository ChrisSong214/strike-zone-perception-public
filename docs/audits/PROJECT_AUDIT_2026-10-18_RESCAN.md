# Updated codebase audit

Reviewed October 18, 2026. This report revisits the updated source and saved artifacts, preserving the [original audit](/Users/chrissong/sp26_sagb_dj/PROJECT_AUDIT_2026-10-18.md).

The update corrects several substantive mathematical and implementation errors. Of the original 24 findings, **6 are resolved in source, 6 are partially addressed, and 12 remain open**. These counts describe source-level remediation, not regenerated or validated production outputs. Saved ratings still do not reproduce the current feature definition, and changes to cross-validation and normalization introduce additional integration and resource concerns.

P1 denotes a result-validity or workflow-blocking issue within the affected workflow. P2 denotes a material correctness, interpretation, or reproducibility issue. A modeling limitation is identified explicitly rather than presented as an execution failure.

## Newly identified issues and incomplete repairs

### R1 — P1: The revised cloud illustration fails before drawing

At [generate_cloud.py:43](/Users/chrissong/sp26_sagb_dj/generate_cloud.py:43), `rv1.cdf(...)` executes before `rv1` is constructed at line 49. The same ordering error occurs for `rv2` at lines 66 and 71. Executing the first expression in isolation reproduces `NameError: name 'rv1' is not defined`.

Moving the calculations after their distribution constructors will remove this blocker. The subplot labels now say schematic, but the overall title at line 85 still calls the figure an LSTM probability cloud. The existing PNG and presentation retain the earlier illustration and probability captions. Regenerate both artifacts and make their schematic status explicit throughout.

### R2 — P2: The exporter silently assigns a requested season to legacy rows

[export_leaderboard.py:24](/Users/chrissong/sp26_sagb_dj/export_leaderboard.py:24) names the output using the requested year. Lines 32–36 apply a season filter only when the table has a season column; otherwise they select every row.

The current database has no season column. A request for 2024 would therefore export all **532 existing ratings** into a 2024-labeled file without establishing their season. This was verified through the equivalent read-only query; no export was written. Reject unknown-season data or perform an explicit, verified migration instead of silently accepting the requested label.

### R3 — P2: Outer-fold labels still select the model used to score that fold

The batter-grouped out-of-fold loop is a substantial improvement. However, [scripts/train_swing_model.py:132](/Users/chrissong/sp26_sagb_dj/scripts/train_swing_model.py:132) passes the outer validation fold into `train`, which uses its labels for early stopping in [src/swing_model.py:88](/Users/chrissong/sp26_sagb_dj/src/swing_model.py:88). That same fold is then scored.

Its labels are excluded from gradient fitting but influence model selection, so this is not fully untouched out-of-fold evaluation. Choose stopping rounds using an inner split of the outer training population, then predict the outer fold. Also label the `--oof-splits <= 1` branch accurately: it retains mixed in-sample and held-out predictions while the subsequent output still calls its AUC out-of-fold.

### R4 — P2: Out-of-fold predictions are not saved for downstream consumers

[scripts/train_swing_model.py:119](/Users/chrissong/sp26_sagb_dj/scripts/train_swing_model.py:119) creates out-of-fold `xgb_p_swing` values only in memory. The model saved at lines 111–116 is the earlier model fitted to the 80% training-batter subset, not a container for those fold predictions.

[src/swing_model.py:365](/Users/chrissong/sp26_sagb_dj/src/swing_model.py:365) preserves a supplied prediction column, but the standalone comparisons load a saved model and recompute predictions: [stowers_kwan_full.py:16](/Users/chrissong/sp26_sagb_dj/stowers_kwan_full.py:16), [compare_stowers_kwan_40.py:40](/Users/chrissong/sp26_sagb_dj/compare_stowers_kwan_40.py:40), [compare_bregman_ramirez.py:20](/Users/chrissong/sp26_sagb_dj/compare_bregman_ramirez.py:20), and [find_elite_decisions.py:8](/Users/chrissong/sp26_sagb_dj/find_elite_decisions.py:8).

These consumers cannot reproduce the new leaderboard's baseline residuals from that saved model. Persist predictions keyed by pitch, with fold assignment and run metadata, and make retrospective analysis use the same prediction artifact.

### R5 — P2: Native XGBoost prediction does not honor the selected stopping round automatically

[src/swing_model.py:94](/Users/chrissong/sp26_sagb_dj/src/swing_model.py:94) trains with early stopping, but line 112 calls the native booster's `predict` without an iteration range. Native `Booster.predict` uses the retained model unless the caller explicitly selects the best iteration or training saves only the best model. This behavior is documented in the [official XGBoost prediction guide](https://xgboost.readthedocs.io/en/stable/prediction.html).

Use `iteration_range=(0, best_iteration + 1)` where applicable, or a callback that saves the best model. The existing saved model's best iteration is 299 of 300, so this review does **not** demonstrate an effect on that artifact. The issue affects runs and folds that stop earlier.

### R6 — P2: Training-only normalization creates large simultaneous copies

The normalization leakage is fixed, but [src/dataset.py:89](/Users/chrissong/sp26_sagb_dj/src/dataset.py:89) copies the training subset by advanced indexing, then line 91 creates another full-sized tensor through `.abs()`. The validity selection at line 94 allocates further data.

For the saved all-years population of 3,895,131 sequences, each 100 × 9 float32, the original tensor is about **13.06 GiB**. An 85% training copy and its absolute-value copy add about **11.10 GiB each**, bringing these three arrays alone to approximately **35.26 GiB** before other allocations. This machine has 18 GiB of physical RAM.

This is an allocation-based capacity finding, not an observed out-of-memory failure; no such allocation was attempted. Compute training-only statistics in chunks and avoid materializing the entire subset and its absolute values together.

## Most consequential remaining issues

### R7 — P1: Saved ratings still reflect obsolete feature geometry

[scripts/build_swing_features.py:108](/Users/chrissong/sp26_sagb_dj/scripts/build_swing_features.py:108) uses rectangular Chebyshev distance. For Kyle Tucker's 2,439 eligible 2025 pitches, the saved chase rate remains exactly reproducible with the old radial definition: **97/914 = 10.6127%**. The current rectangular, restricted-bin definition gives **59/738 = 7.9946%**. This was rechecked against the database.

The saved table also lacks the season and diagnostic fields produced by the updated writer. The two database paths are hard links to the same file, not independent corroborating datasets. Fix upstream issues, then regenerate features, predictions, ratings, and exports as a versioned run. Current source correctness does not establish saved-output correctness.

### R8 — P1, modeling limitation: Learned residual uncertainty is not validated human perception

The inputs reconstruct position, velocity, and constant acceleration from retrospective pitch fits ([scripts/build_training_data.py:35](/Users/chrissong/sp26_sagb_dj/scripts/build_training_data.py:35); [src/physics.py:107](/Users/chrissong/sp26_sagb_dj/src/physics.py:107)). Those quantities already determine the endpoint analytically. [src/model.py:135](/Users/chrissong/sp26_sagb_dj/src/model.py:135) optimizes endpoint-coordinate likelihood; the saved swing labels do not supervise this loss.

The new decision-time coordinate statistics do not establish that sigma measures a batter's sensory uncertainty. Define the observations available at the decision time, compare with the analytical physics baseline, and measure uncertainty coverage and strike-probability calibration. Until then, describe sigma as conditional model residual uncertainty.

### R9 — P2: The decision fraction changes for long trajectories

The builder and scorer retain only the last 100 samples ([scripts/build_training_data.py:96](/Users/chrissong/sp26_sagb_dj/scripts/build_training_data.py:96); [src/lstm_scoring.py:110](/Users/chrissong/sp26_sagb_dj/src/lstm_scoring.py:110)). [src/model.py:124](/Users/chrissong/sp26_sagb_dj/src/model.py:124) selects a fraction of that suffix.

A real 2025 pitch, game 776170 / at-bat 26 / pitch 4, has 115 trajectory samples. Its nominal 40% decision point selects 275 ms, approximately **48.63% of the full flight**, rather than the intended 226.22 ms. Preserve original elapsed time or resample the complete trajectory before choosing a physical-time cutoff.

### R10 — P2: Confidence still cancels from the actual-decision component

[src/swing_model.py:397](/Users/chrissong/sp26_sagb_dj/src/swing_model.py:397) multiplies actual DVA by confidence; lines 433–435 divide by total confidence. Uniform confidence therefore cancels. A one-context reproduction with baseline swing probability 0.5, all swings, and strike probability 1 produces actual DVA **0.5 at both n=1 and n=200**. The qualification threshold was lowered only to expose this algebra.

The tendency component does shrink, and the blend changes accordingly. The actual component is a relative confidence-weighted mean, not absolute shrinkage toward the league. Change either the aggregation or the small-sample claims at lines 406–417.

### R11 — P2: Particle paths still mix different physical times and can admit invalid states

[src/perception.py:112](/Users/chrissong/sp26_sagb_dj/src/perception.py:112) aligns trajectories by row number. A seeded reproduction yielded 83–90 sampled rows versus 86 actual rows, with some posterior paths ending near **y=2.95 ft**, rather than the 1.417 ft plate plane. Plotting final markers on the plate does not repair their coordinates. Interpolate onto a common physical time or distance grid and compute plate intersections explicitly.

The specific missing-release-y fallback is fixed, but generic finite-state validation is still absent. A missing release-z reproduction can produce an accepted trajectory with nonfinite z, yielding `(0.0, 0.0, NaN)` from the physical-metrics calculation. Reject nonfinite trajectories and distinguish missing simulation results from valid precision estimates.

### R12 — P2: Schemas, artifact identity, and metric labels remain inconsistent

The database contains only `statcast` and `swing_ratings`. [scripts/run_eye_rating.py:49](/Users/chrissong/sp26_sagb_dj/scripts/run_eye_rating.py:49) requires a missing physical-metrics table; notebook 05 requires a missing `eye_ratings` table. Physical and LSTM writers still replace a shared table with different columns through [src/db_manager.py:106](/Users/chrissong/sp26_sagb_dj/src/db_manager.py:106).

The swing writer now includes season and a season-specific model name, but still replaces the entire ratings table and overwrites a default model alias. Feature metadata does not identify the cutoff and checkpoint hashes. Several output scripts still hardcode 2025 or 40% labels. Use explicit method schemas and immutable run identifiers.

The reported chase statistic also remains restricted to the far-outside bin ([src/swing_model.py:466](/Users/chrissong/sp26_sagb_dj/src/swing_model.py:466)). Tucker's all-outside-zone swing rate is **324/1,402 = 23.11%**, not the restricted 7.99%. Label these separately. The DVA utility at line 376 remains strike-probability agreement with a universal 0.5 swing threshold, rather than count-aware optimal baseball value.

## Status of all original findings

“Resolved” below means the specific source defect is addressed. Broader validation and artifact regeneration remain separate obligations.

| Original | Status | Current evidence or remaining work |
|---|---|---|
| 1. Builder syntax failure | Resolved | Stray character removed; all 49 Python files parse. |
| 2. Human-perception interpretation | Open | Endpoint likelihood still lacks perceptual validation; R8. |
| 3. Correlation omitted from strike probability | Resolved | Bivariate integration accepts rho, and scoring/feature callers supply it. Numerical checks agree; see verification limits. |
| 4. Unequal baseline evaluation | Partial | Batter-grouped OOF added; outer-label early stopping and unsaved predictions remain; R3–R5. |
| 5. Saved feature/results mismatch | Open | Stored chase rates still match old geometry; R7. |
| 6. Synthetic figure presented as model output | Partial | Subplots now say schematic, but execution order is broken, overall title remains misleading, and saved artifacts are unchanged; R1. |
| 7. Movement-unit conversion | Resolved | Erroneous division by 12 removed from `src/uncertainty.py`. |
| 8. Posterior variance and conflicting uncertainty methods | Partial | Particle variance now matches P×R/(P+R); additive, Bayesian-fusion, and learned-residual paths still need distinct descriptions. |
| 9. Trajectory alignment | Open | Index alignment persists; R11. |
| 10. Missing-release-y fallback | Resolved | Valid extension now produces finite endpoints when release y is NaN. Other nonfinite-state cases remain; R11. |
| 11. Empty cloud encoded as certainty | Resolved | Empty cloud now returns three NaNs. |
| 12. Non-resumable backfill | Open | Primary-key table is still populated by unconditional append in `scripts/backfill_physical_metrics.py:129`; reruns collide. |
| 13. Validation preprocessing and split | Partial | Scaler fits training indices only; random pitch splits remain, tensors lack game/date IDs, and new copies pose capacity risk; R6. |
| 14. Validation at deployed decision time | Partial | 40% coordinate metrics added, but no interval/probability calibration, console summaries still emphasize endpoint metrics, and saved checkpoints predate the change. |
| 15. Long-flight decision timing | Open | Fraction is applied to retained suffix; R9. |
| 16. Confidence cancellation | Open | Actual-component shrinkage claim remains unsupported; R10. |
| 17. Restricted chase-rate label | Open | Far-outside bin is still labeled chase rate; R12. |
| 18. Strike agreement versus optimal utility | Open | Count-dependent behavioral baseline does not make the utility count-dependent; R12. |
| 19. Database/notebook schema mismatch | Open | Required tables missing; method-specific columns still conflict; R12. |
| 20. Season/model provenance | Partial | Season column and model naming added in writer, but old schema/export fallback, table replacement, and missing run identity remain; R2/R12. |
| 21. Missing core dependencies | Resolved | Requirements now declare torch, xgboost, pyarrow, plotly, and other directly used packages. Installation compatibility was not validated. |
| 22. Presentation formula mismatch | Open | Page 20 does not specify the implemented 50/50 tendency/actual blend; page 21 does not complete that definition. |
| 23. Notebook mirrored paths and plate planes | Open | Notebook 03 still combines transformed and raw traces; notebook 02 still hardcodes a pitcher title inconsistent with its selection. |
| 24. Obsolete project guidance | Open | `CLAUDE.md` still describes an obsolete combined pipeline and references a nonexistent test directory. |

## Verification and limits

- Parsed all **49 Python source files** successfully and checked their hashes against the source manifest captured for this rescan. No source changes occurred during this audit.
- Re-ran the **three existing physics boundary unit tests**, which passed using the separate bundled runtime. This is not a full test-suite or model-inference pass.
- Checked the actual extracted bivariate-CDF implementation with bundled NumPy and pure-Python normal/quadrature stand-ins. At a zone corner, rho −0.9/0/+0.9 yielded approximately **0.071783/0.25/0.428217**. Twelve additional nonzero-boundary cases agreed with independent conditional-normal quadrature within **2.30×10⁻¹³**. This validates the formula in these cases, not integration with the project's installed SciPy/PyTorch environment.
- Executed focused checks for corrected posterior variance (**0.5000000000000001** when P=R=1), missing-release-y fallback (**100/100 finite endpoints**), empty-cloud NaNs, remaining path-alignment problems, invalid-state handling, decision timing, and confidence cancellation.
- Inspected source, notebook code/saved outputs, read-only database schemas and selected records, checkpoint metadata, logs, and output provenance. Both LSTM checkpoints remain byte-identical and identify 2025 training; there is no saved all-years comparison checkpoint. The training comparison log does not establish a completed all-years run.
- Rechecked relevant presentation text and the existing cloud image. The presentation remains the April artifact; this rescan did **not** repeat the original audit's full 25-page rendering. Its SHA-256 is `ca1f6cf6adf3b6a769f06c6050fe5478608f182842499d30f04242823d1c3b4f`.
- The project's `.venv` native numerical imports were rejected by macOS with a code-signature/system-policy error. Those attempts explain the reported library alerts. Imports from that environment were stopped; no security settings were changed. No full local-environment test pass, training, feature rebuild, database backfill, or end-to-end artifact regeneration is claimed.
- The folder is not a Git working tree, so there is no commit-based diff baseline. No graphify index was available. Source, database, model, notebook, and existing report files were preserved; this report is the new project deliverable. Tests/imports may have produced caches.

## Recommended repair order

1. Fix the figure execution order and fail safely on unknown-season exports.
2. Complete OOF evaluation: inner model selection, best-iteration prediction, and saved pitch-keyed fold predictions shared by all analyses.
3. Make normalization memory-bounded; preserve game/date identifiers and full-flight timing.
4. Resolve perceptual interpretation, uncertainty calibration, path alignment, confidence aggregation, and metric naming before relying on rankings.
5. Define method-specific schemas and run metadata, then rebuild all dependent artifacts together.
6. Regenerate notebooks and presentation outputs, and update guidance to describe the implemented scoring formula and supported workflows.
