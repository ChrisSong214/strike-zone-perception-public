# Exploratory evidence used to choose presentation ideas

Computed October 18, 2026 from the updated project's six leaderboard CSVs and read-only Statcast SQLite data. This was a small diagnostic for research selection, not a new validated production model or a full audit.

## 1. How much of DVA can two ordinary rates reproduce?

Filter to at least 750 pitches per batter-season. Fit an intercept plus chase rate and zone-swing rate to `swing_rating` using seasons 2020–2024. Apply that fixed mapping to 2025.

- 2025 test population: **354 hitters**.
- Held-out-season rating-reconstruction R²: **0.9056103**.
- Correlation between `tendency_dva` and `actual_dva` across the qualifying six-season sample: **0.911378**.

Interpretation: most cross-player variation in the current rating can be reconstructed from familiar aggression/restraint summaries. This is **not** forecasting 2025 with only pre-2025 data: the mapping uses each test hitter's contemporaneous 2025 chase and zone-swing rates. It also does not prove that the unexplained remainder represents an independent skill.

## 2. Does DVA add to later-season wOBA prediction?

For comparability with the current presentation defaults, use at least 750 pitches and 197 terminal-event plate appearances in both the predictor and target seasons. Merge exactly consecutive years by batter. This conditions on returning, sufficiently active players; it does not describe all players or forecast roster survival.

Fit ordinary least squares using only earlier target years and evaluate target years 2023, 2024, and 2025. Training/test pair counts are respectively **425/270**, **695/285**, and **980/267**. The combined test sample contains **822 player-year forecasts**, with repeated hitters across years.

| Forecast inputs from prior season | Pooled test R² | RMSE in wOBA units |
|---|---:|---:|
| wOBA + chase rate | 0.155811 | 0.033239 |
| wOBA + chase rate + DVA | 0.166540 | 0.033027 |
| wOBA + chase rate + zone-swing rate + log pitches | 0.149286 | 0.033367 |
| Same expanded baseline + DVA | 0.145192 | 0.033447 |

Adding DVA changes R² by **+0.010730** under the smaller baseline and **−0.004094** under the expanded baseline. The expanded baseline itself performs worse here, so adding covariates is not automatically better. A final comparison should also consider appropriately tuned regularization, additional relevant public predictors, and identical evaluation samples.

The smaller baseline's RMSE improvement is approximately **0.000212 wOBA**, or **0.64%** of its error. The expanded baseline's error slightly increases. This supports a claim of **baseline-sensitive incremental value in this exploratory test**, not a general conclusion that DVA does or does not matter.

A 2,000-resample hitter-cluster bootstrap of the fixed paired forecasts gave these intervals for mean squared-error reduction (positive favors adding DVA):

- Smaller baseline: estimate `1.4042e-5`, 95% interval `[1.2103e-6, 2.8753e-5]`.
- Expanded baseline: estimate `−5.3581e-6`, 95% interval `[−1.0743e-5, −1.4349e-7]`.

These intervals condition on already fitted forecasts. They do not include upstream model estimation, training-data uncertainty, model-selection uncertainty, or season-wide dependence. Do not use them as confirmatory significance claims after selecting a research direction from these results.

## What this pilot does not establish

- The aggregate regression stage is temporally held out; the temporal training history of the underlying DVA/LSTM artifacts was not verified. This is not an end-to-end prospective validation.
- No new pitch-level model was trained. No project `.venv`, native model inference, or live database writes were used.
- 2020 is included in early training; the shortened season warrants a sensitivity analysis. Thresholds should be prespecified or varied transparently, not selected for maximum lift.
- Existing SQL outcome definitions were followed for comparability; the current cohort and outcomes should be reconciled to official season totals before a final presentation.
- The 2023–2025 outcomes have now been inspected during idea development. Describe this as exploratory; do not later call the same results an untouched confirmatory test.

## Reproduction artifacts

- [Read-only diagnostic script](/Users/chrissong/sp26_sagb_dj/docs/presentation/research_idea_pilot.py).
- [Computed results and input CSV fingerprints](/Users/chrissong/sp26_sagb_dj/docs/presentation/research_idea_pilot_results.json).

The script requires NumPy/Pandas and access to the project's current CSVs and SQLite database. Rerunning it writes its own JSON result only; it does not refresh or replace any project leaderboard or model. Current-input fingerprints in the saved JSON identify the CSV snapshot used for this pilot.
