# Presentation readiness — October 19, 2026, fourth pass

**Verdict: the core project is presentable as an exploratory trajectory-based decision-evaluation study. Three presentation corrections remain before using the current materials unchanged. Another broad engineering audit is not the priority.**

Reviewed the current uncommitted working copy above `d2f643f`. This pass focused on substantive methods, artifact consistency, figures, and conclusions. No code, production data, or figures were changed.

## Gates now passed

- **Corrected features:** all 724,002 saved rows match current Chebyshev distances, normalized zone labels, and location bins; feature version is `2026.1-chebyshev`.
- **Regenerated results:** all 532 exported batter chase rates match those corrected feature flags. Database and CSV agree across checked ranks, ratings, actual/tendency DVA, EV-DVA EB, and deception values.
- **OOF structure:** both prediction sets contain valid probabilities for 724,002 rows, five outer folds, no duplicate pitch keys, and one fold per batter.
- **Updated top-ten figure:** Leo Rivas 129.4, Gleyber Torres 127.0, and Kyle Tucker 125.7 match the current CSV.
- **Tests:** 81 passed, two warnings, in 4.18 seconds in the isolated audit environment. This is supporting engineering evidence, not a research-validity test.
- **Claim improvements:** the forecasting section now reports uncertain incremental predictive value; the horizon section acknowledges null biomechanical associations; the main takeaway correctly describes an exploratory framework.

## Three remaining presentation corrections

### 1. Replace or omit the Stowers–Kwan “passivity penalty” example

[The current image](/Users/chrissong/sp26_sagb_dj/assets/stowers_vs_kwan.png) still shows Stowers 114.8 versus Kwan 97.1. Current EV-DVA ratings are **Stowers 104.1 / rank 185**, **Kwan 107.8 / rank 110**. The ordering reverses. The [README case study](/Users/chrissong/sp26_sagb_dj/README.md:85) should not use the older image as evidence for the current score.

For presentation, simply omit this case study or explicitly identify it as a historical, differently defined metric. A historical comparison is not a current EV-DVA result.

### 2. Omit the pooled year-over-year superiority headline until the cohort is homogeneous

[Presentation findings](/Users/chrissong/sp26_sagb_dj/docs/presentation/PRESENTATION_FINDINGS_2020_2025.md:40) still quote DVA correlation .8138 versus chase .7192. The [current summary](/Users/chrissong/sp26_sagb_dj/assets/presentation/yoy_reliability_summary.csv) reports DVA **.7898** versus chase **.8340**.

More fundamentally, the analyses tag the historical-alignment-to-2025-EV transition but still include it in the pooled calculation. That is not a clean persistence estimate for a single metric. Labeling the old headline “historical” alone does not repair its supporting cohort.

Use the verified 2025 split-half repeatability curve instead. It shows increasing repeatability with sample size; chase rate remains more repeatable in that comparison. Its saved values are Spearman–Brown-corrected reliability estimates, so label them accordingly rather than raw half-sample Pearson correlations.

### 3. Turn causal scouting claims into observational hypotheses

[Findings line 199](/Users/chrissong/sp26_sagb_dj/docs/presentation/PRESENTATION_FINDINGS_2020_2025.md:199) says certain pitches reliably induce takes or prompt emergency swings. [README line 112](/Users/chrissong/sp26_sagb_dj/README.md:112) treats repeatable residual patterns as demonstrated deception skill.

The fitted regressions support associations and potential follow-up questions. They do not identify what would happen if a pitcher deliberately changed uncertainty. Present these as model-associated swing tendencies and candidate scouting hypotheses requiring later-game validation.

## Defensible presentation scope

The strongest supported story is an interpretable framework using reconstructed trajectory forecasts, correlated zone probabilities, batter-grouped OOF swing baselines, and count-dependent utility assumptions. Its scores show repeatable player variation under the chosen model. Historical alignment forecasting shows a small observed wOBA gain, **ΔR² .0072 with 95% CI [−.0025, .0168]**, so added predictive value remains uncertain. Horizon estimates remain exploratory fitted indices; null associations with bat tracking do not establish a cognitive mechanism.

After those three cuts or corrections, the talk has a coherent methods-and-results story with appropriate limits. The historical PDF remains a separate historical artifact; this readiness judgment does not certify every statement in that deck.
