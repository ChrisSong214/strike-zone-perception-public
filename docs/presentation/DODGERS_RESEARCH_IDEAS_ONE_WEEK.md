# A meaningful Dodgers presentation in one week

Prepared October 18, 2026. Constraints: public data only, approximately one week, reuse the current project. These are research proposals; the small diagnostic below is explicitly exploratory.

## Recommendation

Build the presentation around **“Does a hitter's plate discipline travel to a different pitch mix?”** Use context-specific player cards as the explanation layer. This gives the audience a player-evaluation question, a fair test, and inspectable baseball examples.

The useful contribution would be evidence that separates **what a hitter does** from **the opportunities they receive**, and shows whether that distinction survives later games. Do not promise a first-ever metric or insight into what the Dodgers already know internally. Public work already covers individualized decision value, selective aggression, Bayesian uncertainty, and pitch tunneling; the attached [related-work memo](/Users/chrissong/sp26_sagb_dj/docs/presentation/RELATED_WORK_FOR_RESEARCH_IDEAS.md) maps that literature.

## What the updated project already tells us

The current checkout contains six season leaderboards and new reliability, forecasting, stabilization, and pitcher analyses. A fresh read-only diagnostic on the updated CSVs found:

- A linear mapping from **chase rate and zone-swing rate**, fitted on 2020–2024 ratings, reproduces **90.6% of the variation in 2025 DVA ratings** among 354 hitters with at least 750 pitches. This predicts a contemporaneous rating from contemporaneous rates; it is not future-performance prediction.
- The two DVA components have pooled correlation **0.911** in that qualifying sample. Separate names do not establish independent underlying skills.
- In an exploratory rolling forecast at the season-summary stage, adding DVA modestly helped a prior-wOBA-plus-chase baseline, but did not help after also adding zone-swing rate and playing time. This is baseline sensitivity, not proof that DVA contains no useful information.

This points toward the part of the project worth investigating: **which repeatable differences remain after accounting for simple aggression/restraint and pitch opportunities?** The remaining unexplained rating variation is not automatically a new skill; it could contain context, estimation effects, or noise.

The [pilot evidence and reproduction details](/Users/chrissong/sp26_sagb_dj/docs/presentation/RESEARCH_IDEA_PILOT_EVIDENCE.md) give the exact cohorts, results, assumptions, and limitations.

One presentation claim needs correction before use: [predictive_validity.py:59](/Users/chrissong/sp26_sagb_dj/analysis/predictive_validity.py:59) fits and scores regressions on the same paired observations. A next-year target does not make those regression scores out of sample. The existing presentation's forecasting and biological-skill claims go beyond what that calculation establishes.

## Idea 1 — Does plate discipline travel? Recommended main study

**Question:** Would two hitters with similar chase rates still look similar if they faced the same distribution of counts, locations, and pitch types?

**Baseball use:** Help identify whether a player's rating change reflects a change in decisions or a change in the pitches they are seeing. This can inform acquisition review, opponent preparation, and follow-up scouting.

**Experiment:** Begin with the project's existing count × location × pitch-family bins. Estimate each hitter's swing tendency and descriptive decision score within those contexts, shrinking sparse estimates toward league values. Evaluate every hitter using one fixed reference pitch distribution learned from the development period. Show both their observed score and their score under that common distribution. Require adequate overlap; report how much of the reference distribution is supported for each player rather than inventing performance in unseen situations.

For context-specific score \(q_{ic}\), compare:

\[
Q_i^{observed}=\sum_c w_{ic}q_{ic},\qquad
Q_i^{standardized}=\sum_c w_c^{reference}q_{ic}.
\]

Keep the score definition identical between the two calculations. The difference isolates weighting by opportunity mix within the model; it is not a causal estimate of changing the pitches a hitter actually faces.

**Decisive test:** Freeze the reference mix and player estimates before later games. Test whether the profile predicts subsequent context-specific behavior better than overall chase, zone-swing, and the existing DVA score. Use next-season offensive production as a secondary test. Compare on identical player samples and show interval estimates using game/hitter clusters as appropriate. Any forecast of a future opponent's mix must use information available before that matchup, not its realized pitches.

**Presentation asset:** One rank-change plot, one held-out performance comparison, and two player cards showing *why* the standardized result changes. Include an unchanged case as a control.

**Potential contribution:** A validated distinction between opportunity mix and transferable behavior. Standardization itself is established; the particular empirical finding and its usefulness are what must earn the claim.

**Failure condition:** If changes are concentrated in unsupported cells or fail to persist, report that the adjustment is too unstable to use. This still answers a worthwhile evaluation question.

## Idea 2 — Same chase rate, different development question

**Question:** Can a compact decision profile identify stable tendencies hidden by season averages?

**Baseball use:** Produce a specific review question for a hitter instead of the generic instruction to improve discipline.

**Experiment:** Predefine three or four contexts: early-count takes in the heart of the zone, two-strike swings below the zone, ahead-in-count chase, and one handedness/pitch-family contrast with adequate samples. Start with observed swing/take rates and matched league expectations. Use partial pooling and uncertainty intervals. Select players and contexts in earlier games; assess recurrence in later games before choosing the presentation examples.

The claim is a tendency, not a diagnosis of eyesight or an instruction to swing more. Calling a decision costly requires count-aware swing/take value and attention to player contact/power differences. A strike-probability threshold alone cannot supply that interpretation.

**Decisive test:** Does the profile improve later pitch-level swing prediction over an equally flexible baseline using aggregate chase/zone-swing rates? Report log loss or Brier score and sample counts. If claiming earlier detection, compare methods at the same observation count and false-alert budget.

**Presentation asset:** A one-page player card with the observed tendency, comparison group, uncertainty, later-period result, and a small set of public pitch/video examples. Select examples to explain the measured result, not to establish it.

**Potential contribution:** A reproducible, context-specific signal that is actionable enough to investigate. This is the safest one-week alternative and can share nearly all work with Idea 1.

## Idea 3 — Does the previous pitch change the next decision?

**Question:** After accounting for the current pitch, does the preceding pitch add useful information about swing behavior?

**Baseball use:** A narrow opponent-preparation or sequencing question, rather than a general “most deceptive pitchers” leaderboard.

**One-week scope:** Restrict to one common pitch pair, such as four-seam fastball followed by slider, and a handedness group. Compare a current-pitch model with the same model plus prior-pitch location, speed, movement, and their differences. Include count, pitcher/arsenal, batter tendencies, and conventional separation features. Preserve actual within-plate-appearance ordering and hold out later games.

**Decisive test:** Does sequence information improve held-out swing prediction? Does the association remain within pitcher and count, and across more than one pitcher? Pitch choice and count are selected, not randomized, so this does not prove that changing the sequence causes a chase or additional runs.

**Presentation asset:** Two closely matched current pitches with different preceding contexts, alongside the aggregate held-out result. The examples illustrate an association; the aggregate test supports it.

**Potential contribution:** A narrow behavioral effect beyond a strong current-pitch/tunnel baseline. Pitch tunnels and release resemblance already have extensive prior art. This is the highest-risk option for the deadline; do it only if the first two are already complete.

## Seven-day scope

| Day | Deliverable |
|---|---|
| 1 | Freeze the question, score definition, public-data cohort, temporal split, and baselines. Verify upstream feature/model provenance and required pitch-level files. |
| 2–3 | Implement the modest context profile and common reference mix; inspect overlap and shrinkage. |
| 4 | Evaluate later games, run simple-baseline ablations, and quantify uncertainty. Avoid optimizing the minimum-pitch cutoff to maximize the result. |
| 5 | Review two or three player cases plus a control, using public pitch/video examples where available. |
| 6 | Build five slides: question, method, held-out evidence, player examples, practical implication. |
| 7 | Reproduce outputs, rehearse objections, and state what did and did not hold up. |

For this deadline, use the available through-2025 period. MLB documents a change in plate-location and strike-zone definitions in 2026, which would require explicit harmonization. [Official Statcast CSV documentation](https://baseballsavant.mlb.com/csv-docs).

The current local checkout has six season-level leaderboards but only the 2025 pitch-feature Parquet. Confirm earlier pitch-level artifacts before promising a multi-year context study; use chronological blocks within 2025 as an exploratory fallback. Do not call a later period untouched if it has already been used to select the method or examples.

## Suggested opening and closing

**Opening:** “Two hitters can have the same chase rate while making different decisions against different opportunities. I wanted to know which differences persist when we compare them on equal footing.”

**Closing:** State the measured answer, its uncertainty, and one concrete scouting or player-development question it raises. A well-tested negative result is preferable to presenting a large correlation as proof of a new biological skill.

## Sources that shape the novelty claim

- [Driveline: individualized swing decisions](https://drivelinebaseball.com/blogs/blog/quantifying-swing-decisions-an-individualized-approach) already adjusts decision evaluation to the player and context.
- [Yee and Deshpande: Bayesian plate-discipline evaluation](https://arxiv.org/abs/2305.05752) already studies swing/take value with uncertainty.
- [Sutton-Brown: swing processes through bat and pitch tracking](https://www.baseballprospectus.com/news/article/103703/best-of-bp-2025-understanding-swing-processes-through-bat-and-pitch-tracking/) is close prior art for uncertain early pitch detection.

The recommendation is an incremental empirical contribution with public data, not a claim that these themes are new to professional baseball.
