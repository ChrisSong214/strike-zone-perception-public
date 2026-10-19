# Related work and defensible contributions for the Dodgers presentation

Researched October 18, 2026. Scope: public data, roughly one week of work, existing project infrastructure. These are proposed studies, not completed findings. This is a targeted prior-art review, not an exhaustive novelty search; no claim of first invention is justified.

## Recommendation

The strongest one-week contribution is **testing whether a hitter's apparent decision skill survives a different pitch mix**, followed by **identifying specific decision patterns that predict later improvement**. A fair negative result can be useful: explain exactly when the current metric adds information, when it merely repackages chase/contact/previous production, and when the available public data cannot distinguish the two.

Do not lead with a new leaderboard, a more complex neural model, or the claim that swing decisions should account for uncertainty. Substantial public work already covers those themes. The contribution can instead be a reproducible test, an interpretable failure mode, and a concrete next step for player evaluation.

## What already exists

| Primary source | What its authors already do or report | Consequence for this project |
|---|---|---|
| Dan Aucoin, Driveline, [Quantifying Swing Decisions: An Individualized Approach](https://drivelinebaseball.com/blogs/blog/quantifying-swing-decisions-an-individualized-approach) (2019) | Individualized swing/take value, count and contact adjustments, and Swing Decision+. The analysis compares descriptive, reliability, and future-performance relationships. | Player-specific expected value and future-year association are established ideas. A new name or plus-scale rating is not sufficient differentiation. |
| Robert Orr, [Quantifying the Corey Seager Approach](https://www.baseballprospectus.com/news/article/86572/the-crooked-inning-corey-seager-rangers/) (2023) | SEAGER uses count/location swing-versus-take values to capture selective aggression. The author reports current- and future-season relationships, including a weighted correlation with next-year ISO. | Avoid claiming novelty from penalizing hittable takes or showing a future correlation. Compare against both restraint and aggression baselines. |
| Joshua Mould and David Anderson, [Quantifying Hitter Plate Discipline in Major League Baseball, SABR presentation](https://sabr.org/analytics/presentations/2022) (2022) | EAGLE models outcome probabilities and run expectancy for swinging and taking; the presentation abstract reports stability and future OPS prediction. | Expected runs gained by choosing swing/take also has prior art. The abstract's reported results are authors' findings, not independently replicated here. |
| Ryan Yee and Sameer Deshpande, [Evaluating plate discipline with Bayesian additive regression trees](https://arxiv.org/abs/2305.05752) (2023; journal DOI linked there) | Models called strikes, contact and subsequent run expectancy with game/player context; propagates uncertainty into the swing/take recommendation. | Context and probabilistic recommendations are not novel on their own. The proposed contribution needs a different empirical question or validation design. |
| Thomas Nestico, [Modelling Batter Decision Value](https://tjstats.ca/2023/12/01/modelling-batter-decision-value/) (2023) | Separates swing and take value models, uses count-dependent outcome values, and reports separate in-zone/out-of-zone components. | A two-component decision profile is a useful baseline. Do not assume agreement with the strike zone equals optimal run value. |
| Jeff Long, Harry Pavlidis and Martin Alonso, [Updating Pitch Tunnels](https://www.baseballprospectus.com/news/article/37436/prospectus-feature-updating-pitch-tunnels/) (2018) | Uses batter-perspective separation, time before arrival, and the earliest/greatest pre-tunnel difference of consecutive pitches. | A fixed-versus-time-based tunnel comparison or late-separation score alone is not novel. |
| Michael Rosen, [The Kirby Corollary: Why Batters Don't Swing at Sliders](https://blogs.fangraphs.com/the-kirby-corollary-why-batters-dont-swing-at-sliders/) (2024) | Tests how similarity between fastball and slider horizontal release angles relates to slider swing behavior. | Release resemblance and arsenal context must be comparison features in a sequence/ambiguity study. |
| Justin Brantley and Konrad Kording, [Bayesball: Bayesian Integration in Professional Baseball Batters](https://www.biorxiv.org/content/10.1101/2022.10.12.511934v2) (revised preprint, 2024) | Tests a Bayesian interpretation of prior knowledge and observed pitch information using professional baseball data, including pitch-tipping examples. | Do not present Bayesian fusion itself as the contribution or a model posterior as a direct measurement of the hitter's belief. |
| Stephen Sutton-Brown, [Understanding Swing Processes Through Bat and Pitch Tracking](https://www.baseballprospectus.com/news/article/103703/best-of-bp-2025-understanding-swing-processes-through-bat-and-pitch-tracking/) (2025; linked year-end reprint) | Uses bat tracking to infer directional/timing errors and models uncertain early pitch detection. Reports that early uncertain information helps explain swing initiation. | This is especially close prior art for trajectory uncertainty. A public-data uncertainty model needs comparison with this approach and independent behavioral validation. |

BP and bioRxiv direct fetches were restricted; the search tool returned first-party indexed text. Summaries above are limited to that accessible text. Original analysis and inference should be distinguished from causal claims.

## Idea 1 — Would we grade the hitter the same way against a different pitch mix?

**Question:** How much of a player's decision rating reflects their behavior, and how much reflects which pitches and counts they faced?

**One-week experiment:** Use seasons already present, preferably ending in 2025. Estimate swing behavior within a modest set of count, normalized-location, pitcher-handedness and pitch-family contexts. Use partial pooling or shrinkage so sparse cells do not generate extreme player scores. Evaluate every hitter against the same declared reference mix, restricted to contexts with adequate support. Show observed and standardized profiles side by side.

**The critical comparison:** Freeze estimates before the next period. Compare a baseline containing prior production, chase, zone swing/contact, age and playing time with the same model plus the standardized decision features. Report held-out error change and uncertainty. Use rolling season/time splits; fit all standardization, shrinkage and tuning inside training data. If only a few seasons exist, characterize the result as preliminary rather than general proof.

**Plausible incremental contribution:** A transparent decomposition of opportunity mix versus behavior, plus evidence about transfer to a new mix. This combines established adjustment ideas into a specific test; this review does not establish that nobody has done it before.

**Presentation result:** A paired chart of two hitters with similar chase rates but different standardized aggression/restraint, and a held-out comparison showing whether this distinction persists. Include one case where standardization changes the story and one where it does not.

**Meaning:** Useful for deciding whether a change in a hitter's leaderboard position deserves investigation. It does not estimate what would happen if a coach changed the hitter's approach, and it does not isolate a biological trait called vision.

## Idea 2 — Which specific decision pattern is repeatable enough to act on?

**Question:** Can a small, understandable profile distinguish stable player tendencies from noisy aggregate changes?

**One-week experiment:** Predefine three or four contexts, such as two-strike chase, ahead-in-count chase, and early-count takes in favorable hitting regions. Estimate deviations from a context-matched baseline with shrinkage and intervals. Select candidate cases using only the earlier period, then test the direction and size of the deviation in later games. Do not search many contexts and present only successful ones.

**Comparison:** Aggregate chase/zone-swing rates, prior-period profile, and the current project rating. Prefer future decision behavior as the primary target; future offensive production is a secondary harder test. When claiming a take was costly, use a count-aware value model and check batter contact/power differences. Otherwise call it an unusual tendency, not a mistake.

**Plausible incremental contribution:** Evidence that a compact profile identifies stable, context-specific changes earlier or more clearly than a season leaderboard. This is a validation and interpretation contribution, not an invention of individualized plate discipline.

**Presentation result:** A one-page player card: the tendency, sample count, uncertainty, later confirmation, and selected public pitch/video examples. A useful question for the team is whether that pattern merits closer scouting or development review; the analysis alone does not prescribe a swing change.

## Idea 3 — Does the preceding pitch explain behavior after accounting for the current pitch?

**Question:** Does pitch order add reproducible predictive information beyond location, count, speed, movement and arsenal characteristics?

**One-week scope:** Restrict to one common pair, such as four-seam fastball followed by slider, and one handedness group. Compare a current-pitch model with a model adding previous-pitch characteristics and their differences. Preserve actual within-plate-appearance order; keep plate appearances/games together across splits. Include within-count and pitcher-controlled checks so selection into a sequence is not mistaken for deception.

**Comparison:** A conventional tunnel/separation feature, current-pitch characteristics, pitcher identity/arsenal and previous-pitch type. Evaluate on later games and, if feasible, held-out pitchers. Swapping prior-pitch information within matched contexts can be a diagnostic, but it is not a randomized causal experiment.

**Plausible incremental contribution:** Demonstrating a narrow, repeatable incremental sequence signal and showing concrete examples where traditional separation and behavior disagree. Existing tunnel and release-angle work means that sequence geometry by itself is insufficient novelty.

**Tradeoff:** More confounding and thinner samples make this riskier than the first two in one week. Call the output an association or prediction increment, not the number of runs caused by a particular sequence.

## Where time-varying decision uncertainty fits

A future extension could plot how the preferred action or its confidence changes as the model receives progressively longer trajectory prefixes. That requires calibrated uncertainty, honest prefix-only information, a stated value model, and held-out behavioral validation. A flip in the model's answer is not evidence of a flip in the batter's mind. With uncalibrated model uncertainty and no new training budget, this belongs on the next-step slide rather than as the headline result.

[Higuchi et al. (2016)](https://pubmed.ncbi.nlm.nih.gov/26848742/) experimentally manipulated visible trajectory duration in ten college players hitting machine pitches. The results motivate sensitivity to observation time but do not establish one universal MLB swing-decision cutoff or validate a particular model noise level.

## Public-data boundaries that affect all three studies

MLB's [Statcast CSV documentation](https://baseballsavant.mlb.com/csv-docs) provides pre-pitch count, pitcher/batter identities, pitch-order keys and kinematic parameters. It also states that `plate_x`/`plate_z` refer to the front of the plate through 2025 but the middle of the plate from 2026, and that strike-zone bounds change to ABS definitions in 2026. For a one-week project, ending the principal analysis in 2025 avoids silently crossing that measurement change; a 2026 extension needs explicit harmonization.

Public summary kinematics are not raw, time-stamped measurements available to the hitter during flight. Treat reconstructed prefixes as model-based trajectory summaries, not proof of what a person could perceive at that moment. Present all future-prediction and causal conclusions at the level actually tested.

## Minimum evidence for a meaningful presentation

1. One specific baseball question, fixed before selecting examples.
2. A strong simple baseline and the exact additional information being tested.
3. A genuinely held-out later period, with preprocessing fitted only on earlier data.
4. Effect size and uncertainty, not just a correlation or a sorted leaderboard.
5. Two or three inspectable pitch/player examples that explain the result.
6. A clear failure condition: if the added information does not improve future prediction or survive support checks, say so and show where the approach breaks.

The defensible claim is: **“I tested whether this distinction carries useful information beyond standard summaries, and here is where it does and does not hold up.”**
