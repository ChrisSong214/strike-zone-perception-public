# Leaderboard Datasets & Schema Versioning

This directory stores historical and current season decision value added (DVA) and perceptual eye rating leaderboards.

## Schema Versioning

The leaderboards utilize two schema versions across seasons:

### Version 1 (v1 — 18 Columns): Seasons 2020–2024
Files: `2020_dva_leaderboard.csv` through `2024_dva_leaderboard.csv`
Metric Specification: `metric_definition = 'alignment_dva_legacy'`
- Primary decision value: `dva_mean = 0.5 * (tendency_dva + actual_dva)` (symmetric strike-alignment heuristic).

The v1 schema provides core decision metrics, Bayesian batter tendencies, and baseline comparisons:
1. `batter_id` (int): Statcast MLB player ID.
2. `rank` (int): Season rank ordered by overall `swing_rating`.
3. `player_name` (str): Batter full display name.
4. `swing_rating` (float): Standardized DVA rating (100 = league average, 10 = 1 standard deviation).
5. `chase_rate` (float): Out-of-zone swing rate (pitches outside rulebook strike zone).
6. `zone_swing_rate` (float): In-zone swing rate.
7. `tendency_dva` (float): Batter posterior swing tendency differential from league prior.
8. `actual_dva` (float): Observed strike-alignment decision quality differential.
9. `n_pitches` (int): Total pitch count faced by batter in the qualified season.
10. `season` (int): Championship season year (e.g., 2020, 2024).
11. `dva_mean` (float): Primary decision value score used to standardize the swing rating.
12. `swing_rate` (float): Overall swing percentage across all pitches.
13. `restricted_chase_rate` (float): Chase percentage restricted to extreme chase/waste zones.
14. `deviation_from_league` (float): Overall swing rate deviation from expected league baseline.
15. `chase_deviation` (float): Chase swing rate deviation from league prior across chase bins.
16. `avg_confidence` (float): Average posterior credibility weight ($1 - \text{shrinkage}$) across context bins.
17. `z_score` (float): Raw standard normal score for `dva_mean` relative to qualified season cohort.
18. `updated_at` (timestamp): ISO generation timestamp.

---

### Version 2 (v2 — 27 Columns): Season 2025+
Files: `2025_dva_leaderboard.csv`
Metric Specification: `metric_definition = 'ev_dva'`
- Primary decision value: `dva_mean = ev_dva_eb` (count-dependent expected run value DVA with Empirical Bayes shrinkage).
- Cross-season comparison note: The common loader assigns `metric_definition` to discriminate between legacy alignment DVA and production EV-DVA. YoY and breakout analyses flag the 2024→2025 boundary as a metric transition.

The v2 schema incorporates all v1 columns and introduces 9 advanced econometric and perceptual metrics:
- Retains all 18 v1 core columns.
- Additional columns:
19. `metric_definition` (str): Identifier for mathematical estimand ('ev_dva' vs 'alignment_dva_legacy').
20. `ev_dva` (float): Raw Expected Run Value DVA per pitch using count transition delta matrix.
21. `ev_dva_eb` (float): Empirical Bayes ANOVA-shrunk EV-DVA ($d_i^* = \frac{n_i}{n_i + K^*} \bar{d}_i$).
22. `alignment_dva` (float): Symmetric strike zone alignment decision quality per pitch.
23. `alignment_dva_eb` (float): Empirical Bayes ANOVA-shrunk alignment DVA.
24. `k_shrinkage_ev` (float): Empirical Bayes variance-ratio shrinkage constant $K^* = \sigma^2_{\text{within}} / \tau^2_{\text{between}}$ for EV-DVA.
25. `k_shrinkage_align` (float): Empirical Bayes shrinkage constant $K^*$ for alignment DVA.
26. `deception_swing_oe` (float): Deceptive chase swing rate over expected under LSTM perceptual shifts.
27. `ev_dva_perceptual` (float): Decision-time ($t = 0.40$) perceptual baseline EV-DVA conditioning solely on early flight information.

---

### Additional 2025 Diagnostic Leaderboards
- `2025_commit_horizon_leaderboard.csv`: Batter psychometric commit horizon ($\tau^*$) and information center-of-mass.
- `2025_mirage_batter_leaderboard.csv`: Batter susceptibility to illusion/tunneling discrepancies.
- `2025_mirage_pitcher_leaderboard.csv`: Pitcher deception potency and repertoire mirage generation.
- `2025_uncertainty_sensitivity_leaderboard.csv`: Batter sensitivity ($\beta_u$) to 2D spatial uncertainty volume.
