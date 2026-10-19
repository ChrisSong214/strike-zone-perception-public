The reason your current DVA rating has a **90.6% $R^2$ with simple Z-Swing% and Chase%**—and why the three suggested proposals feel underwhelming—comes down to a mathematical bottleneck in how your repository currently aggregates the LSTM+MDN output.

In `src/lstm_scoring.py` and `src/swing_model.py`, your pipeline takes a rich 5-parameter spatial belief distribution $(\mu_x, \mu_z, \sigma_x, \sigma_z, \rho)$ at $40\%$ flight, integrates it into a single scalar $P(\text{Strike})$, and scores the batter via $\mathbb{I}(\text{Swing}) \cdot P(\text{Strike}) + \mathbb{I}(\text{Take}) \cdot (1 - P(\text{Strike}))$. Because 9-parameter constant-acceleration trajectories are smooth, $P(\text{Strike})$ at $40\%$ flight is highly collinear with the final plate location. When you average that scalar across 2,000+ pitches in a season, the non-linearities wash out and you simply reconstruct `Z-Swing% - O-Swing%`.

The suggestions you were given ("Does plate discipline travel?" / context standardization) just re-weight `Z-Swing%` and `O-Swing%` by count and pitch zone. Every MLB front office already does context-neutral standardization; presenting that to the Dodgers Quantitative Analysis team will not stand out.

To make your presentation genuinely meaningful, **stop throwing away the best parts of your own architecture.** Your codebase already computes three quantities that `Z-Swing% - O-Swing%` cannot see: **temporal sequence dynamics across all 100 timesteps**, **the spatial covariance matrix $\Sigma_t$ ($\sigma_x, \sigma_z, \rho$)**, and **the residual gap between early belief and final crossing**.

Here are three research directions you can execute in a few days using your existing pipeline (`trajectories_2025.pt`, `best_model.pt`, and `src/lstm_scoring.py`) that extract signal orthogonal to `Z-O Swing%`:

---

## 1. Deception vs. Discipline: The "Perceptual Mirage" Decomposition (Recommended)

### Why Current DVA Collapses to Z-O Swing%

Right now, if a pitch crosses 2 inches outside the zone ($y = 1.417\text{ ft}$), traditional Chase% calls a swing a mistake (0 or -1). If your LSTM at $40\%$ flight had $P(\text{Strike}) = 0.65$ because of late break, your DVA gives the batter $+0.65$. However, on 85% of pitches (fastballs in the heart, waste pitches in the dirt), $P(\text{Strike})_{40\%} \approx \mathbb{I}(\text{Strike})_{100\%}$. Those 85% of pitches dominate the season sum, causing the 90.6% correlation with raw Z-O Swing%.

### The Fix: Isolate the "Mirage Residual" ($\Delta P$)

Using `best_model.pt` and `trajectories_2025.pt`, run `LSTMEyeScorer` at $t = 0.40$ (biological commit point) and compare it directly against the ground-truth zone outcome at $t = 1.0$:


$$\Delta P_i = P(\text{Strike} \mid \mathcal{F}_{0.40})_i - \mathbb{I}(\text{True Strike})_i$$

Partition every pitch into two regimes:

1. **Transparent Pitches ($\vert{}\Delta P_i\vert{} < 0.25$):** Pitches that look like strikes early and stay strikes, or look like balls out of the hand and stay balls. (Here, your model *should* equal Z-O Swing%).


2. **Perceptual Mirage Pitches ($\vert{}\Delta P_i\vert{} \ge 0.25$):**
* **Tunnel Chases ($P_{0.40} > 0.60$, True Ball):** Pitches that appear to be strikes at the commit window before breaking out of the zone.


* **Frozen Takes ($P_{0.40} < 0.40$, True Strike):** Backdoor breaking balls or front-hip sinkers that look outside at $40\%$ flight and clip the zone late.



### What Makes This Meaningful to the Dodgers

* **Orthogonal to Z-O Swing%:** By construction, conditioning on the $\vert{}\Delta P_i\vert{} \ge 0.25$ subset strips out the 85% of obvious pitches that cause the 90.6% collinearity.
* **Two Distinct Skills:** You decompose a hitter's discipline into **Baseline Zone Geometry** (how they handle transparent pitches) vs. **Deception Vulnerability** (how they handle high-$\vert{}\Delta P\vert{}$ pitches).
* **Dual-Sided Asset (Hitters & Pitchers):** If you aggregate $\mathbb{E}[\text{Swing} \mid \text{True Ball}, P_{0.40}]$ by **pitcher and pitch type**, your exact same batter pipeline immediately turns into a **Pitch Deception / Tunneling Metric** (`Perceptual Stuff+`), quantifying which Dodgers or opposing pitchers generate the highest $P(\text{Strike})_{0.40}$ on balls that finish outside the zone.



---

## 2. Sensitivity to Visual Uncertainty: Using the MDN Covariance Matrix ($\det \Sigma_{0.40}$)

### Why You Are Underutilizing Your MDN

Your `PitchPerceptionLSTM` predicts 5 parameters: $(\mu_x, \mu_z, \sigma_x, \sigma_z, \rho)$. Currently, `LSTMEyeScorer` integrates over the strike zone to get a single number $P(\text{Strike})$.
Notice the mathematical trap:

* **Pitch A:** $\mu_x = 0.70\text{ ft}$ (right on the zone edge), $\sigma_x = 0.05\text{ ft}$ (low uncertainty). $P(\text{Strike}) \approx 0.50$.
* **Pitch B:** $\mu_x = 0.70\text{ ft}$ (right on the zone edge), $\sigma_x = 0.45\text{ ft}$ (massive perceptual uncertainty). $P(\text{Strike}) \approx 0.50$.

Both pitches yield the exact same $P(\text{Strike}) = 0.50$, so your current DVA treats them identically! You built a Mixture Density Network to estimate spatial variance $(\sigma_x, \sigma_z, \rho)$, and then integrated the variance away.

### The Fix: Condition Swing Decisions on Spatial Entropy

Keep the covariance matrix explicit. Define the **Perceptual Uncertainty Volume** at $40\%$ flight from the determinant of the 2D covariance matrix:


$$\mathcal{U}_{0.40} = \sqrt{\det \mathbf{\Sigma}_{0.40}} = \sigma_x \sigma_z \sqrt{1 - \rho^2}$$

Now model each batter's swing probability as a function of both **Expected Location ($P_{\text{Strike}}$)** and **Perceptual Uncertainty ($\mathcal{U}_{0.40}$)**:


$$\text{logit } P(\text{Swing}_{ij}) = \alpha_i + \beta_{1,i} P(\text{Strike})_{ij} + \beta_{2,i} \mathcal{U}_{0.40, ij} + \text{Count Controls}$$

### What Makes This Meaningful to the Dodgers

* **The $\beta_{2,i}$ Coefficient ("Uncertainty Aversion"):**
* When a pitch is high-uncertainty ($\mathcal{U}_{0.40}$ is large), what is the batter's default policy?
* Hitters with $\beta_{2,i} \ll 0$ **freeze under high visual uncertainty** (e.g., Steven Kwan taking borderline pitches when the belief cloud is wide).


* Hitters with $\beta_{2,i} \gg 0$ **guess/fire under uncertainty** (selling out for contact when they cannot resolve spin early).


* **Zero Overlap with Z-O Swing%:** Standard Z-O Swing% has zero concept of $\sigma_x, \sigma_z, \rho$. Showing that two hitters have identical 25% chase rates, but Hitter A only chases when $\mathcal{U}_{0.40}$ is huge (fooled by spin/tunneluncertainty) while Hitter B chases when $\mathcal{U}_{0.40}$ is small (poor strike zone calibration), gives hitting coaches two completely different interventions.



---

## 3. Inferring Individual Commit Times ($t^*_i$) Across the 100-Step Trajectory

### Why Fixed $40\%$ Flight Limits Your Model

In `scripts/build_swing_features.py`, you hardcode `--decision-frac 0.40` for every single hitter in MLB. Because $0.40$ is static across all batters, everyone is evaluated on the exact same pitch snapshot.

### The Fix: Find Each Hitter's Effective Decision Horizon ($t^*_i$)

Your `PitchPerceptionLSTM` is a streaming model over `(100, 9)` trajectory tensors! Instead of running `build_swing_features.py` only at `0.40`, run inference at a grid of flight fractions:


$$\tau \in \{0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55\}$$

For each hitter $i$, compute the out-of-sample log-likelihood (or Brier score) of their actual swing decisions predicted by $P(\text{Strike} \mid \mathcal{F}_\tau)$ at each timestep $\tau$:


$$\tau^*_i = \arg\max_{\tau \in [0.25, 0.55]} \mathcal{L}\big(\text{Swing}_i \;\big\vert{}\; P(\text{Strike})_\tau, \text{Count}\big)$$

### What Makes This Meaningful to the Dodgers

* **Biomechanical & Visual Latency Inference from Decisions Alone:**
* A hitter whose swing decisions are best explained by $P(\text{Strike})$ at $\tau^* = 0.50$ (50% of flight) is **tracking the ball deeper** before committing (longer visual window, faster bat acceleration/shorter swing path).


* A hitter whose swing decisions peak in likelihood at $\tau^* = 0.32$ (32% of flight) is **committing early** out of the pitcher's hand (long load, slow bat, or guessing pitch type at release).




* **Validation with Public Bat-Tracking Data:** Statcast bat-tracking metrics (`bat_speed`, `swing_length`, `time_to_contact`) are public on Baseball Savant for 2024–2025. If you show that your LSTM-inferred decision horizon $\tau^*_i$ correlates with Statcast `swing_length` or `bat_speed`—or identifies hitters who commit earlier than their bat speed requires—that is a publishable, front-office-grade finding.

---

## Comparison: How to Spend Your One Week

| Metric / Criterion | Suggested Idea 1 ("Does Discipline Travel?") | Option 1: Perceptual Mirage ($\Delta P_{0.40 \to 1.0}$) | Option 2: Uncertainty Sensitivity ($\det \Sigma_{0.40}$) | Option 3: Individual Decision Horizon ($\tau^*_i$) |
| --- | --- | --- | --- | --- |
| **Why it beats Z-O Swing%** | Doesn't really; just re-weights by count/zone. | Isolates the ~15% of pitches where early belief $\neq$ final zone. | Uses the MDN's $\sigma_x, \sigma_z, \rho$ directly instead of integrating them out. | Turns a static $0.40$ cutoff into a player-specific temporal trait. |
| **Code Changes Needed** | Grouping averages in pandas. | 20 lines in `swing_model.py` (residualizing $P_{0.40}$ vs. true zone). | 30 lines in `lstm_scoring.py` (exporting $\sigma_x \sigma_z \sqrt{1-\rho^2}$). | Running `build_swing_features.py` across 6 values of `--decision-frac`.

 |
| **Dodgers R&D Appeal** | Low (standard context neutralization). | **High** (separates pitch deception from batter zone judgment).

 | **High** (proves why the MDN head actually matters mathematically).

 | **Very High** (connects neural perception directly to bat tracking).

 |

### The 15-Minute Narrative Arc for Your Dodgers Interview

Instead of hiding the fact that your original DVA had a 90.6% $R^2$ with Z-Swing% and Chase%, **make that your hook in Minute 4 of the presentation**:

1. **The Setup (Minutes 0–4):** Walk through the `PitchPerceptionLSTM` + MDN and Bayesian DVA framework.


2. **The Critical Self-Audit (Minute 5):** *"When auditing my initial 2025 DVA leaderboard, I found that a linear combination of Z-Swing% and Chase% reconstructed 90.6% of the variance. Why? Because on 85% of pitches, trajectories are transparent enough that integrating the MDN down to a scalar $P(\text{Strike})$ at $40\%$ flight is collinear with plate arrival."* (Interviewers love when a candidate diagnoses their own model's collinearity).
3. **The Breakthrough (Minutes 6–12):** Show how unpacking the MDN into **Option 1 (Mirage Residual $\Delta P$)** and **Option 2 (Covariance Volume $\det \Sigma_{0.40}$)**—or **Option 3 (Individual Commit Horizon $\tau^*_i$)**—extracts the exact orthogonal signal that scalar integration washed out.
4. **Baseball Actionability (Minutes 12–15):** Show how this re-ranks hitters (including your Kwan vs. Stowers script) and simultaneously scores pitchers on deception.


# AGENT IMPLEMENTATION SPECIFICATION: Orthogonal Perceptual Extensions for `strike-zone-perception`

## 0. Context & Architectural Invariants
You are operating inside the `sp26_sagb_dj` repository. Before modifying any files, inspect the existing codebase to confirm column names and tensor shapes:
- `src/model.py`: `PitchPerceptionLSTM` outputs 5 Mixture Density Network (MDN) parameters per timestep: `(mu_x, mu_z, sigma_x, sigma_z, rho)`.
- `src/lstm_scoring.py`: `LSTMEyeScorer` evaluates exact bivariate normal (BVN) CDF over the rectangular strike zone bounds via Owen's T function (`_bvn_cdf`).
- `src/swing_model.py`: `LeagueSwingModel` trains a 5-fold batter-grouped out-of-fold (OOF) XGBoost baseline and calculates Decision Value Added (DVA) with empirical Bayes shrinkage.
- `scripts/build_swing_features.py`: Loads `data/training/trajectories_{year}.pt` (left-padded `(100, 9)` tensors) and `data/models/best_model.pt`, runs inference at `--decision-frac` (default `0.40`), and writes `data/training/swing_features_{year}.parquet`.
- `tests/`: Contains 9 unit tests (`pytest tests/`). **Invariant:** All existing unit tests must continue to pass.

### Objective
Standard season-level DVA has a 90.6% $R^2$ with raw `Z-Swing%` and `Chase%` because integrating the MDN to a scalar $P(\text{Strike})$ at $t=0.40$ is highly collinear with final plate location on the ~85% of pitches that are transparent. We will implement three modular extensions that extract orthogonal signal from the LSTM+MDN without breaking backward compatibility:
1. **Extension 1: Perceptual Mirage Decomposition ($\Delta P = P_{0.40} - \mathbb{I}(\text{True Strike})$)**
2. **Extension 2: MDN Covariance Volume & Uncertainty Sensitivity ($\mathcal{U}_{0.40} = \sigma_x \sigma_z \sqrt{1 - \rho^2}$)**
3. **Extension 3: Individual Batter Decision Horizon Inference ($\tau^*_i \in [0.25, 0.55]$)**

---

## PHASE 1: Expose Full MDN Spatial Parameters & Ground-Truth Indicators

### 1.1 Modify `src/lstm_scoring.py`
Currently, `LSTMEyeScorer` computes `p_strike` from the MDN output `(mu_x, mu_z, sigma_x, sigma_z, rho)` and discards the raw distributional parameters or only returns `p_strike`.
1. Inspect `LSTMEyeScorer` in `src/lstm_scoring.py`.
2. Add a helper method or update the scoring method (preserving the default return signature or adding `return_mdn_params: bool = False`) so that for each pitch evaluated at `decision_frac`, we extract:
   - `p_strike`: Exact BVN integral over `[plate_x_min, plate_x_max] x [sz_bot, sz_top]`.
   - `mu_x`, `mu_z`: Predicted mean plate-crossing coordinates at `decision_frac`.
   - `sigma_x`, `sigma_z`, `rho`: Predicted standard deviations and correlation at `decision_frac`.
   - `uncertainty_vol`: The 2D Gaussian differential entropy volume proxy:
     $$\mathcal{U}_t = \sqrt{\det \mathbf{\Sigma}_t} = \sigma_x \cdot \sigma_z \cdot \sqrt{\max(1 - \rho^2, 10^{-6})}$$
   - `mahalanobis_edge_dist`: Distance from $(\mu_x, \mu_z)$ to the nearest strike-zone boundary normalized by directional variance.

### 1.2 Modify `scripts/build_swing_features.py`
1. Inspect `scripts/build_swing_features.py` to see how `swing_features_{year}.parquet` is constructed.
2. Ensure the following columns are computed and saved into `data/training/swing_features_{year}.parquet` (and if `--decision-frac` is non-default, allow an optional `--output` path so multi-horizon runs do not overwrite the default `0.40` parquet):
   - `p_strike` (at `decision_frac`)
   - `mu_x`, `mu_z`, `sigma_x`, `sigma_z`, `rho`
   - `uncertainty_vol`: $\sigma_x \sigma_z \sqrt{1 - \rho^2}$
   - `log_uncertainty_vol`: $\log(\text{uncertainty\_vol} + 10^{-6})$
   - `true_strike`: Binary indicator $\mathbb{I}(\vert{}\text{plate\_x}\vert{} \le 0.833 \text{ and } \text{sz\_bot} \le \text{plate\_z} \le \text{sz\_top})$ using the actual Statcast plate arrival coordinates (`plate_x`, `plate_z`, `sz_bot`, `sz_top`).
   - `mirage_delta`: $\Delta P = \text{p\_strike} - \text{true\_strike} \in [-1, 1]$.

---

## PHASE 2: Extension 1 — Perceptual Mirage Decomposition ($\Delta P$)

### 2.1 Mathematical Formulation
Every pitch $j$ faced by batter $i$ (and thrown by pitcher $p$) has a **Mirage Residual**:
$$\Delta P_j = P(\text{Strike} \mid \mathcal{F}_{0.40})_j - \mathbb{I}(\text{True Strike}_{1.0})_j$$

Classify every pitch into 4 mutually exclusive perceptual regimes using threshold $\tau_{\text{mirage}} = 0.25$ (and verify sensitivity at $0.20$ and $0.30$):
1. **Transparent Strike (`transparent_strike`):** $\text{true\_strike} = 1$ and $P_{0.40} \ge 0.75$ ($\Delta P \in [-0.25, 0]$).
2. **Transparent Ball (`transparent_ball`):** $\text{true\_strike} = 0$ and $P_{0.40} \le 0.25$ ($\Delta P \in [0, +0.25]$).
3. **Tunnel Mirage / Deceptive Ball (`tunnel_mirage`):** $\text{true\_strike} = 0$ and $P_{0.40} > 0.50$ ($\Delta P > +0.50$, or moderate $+0.25 \le \Delta P \le +0.50$). Looks like a strike at commit time, finishes as a ball.
4. **Frozen Mirage / Late-Breaking Strike (`frozen_mirage`):** $\text{true\_strike} = 1$ and $P_{0.40} < 0.50$ ($\Delta P < -0.50$, or moderate $-0.50 \le \Delta P \le -0.25$). Looks like a ball at commit time, clips the zone late.

### 2.2 Create `scripts/run_mirage_decomposition.py`
Write a standalone CLI script `scripts/run_mirage_decomposition.py` with arguments `--years 2020 2021 2022 2023 2024 2025` (or single `--year 2025`) and `--min-pitches 500`:
1. **Load Features:** Load `swing_features_{year}.parquet` and OOF baseline predictions `p_swing_league` from `LeagueSwingModel`.
2. **Compute Regime-Specific DVA & Swing Rates per Batter:**
   - `dva_transparent`: DVA accumulated on `|\Delta P| < 0.25` pitches.
   - `dva_mirage`: DVA accumulated on `|\Delta P| >= 0.25` pitches.
   - `tunnel_chase_rate`: Raw swing rate on `tunnel_mirage` pitches ($\text{true\_strike} = 0, P_{0.40} \ge 0.50$), plus shrunken residual `tunnel_chase_res = mean(swing - p_swing_league)` shrunk via Empirical Bayes with $K = 50$ pitches.
   - `transparent_chase_rate`: Raw swing rate on `transparent_ball` pitches ($\text{true\_strike} = 0, P_{0.40} < 0.25$).
   - `frozen_take_rate`: Take rate on `frozen_mirage` pitches ($\text{true\_strike} = 1, P_{0.40} < 0.50$).
3. **Orthogonality & Predictive Audit (Crucial for Presentation):**
   - Regress `dva_mirage` and `tunnel_chase_res` against `(z_swing_pct, o_swing_pct)`. Report the $R^2$ (demonstrating that while overall DVA is 90.6% explained by Z-O Swing%, `dva_mirage` and `tunnel_chase_res` capture distinct variance).
   - Compute split-half reliability (odd vs. even games in 2025) and year-over-year correlation (if multi-year parquets exist) for `tunnel_chase_res` vs. `transparent_chase_rate`.
4. **Dual-Sided Pitcher Deception Leaderboard (`Perceptual Stuff`):**
   - Group by `pitcher_id` and `pitch_type` (minimum 100 pitches).
   - Calculate `mean_tunnel_mirage = mean(max(p_strike - true_strike, 0))` on true balls, and `induced_mirage_chase_oe` (swings above `p_swing_league` induced specifically via high $\Delta P$).
5. **Outputs:**
   - Save batter leaderboard to `data/leaderboards/2025_mirage_batter_leaderboard.csv`.
   - Save pitcher deception leaderboard to `data/leaderboards/2025_mirage_pitcher_leaderboard.csv`.
   - Generate figure `assets/mirage_orthogonality_quadrant.png`: A 2D scatter plot of qualifying 2025 batters with `transparent_chase_rate` on the X-axis and `tunnel_chase_res` on the Y-axis, highlighting archetypes (e.g., Steven Kwan, Kyle Stowers, Juan Soto, Shohei Ohtani, Freddie Freeman, Mookie Betts).

---

## PHASE 3: Extension 2 — Visual Uncertainty Sensitivity ($\det \Sigma_{0.40}$)

### 3.1 Mathematical Formulation
Two pitches can have identical $P(\text{Strike})_{0.40} = 0.50$ while having vastly different spatial uncertainty volumes $\mathcal{U}_{0.40} = \sigma_x \sigma_z \sqrt{1 - \rho^2}$.
We estimate each batter's **Uncertainty Policy ($\beta_{\mathcal{U}, i}$)** via a hierarchical/penalized logistic mixed-effects model on borderline pitches ($P(\text{Strike})_{0.40} \in [0.15, 0.85]$):
$$\text{logit } P(\text{Swing}_{ij}) = \text{logit}(\hat{p}_{\text{league}, ij}) + \alpha_i + \beta_{P, i} \cdot \tilde{P}(\text{Strike})_{ij} + \beta_{\mathcal{U}, i} \cdot \tilde{\mathcal{U}}_{0.40, ij}$$
where:
- $\text{logit}(\hat{p}_{\text{league}, ij})$ is an offset from the OOF XGBoost league swing model (trained *without* $\mathcal{U}_{0.40}$ so the offset only captures count, handedness, and expected location).
- $\tilde{\mathcal{U}}_{0.40, ij}$ is the z-scored log uncertainty volume $\log(\sigma_x \sigma_z \sqrt{1 - \rho^2})$, standardized within pitch-type group or overall so $\beta_{\mathcal{U}, i}$ is in interpretable log-odds per SD of perceptual uncertainty.
- Random slopes $(\alpha_i, \beta_{P, i}, \beta_{\mathcal{U}, i}) \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Omega})$ are estimated via L2-penalized ridge logistic regression per batter (Empirical Bayes shrinkage with penalty $\lambda$ tuned via 5-fold cross-validation on held-out log-loss).

### 3.2 Create `scripts/run_uncertainty_sensitivity.py`
Write `scripts/run_uncertainty_sensitivity.py`:
1. **Filter to Competitive Decision Window:** Restrict to pitches where $P(\text{Strike})_{0.40} \in [0.15, 0.85]$ (where perceptual uncertainty actively affects the decision boundary).
2. **Standardize Features:**
   - Compute `u_zscore = (log_uncertainty_vol - mean) / std`.
   - Compute `p_centered = p_strike - 0.50`.
   - Compute `league_logit = clip(logit(p_swing_league), -6, 6)`.
3. **Fit Empirical Bayes Batter Slopes:**
   - For each qualifying batter ($N \ge 200$ borderline pitches), fit a penalized logistic regression with offset `league_logit` and features `[1, p_centered, u_zscore]` using L2 prior precision $\Lambda = \text{diag}(\lambda_0, \lambda_1, \lambda_2)$ calibrated from between-batter variance.
   - Extract posterior mean and standard error for:
     - $\alpha_i$: Baseline aggressiveness above league average.
     - $\beta_{P, i}$: Strike-zone sharpness (steepness of psychometric curve with respect to $P(\text{Strike})$).
     - $\beta_{\mathcal{U}, i}$: **Uncertainty Aversion / Ambiguity Tolerance**.
       - $\beta_{\mathcal{U}, i} < 0$: Batter **freezes/takes** when the MDN covariance ellipse is wide.
       - $\beta_{\mathcal{U}, i} > 0$: Batter **guesses/fires** when the MDN covariance ellipse is wide.
4. **Validation & Held-Out Log-Likelihood Test:**
   - Evaluate 5-fold out-of-sample log-loss and Brier score comparing:
     - Model A: `league_logit + alpha_i + beta_P_i * p_centered`
     - Model B: `league_logit + alpha_i + beta_P_i * p_centered + beta_U_i * u_zscore`
   - Report the exact log-likelihood improvement ($\Delta \text{LL}$) and likelihood-ratio p-value across the league and for top batters.
   - Compute the $R^2$ of $(\beta_{P, i}, \beta_{\mathcal{U}, i})$ with `(z_swing_pct, o_swing_pct)` to prove orthogonality.
5. **Outputs:**
   - Save `data/leaderboards/2025_uncertainty_sensitivity_leaderboard.csv` containing `batter_id, player_name, n_borderline, alpha_i, beta_P_i, beta_U_i, beta_U_se, z_swing_pct, o_swing_pct, dva`.
   - Generate `assets/uncertainty_response_curves.png`: Plot $P(\text{Swing})$ as a function of Uncertainty Volume $\mathcal{U}_{0.40}$ (holding $P(\text{Strike}) = 0.50$ fixed) for two contrasting pairs of hitters (e.g., Steven Kwan vs. Kyle Stowers, and two hitters with identical 25% Chase% but opposite $\beta_{\mathcal{U}, i}$ signs).

---

## PHASE 4: Extension 3 — Individual Batter Commit Horizon ($\tau^*_i$)

### 4.1 Mathematical Formulation
Instead of assuming every MLB hitter commits at $\tau = 0.40$ (40% of pitch flight), we leverage the streaming `(100, 9)` nature of `PitchPerceptionLSTM` across a temporal grid:
$$\mathcal{T} = \{0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55\}$$
(corresponding to indices `t_idx = [24, 29, 34, 39, 44, 49, 54]` in the 100-step trajectory tensor).

For each pitch $j$ and each candidate horizon $\tau \in \mathcal{T}$, compute $P(\text{Strike} \mid \mathcal{F}_\tau)_j$.
For each batter $i$, fit a count-adjusted psychometric swing