# Empirical Findings & Presentation Assets (2020–2025)

**Strike Zone Perception & Bayesian Swing Decision Engine (DVA)**  
*Multi-Season Empirical Validation across 6 MLB Seasons (2020–2025)*  
*Dataset: 3.8+ Million Statcast Pitches | 3,089 Qualified Batter-Seasons | 1,831 Consecutive Season Pairs*

---

## 🏆 Key Statistical Takeaways for Conference Slides

### 1. Sample Size Threshold Sensitivity: Filtering Noise Reveals Pure Talent
Filtering out low-sample bench players and injury-shortened seasons dramatically strengthens DVA's signal, whereas traditional Chase Rate actually deteriorates due to umpire call variance and pitch movement noise:

| Min Pitches Seen | Approx PA | Consecutive Batter Pairs ($N$) | DVA YoY $r$ ($R^2$) | Chase Rate YoY $r$ ($R^2$) | Z-O Swing% YoY $r$ ($R^2$) | Future wOBA $\Delta R^2$ (+DVA) | % Lift | $F$-Statistic | $p$-value |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$\ge 300$** | $\sim 78$ | 1,845 | 0.7771 (0.604) | 0.7093 (0.503) | 0.6012 (0.362) | +0.0059 | +4.1% | 12.80 | $p = 0.00036$ |
| **$\ge 500$** | $\sim 131$ | 1,557 | 0.8005 (0.641) | 0.7208 (0.520) | 0.6119 (0.374) | +0.0049 | +2.8% | 9.24 | $p = 0.0024$ |
| **$\ge 750$** | $\sim 197$ | 1,247 | 0.8138 (0.662) | 0.7192 (0.517) | 0.6284 (0.395) | +0.0091 | +5.5% | 13.74 | $p = 0.00022$ |
| **$\ge 1,000$** | $\sim 263$ | 948 | **0.8332 (0.694)** | 0.7076 (0.501) | 0.6340 (0.402) | +0.0088 | +4.0% | 10.82 | $p = 0.0010$ |
| **$\ge 1,200$** | $\sim 315$ | 770 | 0.8243 (0.680) | 0.6842 (0.468) | 0.6143 (0.377) | **+0.0180** | **+8.1%** | **18.12** | **$p = 0.000023$** |
| **$\ge 1,500$** | $\sim 394$ | 580 | **0.8346 (0.697)** | 0.6829 (0.466) | 0.6250 (0.391) | +0.0163 | +7.7% | 12.15 | $p = 0.00053$ |

> **Slide Recommendation / Key Insight:**
> 1. **DVA Completely Dominates Z-O Swing%:** While analysts often cite Z-O Swing% (Zone Swing minus Chase Rate) as a single-number plate discipline proxy, it has significantly weaker persistence ($r \approx 0.60 - 0.63, R^2 \approx 39\%$). DVA explains **66.2% to 69.7% of next-season variance — an extra +27 to +30 percentage points of explained true-talent stickiness**.
> 2. **Why Z-O Swing% Underperforms Mathematically:** Hitters with aggressive approaches tend to swing more at both Zone AND Out-of-Zone pitches (positive covariance). Taking the difference $\text{Var}(Z - O) = \text{Var}(Z) + \text{Var}(O) - 2\text{Cov}(Z, O)$ subtracts away the batter's true stylistic signal while compounding measurement noise from umpire edge variance. DVA solves this by modeling flight uncertainty and point-of-decision value directly.
> 3. **Chase Rate Plateaus and Decays:** Traditional Chase Rate peaks at $r = 0.721$ and drops to $r = 0.683$ among qualified regulars.
> 4. **Triple the Predictive Lift on Future wOBA:** For regular starting hitters ($\ge 1,200$ pitches / $\approx 315$ PA), adding DVA to prior wOBA and Chase Rate yields an **+8.1% improvement in predictive power ($\Delta R^2 = +0.0180$, $F = 18.12, p = 2.3 \times 10^{-5}$)**.

---

### 2. True Talent Stickiness: DVA vs. Plate Discipline & Hitting Metrics ($\ge 750$ Pitches)
Across $N = 1,247$ qualified consecutive regular-season batter pairs (2020 $\to$ 2025):

| Metric | Sample ($N$) | Pearson $r$ | $R^2$ | Spearman $\rho$ | $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DVA Swing Rating** | **1,247** | **0.8138** | **0.6622** | **0.7987** | **$p < 10^{-295}$** |
| **Micro Execution DVA (`actual_dva`)** | 1,247 | 0.8026 | 0.6441 | 0.7851 | $p < 10^{-281}$ |
| **Macro Tendency DVA (`tendency_dva`)** | 1,247 | 0.7911 | 0.6258 | 0.7749 | $p < 10^{-268}$ |
| **Zone Swing Rate (Z-Swing%)** | 1,247 | 0.7855 | 0.6170 | 0.7778 | $p < 10^{-261}$ |
| **Traditional Chase Rate (O-Swing%)** | 1,247 | 0.7192 | 0.5172 | 0.7189 | $p = 4.3 \times 10^{-199}$ |
| **Z-O Swing% (Zone Swing - Chase)** | 1,247 | 0.6284 | 0.3949 | 0.6367 | $p = 5.5 \times 10^{-138}$ |
| **Offensive wOBA** | 1,247 | 0.4060 | 0.1648 | 0.3675 | $p = 1.1 \times 10^{-50}$ |

> **Presentation Punchline:** *At the qualified starter threshold ($\ge 750$ pitches), DVA explains **66.2% of next-season decision variance ($R^2 = 0.662$)**, beating traditional Chase Rate (51.7%) by **14.5 percentage points** and Z-O Swing% (39.5%) by **26.7 percentage points of explained variance**.*

---

### 3. Predictive Validity: Forecasting Future Performance ($N \to N+1$, $\ge 750$ Pitches)
Out-of-sample regressions evaluated using 5-fold `GroupKFold` grouped strictly by `batter_id` (preventing player-specific fixed effect leakage across consecutive seasons) and reporting uncensored $\Delta R^2$:

| Forecast Target | Baseline (Chase) $R^2$ | Baseline (Z-O Swing%) $R^2$ | Extended (+ DVA) $R^2$ | Uncensored $\Delta R^2$ | % Gain | In-Sample Nested $F$ | In-Sample $p$-value | Out-of-Sample Bootstrap 95% CI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Next-Year wOBA ($\text{wOBA}_{N+1}$)** | **0.1670** | **0.1749** | **0.1761** | **+0.0091** | **+5.5%** | **13.74** | **$p = 0.000219$** | **[+0.0035, +0.0152] ($p < 0.001$)** |
| **Next-Year Walk Rate ($\text{BB\%}_{N+1}$)** | 0.0405 | 0.0334 | 0.0437 | +0.0032 | +8.0% | 4.19 | $p = 0.0410$ | [+0.0002, +0.0071] ($p = 0.038$) |
| **Next-Year Strikeout Rate ($\text{K\%}_{N+1}$)** | 0.0700 | 0.0700 | 0.0705 | +0.0006 | +0.8% | 0.78 | $p = 0.378$ | [-0.0009, +0.0024] ($p = 0.354$) |

> **Methodological Rigor Highlights:**
> 1. **GroupKFold Integrity:** Rather than random pooling that leaks player fixed effects across multi-year pairs ($2020\to2021, \dots, 2024\to2025$), grouping by `batter_id` ensures held-out validation evaluates true out-of-sample player generalization.
> 2. **Uncensored Metrics:** Negative changes are never clamped to zero; raw $\Delta R^2 = R^2_{\text{full}} - R^2_{\text{base}}$ is reported transparently.
> 3. **Hypothesis Testing Separation:** Classical $F$-tests are reported as in-sample nested OLS model evaluations, whereas out-of-sample predictive improvement is validated with 1,000 batter-clustered bootstrap resamples and rolling chronological evaluations.

---

### 4. Sample Size Stabilization Rate (Production EV-DVA & Alignment DVA)
Using odd/even pitch split-half reliability with full production Empirical Bayes shrinkage ($K^* = \sigma^2_w / \tau^2_b$) across sample sizes from 50 to 1,000 pitches:
* **200 Pitches:** $r = 0.517$ (Moderate signal)
* **500 Pitches:** $r = 0.629$ (Strong signal)
* **750 Pitches:** $r = 0.706$ (**Crosses the standard $r = 0.70$ sabermetric stabilization threshold**)
* **1,000 Pitches:** $r = 0.755$
*(Note: Evaluated on the exact production EV-DVA and Alignment DVA metrics rather than simplified proxies, verifying that the published Bayesian ratings stabilize rapidly.)*

---

### 5. Dual Baseline Decoupling: Objective Post-Hoc vs. Perceptual Decision-Time
The platform maintains two explicitly decoupled league swing baselines to address the information boundary between physical reality and batter perception:
1. **Objective Post-Hoc Baseline (`FINAL_LOCATION_FEATURES`):** Conditions on where the pitch arrived at the plate (`plate_x, plate_z, loc_x_norm, loc_z_norm, dist_from_center, in_zone`).
2. **Perceptual Decision-Time Baseline (`DECISION_TIME_FEATURES`):** Conditions exclusively on early trajectory information available at the biological decision point $t=0.40$ (`lstm_p_strike, mu_x, mu_z, sigma_x, sigma_z, rho, balls, strikes, stand, p_throws, pitch_type, release_speed`).
3. **Deception Over Expected:** The difference $\hat{p}_{\text{swing}}(\text{Decision-Time}) - \hat{p}_{\text{swing}}(\text{Final-Location})$ directly measures pitch deception—how much a pitch's early trajectory tricks the league into swinging beyond where it actually finishes.

---

### 6. Deception vs. Discipline: The Perceptual Mirage Decomposition ($\Delta P$)

By comparing the LSTM's perceptual belief distribution at the **biological commit point ($t = 0.40$, ~175 ms)** against the **ground-truth plate arrival ($t = 1.0$)**, we define the **Mirage Residual**:
$$\Delta P_i = P(\text{Strike} \mid \mathcal{F}_{0.40})_i - \mathbb{I}(\text{True Strike})_i$$

- **Separating True Perceptual Reversals from General Discrepancies:**
  - **True Perceptual Reversals (Strict Sign-Flip Across 0.50 Decision Boundary):** Pitches where early flight indicated a strike with majority confidence ($P_{0.40} \ge 0.50$) but missed the zone late (**Tunnel Reversals**), or indicated a ball ($P_{0.40} \le 0.50$) but clipped the zone late (**Frozen Reversals**). A ball with $P=0.10$ is never mislabeled as a tunnel reversal.
  - **Prediction Discrepancy Threshold ($|\Delta P| \ge 0.08$):** Identifies boundary-uncertainty discrepancies ($11,884$ pitches in 2025).
  - **Continuous Deception-Weighted Chase OE:**
    $$\text{Deceptive Chase OE}_i = \frac{\sum_{j \in \text{Balls}} P(\text{Strike})_{0.40, j} \cdot (\text{Swing}_{ij} - \hat{p}_{\text{league}, ij})}{\sum_{j \in \text{Balls}} P(\text{Strike})_{0.40, j}}$$
    Gives appropriate continuous weight to every out-of-zone pitch based on early strike belief without requiring arbitrary thresholds.

Calibrated at $|\Delta P| \ge 0.08$ across 724,002 Statcast pitches in 2025:
* **Transparent Pitches ($|\Delta P| < 0.08$):** $98.4\%$ ($712,118$ pitches)
* **Perceptual Mirage Pitches ($|\Delta P| \ge 0.08$):** $1.6\%$ ($11,884$ pitches)
  * **Tunnel Chases ($\Delta P \ge +0.08$):** $5,595$ pitches (pitches appearing in-zone early that break out)
  * **Frozen Takes ($\Delta P \le -0.08$):** $6,289$ pitches (pitches appearing out-of-zone early that clip late)

#### 📊 Continuous Deception-Weighted Chase: The True Measure of Tunnel Resistance
To eliminate threshold sensitivity across sample sizes, we define the **Deception-Weighted Chase Rate**:
$$\text{Weighted Tunnel Chase} = \frac{\sum \text{Swing}_i \cdot P(\text{Strike} \mid \mathcal{F}_{0.40})_i \cdot (1 - \text{True Strike}_i)}{\sum P(\text{Strike} \mid \mathcal{F}_{0.40})_i \cdot (1 - \text{True Strike}_i)}$$
Across 421 qualified hitters in 2025, Weighted Tunnel Chase averages $60.2\%$ (spanning $10.4\%$ to $99.9\%$). Crucially, its correlation with raw chase rate is only $r = 0.284$ ($R^2 = 0.081$), demonstrating that traditional chase rate explains only $8.1\%$ of the linear variation, leaving **$91.9\%$ of unexplained linear variation in pitch deception susceptibility.**

#### 📊 Variance Decomposition Audit: Breaking the 90% Collinearity with Z-O Swing%
Regressing metrics against linear combinations of `(Z-Swing%, Chase Rate)`:

| Metric | Linear $R^2$ with (Z-Swing%, Chase%) | Unexplained Linear Variance ($1 - R^2$) | Non-Linear GBDT $R^2$ | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Overall DVA (Total)** | 0.649 | 35.1% | 0.672 | Influenced by unambiguous strikes and balls |
| **Transparent DVA ($|\Delta P| < 0.08$)** | 0.648 | 35.2% | 0.670 | Baseline strike zone geometry calibration |
| **Mirage DVA ($|\Delta P| \ge 0.08$)** | **0.002** | **99.8%** | **0.005** | **Virtually uncaptured by traditional plate discipline!** |
| **Tunnel Chase Residual (`tunnel_chase_res`)** | **0.209** | **79.1%** | **0.224** | Measures pure susceptibility to late pitch tunneling |

> **Key Discovery:** A batter's **Transparent Chase Rate** and **Tunnel Chase Residual** share an $R^2$ of only **0.135** ($r = 0.368, p = 6.4 \times 10^{-15}$), leaving over **86.5% of unexplained linear variation** between baseline zone judgment and pitch tunneling vulnerability. Non-linear GBDT regression similarly yields $R^2 = 0.152$, confirming that deception vulnerability is distinct from baseline zone discipline. Split-half odd/even pitch reliability is $r = 0.813$ (Spearman-Brown reliability: $0.897$).

#### 🔄 Dynamic Trajectory Belief Reversal & Pitcher Deception Repeatability
To separate true pitch deception from ordinary binomial forecast noise on borderline pitches ($P = 0.60$ landing outside 40% of the time):
1. **Trajectory Belief Reversal ($\text{belief\_shift}$):** Evaluates the active dynamic drift in early-flight strike belief $\Delta P_{\text{shift}} = P(\text{Strike})_{0.30} - P(\text{Strike})_{0.55}$, measuring whether late break genuinely diverged from the early trajectory vector.
2. **Pitcher Induced Deception Over Expected (`induced_mirage_chase_oe`):** Models batter swing residuals after controlling for early strike probability, Euclidean edge distance, count, and pitch type, shrunk by pitcher via Empirical Bayes.
3. **Split-Half Reliability:** Testing across odd-game vs. even-game splits for 602 MLB pitchers in 2025 demonstrates that deception is a **statistically significant, repeatable pitcher skill**:
   - `induced_mirage_chase_oe`: $r = 0.233$ ($p = 7.7 \times 10^{-9}$)
   - `mean_belief_shift`: $r = 0.280$ ($p = 2.7 \times 10^{-12}$)

#### 🔍 Batter Archetypes Spotlight: Kwan vs. Stowers vs. Dodgers Stars
Using Empirical Bayes shrinkage ($K = 50$) to create the **Deception Vulnerability Index (DVI)** (100 = league avg, higher = chases more pitch tunnels):

* **Steven Kwan (72.6 DVI):** 25.8% Transparent Chase | **20.0% Tunnel Chase** | 35.7% Weighted Tunnel Chase. Kwan possesses superhuman immunity to pitch tunneling; even when a pitch fools 60% of the league, he stays off it.
* **Kyle Stowers (119.5 DVI):** 33.7% Transparent Chase | **63.6% Tunnel Chase** | 72.8% Weighted Tunnel Chase. While Stowers has average baseline zone control, late secondary break fools him on nearly 2 out of 3 deceptive balls.
* **Juan Soto (68.1 DVI):** 19.3% Transparent Chase | **37.5% Tunnel Chase** | 44.1% Weighted Tunnel Chase. Elite in both baseline zone discipline and tunneling resistance.
* **Mookie Betts (59.2 DVI):** 25.8% Transparent Chase | **30.0% Tunnel Chase** | 39.8% Weighted Tunnel Chase. The premier deception-immune hitter in MLB; severely outperforms league expectation on deceptive pitches.
* **Shohei Ohtani (118.8 DVI):** 30.6% Transparent Chase | **61.9% Tunnel Chase** | 71.4% Weighted Tunnel Chase. Highly aggressive ambush profile; vulnerable to elite late tunneling breaks.

#### 🎯 Dual-Sided Pitcher "Perceptual Stuff+" Leaderboard ($N \ge 250$ Pitches, $N_{\text{dec}} \ge 15$)
Requiring $N \ge 250$ pitches and $N_{\text{dec}} \ge 15$ deceptive balls isolates qualified arms with genuine late deception, eliminating small-sample reliever noise:
1. **Bailey Ober (FF):** **155.9 Perceptual Stuff+ (Rank #1 in MLB)** | 31 Deceptive Balls | 38.5% Chase Induced on Balls | $+3.4\%$ Swings Above Expectation.
2. **Carlos Vargas (SI):** **151.3 Perceptual Stuff+ (Rank #2 in MLB)** | 18 Deceptive Balls | 34.6% Chase Induced on Balls | $+15.2\%$ Swings Above Expectation.
3. **Trevor Rogers (FF):** **135.1 Perceptual Stuff+ (Rank #3 in MLB)** | 18 Deceptive Balls | 35.2% Chase Induced on Balls | $+2.0\%$ Swings Above Expectation.
4. **Kyle Hendricks (CH):** **129.8 Perceptual Stuff+ (Rank #4 in MLB)** | 38 Deceptive Balls | 35.6% Chase Induced on Balls | Master of late fade and velocity tunneling.
5. **Kris Bubic (FF):** **127.0 Perceptual Stuff+ (Rank #5 in MLB)** | 17 Deceptive Balls | 35.6% Chase Induced on Balls | $+9.1\%$ Swings Above Expectation.
* **Yoshinobu Yamamoto:** Ranks #19 overall in MLB (**113.9 Perceptual Stuff+**, 26 deceptive balls across arsenal); Yamamoto's **Cutter (FC)** specifically leads MLB pitch types at **172.9 Perceptual Stuff+** (Rank #1 among all pitch types with $\ge 5$ deceptive balls).

---

### 6. The Pitcher's Perspective: Deception Leaders (Traditional DVA)
Top MLB pitchers who induce the most negative DVA in opposing hitters (causing the highest frequency of wrong decisions at the 40% flight decision point):
1. **Andrew Kittredge** (Deception: +10.41 | Chase Induced: 41.6% | Frozen Strike%: 34.4%)
2. **Graham Ashcraft** (Deception: +10.64 | Chase Induced: 38.4% | Frozen Strike%: 36.5%)
3. **Ryan Thompson** (Deception: +10.71 | Chase Induced: 36.0% | Frozen Strike%: 37.6%)
4. **Griffin Jax** (Deception: +11.61 | Chase Induced: 38.1% | Frozen Strike%: 35.9%)
5. **Aaron Nola** (Deception: +12.80 | Chase Induced: 33.4% | Frozen Strike%: 38.7%)

---

### 7. Sensitivity to Visual Uncertainty: The MDN Covariance Matrix ($\det \mathbf{\Sigma}_{0.40}$)

Rather than treating model uncertainty as an unobserved nuisance parameter, the LSTM+MDN explicitly outputs the full bivariate spatial covariance matrix at the early-flight decision point ($t = 0.40$):
$$\mathbf{\Sigma}_{0.40} = \begin{pmatrix} \sigma_x^2 & \rho \sigma_x \sigma_z \\ \rho \sigma_x \sigma_z & \sigma_z^2 \end{pmatrix}$$

The **Uncertainty Volume** represents the area of the 1-standard-deviation confidence ellipse:
$$\text{Vol}(\mathbf{\Sigma}_{0.40}) = \pi \sqrt{\det \mathbf{\Sigma}_{0.40}} = \pi \sigma_x \sigma_z \sqrt{1 - \rho^2}$$

#### 📐 Boundary-Orthogonal Uncertainty Controls & Predictive Validity
To ensure that spatial uncertainty is not merely acting as a proxy for distance to the strike-zone edge:
1. **Euclidean Edge Distance ($d_{\text{edge}}$):** Exact geometric distance from predicted mean location $(\mu_x, \mu_z)$ to the nearest strike-zone rectangle boundary.
2. **Boundary-Orthogonal Standard Deviation ($\sigma_{\perp}$):** Directional standard deviation along the axis perpendicular to the nearest edge ($\sigma_x$ for side edges, $\sigma_z$ for top/bottom edges).
3. **Out-of-Sample Log-Likelihood Test:** Adding $\log \text{Vol}(\mathbf{\Sigma}_{0.40})$ and $\sigma_{\perp}$ to a baseline controlling for count-conditioned league swing logits, early strike probability $P(\text{Strike})$, and Euclidean edge distance yields a statistically significant out-of-sample log-likelihood improvement across 5-fold cross-validation ($\Delta \text{LL} > 0, p < 0.001$), proving that spatial uncertainty drives batter choices beyond simple boundary distance.

On borderline pitches ($0.30 \le P(\text{Strike}) \le 0.70$), we estimate each batter's **Ambiguity Policy Parameter ($\beta_{U,i}$)**:
$$\text{logit } P(\text{Swing}_i) = \alpha_i + \beta_{P,i}(P - 0.50) + \beta_{U,i} \ln\left(\frac{\text{Vol}_i}{\text{Vol}_{\text{league}}}\right) + \beta_{\text{edge}} d_{\text{edge}} + \beta_{\perp} \sigma_{\perp}$$

#### 🎯 Strategic Archetypes: Ambiguity-Seeking vs. Ambiguity-Averse
Across 353 qualified hitters in 2025:

* **Ambiguity-Seeking ($\beta_{U,i} > 0$, "When in Doubt, Swing"):** Hitters who expand their swing aggression under visual uncertainty to protect the plate:
  1. **Tyler Stephenson:** $\beta_U = +1.98$ ($p < 0.001$, Rank #1 Ambiguity-Seeking)
  2. **Tyler Soderstrom:** $\beta_U = +1.77$ ($p < 0.001$)
  3. **Luis García:** $\beta_U = +1.57$ ($p < 0.001$)
  4. **Lars Nootbaar:** $\beta_U = +1.37$ ($p < 0.001$)
  5. **Bryan Reynolds:** $\beta_U = +1.28$ ($p < 0.001$)

* **Ambiguity-Averse ($\beta_{U,i} < 0$, "When in Doubt, Take"):** Hitters who refuse to offer unless trajectory confidence is exceptionally clear:
  1. **Mike Trout:** $\beta_U = -2.10$ ($p < 0.001$, Rank #1 Ambiguity-Averse in MLB)
  2. **Ronald Acuña Jr.:** $\beta_U = -2.06$ ($p < 0.001$)
  3. **Blaze Alexander:** $\beta_U = -1.52$ ($p < 0.001$)
  4. **Ryan McMahon:** $\beta_U = -1.37$ ($p < 0.001$)
  5. **Trent Grisham:** $\beta_U = -1.30$ ($p < 0.001$)
  6. **Marcell Ozuna:** $\beta_U = -1.19$ ($p < 0.001$)

> **Dodgers Advance Scouting Application:** Pitchers attacking Mike Trout or Ronald Acuña Jr. should throw high-uncertainty pitches (high movement variance) on borderline counts, as they reliably freeze and take. Conversely, attacking Tyler Stephenson or Luis García with ambiguous movement induces out-of-zone emergency swings.

---

### 8. Inferring Individual Information-Use Horizons ($\tau^*_i$) & Statcast Bat-Tracking Validation

Instead of assuming every MLB hitter commits to a swing at a static 40% flight, we infer each batter's unique **Information-Use Horizon ($\tau^*_i \in [0.25, 0.55]$)** using a smooth cumulative **Information Center of Mass**:
$$\tau^*_i = \tau_0 + \frac{\sum_{k=1}^K \bar{\tau}_k \cdot \max(w_{ik}, 0)}{\sum_{k=1}^K \max(w_{ik}, 0)}$$
where $w_{ik}$ are the non-negative ridge regression weights on sequential trajectory updates $\Delta P_{\tau_{k-1} \to \tau_k}$, and $\text{SE}(\tau^*_i)$ is computed in closed form via the delta method.

#### ⏱️ Population Inferred Information-Use Distribution (421 Qualified Hitters in 2025)
* **$\tau^* \in [0.25, 0.32]$ ($100\text{--}130\text{ ms}$ after release):** $110$ hitters ($26.1\%$) — Early Information Cutoff
* **$\tau^* \in (0.32, 0.42]$ ($130\text{--}170\text{ ms}$ after release):** $137$ hitters ($32.5\%$) — League-Median Decision Window
* **$\tau^* \in (0.42, 0.55]$ ($170\text{--}220\text{ ms}$ after release):** $174$ hitters ($41.3\%$) — Deep Trajectory Trackers

#### ⚾ Biomechanical Grounding: Independent Statcast Bat-Tracking Validation
Framing bat tracking as an **independent external biomechanical validation** confirms that physical swing mechanics strongly constrain the cognitive information horizon:
* **Giancarlo Stanton ($\tau^* = 0.30 \pm 0.04$, $120\text{ ms}$):** Massive rotational inertia ($79.0\text{ mph}$ bat speed, $8.72\text{ ft}$ swing length). Stanton's long acceleration arc requires an earlier information cutoff.
* **Mookie Betts ($\tau^* = 0.28 \pm 0.03$, $110\text{ ms}$):** Ultra-fast read-and-commit profile ($66.7\text{ mph}$ bat speed, $6.95\text{ ft}$ swing length).
* **Steven Kwan ($\tau^* = 0.35 \pm 0.03$, $140\text{ ms}$):** Extremely compact stroke ($61.5\text{ mph}$ bat speed, $6.17\text{ ft}$ swing length).
* **Shohei Ohtani ($\tau^* = 0.40 \pm 0.03$, $160\text{ ms}$):** Balanced decision point ($74.1\text{ mph}$ bat speed, $7.90\text{ ft}$ swing length).
* **Aaron Judge ($\tau^* = 0.44 \pm 0.03$, $175\text{ ms}$):** Elite pitch tracking capability ($74.8\text{ mph}$ bat speed, $8.04\text{ ft}$ swing length).
* **Juan Soto ($\tau^* = 0.46 \pm 0.03$, $185\text{ ms}$):** Deep, patient vision allowing late pitch resolution ($71.0\text{ mph}$ bat speed, $6.97\text{ ft}$ swing length).
* **Freddie Freeman ($\tau^* = 0.49 \pm 0.03$, $195\text{ ms}$):** Short, inside-out stroke ($68.4\text{ mph}$ bat speed, $6.81\text{ ft}$ swing length) allowing him to utilize trajectory information deeper than $90\%$ of MLB hitters.

---

### 9. Full-Pipeline Holdout & Acceleration Sensitivity Audit

To establish uncompromising methodological rigor, we evaluate the entire two-stage modeling pipeline (LSTM Trajectory Perception $\to$ XGBoost League Baseline $\to$ Bayesian EV-DVA) across temporal, group, and kinematic holdouts:

#### 📅 Temporal Holdout Stability (First-Half vs. Second-Half 2025)
* **Sample Split:** $369,169$ pitches in First-Half (pre-July 1) vs. $354,833$ pitches in Second-Half (post-July 1).
* **Upstream LSTM Calibration Invariance:**
  - 1st Half: Brier = $0.0875$ | ECE = $0.0908$ | AUC-ROC = $0.9746$ | Log-Loss = $2.1508$
  - 2nd Half: Brier = $0.0866$ | ECE = $0.0899$ | AUC-ROC = $0.9747$ | Log-Loss = $2.1378$
* **Downstream EV-DVA Split-Half Stability:** Across 342 qualified batters ($\ge 250$ pitches in both halves), split-half correlation is **$r = 0.721$ ($p = 3.58 \times 10^{-56}$, Spearman $\rho = 0.692$)**, closely mirroring benchmark Chase Rate ($r = 0.784$) and Zone-Swing% ($r = 0.768$).

#### 👥 Group Holdout Generalization (Unseen Pitchers & Unseen Batters)
* **Unseen Pitchers (5-Fold GroupKFold):** Brier = $0.0872 \pm 0.0011$ | ECE = $0.0904 \pm 0.0011$ | AUC = $0.9746 \pm 0.0005$
* **Unseen Batters (5-Fold GroupKFold):** Brier = $0.0871 \pm 0.0004$ | ECE = $0.0904 \pm 0.0004$ | AUC = $0.9747 \pm 0.0005$
* **Interpretation:** The perception model generalizes seamlessly across release geometries and batting stances with zero degradation.

#### 🚀 Kinematic Sensitivity & The Perceptual Proxy Framework
* **Linear Tangent Projection from $t = 0.40$:** Extrapolating plate crossing using only $(x, y, z, v_x, v_y, v_z)$ at $40\%$ flight (assuming zero late acceleration) achieves:
  - Plate X: $r = 0.9684$ ($R^2 = 0.9377$) | MAE = $2.16$ inches
  - Plate Z: $r = 0.9727$ ($R^2 = 0.9462$) | MAE = $5.29$ inches
  - **Result:** Over **$93.8\%$--$94.6\%$ of final plate coordinate variance is already determined by early release position and velocity.**
* **Trajectory-Based Perceptual Proxy vs. Biological Perception:**
  Statcast's 9-parameter inputs provide the normative physical upper bound of strike-location information contained in early ball flight ($P(\text{Strike} \mid \mathcal{F}_{0.40})$). Human visual perception carries sensory transmission delays (~150 ms) and retinal slip; thus, $P(\text{Strike} \mid \mathcal{F}_{0.40})$ serves as an objective, leak-free physical proxy for evaluating decision value rather than a direct physiological measurement.

---

## 🎨 Generated Visual Assets (Ready for Slides)

All figures and tables have been rendered in [`assets/presentation/`](../../assets/presentation/):

1. **[`threshold_sensitivity.png`](../../assets/presentation/threshold_sensitivity.png)**: Dual-panel plot of Metric Autocorrelation ($r$) vs. Sample Threshold and Out-of-Sample $\Delta R^2$ Gain vs. Threshold.
2. **[`yoy_reliability.png`](../../assets/presentation/yoy_reliability.png)**: 4-panel regression comparison (DVA vs. Tendency vs. Chase Rate vs. wOBA).
3. **[`predictive_power.png`](../../assets/presentation/predictive_power.png)**: Grouped bar chart of out-of-sample $R^2$ improvement predicting Year $N+1$ performance.
4. **[`stabilization_curve.png`](../../assets/presentation/stabilization_curve.png)**: Split-half reliability curve across pitch counts ($N=50$ to $N=1000$).
5. **[`quadrant_archetypes.png`](../../assets/presentation/quadrant_archetypes.png)**: 2D scatter plot of Macro Tendency vs. Micro Execution with player callouts.
6. **[`breakout_candidates.png`](../../assets/presentation/breakout_candidates.png)**: $\Delta\text{DVA}$ vs. $\Delta\text{wOBA}$ breakout trajectory plot.
7. **[`pitcher_deception.png`](../../assets/presentation/pitcher_deception.png)**: Graphic table of the top deceptive pitchers inducing batter errors.
8. **[`mirage_orthogonality_quadrant.png`](../../assets/presentation/mirage_orthogonality_quadrant.png)**: 2D scatter plot of Transparent Chase Rate vs. Tunnel Chase Residual with key player annotations.
9. **[`uncertainty_response_curves.png`](../../assets/presentation/uncertainty_response_curves.png)**: Psychometric response curves showing individual batter swing probability shifts under visual uncertainty ($\det \mathbf{\Sigma}_{0.40}$ & $\sigma_{\perp}$).
10. **[`commit_horizon_bat_tracking.png`](../../assets/presentation/commit_horizon_bat_tracking.png)**: Dual-panel plot showing the population distribution of individual Information-Use Horizons $\tau^*_i$ and biomechanical validation against Statcast bat speed and swing length.
11. **[`full_pipeline_holdout_audit.csv`](../../assets/presentation/full_pipeline_holdout_audit.csv)**: Summary table of temporal split calibration, group holdout generalization, downstream EV-DVA stability, and kinematic sensitivity.
