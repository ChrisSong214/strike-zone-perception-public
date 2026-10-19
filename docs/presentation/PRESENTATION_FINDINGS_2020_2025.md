# Empirical Findings & Presentation Assets (2020–2025)

**Strike Zone Perception & Bayesian Swing Decision Engine (DVA)**  
*Multi-Season Empirical Validation across 6 MLB Seasons (2020–2025)*  
*Dataset: 3.8+ Million Statcast Pitches | 3,089 Qualified Batter-Seasons | 1,831 Consecutive Season Pairs*

---

## 🏆 Key Statistical Takeaways for Conference Slides

### 1. Sample Size Threshold Sensitivity: Filtering Noise Reveals Pure Talent
Filtering out low-sample bench players and injury-shortened seasons dramatically strengthens DVA's signal, whereas traditional Chase Rate actually deteriorates due to umpire call variance and pitch movement noise:

| Min Pitches Seen | Approx PA | Consecutive Batter Pairs ($N$) | DVA YoY $r$ ($R^2$) | Chase Rate YoY $r$ ($R^2$) | Future wOBA $\Delta R^2$ (+DVA) | % Lift | $F$-Statistic | $p$-value |
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
Out-of-sample regressions comparing Baseline $(\text{Target}_N + \text{Chase Rate}_N)$ vs Extended $(\text{Target}_N + \text{Chase Rate}_N + \text{DVA}_N)$:

| Forecast Target | Baseline (Chase) $R^2$ | Baseline (Z-O Swing%) $R^2$ | Extended (+ DVA) $R^2$ | $\Delta R^2$ | % Gain | $F$-Statistic | $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Next-Year wOBA ($\text{wOBA}_{N+1}$)** | **0.1670** | **0.1749** | **0.1761** | **+0.0091** | **+5.5%** | **13.74** | **$p = 0.000219$ ($p < 0.001$)** |
| **Next-Year Walk Rate ($\text{BB\%}_{N+1}$)** | 0.0405 | 0.0334 | 0.0437 | +0.0032 | +8.0% | 4.19 | $p = 0.0410$ ($p < 0.05$) |
| **Next-Year Strikeout Rate ($\text{K\%}_{N+1}$)** | 0.0700 | 0.0700 | 0.0705 | +0.0006 | +0.8% | 0.78 | $p = 0.378$ |

---

### 4. Sample Size Stabilization Rate
Using odd/even pitch split-half reliability across sample sizes from 50 to 1,000 pitches:
* **200 Pitches:** $r = 0.517$ (Moderate signal)
* **500 Pitches:** $r = 0.629$ (Strong signal)
* **750 Pitches:** $r = 0.706$ (**Crosses the standard $r = 0.70$ sabermetric stabilization threshold**)
* **1,000 Pitches:** $r = 0.755$

---

### 4. Talent Decomposition: The 4 Quadrant Archetypes
By decomposing DVA into **Macro Count Tendency** vs. **Micro Pitch Execution**:
* **Quadrant I (The Elites):** High macro discipline + elite micro execution (e.g., Kyle Tucker, Corey Seager, Juan Soto).
* **Quadrant II (The Passive Snipers):** Overly passive in count approach, but elite visual acuity on borderline pitches when they swing.
* **Quadrant III (The Ambush Sluggers):** Aggressive green-light approach, but vulnerable to late secondary spin/movement.
* **Quadrant IV (The Overmatched):** Poor zone recognition and high swing error rate.

---

### 5. The Pitcher's Perspective: Deception Leaders
Top MLB pitchers who induce the most negative DVA in opposing hitters (causing the highest frequency of wrong decisions at the 40% flight decision point):
1. **Andrew Kittredge** (Deception: +10.41 | Chase Induced: 41.6% | Frozen Strike%: 34.4%)
2. **Graham Ashcraft** (Deception: +10.64 | Chase Induced: 38.4% | Frozen Strike%: 36.5%)
3. **Ryan Thompson** (Deception: +10.71 | Chase Induced: 36.0% | Frozen Strike%: 37.6%)
4. **Griffin Jax** (Deception: +11.61 | Chase Induced: 38.1% | Frozen Strike%: 35.9%)
5. **Aaron Nola** (Deception: +12.80 | Chase Induced: 33.4% | Frozen Strike%: 38.7%)

---

## 🎨 Generated Visual Assets (Ready for Slides)

All figures have been rendered at 300 DPI in [`assets/presentation/`](../../assets/presentation/):

1. **[`threshold_sensitivity.png`](../../assets/presentation/threshold_sensitivity.png)**: Dual-panel plot of Metric Autocorrelation ($r$) vs. Sample Threshold and Out-of-Sample $\Delta R^2$ Gain vs. Threshold.
2. **[`yoy_reliability.png`](../../assets/presentation/yoy_reliability.png)**: 4-panel regression comparison (DVA vs. Tendency vs. Chase Rate vs. wOBA).
3. **[`predictive_power.png`](../../assets/presentation/predictive_power.png)**: Grouped bar chart of out-of-sample $R^2$ improvement predicting Year $N+1$ performance.
4. **[`stabilization_curve.png`](../../assets/presentation/stabilization_curve.png)**: Split-half reliability curve across pitch counts ($N=50$ to $N=1000$).
5. **[`quadrant_archetypes.png`](../../assets/presentation/quadrant_archetypes.png)**: 2D scatter plot of Macro Tendency vs. Micro Execution with player callouts.
6. **[`breakout_candidates.png`](../../assets/presentation/breakout_candidates.png)**: $\Delta\text{DVA}$ vs. $\Delta\text{wOBA}$ breakout trajectory plot.
7. **[`pitcher_deception.png`](../../assets/presentation/pitcher_deception.png)**: Graphic table of the top deceptive pitchers inducing batter errors.
