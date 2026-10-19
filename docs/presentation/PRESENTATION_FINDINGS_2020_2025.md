# Empirical Findings & Presentation Assets (2020–2025)

**Strike Zone Perception & Bayesian Swing Decision Engine (DVA)**  
*Multi-Season Empirical Validation across 6 MLB Seasons (2020–2025)*  
*Dataset: 3.8+ Million Statcast Pitches | 3,089 Qualified Batter-Seasons | 1,831 Consecutive Season Pairs*

---

## 🏆 Key Statistical Takeaways for Conference Slides

### 1. True Talent Stickiness: DVA is More Stable Than Chase Rate
Across **$N = 1,831$ consecutive batter seasons (2020 $\to$ 2025)**, DVA demonstrates elite Year-over-Year autocorrelation, surpassing traditional plate discipline metrics and vastly outperforming surface hitting statistics:

| Metric | Sample ($N$) | Pearson $r$ | $R^2$ | Spearman $\rho$ | $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DVA Swing Rating** | **1,831** | **0.7775** | **0.6045** | **0.7577** | **$p < 10^{-300}$** |
| **Micro Execution DVA (`actual_dva`)** | 1,831 | 0.7613 | 0.5796 | 0.7464 | $p < 10^{-300}$ |
| **Macro Tendency DVA (`tendency_dva`)** | 1,831 | 0.7560 | 0.5715 | 0.7310 | $p < 10^{-300}$ |
| **Zone Swing Rate (Z-Swing%)** | 1,831 | 0.7538 | 0.5683 | 0.7379 | $p < 10^{-300}$ |
| **Traditional Chase Rate (O-Swing%)** | 1,831 | 0.7097 | 0.5036 | 0.7093 | $p = 1.7 \times 10^{-280}$ |
| **Offensive wOBA** | 1,831 | 0.3767 | 0.1419 | 0.3632 | $p = 7.9 \times 10^{-63}$ |

> **Presentation Punchline:** *DVA explains **60.5% of next-season decision variance ($R^2 = 0.605$)**, compared to just 50.4% for traditional Chase Rate and 14.2% for wOBA. Evaluating swing decisions at the biological decision point under flight uncertainty captures a sticky, intrinsic human perceptual skill rather than random noise.*

---

### 2. Predictive Validity: Forecasting Future wOBA ($N \to N+1$)
Does decision quality today forecast who hits tomorrow? We tested whether adding Year $N$ DVA to a baseline regression of $(\text{wOBA}_N + \text{Chase Rate}_N)$ significantly improves out-of-sample prediction of $\text{wOBA}_{N+1}$:

| Forecast Target | Baseline $R^2$ | Extended (+ DVA) $R^2$ | $\Delta R^2$ | % Gain | $F$-Statistic | $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Next-Year wOBA ($\text{wOBA}_{N+1}$)** | **0.1442** | **0.1504** | **+0.0062** | **+4.3%** | **13.42** | **$p = 0.000256$ ($p < 0.001$)** |
| **Next-Year Walk Rate ($\text{BB\%}_{N+1}$)** | 0.0435 | 0.0453 | +0.0017 | +4.0% | 3.33 | $p = 0.068$ |
| **Next-Year Strikeout Rate ($\text{K\%}_{N+1}$)** | 0.0581 | 0.0582 | +0.0001 | +0.2% | 0.22 | $p = 0.641$ |

> **Presentation Punchline:** *Adding DVA produces a **statistically significant incremental gain in forecasting future offensive wOBA ($F = 13.42, p < 0.001$)**. Because traditional chase rate penalizes hitters for hindsight umpire calls on late-breaking pitches, DVA isolates the pure decision quality that translates into sustainable offensive value.*

---

### 3. Sample Size Stabilization Rate
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

1. **[`yoy_reliability.png`](../../assets/presentation/yoy_reliability.png)**: 4-panel regression comparison (DVA vs. Tendency vs. Chase Rate vs. wOBA).
2. **[`predictive_power.png`](../../assets/presentation/predictive_power.png)**: Grouped bar chart of out-of-sample $R^2$ improvement predicting Year $N+1$ performance.
3. **[`stabilization_curve.png`](../../assets/presentation/stabilization_curve.png)**: Split-half reliability curve across pitch counts ($N=50$ to $N=1000$).
4. **[`quadrant_archetypes.png`](../../assets/presentation/quadrant_archetypes.png)**: 2D scatter plot of Macro Tendency vs. Micro Execution with player callouts.
5. **[`breakout_candidates.png`](../../assets/presentation/breakout_candidates.png)**: $\Delta\text{DVA}$ vs. $\Delta\text{wOBA}$ breakout trajectory plot.
6. **[`pitcher_deception.png`](../../assets/presentation/pitcher_deception.png)**: Graphic table of the top deceptive pitchers inducing batter errors.
