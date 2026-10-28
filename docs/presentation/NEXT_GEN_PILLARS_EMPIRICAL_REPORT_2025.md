# Next-Generation Research Extensions: Empirical Findings & Architectural Report (2025)

**Pillars 1–4: Advanced Sensorimotor Perception, Stochastic Control, and Individualized Decision Utility**  
*Execution Environment: Python 3.12, PyTorch 2.x, XGBoost, Scikit-Learn, SciPy, SQLite (`data/mlb.db`)*  
*Dataset: 2025 MLB Regular Season | 724,002 Statcast Pitches | 421 Qualified Batters | 640 Evaluated Pitchers*  
*Evaluation Protocol: Strict 5-Fold Batter-Grouped Out-of-Fold (OOF) Provenance (`GroupKFold(n_splits=5, groups=batter)`)*

---

> [!IMPORTANT]
> **Architectural & Methodological Integrity Invariant:**  
> All downstream decision models, psychometric response curves, spatial contact surfaces, and Decision Value Added (DVA) decompositions in this report maintain **strict 5-fold Batter-Grouped Out-of-Fold (OOF) provenance**. Zero pitch-level feature, spatial contact quality, or player random effect is evaluated on training-fold data. Across all 5 folds, the batter overlap between training and testing splits is identically zero ($N_{\text{overlap}} = 0$).

---

## 1. Executive Summary & System Architecture

The standard strike zone perception framework models batter swing decisions against league-average run expectancy transitions and static Cartesian trajectories. The Next-Generation Research Extensions specification expands this framework into four neurobiologically and physically grounded pillars:

1. **Pillar 4 (Retinal Optics & Psychophysics):** Replaces Cartesian $(x, y, z, v_x, v_y, v_z, a_x, a_y, a_z)$ inputs with spherical retinal optic flow $(\theta, \phi, \psi, \dot{\theta}, \dot{\phi}, \dot{\psi}/\psi)$ and Weber-Fechner sensory uncertainty, proving zero leakage of constant Magnus acceleration into initial perception.
2. **Pillar 1 (Cognitive & Sequential Bayesian Arsenal Fusion):** Models pre-release repertoire priors $\pi_k^{(0)}$ via Dirichlet-Multinomial Bayesian shrinkage and evaluates information-theoretic tunneling deception via closed-form Kullback-Leibler divergence surprise $\mathcal{S}_{\text{tunnel}} = D_{\text{KL}}(P_{0.40} \parallel P_{0.20})$.
3. **Pillar 2 (Continuous-Time Stochastic Control & Drift-Diffusion):** Formulates decision commitment via a Drift-Diffusion Model (DDM) with dynamic collapsing boundaries, biomechanically anchored to individual Statcast bat-tracking execution windows ($T_{\text{swing}, i} \approx 2 \cdot \text{swing length}_i / (\text{bat speed}_i \cdot 1.46667)$).
4. **Pillar 3 (Individualized Damage Surfaces & Minimax Equilibrium):** Disentangles **Universal Zone Recognition DVA** from **Idiosyncratic Barrel Damage DVA** via an exact additive orthogonal factorization ($|\text{Residual}| < 10^{-16}$) and solves game-theoretic Minimax pitch optimization against individualized damage profiles.

```mermaid
flowchart TD
    Raw["Raw 2025 Statcast & Hawkeye Data\n(trajectories_2025.pt & statcast table)"] --> P4["Pillar 4: Retinal Optics\n• Spherical Coordinates (θ, φ, ψ)\n• Looming Rate (ψ̇/ψ) & Angular Velocity\n• Weber-Fechner Sensory Noise Injection\n• Excludes Constant (ax, ay, az)"]
    Raw --> P1["Pillar 1: Bayesian Arsenal Fusion\n• Dirichlet-Multinomial Repertoire Prior π_k^(0)\n• Conjugate Gaussian Precision Addition\n• KL Divergence Tunnel Surprise S_tunnel\n• Tunnel Amplification Factor (TAF)"]
    P4 --> Filter["Streaming Trajectory Perception Engine\nMulti-Horizon Gaussian Beliefs (μ_τ, Σ_τ)"]
    P1 --> Filter
    Filter --> P2["Pillar 2: Stochastic Control & DDM\n• Biomechanical Deadlines (bat_speed, swing_length)\n• Dynamic Collapsing Boundaries B_i(t)\n• First-Passage Integration & Policy Fitting"]
    Filter --> P3["Pillar 3: Individualized Damage Surfaces\n• 2D Spatial xwOBAcon Surface x̂wOBAcon_i\n• Universal DVA vs. Damage Alignment DVA\n• Game-Theoretic Minimax Pitch Selection"]
    P2 --> OOF["Strict 5-Fold Batter-Grouped OOF Engine\n(GroupKFold: Zero Batter Overlap across Folds)"]
    P3 --> OOF
```

---

## 2. Pillar 4: Physics-Informed Retinal Optics & Psychophysics

### 2.1 The Constant Acceleration Leakage Problem
Statcast 9-parameter kinematic fits $(x, y, z, v_x, v_y, v_z, a_x, a_y, a_z)$ fit full-flight parabolas under an assumption of constant acceleration. Supplying $(a_x, a_y, a_z)$ at ball release ($t = 0$) allows neural networks to "cheat" by reading the full flight's Magnus movement before the ball has traveled 5 feet.

### 2.2 Mathematical Formulation
We project the 3D Cartesian coordinates to the spherical coordinate system of the batter's eye $\mathbf{p}_{\text{eye}} = (x_{\text{stand}}, y_{\text{plate}} + 1.417, z_{\text{eye}})$:
- $x_{\text{stand}} = -1.75\text{ ft}$ (RHB in left box) or $+1.75\text{ ft}$ (LHB in right box)
- $y_{\text{eye}} = 1.417\text{ ft}$ (distance behind front point of plate)
- $z_{\text{eye}} = \text{sz top} + 0.65\text{ ft}$

The relative displacement vector and distance are:
$$\Delta \mathbf{r}(t) = \mathbf{p}(t) - \mathbf{p}_{\text{eye}}, \quad d(t) = \|\Delta \mathbf{r}(t)\|_2$$

The 9D Retinal Optic Flow channels are defined as:
1. **Azimuth Angle:** $\theta(t) = \arctan2(\Delta x(t), \Delta y(t))$
2. **Elevation Angle:** $\phi(t) = \arctan2\left(\Delta z(t), \sqrt{\Delta x(t)^2 + \Delta y(t)^2}\right)$
3. **Subtended Visual Angle:** $\psi(t) \approx \frac{2 R_{\text{ball}}}{\max(d(t), 1.0)}$, where $R_{\text{ball}} = 0.1208\text{ ft}$
4. **Angular Azimuth Velocity:** $\dot{\theta}(t) = \frac{v_x(t) \Delta y(t) - v_y(t) \Delta x(t)}{\Delta x(t)^2 + \Delta y(t)^2}$
5. **Angular Elevation Velocity:** $\dot{\phi}(t) = \frac{v_z(t)(\Delta x(t)^2 + \Delta y(t)^2) - \Delta z(t)(\Delta x(t) v_x(t) + \Delta y(t) v_y(t))}{d(t)^2 \sqrt{\Delta x(t)^2 + \Delta y(t)^2}}$
6. **Retinal Looming Rate (Tau-Dot):** $\frac{\dot{\psi}(t)}{\psi(t)} = -\frac{\Delta \mathbf{r}(t) \cdot \mathbf{v}(t)}{d(t)^2}$
7. **Weber Azimuth Noise:** $\sigma_{\dot{\theta}}(t) = \sqrt{\sigma_{0, \text{ang}}^2 + w_{\text{weber}}^2 \dot{\theta}(t)^2}$
8. **Weber Elevation Noise:** $\sigma_{\dot{\phi}}(t) = \sqrt{\sigma_{0, \text{ang}}^2 + w_{\text{weber}}^2 \dot{\phi}(t)^2}$
9. **Weber Looming Noise:** $\sigma_{\text{loom}}(t) = \sqrt{\sigma_{0, \text{loom}}^2 + w_{\text{loom}}^2 \left(\frac{\dot{\psi}(t)}{\psi(t)}\right)^2}$

### 2.3 Proof of Zero Acceleration Leakage
Because the angular projections and instantaneous rates depend strictly on displacement $\Delta \mathbf{r}(0)$ and velocity $\mathbf{v}(0)$, modifying $(a_x, a_y, a_z)$ at $t = 0$ produces an identical initial retinal state:
$$\left\|\mathbf{R}_{\text{retinal}}(t=0 \mid a) - \mathbf{R}_{\text{retinal}}(t=0 \mid a')\right\|_\infty \equiv 0.000000$$
Verified deterministically in `tests/test_next_gen_pillars.py::test_retinal_projection_zero_leakage`.

---

## 3. Pillar 1: Cognitive & Sequential Bayesian Arsenal Fusion

### 3.1 Repertoire Prior Formulation
Prior to release, batters form expectation weights across pitch types $k \in \{1, \dots, K\}$ via Dirichlet-Multinomial Bayesian shrinkage against league count usage:
$$\alpha_k = \alpha_0 \cdot \bar{\pi}_k(b, s) + N_{p, b, s, k}, \quad \pi_k^{(0)} = \frac{\alpha_k}{\sum_{j=1}^K \alpha_j}$$

### 3.2 Precision Addition & Closed-Form Tunneling Surprise
During flight, conjugate Gaussian updates combine prior expectations with noisy sensory measurements:
$$\bm{\Sigma}_{\text{post}}(\tau) = \left[(\bm{\Sigma}_{\text{prior}})^{-1} + (\bm{\Sigma}_{\text{vis}}(\tau))^{-1}\right]^{-1}$$
$$\det \bm{\Sigma}_{\text{post}} \le \min\left(\det \bm{\Sigma}_{\text{prior}}, \det \bm{\Sigma}_{\text{vis}}\right)$$

Information-theoretic tunneling surprise is evaluated in closed form between the recognition horizon ($\tau = 0.20$) and commitment horizon ($\tau = 0.40$):
$$\mathcal{S}_{\text{tunnel}} = D_{\text{KL}}\Big(p(\mathbf{x}_{\text{plate}} \mid \mathcal{F}_{0.40}) \;\Big\|\; p(\mathbf{x}_{\text{plate}} \mid \mathcal{F}_{0.20})\Big)$$
$$D_{\text{KL}}(P_{0.40} \parallel P_{0.20}) = \frac{1}{2}\left[\operatorname{tr}(\bm{\Sigma}_{0.20}^{-1}\bm{\Sigma}_{0.40}) + (\bm{\mu}_{0.20} - \bm{\mu}_{0.40})^\top \bm{\Sigma}_{0.20}^{-1}(\bm{\mu}_{0.20} - \bm{\mu}_{0.40}) - 2 + \ln\left(\frac{\det\bm{\Sigma}_{0.20}}{\det\bm{\Sigma}_{0.40}}\right)\right]$$

### 3.3 Inter-Pitch Tunnel Amplification Factor (TAF)
For consecutive pitches within the same plate appearance:
$$\text{TAF}_n = \frac{\|\mathbf{x}_{\text{plate}, n} - \mathbf{x}_{\text{plate}, n-1}\|_2}{\max(\|\mathbf{r}_n(t_{0.40}) - \mathbf{r}_{n-1}(t_{0.40})\|_2, 0.05)}$$

### 3.4 2025 MLB Tunneling Deception Leaderboard
Evaluated on all 724,002 pitches ($N \ge 200$ pitches, 640 pitchers):

| Rank | Pitcher Name | Pitches Evaluated | Mean KL Surprise $\mathbb{E}[\mathcal{S}_{\text{tunnel}}]$ | Max KL Surprise | Mean TAF | Median TAF | Induced Swing% |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Devin Williams** | 1,194 | **0.7483** | 0.8241 | **1.5819** | 1.3410 | 48.2% |
| **2** | **Alek Jacob** | 586 | **0.7438** | 0.8198 | 1.2933 | 1.1120 | 47.6% |
| **3** | **AJ Blubaugh** | 507 | **0.7397** | 0.8012 | 1.3055 | 1.1450 | 46.1% |
| **4** | **Logan Henderson** | 410 | **0.7371** | 0.7984 | 1.3498 | 1.2014 | 45.9% |
| **5** | **Jacob Webb** | 978 | **0.7366** | 0.8115 | 1.2628 | 1.0980 | 47.9% |
| **6** | **Justin Bruihl** | 231 | **0.7364** | 0.7942 | 0.8739 | 0.7510 | 43.3% |
| **7** | **Jose A. Ferrer** | 1,208 | **0.7359** | 0.8087 | 1.1717 | 1.0250 | 50.1% |
| **8** | **Robert Suarez** | 1,125 | **0.7349** | 0.8041 | 1.1580 | 1.0410 | 51.4% |
| **9** | **Shelby Miller** | 704 | **0.7342** | 0.8016 | 1.4012 | 1.2180 | 46.8% |
| **10** | **Hogan Harris** | 1,125 | **0.7340** | 0.8093 | 1.3996 | 1.1940 | 48.0% |

> **Key Empirical Finding:** Devin Williams' "Airbender" changeup paired with his high four-seamer generates the highest information-theoretic surprise in baseball ($\mathbb{E}[\mathcal{S}_{\text{tunnel}}] = 0.748$, TAF = 1.58). Hitters commit to beliefs formed at 175 ms ($\tau = 0.20$) that diverge by over 1.5 feet at plate crossing.

---

## 4. Pillar 2: Continuous-Time Optimal Stopping & Drift-Diffusion Control

### 4.1 Biomechanical Motor Deadlines
Rather than assuming an arbitrary static decision horizon, each batter's motor commitment deadline is anchored to verified Statcast bat tracking metrics (`bat_speed` and `swing_length`):
$$T_{\text{swing}, i} \approx \frac{2 \cdot \text{swing length}_i}{\text{bat speed}_i \cdot 1.46667}$$
$$T_{\text{deadline}, ij} = T_{\text{plate}, j} - T_{\text{swing}, i}$$

### 4.2 Dynamic Collapsing Boundary Formulation
Evidence accumulates via the stochastic differential equation:
$$dE_i(t) = \nu_i \cdot u(t)\,dt + \sigma_{\text{sensory}}\,dW_t$$
where momentary drift $u(t) = \frac{P(\text{Strike} \mid \mathcal{F}_t) - 0.50}{\sqrt{\det \bm{\Sigma}_t + \epsilon}}$. The decision boundaries collapse monotonically toward zero:
$$B_i(t) = \pm B_{0, i} \left(1 - \kappa_i \frac{t}{T_{\text{deadline}, ij}}\right)_+^{\gamma_i}$$
At $t = T_{\text{deadline}}$, $B_i(T_{\text{deadline}}) = 0.0$ when $\kappa = 1.0$, forcing a terminal decision before the physical point of no return.

### 4.3 2025 MLB DDM Batter Policy Empirical Analysis
Fitted across 421 qualified hitters ($N \ge 500$ pitches):

| Player Name | Pitches | Drift Rate $\nu_i$ | Boundary $B_{0, i}$ | Urgency $\kappa_i$ | Bat Speed | Swing Length | $T_{\text{swing}}$ | Motor Deadline | Policy Archetype |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Giancarlo Stanton** | 1,330 | 4.935 | 0.263 | 1.000 | **79.0 mph** | 8.72 ft | 150.6 ms | **269.4 ms** | High Urgency Collapse |
| **Aaron Judge** | 2,736 | 4.915 | 0.386 | 1.000 | **74.8 mph** | 8.04 ft | 146.6 ms | **273.4 ms** | High Urgency Collapse |
| **Shohei Ohtani** | 3,169 | 4.920 | 0.331 | 1.000 | **74.1 mph** | 7.90 ft | 145.3 ms | **274.7 ms** | High Urgency Collapse |
| **Kyle Stowers** | 1,712 | 4.912 | 0.386 | 0.999 | 73.0 mph | **6.64 ft** | **123.9 ms** | **296.1 ms** | High Urgency Collapse |
| **Juan Soto** | 2,968 | 4.912 | 0.384 | 1.000 | 71.0 mph | 6.97 ft | 133.8 ms | **286.2 ms** | High Urgency Collapse |
| **Kyle Tucker** | 2,439 | 4.896 | **0.484** | 1.000 | 69.6 mph | 7.43 ft | 145.6 ms | 274.4 ms | High Urgency Collapse |
| **Mookie Betts** | 2,780 | 4.916 | 0.337 | 1.000 | 66.7 mph | 6.95 ft | 142.2 ms | 277.8 ms | High Urgency Collapse |
| **Steven Kwan** | 2,829 | 4.918 | 0.312 | 1.000 | **61.5 mph** | **6.17 ft** | 136.8 ms | **283.2 ms** | High Urgency Collapse |

> **Biomechanical Takeaway:** Kyle Stowers' ultra-compact 6.64-ft swing affords him a **123.9 ms swing duration**, granting him a motor commitment deadline of **296.1 ms**—nearly 27 ms more visual tracking time than Giancarlo Stanton (269.4 ms), whose massive 8.72-ft swing length requires an earlier commitment despite elite 79.0 mph bat speed.

---

## 5. Pillar 3: Individualized Damage Surfaces & Minimax Equilibrium

### 5.1 Decoupling Zone Recognition from Barrel Damage
Universal DVA rewards hitters for taking pitches outside the rulebook zone and swinging inside. However, elite sluggers generate substantial positive expected value swinging at personal "honey-holes" on the zone perimeter, whereas contact hitters destroy value swinging at borderline pitches where they cannot generate exit velocity.

### 5.2 Additive Orthogonal Decomposition
We decouple every pitch decision into:
$$\text{Total Indiv EV-DVA}_{ij} = \text{Universal EV-DVA}_{ij} + \text{Damage Alignment DVA}_{ij}$$
where:
$$\text{Universal EV-DVA}_{ij} = (\text{Swing}_{ij} - \hat{p}_{\text{oof}, ij}) \cdot \left[\text{EV}^{\text{league}}(\text{Swing}_{ij}) - \text{EV}(\text{Take}_{ij})\right]$$
$$\text{Damage Alignment DVA}_{ij} = (\text{Swing}_{ij} - \hat{p}_{\text{oof}, ij}) \cdot \left[\text{EV}_i^{\text{indiv}}(\text{Swing}_{ij}) - \text{EV}^{\text{league}}(\text{Swing}_{ij})\right]$$
$$\text{Total Indiv EV-DVA}_{ij} = (\text{Swing}_{ij} - \hat{p}_{\text{oof}, ij}) \cdot \left[\text{EV}_i^{\text{indiv}}(\text{Swing}_{ij}) - \text{EV}(\text{Take}_{ij})\right]$$

Across all 724,002 pitches in 2025, the additive identity holds with exact numerical precision:
$$\max_{j} \left|\text{Total Indiv DVA}_j - (\text{Universal DVA}_j + \text{Damage Alignment DVA}_j)\right| = 2.78 \times 10^{-17} < 10^{-6}$$

### 5.3 2025 MLB Personalized DVA Leaderboard
Evaluated with strict 5-fold Batter-Grouped OOF cross-validation ($N \ge 500$ pitches):

| Rank | Player Name | Pitches | Total Indiv DVA | Universal DVA (Zone Eye) | Damage Alignment DVA (Barrel Value) | Swing Rating | Decision Archetype |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **Freddie Freeman** | 2,703 | **+14.69** | +13.02 | **+1.67** | **123.4** | Balanced Zone Operator |
| **2** | **Kyle Tucker** | 2,439 | **+14.45** | **+14.05** | +0.40 | **125.6** | Balanced Zone Operator |
| **3** | **Fernando Tatís Jr.** | 2,677 | **+14.44** | +13.10 | **+1.34** | **122.9** | Balanced Zone Operator |
| **4** | **Brent Rooker** | 2,575 | **+13.57** | +10.98 | **+2.59** | **122.7** | Balanced Zone Operator |
| **5** | **George Springer** | 2,573 | **+11.96** | +12.08 | -0.12 | 119.5 | Balanced Zone Operator |
| **6** | **Bryan Reynolds** | 2,631 | **+11.75** | +11.15 | +0.60 | 118.9 | Balanced Zone Operator |
| **7** | **Aaron Judge** | 2,736 | **+11.53** | +10.00 | **+1.53** | 117.8 | Balanced Zone Operator |
| **8** | **Corey Seager** | 1,668 | **+11.03** | +9.67 | **+1.37** | 119.8 | Balanced Zone Operator |
| **9** | **Christian Walker** | 2,513 | **+10.84** | +10.08 | +0.76 | 117.6 | Balanced Zone Operator |
| **10** | **Marcus Semien** | 2,075 | **+10.72** | +11.19 | -0.47 | 118.8 | Balanced Zone Operator |

### 5.4 Spotlight: Passivity vs. Power Tradeoffs (Steven Kwan vs. Slugging Hitters)

| Batter Name | Total Indiv DVA | Universal DVA (Recognition) | Damage Alignment DVA | Swing% | Key Driver |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Brent Rooker** | **+13.57** | +10.98 | **+2.59** | 46.8% | Dual threat: elite zone control + massive barrel damage |
| **Freddie Freeman** | **+14.69** | +13.02 | **+1.67** | 43.1% | Master of low-whiff middle-in barrel optimization |
| **Kyle Stowers** | **+7.24** | **+7.92** | -0.68 | 49.2% | High passivity in non-damage counts |
| **Steven Kwan** | **-3.83** | **-4.33** | **+0.50** | 38.6% | Extreme contact specialist; minimal barrel damage ceiling |
| **Shohei Ohtani** | **-3.16** | -1.69 | -1.47 | 49.5% | Hyper-aggressive in chase regions penalized by baseline |
| **Giancarlo Stanton** | **-5.24** | -4.22 | -1.02 | 44.8% | Vulnerable to low-away sliders due to long swing commitment |
| **Juan Soto** | **-8.72** | -0.81 | -7.91 | 35.5% | Penalized in 2025 by high take rates in personal barrel zones |

---

## 6. Verification Suite & Unit Tests Summary

All four pillars and strict OOF constraints are verified by 20 deterministic unit tests in `tests/test_next_gen_pillars.py`:

| Test Name | Pillar | Invariant Verified | Status |
| :--- | :---: | :--- | :---: |
| `test_retinal_projection_zero_leakage` | Pillar 4 | $\|\mathbf{R}(t=0 \mid a) - \mathbf{R}(t=0 \mid a')\|_\infty = 0$ | **PASSED** |
| `test_retinal_projection_shape_and_range` | Pillar 4 | Output shape $(B, T, 9)$, valid spherical angles $\theta, \phi, \psi$ | **PASSED** |
| `test_retinal_handedness_eye_offset` | Pillar 4 | Eye positions: RHB $x = -1.75\text{ ft}$, LHB $x = +1.75\text{ ft}$ | **PASSED** |
| `test_retinal_looming_rate_expansion` | Pillar 4 | Looming rate $\dot{\psi}/\psi > 0$ for approaching trajectory | **PASSED** |
| `test_weber_fechner_noise_scaling` | Pillar 4 | Noise $\sigma \ge \sigma_0$, scales monotonically with optical velocity | **PASSED** |
| `test_weber_noise_injection_stochasticity` | Pillar 4 | Injects noise into velocity channels without corrupting angles | **PASSED** |
| `test_bayesian_precision_addition` | Pillar 1 | $\det \bm{\Sigma}_{\text{post}} \le \min(\det \bm{\Sigma}_{\text{prior}}, \det \bm{\Sigma}_{\text{vis}})$ | **PASSED** |
| `test_bayesian_conjugate_mean_update` | Pillar 1 | Precision-weighted mean combination lies in convex hull | **PASSED** |
| `test_dirichlet_multinomial_shrinkage` | Pillar 1 | Shrinks small samples to league baseline, simplex constraint $\sum \pi = 1$ | **PASSED** |
| `test_gaussian_kl_divergence_nonnegativity_and_identity` | Pillar 1 | $D_{\text{KL}}(P \parallel P) = 0$, $D_{\text{KL}}(P_1 \parallel P_2) \ge 0$ | **PASSED** |
| `test_tunnel_amplification_factor` | Pillar 1 | Tight early tunnel with late divergence yields $\text{TAF} \gg 1$ | **PASSED** |
| `test_sequential_arsenal_filter_dimensions` | Pillar 1 | Neural filter outputs valid simplex priors and positive variances | **PASSED** |
| `test_ddm_boundary_collapse` | Pillar 2 | $B(0) = B_0$, monotonically non-increasing, $B(T_{\text{deadline}}) \equiv 0$ | **PASSED** |
| `test_ddm_euler_maruyama_first_passage` | Pillar 2 | Generates binary decision with commit time $t \in [0, T_{\text{deadline}}]$ | **PASSED** |
| `test_ddm_drift_monotonicity` | Pillar 2 | Strike probability increases monotonically with drift $\nu \cdot u$ | **PASSED** |
| `test_bat_tracking_execution_timing` | Pillar 2 | $T_{\text{swing}}$ scales inversely with bat speed, in realistic range $[100, 250]\text{ ms}$ | **PASSED** |
| `test_dva_additive_identity` | Pillar 3 | $|\text{Total Indiv DVA} - (\text{Universal DVA} + \text{Damage DVA})| < 10^{-6}$ | **PASSED** |
| `test_strict_oof_partition` | Architecture | GroupKFold: 0 overlapping batters across all training and test folds | **PASSED** |
| `test_game_theoretic_minimax_pitch` | Pillar 3 | Minimax optimizer selects pitch type and location minimizing batter RV | **PASSED** |
| `test_spatial_contact_utility_shrinkage` | Pillar 3 | Spatial xwOBAcon and probabilities respect physical bounds | **PASSED** |

### Complete Test Suite Execution Result:
```text
============================= 101 passed in 5.85s =============================
- Existing Regression Tests: 81 passed (tests/test_audit_cycle1-3, test_scoring, test_commit_horizon, etc.)
- Next-Gen Pillars Suite: 20 passed (tests/test_next_gen_pillars.py)
- Zero warnings, zero regressions.
```

---

## 7. Deliverables & Implementation Artifacts

All code, pipeline runners, unit tests, and persistent out-of-sample data artifacts have been committed to the repository:

1. **Modules (`src/`):**
   - `src/retinal_physics.py`: Physics-informed coordinate projections, looming rates, and Weber-Fechner noise injection.
   - `src/bayesian_tunneling.py`: Dirichlet-Multinomial priors, precision addition, closed-form Gaussian KL divergence, and neural filter.
   - `src/drift_diffusion.py`: Continuous-time Collapsing Boundary DDM anchored to Statcast bat tracking.
   - `src/personalized_utility.py`: 2D spatial contact utility surfaces, additive orthogonal DVA decomposition, and minimax pitch solver.
2. **Pipelines & Runners (`scripts/`):**
   - `scripts/train_retinal_model.py`: Retinal perception training and uncertainty calibration pipeline.
   - `scripts/run_bayesian_tunneling.py`: Vectorized KL surprise and TAF leaderboard generator.
   - `scripts/run_commit_horizon.py --method ddm`: Biomechanical DDM policy estimation runner.
   - `scripts/run_personalized_dva.py`: Strict 5-fold OOF personalized DVA engine.
3. **Unit Tests (`tests/`):**
   - `tests/test_next_gen_pillars.py`: 20 deterministic mathematical and architectural verification tests.
4. **Data Artifacts (`data/`):**
   - `data/processed/oof_next_gen_predictions_2025.parquet`: 724,002 out-of-sample pitch predictions with decomposed DVA.
   - `data/leaderboards/2025_tunnel_surprise_leaderboard.csv`: 640 pitchers ranked by induced tunneling deception.
   - `data/leaderboards/2025_ddm_batter_policy.csv`: 421 batters parameterized by drift rate, conservatism, and collapse rate.
   - `data/leaderboards/2025_personalized_dva_leaderboard.csv`: 421 batters with Universal vs. Damage Alignment DVA factorization.
