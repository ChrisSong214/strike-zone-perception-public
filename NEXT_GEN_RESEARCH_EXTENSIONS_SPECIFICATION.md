# NEXT-GENERATION RESEARCH SPECIFICATION: PILLARS 1–4
## Advanced Sensorimotor Perception, Stochastic Control, and Individualized Decision Utility

**Repository Target:** `sp26_sagb_dj` / `strike-zone-perception`  
**Execution Environment:** Python 3.10+, PyTorch 2.x, XGBoost, Scikit-Learn, SciPy, SQLite (`data/mlb.db`), Parquet (`data/processed/`)  
**Strict Architectural Invariant:** All downstream decision models, psychometric curves, contact surfaces, and DVA evaluations must maintain **strict 5-fold Batter-Grouped Out-of-Fold (OOF) provenance** (`GroupKFold(n_splits=5, groups=batter_id)`). Zero pitch-level feature or player random effect may be evaluated on training-fold data.

---

## 0. SYSTEM ARCHITECTURE & DATA DEPENDENCY GRAPH

```
                  ┌───────────────────────────────────────────────┐
                  │          Raw Statcast / Hawkeye Data          │
                  │ (trajectories_2025.pt & statcast table in db) │
                  └──────────────────────┬────────────────────────┘
                                         │
        ┌────────────────────────────────┴────────────────────────────────┐
        ▼                                                                 ▼
┌─────────────────────────────────┐                     ┌──────────────────────────────────┐
│   PILLAR 4: RETINAL OPTICS      │                     │   PILLAR 1: BAYESIAN ARSENAL     │
│ • Coordinate translation        │                     │ • Pitch-sequence tunnel vector   │
│ • Retinal optic flow (θ, φ, ψ̇/ψ)│                     │ • Dirichlet-Multinomial prior    │
│ • Weber-Fechner noise injection │                     │ • Sequential Kalman / EKF update │
└───────────────┬─────────────────┘                     └────────────────┬─────────────────┘
                │                                                        │
                └────────────────────────┬───────────────────────────────┘
                                         │
                                         ▼
                 ┌───────────────────────────────────────────────┐
                 │       Streaming Trajectory Filter/LSTM        │
                 │   Updated Posteriors: μ(t), Σ(t), KL-Surprise │
                 └───────────────────────┬───────────────────────┘
                                         │
        ┌────────────────────────────────┴────────────────────────────────┐
        ▼                                                                 ▼
┌─────────────────────────────────┐                     ┌──────────────────────────────────┐
│   PILLAR 2: STOCHASTIC CONTROL  │                     │   PILLAR 3: INDIVIDUAL UTILITY   │
│ • Drift-Diffusion Model (DDM)   │                     │ • 2D Spatial xwOBAcon surface    │
│ • Bat-tracking execution time   │                     │ • Universal vs Damage DVA split  │
│ • Dynamic collapsing boundaries │                     │ • Game-theoretic Minimax pitch   │
└───────────────┬─────────────────┘                     └────────────────┬─────────────────┘
                │                                                        │
                └────────────────────────┬───────────────────────────────┘
                                         │
                                         ▼
                 ┌───────────────────────────────────────────────┐
                 │     STRICT 5-FOLD BATTER-GROUPED OOF ENGINE   │
                 │ • Unbiased Out-of-Sample Evaluation           │
                 │ • Year-over-Year Reliability & Validation     │
                 └───────────────────────────────────────────────┘
```

---

## PILLAR 1: Cognitive & Sequential Bayesian Arsenal Fusion

### 1.1 Objective & Scientific Motivation
Batters do not perceive pitches in isolation. Prior to ball release ($t = 0$), a batter holds an arsenal prior conditioned on count state $(b, s)$, pitcher repertoire, and pitch sequence history $\mathcal{H}_{n-1}$. During flight, visual evidence updates this prior. When a pitcher tunnels a slider to mirror a fastball's early path, the batter experiences belief revision across discrete hypothesis classes rather than simple continuous spatial dispersion.

### 1.2 Mathematical Formulation
1. **Pre-Release Repertoire Prior ($t = 0$):**
   For pitcher $p$ in count $(b, s)$, the prior probability over pitch type $k \in \{1, \dots, K\}$ is:
   $$\pi_k^{(0)} = P(\text{PitchType} = k \mid p, b, s, \mathcal{H}_{n-1})$$
   Estimated via Dirichlet-Multinomial Bayesian shrinkage against league count-level usage rates.

2. **Sequential Multi-Hypothesis Filtering ($t \in [0.10, 0.55]$):**
   At each timestep $\tau$, the model computes posterior mixture weights $\pi_k(\tau)$ and component distributions $(\bm{\mu}_k(\tau), \bm{\Sigma}_k(\tau))$:
   $$p(\mathbf{x}_{\text{plate}} \mid \mathcal{F}_\tau) = \sum_{k=1}^K \pi_k(\tau) \, \mathcal{N}\big(\mathbf{x}_{\text{plate}} \,;\, \bm{\mu}_k(\tau), \bm{\Sigma}_k(\tau)\big)$$
   Each Gaussian component updates analytically via conjugate precision weighting:
   $$\bm{\Sigma}_{k, \text{post}}(\tau) = \left[(\bm{\Sigma}_{k}^{(0)})^{-1} + \bm{\Sigma}_{\text{vis}}(\tau)^{-1}\right]^{-1}$$
   $$\bm{\mu}_{k, \text{post}}(\tau) = \bm{\Sigma}_{k, \text{post}}(\tau)\left[(\bm{\Sigma}_{k}^{(0)})^{-1}\bm{\mu}_{k}^{(0)} + \bm{\Sigma}_{\text{vis}}(\tau)^{-1}\bm{\mu}_{\text{vis}}(\tau)\right]$$

3. **Information-Theoretic Tunneling Deception (KL-Divergence Surprise):**
   The perceptual disruption between recognition horizon ($\tau = 0.20$) and decision commit point ($\tau = 0.40$) is measured via Kullback-Leibler divergence:
   $$\mathcal{S}_{\text{tunnel}} = D_{\text{KL}}\Big(p(\mathbf{x}_{\text{plate}} \mid \mathcal{F}_{0.40}) \;\Big\|\; p(\mathbf{x}_{\text{plate}} \mid \mathcal{F}_{0.20})\Big)$$
   For Gaussian distributions, this is evaluated in closed form:
   $$D_{\text{KL}}(P_{\text{late}} \parallel P_{\text{early}}) = \frac{1}{2}\left[\operatorname{tr}(\bm{\Sigma}_{\text{early}}^{-1}\bm{\Sigma}_{\text{late}}) + (\bm{\mu}_{\text{early}} - \bm{\mu}_{\text{late}})^\top \bm{\Sigma}_{\text{early}}^{-1}(\bm{\mu}_{\text{early}} - \bm{\mu}_{\text{late}}) - 2 + \ln\left(\frac{\det\bm{\Sigma}_{\text{early}}}{\det\bm{\Sigma}_{\text{late}}}\right)\right]$$

4. **Inter-Pitch Tunneling Geometry:**
   For consecutive pitches within the same plate appearance (`game_pk`, `at_bat_number`):
   - Spatial divergence at decision point: $\Delta \mathbf{r}_{0.40} = \|\mathbf{r}_n(t_{0.40}) - \mathbf{r}_{n-1}(t_{0.40})\|_2$
   - Tunnel Amplification Factor: $\text{TAF}_n = \frac{\|\mathbf{x}_{\text{plate}, n} - \mathbf{x}_{\text{plate}, n-1}\|_2}{\max(\|\mathbf{r}_n(t_{0.40}) - \mathbf{r}_{n-1}(t_{0.40})\|_2, 0.05)}$

### 1.3 Implementation Blueprint
* **Module:** `src/bayesian_tunneling.py`
* **Class:** `SequentialArsenalFilter(nn.Module)`
  ```python
  import torch
  import torch.nn as nn

  class SequentialArsenalFilter(nn.Module):
      def __init__(self, n_pitch_types: int = 8, hidden_dim: int = 128):
          super().__init__()
          self.prior_net = nn.Sequential(
              nn.Linear(12 + n_pitch_types + 4, 64),
              nn.ReLU(),
              nn.Linear(64, n_pitch_types),
              nn.Softmax(dim=-1)
          )
          self.likelihood_head = nn.Linear(hidden_dim, n_pitch_types * 5)

      def forward(self, h_t: torch.Tensor, context_vector: torch.Tensor):
          pi_0 = self.prior_net(context_vector)
          params = self.likelihood_head(h_t)
          return pi_0, params
  ```
* **Pipeline Script:** `scripts/run_bayesian_tunneling.py`
  - Outputs: `data/leaderboards/{year}_tunnel_surprise_leaderboard.csv`
  - Metrics: Pitcher induced surprise $\mathbb{E}[\mathcal{S}_{\text{tunnel}}]$, batter resilience to tunneling shock.

---

## PILLAR 2: Continuous-Time Optimal Stopping & Drift-Diffusion Control

### 2.1 Objective & Scientific Motivation
Discrete multi-step horizon searches ($\tau \in \{0.25, \dots, 0.55\}$) approximate continuous decision dynamics. A continuous-time **Drift-Diffusion Model (DDM) with Collapsing Boundaries** grounds commit decisions in cognitive neurobiology, directly incorporating Statcast bat-tracking biomechanics (`bat_speed`, `swing_length`) to define individual decision deadlines.

### 2.2 Mathematical Formulation
1. **Continuous Evidence Accumulation:**
   The batter's decision state $E_i(t)$ evolves as a stochastic differential equation:
   $$dE_i(t) = \nu_i \cdot u(t)\,dt + \sigma_{\text{sensory}}\,dW_t$$
   where $W_t$ is standard Brownian motion, and momentary drift $u(t)$ is driven by the precision-weighted strike utility:
   $$u(\tau) = \frac{P(\text{Strike} \mid \mathcal{F}_\tau) - \bar{p}_{\text{thresh}}(b, s)}{\sqrt{\det \bm{\Sigma}_\tau + \epsilon}}$$

2. **Biomechanically Anchored Dynamic Boundaries:**
   Let $T_{\text{plate}}$ denote total flight time. The physical execution window $T_{\text{swing}, i}$ is derived from Statcast bat metrics:
   $$T_{\text{swing}, i} \approx \frac{2 \cdot \text{swing\_length}_i}{\text{bat\_speed}_i}$$
   The motor commitment deadline is $T_{\text{deadline}, ij} = T_{\text{plate}, j} - T_{\text{swing}, i}$.
   The decision boundaries collapse toward zero as the deadline nears:
   $$B_i(t) = \pm B_{0, i} \left(1 - \kappa_i \frac{t}{T_{\text{deadline}, ij}}\right)_+^{\gamma_i}$$
   - Crossing $+B_i(t)$: triggers a **Swing** at time $t$.
   - Crossing $-B_i(t)$ or reaching $T_{\text{deadline}}$ with $E_i(t) < 0$: triggers a **Take**.

3. **First-Passage Probability Likelihood:**
   Evaluating transition densities over trajectory checkpoints yields the structural parameter set $\bm{\theta}_i = (\nu_i, B_{0, i}, \kappa_i)$:
   - $\nu_i$: Sensory processing speed / drift rate.
   - $B_{0, i}$: Baseline conservatism / threshold separation.
   - $\kappa_i$: Urgency collapse rate under temporal pressure.

### 2.3 Implementation Blueprint
* **Module:** `src/drift_diffusion.py`
* **Class:** `CollapsingBoundDDM`
  ```python
  import numpy as np
  from scipy.optimize import minimize

  class CollapsingBoundDDM:
      def __init__(self, dt: float = 0.005):
          self.dt = dt

      def simulate_first_passage(self, v: float, b0: float, kappa: float, t_deadline: float):
          n_steps = int(t_deadline / self.dt)
          time_steps = np.linspace(0, t_deadline, n_steps)
          boundary = b0 * np.maximum(0.0, 1.0 - kappa * (time_steps / t_deadline))
          
          # Euler-Maruyama integration
          noise = np.random.normal(0, np.sqrt(self.dt), size=n_steps)
          evidence = np.cumsum(v * self.dt + noise)
          
          swing_cross = np.where(evidence >= boundary)[0]
          take_cross = np.where(evidence <= -boundary)[0]
          
          first_swing = swing_cross[0] if len(swing_cross) > 0 else np.inf
          first_take = take_cross[0] if len(take_cross) > 0 else np.inf
          
          return first_swing < first_take, min(first_swing, first_take) * self.dt
  ```
* **Pipeline Script:** `scripts/run_commit_horizon.py --method ddm`
  - Merges with `data/raw_data/statcast_bat_tracking_{year}.csv`.
  - Saves: `data/leaderboards/{year}_ddm_batter_policy.csv`.

---

## PILLAR 3: Individualized Damage Surfaces & Minimax Equilibrium

### 3.1 Objective & Scientific Motivation
Standard EV-DVA evaluates decisions against league-average run expectancy transitions. However, a borderline pitch is suboptimal for a contact specialist with a .200 $\text{xwOBAcon}$ on edge pitches, but strongly positive expected value for an elite slugger generating high exit velocities. We decouple **Universal Zone Recognition DVA** from **Idiosyncratic Barrel Damage DVA**.

### 3.2 Mathematical Formulation
1. **Batter Spatial Contact Utility Surface:**
   For each batter $i$, fit an out-of-fold regularized spatial surface over predicted coordinates $(\mu_x, \mu_z)_{0.40}$:
   $$\widehat{\text{xwOBAcon}}_i(\mu_x, \mu_z) = \text{GP}_i\big(\mu_x, \mu_z \mid \text{BIP}\big)$$
   $$\widehat{P}_i(\text{Whiff} \mid \mu_x, \mu_z, \sigma_\perp) = \text{LogisticRegression}_i\big(\mu_x, \mu_z, \sigma_\perp\big)$$

2. **Personalized Swing Expected Value:**
   $$\text{EV}_i^{\text{indiv}}(\text{Swing} \mid \mathcal{F}_{0.40}) = \iint_{\mathbb{R}^2} \Big[ P_i(\text{Whiff})\Delta\text{RE}_{\text{strike}} + P_i(\text{Foul})\Delta\text{RE}_{\text{foul}} + P_i(\text{BIP})\text{RV}\big(\widehat{\text{xwOBAcon}}_i\big) \Big] \mathcal{N}(\mathbf{x}; \bm{\mu}, \bm{\Sigma})\,d\mathbf{x}$$

3. **Additive Orthogonal Factorization of DVA:**
   $$\underbrace{\text{Total Indiv EV-DVA}_{ij}}_{\text{Overall Decision Value}} = \underbrace{\text{Universal EV-DVA}_{ij}}_{\text{Pure Visual Zone Recognition}} + \underbrace{\text{Damage Alignment DVA}_{ij}}_{\text{Swinging at Personal Barrel Zones}}$$
   where:
   $$\text{Damage Alignment DVA}_{ij} = (\text{Swing}_{ij} - \hat{p}_{\text{oof}, ij}) \cdot \left[\text{EV}_i^{\text{indiv}}(\text{Swing}_{ij}) - \text{EV}^{\text{league}}(\text{Swing}_{ij})\right]$$

4. **Game-Theoretic Minimax Pitch Optimization:**
   For pitcher $p$ against batter profile $\bm{\Psi}_i = (\beta_{\mathcal{U}, i}, \tau^*_i, \text{DamageSurface}_i)$ in count $(b, s)$:
   $$(m^*, \mathbf{x}^*) = \arg\min_{m \in \mathcal{M}_p, \mathbf{x}} \left[ \hat{p}_{\text{swing}, i}(m, \mathbf{x}) \cdot \text{EV}_i^{\text{indiv}}(\text{Swing}) + (1 - \hat{p}_{\text{swing}, i}(m, \mathbf{x})) \cdot \text{EV}(\text{Take}) \right]$$

### 3.3 Implementation Blueprint
* **Module:** `src/personalized_utility.py`
* **Class:** `IndividualContactUtilityModel`
  - Fits 5-fold OOF spatial surfaces for Whiff, Foul, and BIP xwOBAcon.
  - Decomposes every pitch decision into Universal DVA and Damage Alignment DVA.
* **Pipeline Script:** `scripts/run_personalized_dva.py`
  - Saves: `data/leaderboards/{year}_personalized_dva_leaderboard.csv`
  - Diagnostics: Validates passivity vs. power tradeoffs (e.g., Steven Kwan vs. Kyle Stowers).

---

## PILLAR 4: Physics-Informed Retinal Optics & Psychophysics

### 4.1 Objective & Scientific Motivation
Statcast's 9-parameter kinematic fit assumes constant acceleration across flight. Feeding $(a_x, a_y, a_z)$ at $t = 0.05$ leaks full-trajectory Magnus movement to the neural network. We replace Cartesian $(x, y, z, v, a)$ inputs with biologically accurate **Retinal Spherical Optics** and inject **Weber-Fechner observation noise**, forcing the neural filter to infer trajectory deflection from optical expansion.

### 4.2 Mathematical Formulation
1. **Retinal Coordinate Projection:**
   Define the displacement vector from the batter's eye position $\mathbf{p}_{\text{eye}} = (x_{\text{stand}}, y_{\text{plate}} + 1.417, z_{\text{eye}})$ to the ball $\mathbf{p}(t)$:
   $$\Delta \mathbf{r}(t) = \mathbf{p}(t) - \mathbf{p}_{\text{eye}}, \quad d(t) = \|\Delta \mathbf{r}(t)\|_2$$
   - Azimuth Angle: $\theta(t) = \arctan2(\Delta x(t), \Delta y(t))$
   - Elevation Angle: $\phi(t) = \arctan2\left(\Delta z(t), \sqrt{\Delta x(t)^2 + \Delta y(t)^2}\right)$
   - Subtended Visual Angle: $\psi(t) \approx \frac{2 R_{\text{ball}}}{d(t)}$
   - Retinal Looming Rate (Tau-dot): $\frac{\dot{\psi}(t)}{\psi(t)} = -\frac{\Delta \mathbf{r}(t) \cdot \mathbf{v}(t)}{d(t)^2}$
   - Angular Velocities $(\dot{\theta}, \dot{\phi})$: Computed via analytical finite differences across timesteps.

2. **Weber-Fechner Psychophysical Noise Injection:**
   Sensory noise scales with angular velocity and inversely with viewing duration:
   $$\sigma_{\dot{\theta}}(t) = \sqrt{\sigma_{0, \text{ang}}^2 + w_{\text{weber}}^2 \cdot \dot{\theta}(t)^2}, \quad \sigma_{\text{loom}}(t) = \sqrt{\sigma_{0, \text{loom}}^2 + w_{\text{loom}}^2 \cdot \left(\frac{\dot{\psi}(t)}{\psi(t)}\right)^2}$$
   Acceleration parameters $(a_x, a_y, a_z)$ are excluded from the observation tensor.

### 4.3 Implementation Blueprint
* **Module:** `src/retinal_physics.py`
  ```python
  import torch

  def cartesian_to_retinal_tensor(traj_xyz_v: torch.Tensor, stand: torch.Tensor, sz_top: torch.Tensor):
      # traj_xyz_v: (batch, seq_len, 6) -> [x, y, z, vx, vy, vz]
      eye_x = torch.where(stand == 'R', -1.75, 1.75)
      eye_y = 1.417
      eye_z = sz_top + 0.65
      
      dx = traj_xyz_v[..., 0] - eye_x.unsqueeze(-1)
      dy = traj_xyz_v[..., 1] - eye_y
      dz = traj_xyz_v[..., 2] - eye_z.unsqueeze(-1)
      dist = torch.sqrt(dx**2 + dy**2 + dz**2)
      
      theta = torch.atan2(dx, dy)
      phi = torch.atan2(dz, torch.sqrt(dx**2 + dy**2))
      psi = 2.0 * 0.1208 / torch.clamp(dist, min=1.0)
      
      return torch.stack([theta, phi, psi], dim=-1)
  ```
* **Pipeline Script:** `scripts/train_retinal_model.py`
  - Trains `PitchPerceptionLSTM(input_mode="retinal_9d")`.
  - Verifies that trajectory uncertainty $\bm{\Sigma}_{0.40}$ reflects visual break rather than training-fold residual variance.

---

## 5. STRICT OUT-OF-FOLD (OOF) PROTOCOL & DATA INTEGRITY

1. **Partitioning:** All models across Pillars 1–4 must utilize `GroupKFold(n_splits=5, groups=batter_id)`.
2. **Leakage Prevention:**
   - Batter contact surfaces ($\widehat{\text{xwOBAcon}}_i$) must be estimated on Fold $k$ training splits prior to scoring decisions on Fold $k$ test splits.
   - Empirical Bayes priors must compute ANOVA variance ratios strictly within training folds.
3. **Artifact Persistence:**
   - Store all out-of-sample inferences in `data/processed/oof_next_gen_predictions_{year}.parquet`.

---

## 6. VERIFICATION SUITE & UNIT TESTS

Create `tests/test_next_gen_pillars.py` containing deterministic unit tests covering the four pillars:
1. `test_retinal_projection_zero_leakage()`: Confirm modifying $(a_x, a_y, a_z)$ at $t=0$ yields identical retinal values at the initial timestep.
2. `test_bayesian_precision_addition()`: Verify that $\det \bm{\Sigma}_{\text{post}} \le \min(\det \bm{\Sigma}_{\text{prior}}, \det \bm{\Sigma}_{\text{vis}})$.
3. `test_ddm_boundary_collapse()`: Verify that decision boundary $B_i(t)$ monotonically decreases and equals zero at $t = T_{\text{deadline}}$.
4. `test_dva_additive_identity()`: Confirm that $|\text{Total Indiv DVA} - (\text{Universal DVA} + \text{Damage Alignment DVA})| < 10^{-6}$.
5. `test_strict_oof_partition()`: Ensure no overlapping batter IDs exist between training and validation splits within any fold.