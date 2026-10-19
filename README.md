# Strike Zone Perception & Bayesian Swing Decision Engine

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-31%20passed-brightgreen.svg)]()

> **Research and presentation by Chris Song** presented at the **SABR / SAGB Conference**.  
> Materials and slides available in [`docs/presentation/`](docs/presentation/).

---

## ⚾ Overview

In Major League Baseball, traditional plate discipline metrics like **Chase Rate (O-Swing%)**, **Zone Swing Rate (Z-Swing%)**, and **SwStr%** evaluate batter decisions against a binary, rectangular strike zone using final pitch locations measured at home plate ($y = 1.417\text{ ft}$).

In reality, human batters do not possess 20/20 hindsight. Batters must commit to a swing or take decision **~175 ms after release** (approximately **40% of pitch flight**), when the ball is still 30+ feet away:

<p align="center">
  <img src="assets/probability_cloud.png" alt="Probability Cloud at the Decision Point" width="90%"/>
</p>

This repository implements a **deep streaming perception model** and **hierarchical Bayesian decision engine** that evaluates swing decisions under perceptual uncertainty:

1. **Streaming Trajectory Perception:** An LSTM with a Mixture Density Network (MDN) processes 9D kinematic features $(x, y, z, v_x, v_y, v_z, a_x, a_y, a_z)$ at each millisecond of flight, estimating a 2D Gaussian belief distribution $(\mu_x, \mu_z, \sigma_x, \sigma_z, \rho)$ over plate-crossing location.
2. **Exact Chebyshev Strike Zone Integration:** At the 40% flight decision point, we evaluate the exact bivariate normal CDF over the rectangular strike zone bounds ($L_\infty$) via Owen's T function, fully incorporating cross-axis spatial correlation $\rho$.
3. **Context-Aware XGBoost League Baseline:** Predicts league-expected swing probability $P(\text{Swing})_{\text{league}}$ using game situation (balls, strikes), pitcher hand, velocity, movement, and measured final pitch location $(x, z)$ via 5-fold batter-grouped out-of-fold cross-validation.
4. **Hierarchical Bayesian DVA Scoring:** Evaluates **Decision Value Added (DVA)** by decomposing batter decisions into true-talent swing tendency vs. acute pitch-level execution with empirical Bayes shrinkage.

> [!NOTE]
> **Scientific Modeling Caveat:** Trajectories are reconstructed from Statcast initial conditions under constant acceleration kinematics to model perceptual uncertainty at the biological decision point. The XGBoost baseline conditions on measured final pitch locations to reflect objective pitch quality under league conditions. DVA measures baseline-adjusted strike-probability agreement rather than a full count-specific run-expectancy utility model.

---

## 🏆 2025 DVA Leaderboard Highlights

### Top 10 Decision Makers (2025 MLB Season)
<p align="center">
  <img src="assets/dva_top10.png" alt="2025 DVA Top 10 Leaders" width="85%"/>
</p>

### Bottom 10 Decision Makers (2025 MLB Season)
<p align="center">
  <img src="assets/dva_bottom10.png" alt="2025 DVA Bottom 10 Leaders" width="85%"/>
</p>

*Full 532-player leaderboard is available at [`data/leaderboards/2025_dva_leaderboard.csv`](data/leaderboards/2025_dva_leaderboard.csv).*

---

## 🔬 Modeling Tracks

The repository contains three complementary modeling approaches:

### Track 1: Modern LSTM Perception & Bayesian DVA Engine *(Primary Pipeline)*
- **Trajectory Sequencing:** Left-padded $(100, 9)$ tensors resampled uniformly across flight duration.
- **Model Architecture:** [`PitchPerceptionLSTM`](src/model.py) with 2-layer LSTM + MDN head predicting $(\mu_x, \mu_z, \sigma_x, \sigma_z, \rho)$.
- **Scoring Engine:** [`LSTMEyeScorer`](src/lstm_scoring.py) computes $P(\text{Strike})$ at 40% flight using bivariate normal integration (`_bvn_cdf`).
- **Baseline Engine:** [`LeagueSwingModel`](src/swing_model.py) trained with untouched 5-fold batter-grouped out-of-fold cross-validation.
- **Metric Formulation:**
  $$\text{DVA} = \text{Tendency DVA} + \text{Actual DVA}$$
  - $\text{Tendency DVA}$: True-talent posterior swing tendency deviation from league prior via Beta-Binomial conjugacy.
  - $\text{Actual DVA}$: Observed swing decision value relative to baseline expectation, shrunk toward zero via empirical Bayes prior ($K = 15.0$).

### Track 2: Classical Additive Physical Eye Scorer
- **Module:** [`FastEyeScorer`](src/scoring.py)
- **Mechanism:** Vectorized maximum-likelihood estimation (MLE) over swing decisions assuming an additive variance model:
  $$\text{Var}_{\text{total}} = \text{Var}_{\text{phys}} + \sigma_{\text{vis}}^2$$

### Track 3: Biophysical Monte Carlo & Bayesian Fusion
- **Modules:** [`generate_pitch_cloud`](src/uncertainty.py) & [`PerceptualCloud`](src/perception.py)
- **Mechanism:** Perturbs initial release mechanics with multivariate Gaussian physical noise, then performs Kalman-style continuous Bayesian fusion with distance-dependent visual acuity $\sigma_{\text{vis}}(y)$.
- **3D Visualization:** Interactive 3D trajectory cloud available at [`assets/perceptual_cloud_3d.html`](assets/perceptual_cloud_3d.html).

---

## 📊 Analytical Insights: Case Studies

### Kyle Stowers vs. Steven Kwan (40% Decision Point)
Traditional metrics credit Steven Kwan with elite plate discipline due to low chase rates. However, when evaluating the biological decision point under uncertainty, Kwan frequently takes borderline pitches that appear to be strikes, whereas Kyle Stowers displays aggressive, high-expected-value swing decisions on pitches with high strike probabilities:

<p align="center">
  <img src="assets/stowers_vs_kwan.png" alt="Stowers vs Kwan Decision Comparison" width="75%"/>
</p>

Run this comparison directly:
```bash
python3 analysis/compare_stowers_kwan_40.py --year 2025
```

---

## 🧠 Advanced Perceptual Extensions

This repository includes three cutting-edge perceptual modeling extensions designed to evaluate late movement, biological uncertainty, and individual decision timing:

### 1. Deception vs. Discipline: The Perceptual Mirage Decomposition ($\Delta P$)
Compares the streaming LSTM belief at the biological commit point ($t = 0.40$, ~175 ms) against ground-truth plate arrival ($t = 1.0$):
$$\Delta P_i = P(\text{Strike} \mid \mathcal{F}_{0.40})_i - \mathbb{I}(\text{True Strike})_i$$
- **Decoupled Dimensions:** Decouples baseline strike zone control ($|\Delta P| < 0.08$) from late pitch tunneling vulnerability ($|\Delta P| \ge 0.08$), achieving $>86.5\%$ orthogonality ($R^2 < 0.14$) against traditional Chase Rate.
- **Continuous Deception-Weighted Chase:** Evaluates chase rate weighted by early perceptual strike probability across all out-of-zone pitches ($r = 0.284$ with raw chase, $91.9\%$ independent).
- **Dual-Sided Pitcher Perceptual Stuff+:** Quantifies induced swings above expectation on deceptive balls, isolating arms with genuine late movement (e.g. Yoshinobu Yamamoto's cutter FC leads MLB pitch types at $172.9$ Perceptual Stuff+).

<p align="center">
  <img src="assets/presentation/mirage_orthogonality_quadrant.png" alt="Perceptual Mirage Orthogonality Quadrant" width="85%"/>
</p>

```bash
python3 scripts/run_mirage_decomposition.py --years 2025 --mirage-thresh 0.08
```

### 2. Sensitivity to Visual Uncertainty: The MDN Covariance Matrix ($\det \mathbf{\Sigma}_{0.40}$)
Leverages the full bivariate spatial covariance matrix output by the Mixture Density Network:
$$\mathbf{\Sigma}_{0.40} = \begin{pmatrix} \sigma_x^2 & \rho \sigma_x \sigma_z \\ \rho \sigma_x \sigma_z & \sigma_z^2 \end{pmatrix}, \quad \text{Vol}(\mathbf{\Sigma}_{0.40}) = \pi \sqrt{\det \mathbf{\Sigma}_{0.40}} = \pi \sigma_x \sigma_z \sqrt{1 - \rho^2}$$
- **Ambiguity Policy Estimation:** Fits a count-controlled psychometric logistic model on competitive borderline pitches ($0.30 \le P(\text{Strike}) \le 0.70$) to extract each batter's uncertainty sensitivity $\beta_{U,i}$.
- **Strategic Policy Archetypes:** Discovers **Ambiguity-Seeking** hitters ("when in doubt, swing", e.g. Tyler Stephenson $\beta_U = +1.98$) vs. **Ambiguity-Averse** hitters ("when in doubt, take", e.g. Mike Trout $\beta_U = -2.10$, Ronald Acuña Jr. $\beta_U = -2.06$).

<p align="center">
  <img src="assets/presentation/uncertainty_response_curves.png" alt="Visual Uncertainty Response Curves" width="85%"/>
</p>

```bash
python3 scripts/run_uncertainty_sensitivity.py --year 2025
```

### 3. Inferring Individual Commit Horizons ($\tau^*_i$) & Statcast Bat Tracking
Replaces the static $\tau = 0.40$ assumption by inferring each batter's individual commitment horizon $\tau^*_i \in [0.25, 0.55]$ along the 100-step trajectory via the **Sequential Information Gain / Marginal Forward Update Test**:
$$\text{logit } P(\text{Swing}) = \alpha + \beta_{\text{base}} \cdot (P_{\tau_k} - 0.50) + \beta_{\Delta} \cdot (P_{\tau_{k+1}} - P_{\tau_k})$$
- **Physiological Horizon Distribution:** Identifies the elbow point where late trajectory updates cease to affect swing probability ($\beta_\Delta \ge 0.50 \cdot \max \boldsymbol{\beta}_\Delta$), yielding a balanced interior distribution spanning $100\text{ ms}$ ($\tau^* = 0.25$) to $200\text{ ms}$ ($\tau^* = 0.50$).
- **Biomechanical Grounding:** Hitters with explosive swing lengths and heavy rotational inertia (Giancarlo Stanton: $\tau^* = 0.30$, $79.0\text{ mph}$ bat speed, $8.72\text{ ft}$ swing length) commit early, while compact hitters with short inside-out strokes (Freddie Freeman: $\tau^* = 0.50$, $68.4\text{ mph}$, $6.81\text{ ft}$) track pitches deep into flight.

<p align="center">
  <img src="assets/presentation/commit_horizon_bat_tracking.png" alt="Commit Horizon vs Bat Tracking" width="85%"/>
</p>

```bash
python3 scripts/run_commit_horizon.py --year 2025
```

---

## 📁 Repository Structure

```text
sp26_sagb_dj/
├── assets/                     # Visual figures, plots, and interactive HTML
│   ├── presentation/           # High-resolution (300 DPI) conference presentation assets
│   │   ├── mirage_orthogonality_quadrant.png
│   │   ├── uncertainty_response_curves.png
│   │   └── commit_horizon_bat_tracking.png
│   ├── dva_top10.png
│   ├── dva_bottom10.png
│   ├── dva_scatter_plots.png
│   ├── probability_cloud.png
│   ├── stowers_vs_kwan.png
│   └── perceptual_cloud_3d.html
├── data/
│   ├── leaderboards/           # Tracked CSV exports (DVA, Mirage, Uncertainty, Horizon)
│   ├── models/                 # Model checkpoints (best_model.pt, swing_model.xgb)
│   ├── raw_data/               # Statcast feathers (gitignored; see setup)
│   └── training/               # Trajectories and parquet features (gitignored)
├── docs/
│   ├── audits/                 # Comprehensive code & pipeline audit logs
│   └── presentation/           # Empirical findings, presentation slides, & paper notes
├── notebooks/                  # Interactive Jupyter exploration notebooks
├── scripts/                    # Core pipeline and training CLI scripts
│   ├── run_mirage_decomposition.py   # Deception vs. Discipline (Mirage Residual ΔP)
│   ├── run_uncertainty_sensitivity.py # MDN Covariance Sensitivity (det Σ_0.40)
│   ├── run_commit_horizon.py         # Individual Commit Times (τ*_i) & Bat Tracking
│   ├── build_training_data.py        # Converts Statcast pitches to (100, 9) tensors
│   ├── train_model.py                # Trains LSTM+MDN perception model
│   ├── build_swing_features.py       # Evaluates P(Strike) at decision point
│   ├── train_swing_model.py          # Trains XGBoost out-of-fold baseline
│   ├── run_eye_rating_lstm.py        # Computes LSTM eye ratings & saves to DB
│   └── export_leaderboard.py         # Exports swing_ratings from SQLite to CSV
├── analysis/                   # Predictive validity, case studies, & figure generators
│   ├── predictive_validity.py        # 5-fold cross-validated out-of-sample regressions
│   ├── compare_stowers_kwan_40.py
│   └── ...
├── src/                        # Core Python library
│   ├── db_manager.py           # SQLite connection & schema management
│   ├── dataset.py              # PyTorch Dataset for streaming pitch tracks
│   ├── model.py                # PitchPerceptionLSTM neural network
│   ├── lstm_scoring.py         # Exact BVN strike zone integration
│   ├── swing_model.py          # League XGBoost baseline & DVA calculation
│   ├── physics.py              # 9-parameter constant acceleration trajectory
│   ├── uncertainty.py          # Monte Carlo release parameter perturbation
│   └── scoring.py              # Classical FastEyeScorer MLE
├── tests/                      # Automated test suite (31 tests in pytest)
│   ├── test_mirage_decomposition.py     # Deception residual & weighted chase
│   ├── test_uncertainty_sensitivity.py  # Ridge ambiguity policy & volume
│   ├── test_commit_horizon.py           # Sequential forward update elbow test
│   ├── test_lstm_scoring.py             # BVN CDF & strike zone integration
│   ├── test_oof_provenance.py           # Strict out-of-fold leak guards
│   ├── test_physics_boundaries.py       # Trajectory kinematics & physics
│   └── test_scoring.py                  # DVA scoring & shrinkage
├── .gitignore                  # Production gitignore for large ML datasets
├── pyproject.toml              # Modern PEP 517/621 package build config
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── README.md
```

---

## 🚀 Quickstart & Installation

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/ChrisSong214/strike-zone-perception.git
cd strike-zone-perception

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies and editable package
pip install -r requirements.txt
pip install -e .
```

### 2. Run the Test Suite

```bash
pytest tests/
```

All 31 unit tests verify boundary conditions, bivariate normal Owen's T integrals, single-particle stability, trajectory kinematics, FastEyeScorer filtering and minimum observation guards, strict out-of-fold provenance validation, perceptual mirage decompositions, visual uncertainty sensitivity, and sequential commit horizon forward updates.

### 3. Export the 2025 Leaderboard

Export the pre-calculated DVA rankings from the local SQLite database to CSV:

```bash
python3 scripts/export_leaderboard.py --year 2025
```

---

## 🔄 End-to-End Pipeline Execution

To retrain and execute the full pipeline from raw Statcast pitches to final DVA ratings:

### Database Setup Prerequisite

Before running training data builders on a fresh checkout, ensure the `statcast` table in `data/mlb.db` is populated.

- **Option A (Recommended — Existing Database):**
  Place an existing SQLite database containing the populated `statcast` table directly into `data/mlb.db`.

- **Option B (Fresh Download & Sync from Statcast):**
  1. Download pitch records via `pybaseball` and cache locally to `data/feather/`:
     ```bash
     python3 -c "from src.data_handler import StatcastDataHandler; StatcastDataHandler().update_local_sc(just_current=False, specific_year=2025)"
     ```
  2. Populate the SQLite `statcast` table and generate indexes:
     ```bash
     python3 -c "from src.db_manager import sync_feather_to_sql; sync_feather_to_sql([2025])"
     ```
     *(Note: `sync_feather_to_sql` initializes/replaces the `statcast` table on its initial write. Skip this step if you have existing manual edits or multiple historical seasons already stored in `data/mlb.db`.)*

### Step 1: Precompute Trajectory Tensors
```bash
python3 scripts/build_training_data.py --years 2025
# (positional syntax also supported: python3 scripts/build_training_data.py 2025)
```
Converts Statcast pitch tracks into left-padded $(100, 9)$ tensors saved to `data/training/trajectories_2025.pt`.

### Step 2: Train Streaming LSTM Perception Model
```bash
python3 scripts/train_model.py --years 2025 --epochs 20 --batch-size 512 --device cuda --patience 5
```
Trains `PitchPerceptionLSTM` with MDN loss and saves the optimal weights to `data/models/best_model.pt`.

### Step 3: Extract Features at the Decision Point (40% Flight)
```bash
python3 scripts/build_swing_features.py 2025 --device cuda --decision-frac 0.40
```
Runs inference through the LSTM model at the biological decision point (40% distance) and outputs `data/training/swing_features_2025.parquet`.

### Step 4: Fit Out-of-Fold Baseline Swing Model
```bash
python3 scripts/train_swing_model.py --year 2025 --kappa 30 --oof-splits 5
```
Fits the XGBoost baseline using 5-fold batter-grouped out-of-fold cross-validation, shrinks DVA using empirical Bayes, and writes final ratings to `data/mlb.db`.

### Step 5: Generate Visualizations & Case Studies
```bash
python3 analysis/generate_leaderboard_image.py --year 2025
python3 analysis/compare_stowers_kwan_40.py --year 2025
python3 analysis/compare_bregman_ramirez.py --year 2025
```

---

## 📖 Citation & Acknowledgments

If you use this codebase or methodology in your research, please cite:

```bibtex
@misc{song2026strikezoneperception,
  author = {Song, Chris},
  title = {Strike Zone Perception and Bayesian Swing Decisions Through Deep Streaming Kinematics},
  year = {2026},
  howpublished = {SABR / SAGB Conference Presentation},
  note = {Codebase: sp26_sagb_dj}
}
```

- Data provided via Major League Baseball Statcast via [pybaseball](https://github.com/jldbc/pybaseball).
- Research presented at the **SAGB Conference 2026**.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
