# CLAUDE.md

This file provides architectural guidance and documentation for the Strike Zone Perception & Swing Decision codebase (`sp26_sagb_dj`).

## 🌐 Three Distinct Modeling Tracks

The repository contains three distinct modeling tracks developed for analyzing batter perception and swing decisions:

### Track 1: Modern LSTM Perception & Bayesian Swing Decision Engine (Primary Pipeline)
This is the production pipeline for the Swing Decision Leaderboard:
1. **Trajectory Sequencing:** `scripts/build_training_data.py` converts Statcast pitch tracks into left-padded (100, 9) tensors (`trajectories_{year}.pt`). If pitch duration exceeds 100 timesteps, trajectories are uniformly resampled across flight to strictly preserve flight duration.
2. **Streaming Perception Model:** `src/model.py` (`PitchPerceptionLSTM`) with a Mixture Density Network (MDN) head processes streaming $(x, y, z, v_x, v_y, v_z, a_x, a_y, a_z)$ features and outputs a 2D Gaussian belief $(\mu_x, \mu_z, \sigma_x, \sigma_z, \rho)$ at each timestep.
   * *Interpretation Note:* The learned $\sigma$ represents conditional model residual uncertainty given reconstructed physics, not direct measurement of human sensory acuity.
3. **Biological Decision Point Scoring:** `src/lstm_scoring.py` evaluates belief at 40% of flight (~175 ms decision point). Evaluates exact bivariate normal strike zone integrals via Owen's T function (`_bvn_cdf`), fully incorporating correlation $\rho$ over rectangular Chebyshev bounds ($L_\infty$).
4. **Context-Aware XGBoost Baseline:** `src/swing_model.py` (`LeagueSwingModel`) predicts expected league swing probability $P(\text{Swing})_{\text{league}}$ using count, velocity, movement, and location features. Evaluated via untouched batter-grouped out-of-fold cross-validation (with inner early-stopping split). Baseline predictions are persisted to `data/training/oof_predictions_{year}.parquet`.
5. **Hierarchical Bayesian DVA Scoring:** Evaluates Decision Value Added (DVA):
   * $\text{Tendency DVA}$: True-talent posterior swing tendency deviation from league prior via Beta-Binomial conjugacy.
   * $\text{Actual DVA}$: Observed swing decision value relative to baseline expectation, shrunk toward league baseline ($0.0$) using empirical Bayes pseudo-count prior ($K_{\text{actual}} = 15.0$).
   * Saves to `swing_ratings` table in SQLite DB (`data/mlb.db`).

### Track 2: Classical Additive Physical Eye Scorer
* **Module:** `src/scoring.py` (`FastEyeScorer`).
* **Mechanism:** Additive variance model ($\text{Var}_{\text{total}} = \text{Var}_{\text{phys}} + \sigma_{\text{vis}}^2$) fitted per batter via MLE over swing decisions assuming $P(\text{Swing}) \approx P(\text{Strike})$.
* **Database Target:** `physical_eye_ratings` (and legacy `eye_ratings` table for notebook compatibility).

### Track 3: Biophysical Monte Carlo & Bayesian Fusion
* **Modules:** `src/uncertainty.py` & `src/perception.py`.
* **Mechanism:** Generates physical particle cloud by perturbing release mechanics with multivariate Gaussian noise in consistent physical units (feet). Fuses with distance-dependent visual uncertainty $\sigma_{\text{vis}}(y)$ via continuous distance grid interpolation onto the plate plane ($y = 1.417$ ft) with exact posterior variance scaling.
* **Database Target:** `physical_pitch_metrics` via `scripts/backfill_physical_metrics.py`.

---

## 🛠️ Common Workflows & Scripts

### 1. Retraining the Perception Model (GPU/VM)
```bash
python3 scripts/train_model.py --years 2025 --epochs 20 --batch-size 512 --device cuda --patience 5
```

### 2. Building Features & Persisting Baseline Predictions
```bash
python3 scripts/build_swing_features.py 2025 --device cuda --decision-frac 0.40
python3 scripts/train_swing_model.py --year 2025 --kappa 30 --oof-splits 5
```

### 3. Exporting & Analyzing Results
```bash
python3 scripts/export_leaderboard.py --year 2025
python3 analysis/compare_stowers_kwan_40.py --year 2025
```

### 4. Running Unit Tests
```bash
pytest tests/
```

---

## ⚠️ Key Conventions & Guidelines

1. **Strike Zone Geometry:** Always use the rectangular Chebyshev ($L_\infty$) norm: $\max(|\text{loc\_x\_norm}|, |\text{loc\_z\_norm}|)$. The strike zone is an 8.5-inch half-width rectangle, NOT a circle.
2. **Bivariate Integration:** Never assume independence of $X$ and $Z$ axes when calculating strike zone probabilities. Always use bivariate normal integration incorporating correlation $\rho$.
3. **Out-of-Fold Validation:** Keep baseline swing predictions strictly out-of-fold by batter. Never use outer validation labels for early stopping or in-sample evaluation.
4. **Database Provenance:** All rating tables (`swing_ratings`, `physical_eye_ratings`, `lstm_eye_ratings`) must preserve multi-season history by keying on `season` and performing non-destructive updates.
5. **Physical Units:** Statcast movement metrics (`pfx_x`, `pfx_z`) are stored in feet. Do not divide by 12.