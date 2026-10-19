"""
Swing Decision Model with Hierarchical Bayesian Scoring

Two components:
    1. LeagueSwingModel (XGBoost): predicts league-average P(Swing | context)
    2. BayesianBatterModel: per-batter Beta-Binomial posteriors in 27 context bins

Together they produce a Decision Value Added (DVA) rating for each batter.
"""

import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from scipy.special import expit  # sigmoid


# ---- XGBoost League Model ----

# ---- Feature Definitions for Dual Baselines (Finding 1) ----

# Objective Post-Hoc Baseline: Conditions on measured final plate arrival
FINAL_LOCATION_FEATURES = [
    # Count
    'balls', 'strikes', 'two_strikes',
    # Handedness
    'stand_R', 'p_throws_R', 'platoon',
    # Pitch kinematics
    'release_speed', 'pfx_x', 'pfx_z',
    # Measured final plate location
    'plate_x', 'plate_z', 'loc_x_norm', 'loc_z_norm', 'dist_from_center', 'in_zone',
    # Game state
    'outs_when_up', 'inning', 'runners_on', 'score_diff',
    # LSTM perception at decision point
    'lstm_sigma_x', 'lstm_sigma_z', 'lstm_p_strike',
]

# Perceptual Decision-Time Baseline: Conditions SOLELY on information available at t = 0.40
DECISION_TIME_FEATURES = [
    # Count
    'balls', 'strikes', 'two_strikes',
    # Handedness
    'stand_R', 'p_throws_R', 'platoon',
    # Pitch kinematics known early (release speed)
    'release_speed',
    # Game state
    'outs_when_up', 'inning', 'runners_on', 'score_diff',
    # LSTM perception at decision point (t = 0.40)
    'lstm_mu_x', 'lstm_mu_z', 'lstm_sigma_x', 'lstm_sigma_z', 'lstm_rho', 'lstm_p_strike',
]

# Canonical default alias for backwards compatibility
XGBOOST_FEATURES = FINAL_LOCATION_FEATURES

# Columns that might need special handling
CATEGORICAL_FEATURES = []  # XGBoost handles numerics; categoricals already encoded


class LeagueSwingModel:
    """
    XGBoost model predicting league-average P(Swing | context).
    
    Supports two explicit baseline modes:
    1. 'final_location' (Objective Post-Hoc Baseline): Conditions on measured final pitch location.
    2. 'decision_time'  (Perceptual Baseline): Conditions solely on information available at t=0.40.
    """
    
    def __init__(self, model=None, feature_mode='final_location', feature_names=None):
        self.model = model
        self.feature_mode = feature_mode
        self.feature_names = feature_names
        self.feature_importances_ = None
        self.best_iteration = None
        self.year = None
    
    def train(self, df, val_df=None, params=None):
        """
        Train XGBoost on the full league dataset.
        
        Args:
            df: training DataFrame with feature columns + 'is_swing'
            val_df: optional validation DataFrame
            params: XGBoost parameters (or use defaults)
        """
        import xgboost as xgb
        
        # Select features based on configured feature_mode
        if self.feature_names is not None:
            candidate_features = self.feature_names
        elif self.feature_mode == 'decision_time':
            candidate_features = DECISION_TIME_FEATURES
        else:
            candidate_features = FINAL_LOCATION_FEATURES

        available = [f for f in candidate_features if f in df.columns]
        self.feature_names = available
        
        X_train = df[available].values.astype(np.float32)
        y_train = df['is_swing'].values.astype(np.float32)
        
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=available)
        
        if params is None:
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 50,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'tree_method': 'hist',
                'verbosity': 1,
            }
        
        evals = [(dtrain, 'train')]
        if val_df is not None:
            X_val = val_df[available].values.astype(np.float32)
            y_val = val_df['is_swing'].values.astype(np.float32)
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=available)
            evals.append((dval, 'val'))
        
        self.model = xgb.train(
            params, dtrain,
            num_boost_round=300,
            evals=evals,
            early_stopping_rounds=20,
            verbose_eval=25,
        )
        self.best_iteration = getattr(self.model, 'best_iteration', None)
        self.feature_importances_ = self.model.get_score(importance_type='gain')
        
        return self
    
    def predict(self, df):
        """Predict P(Swing) for each pitch using best iteration from early stopping."""
        import xgboost as xgb
        
        X = df[self.feature_names].values.astype(np.float32)
        dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
        best_iter = getattr(self, 'best_iteration', None)
        if best_iter is None:
            best_iter = getattr(self.model, 'best_iteration', None)
            
        if best_iter is not None:
            return self.model.predict(dmatrix, iteration_range=(0, best_iter + 1))
        return self.model.predict(dmatrix)
    
    def save(self, path, year=None):
        """Save model and metadata to disk."""
        import json
        self.model.save_model(path + '.xgb')
        meta = {
            'year': int(year) if year is not None else None,
            'feature_mode': getattr(self, 'feature_mode', 'final_location'),
            'feature_names': self.feature_names,
            'feature_importances': self.feature_importances_,
            'best_iteration': getattr(self, 'best_iteration', getattr(self.model, 'best_iteration', None)),
        }
        with open(path + '.meta', 'w') as f:
            json.dump(meta, f, indent=2)
    
    def load(self, path):
        """Load model from disk."""
        import xgboost as xgb
        import json
        
        self.model = xgb.Booster()
        self.model.load_model(path + '.xgb')
        meta = None
        if os.path.exists(path + '.meta'):
            try:
                with open(path + '.meta', 'r') as f:
                    meta = json.load(f)
            except Exception:
                try:
                    with open(path + '.meta', 'rb') as f:
                        meta = pickle.load(f)
                except Exception:
                    meta = {}
        else:
            meta = {}
            
        self.feature_mode = meta.get('feature_mode', 'final_location')
        self.feature_names = meta.get('feature_names')
        self.feature_importances_ = meta.get('feature_importances', {})
        self.best_iteration = meta.get('best_iteration')
        self.year = meta.get('year')
        return self


# ---- Hierarchical Bayesian Batter Model ----

# Context bin definitions
COUNT_GROUPS = ['ahead', 'even', 'behind']
LOCATION_BINS = ['zone', 'edge', 'chase']
PITCH_GROUPS = ['fastball', 'breaking', 'offspeed']


class BayesianBatterModel:
    """
    Hierarchical Beta-Binomial model for individual batter swing tendencies.
    
    For each batter × context bin:
        Prior:     Beta(κ·p̂, κ·(1-p̂))   where p̂ = XGBoost league avg
        Posterior: Beta(α₀ + k, β₀ + n-k) where k = swings, n = pitches
    
    κ (kappa) controls how much we trust the league prior.
    κ = 30 means "the league average is worth ~30 pitches of evidence."
    """
    
    def __init__(self, kappa=30):
        """
        Args:
            kappa: strength of the league prior (equivalent sample size).
                   Higher = more shrinkage toward league avg.
                   Lower = more individual expression.
        """
        self.kappa = kappa
        # batter_posteriors[batter_id][bin_key] = {'alpha': float, 'beta': float, 'n': int}
        self.batter_posteriors = {}
        # League-level bin priors
        self.league_priors = {}
    
    def fit(self, df, league_p_swing):
        """
        Compute posteriors for all batters.
        
        Args:
            df: DataFrame with context bin columns and 'is_swing'
            league_p_swing: array of league-average P(Swing) per pitch (from XGBoost)
        """
        df = df.copy()
        df['league_p_swing'] = league_p_swing
        
        # Step 1: Compute league-level bin priors (average P(Swing) per bin)
        for cg in COUNT_GROUPS:
            for lb in LOCATION_BINS:
                for pg in PITCH_GROUPS:
                    bin_key = (cg, lb, pg)
                    mask = ((df['count_group'] == cg) & 
                            (df['loc_bin'] == lb) & 
                            (df['pitch_group'] == pg))
                    bin_data = df[mask]
                    
                    if len(bin_data) > 0:
                        self.league_priors[bin_key] = {
                            'mean_p': bin_data['league_p_swing'].mean(),
                            'actual_rate': bin_data['is_swing'].mean(),
                            'n': len(bin_data),
                        }
                    else:
                        self.league_priors[bin_key] = {
                            'mean_p': 0.5, 'actual_rate': 0.5, 'n': 0,
                        }
        
        # Step 2: Compute per-batter posteriors
        batters = df['batter'].unique()
        
        for batter_id in batters:
            batter_df = df[df['batter'] == batter_id]
            self.batter_posteriors[batter_id] = {}
            
            for cg in COUNT_GROUPS:
                for lb in LOCATION_BINS:
                    for pg in PITCH_GROUPS:
                        bin_key = (cg, lb, pg)
                        mask = ((batter_df['count_group'] == cg) & 
                                (batter_df['loc_bin'] == lb) & 
                                (batter_df['pitch_group'] == pg))
                        bin_data = batter_df[mask]
                        
                        n = len(bin_data)
                        k = bin_data['is_swing'].sum() if n > 0 else 0
                        
                        # Prior from league model
                        p_prior = self.league_priors[bin_key]['mean_p']
                        p_prior = np.clip(p_prior, 0.01, 0.99)
                        
                        alpha_0 = self.kappa * p_prior
                        beta_0 = self.kappa * (1 - p_prior)
                        
                        # Posterior
                        alpha_post = alpha_0 + k
                        beta_post = beta_0 + (n - k)
                        
                        self.batter_posteriors[batter_id][bin_key] = {
                            'alpha': alpha_post,
                            'beta': beta_post,
                            'n': n,
                            'k': int(k),
                            'p_prior': p_prior,
                            'p_posterior': alpha_post / (alpha_post + beta_post),
                            'shrinkage': self.kappa / (self.kappa + n),
                        }
        
        print(f"Computed posteriors for {len(batters)} batters across {len(self.league_priors)} bins")
        return self
    
    def get_batter_swing_prob(self, batter_id, count_group, loc_bin, pitch_group):
        """Get posterior P(Swing) for a batter in a specific context bin."""
        if batter_id not in self.batter_posteriors:
            # Fall back to league average
            bin_key = (count_group, loc_bin, pitch_group)
            return self.league_priors.get(bin_key, {}).get('mean_p', 0.5)
        
        bin_key = (count_group, loc_bin, pitch_group)
        post = self.batter_posteriors[batter_id].get(bin_key)
        if post is None:
            return self.league_priors.get(bin_key, {}).get('mean_p', 0.5)
        
        return post['p_posterior']
    
    def get_batter_profile(self, batter_id):
        """
        Get a summary of a batter's swing tendencies vs league avg.
        
        Returns DataFrame with one row per context bin.
        """
        if batter_id not in self.batter_posteriors:
            return None
        
        rows = []
        for bin_key, post in self.batter_posteriors[batter_id].items():
            cg, lb, pg = bin_key
            rows.append({
                'count_group': cg,
                'loc_bin': lb,
                'pitch_group': pg,
                'n_pitches': post['n'],
                'swings': post['k'],
                'p_prior': post['p_prior'],
                'p_posterior': post['p_posterior'],
                'deviation': post['p_posterior'] - post['p_prior'],
                'shrinkage': post['shrinkage'],
            })
        
        return pd.DataFrame(rows).sort_values('deviation', ascending=False)


# ---- Count-Dependent Expected Value (EV-DVA) & RE24 Transitions ----

# Standard MLB RE24 Empirical Count Transition Run Values across the 12 (balls, strikes) states
# Derived from Statcast delta_run_exp across 3.8M+ pitches
# Values:
#   rv_ball: delta_run_exp of an unhit ball (e.g. walk on 3 balls)
#   rv_called_strike: delta_run_exp of a called strike (e.g. strikeout on 2 strikes)
#   rv_swing_ball: empirical delta_run_exp of a swing on an out-of-zone pitch
#   rv_swing_strike: empirical delta_run_exp of a swing on an in-zone pitch
COUNT_RUN_VALUES = {
    (0, 0): {'rv_ball': 0.0382, 'rv_called_strike': -0.0405, 'rv_swing_ball': -0.0384, 'rv_swing_strike': 0.0026},
    (0, 1): {'rv_ball': 0.0301, 'rv_called_strike': -0.0553, 'rv_swing_ball': -0.0421, 'rv_swing_strike': 0.0016},
    (0, 2): {'rv_ball': 0.0245, 'rv_called_strike': -0.1684, 'rv_swing_ball': -0.0561, 'rv_swing_strike': 0.0271},
    (1, 0): {'rv_ball': 0.0621, 'rv_called_strike': -0.0493, 'rv_swing_ball': -0.0516, 'rv_swing_strike': -0.0137},
    (1, 1): {'rv_ball': 0.0503, 'rv_called_strike': -0.0627, 'rv_swing_ball': -0.0507, 'rv_swing_strike': -0.0063},
    (1, 2): {'rv_ball': 0.0426, 'rv_called_strike': -0.1878, 'rv_swing_ball': -0.0644, 'rv_swing_strike': 0.0191},
    (2, 0): {'rv_ball': 0.1130, 'rv_called_strike': -0.0619, 'rv_swing_ball': -0.0724, 'rv_swing_strike': -0.0373},
    (2, 1): {'rv_ball': 0.1047, 'rv_called_strike': -0.0716, 'rv_swing_ball': -0.0666, 'rv_swing_strike': -0.0282},
    (2, 2): {'rv_ball': 0.0951, 'rv_called_strike': -0.2236, 'rv_swing_ball': -0.0842, 'rv_swing_strike': 0.0018},
    (3, 0): {'rv_ball': 0.1306, 'rv_called_strike': -0.0694, 'rv_swing_ball': -0.0721, 'rv_swing_strike': -0.0410},
    (3, 1): {'rv_ball': 0.1941, 'rv_called_strike': -0.0828, 'rv_swing_ball': -0.1005, 'rv_swing_strike': -0.0665},
    (3, 2): {'rv_ball': 0.2775, 'rv_called_strike': -0.3137, 'rv_swing_ball': -0.1283, 'rv_swing_strike': -0.0395},
}


def estimate_ev_take(p_strike, balls, strikes, rv_table=None):
    """
    Estimate expected run value of taking a pitch given early-flight strike probability P_strike:
        EV_take(P_strike, b, s) = P_strike * RV_called_strike(b, s) + (1 - P_strike) * RV_ball(b, s)
    """
    if rv_table is None:
        rv_table = COUNT_RUN_VALUES
    
    p_s = np.asarray(p_strike, dtype=np.float64)
    b_arr = np.clip(np.asarray(balls, dtype=int), 0, 3)
    s_arr = np.clip(np.asarray(strikes, dtype=int), 0, 2)
    
    ev_take = np.zeros_like(p_s)
    for (b, s), vals in rv_table.items():
        mask = (b_arr == b) & (s_arr == s)
        if np.any(mask):
            ev_take[mask] = p_s[mask] * vals['rv_called_strike'] + (1.0 - p_s[mask]) * vals['rv_ball']
            
    return ev_take


def estimate_ev_swing(p_strike, balls, strikes, rv_table=None):
    """
    Estimate expected run value of swinging at a pitch given early-flight strike probability P_strike:
        EV_swing(P_strike, b, s) = P_strike * RV_swing_strike(b, s) + (1 - P_strike) * RV_swing_ball(b, s)
    interpolating empirical run values of swings across the 12 count states.
    """
    if rv_table is None:
        rv_table = COUNT_RUN_VALUES
        
    p_s = np.asarray(p_strike, dtype=np.float64)
    b_arr = np.clip(np.asarray(balls, dtype=int), 0, 3)
    s_arr = np.clip(np.asarray(strikes, dtype=int), 0, 2)
    
    ev_swing = np.zeros_like(p_s)
    for (b, s), vals in rv_table.items():
        mask = (b_arr == b) & (s_arr == s)
        if np.any(mask):
            ev_swing[mask] = p_s[mask] * vals['rv_swing_strike'] + (1.0 - p_s[mask]) * vals['rv_swing_ball']
            
    return ev_swing


def estimate_eb_shrinkage(df, residual_col='ev_dva', batter_col='batter', min_pitches_for_var=50):
    """
    Estimate Empirical Bayes shrinkage constant K* = sigma2_within / tau2_between
    from one-way random effects ANOVA on pitch-level residuals.
    
    Model:
        d_ij = mu + u_i + eps_ij
        u_i ~ N(0, tau2_between)
        eps_ij ~ N(0, sigma2_within)
    """
    if batter_col not in df.columns or residual_col not in df.columns:
        return 50.0, 1.0, 0.02
        
    counts = df[batter_col].value_counts()
    valid_batters = counts[counts >= min_pitches_for_var].index
    if len(valid_batters) < 5:
        return 50.0, 1.0, 0.02
        
    sub = df[df[batter_col].isin(valid_batters)]
    grouped = sub.groupby(batter_col)[residual_col]
    n_i = grouped.count().values.astype(np.float64)
    mean_i = grouped.mean().values.astype(np.float64)
    var_i = grouped.var(ddof=1).fillna(0).values.astype(np.float64)
    
    df_within = np.sum(n_i - 1.0)
    if df_within <= 0:
        return 50.0, 1.0, 0.02
        
    ss_within = np.sum((n_i - 1.0) * var_i)
    sigma2_within = float(ss_within / df_within)
    
    grand_mean = float(np.sum(n_i * mean_i) / np.sum(n_i))
    k_groups = len(n_i)
    ss_between = float(np.sum(n_i * (mean_i - grand_mean)**2))
    ms_between = ss_between / max(k_groups - 1, 1)
    ms_within = sigma2_within
    
    n_eff = float(np.sum(n_i) - np.sum(n_i**2) / np.sum(n_i)) / max(k_groups - 1, 1)
    tau2_between = float(max(1e-6, (ms_between - ms_within) / max(n_eff, 1.0)))
    
    k_shrinkage = float(np.clip(sigma2_within / tau2_between, 5.0, 500.0))
    return k_shrinkage, sigma2_within, tau2_between


# ---- Decision Value Added (DVA) Scoring ----


def _build_posterior_lookup(bayesian_model):
    """
    Build vectorized lookup arrays from the Bayesian model's posteriors.
    
    Returns:
        dict mapping (batter_id, count_group, loc_bin, pitch_group) 
        → (p_posterior, confidence)
    """
    lookup = {}
    for batter_id, bins in bayesian_model.batter_posteriors.items():
        for bin_key, post in bins.items():
            confidence = 1.0 - post['shrinkage']
            lookup[(batter_id, *bin_key)] = (post['p_posterior'], confidence)
    return lookup


def compute_dva(df, league_model, bayesian_model, league_model_perceptual=None):
    """
    Compute Decision Value Added for each pitch using Bayesian posteriors and
    count-dependent Expected Run Value transitions (EV-DVA).
    
    Supports dual baselines:
    1. **Objective Post-Hoc Baseline** (`league_model`):
       Conditions on final location. Evaluates decision quality relative to objective outcome.
    2. **Perceptual Decision-Time Baseline** (`league_model_perceptual`, optional):
       Conditions strictly on t=0.40 early flight information.
       
    Evaluates:
    1. **EV-DVA (Expected Run Value DVA)**:
       - EV_take(P_strike, b, s) = P_strike * RV_called_strike(b, s) + (1 - P_strike) * RV_ball(b, s)
       - EV_swing(P_strike, b, s) from out-of-fold empirical run values of swings
       - ev_dva = EV(observed_action) - [p_swing_league * EV_swing + (1 - p_swing_league) * EV_take]
       
    2. **Alignment DVA (Symmetric strike alignment quality)**:
       - Retained alongside EV-DVA for direct baseline comparison.
       - actual_dva = is_swing * P_strike + (1-is_swing) * (1-P_strike)
                      - [p_league * P_strike + (1-p_league) * (1-P_strike)]
                      
    3. **Tendency DVA**:
       - Difference in expected quality between batter posterior swing tendency and league prior.
    """
    df = df.copy()
    
    # ---- Vectorised posterior lookup (replaces slow row-by-row .apply) ----
    posterior_lut = _build_posterior_lookup(bayesian_model)
    
    # Build composite keys for every pitch
    keys = list(zip(
        df['batter'].values,
        df['count_group'].values,
        df['loc_bin'].values,
        df['pitch_group'].values,
    ))
    
    # Default fallback: league prior, zero confidence
    league_priors = bayesian_model.league_priors  # {bin_key: {'mean_p': ...}}
    
    p_post_arr  = np.empty(len(df), dtype=np.float64)
    conf_arr    = np.empty(len(df), dtype=np.float64)
    p_prior_arr = np.empty(len(df), dtype=np.float64)
    
    for i, key in enumerate(keys):
        batter_id, cg, lb, pg = key
        bin_key = (cg, lb, pg)
        entry = posterior_lut.get(key)
        prior_p = league_priors.get(bin_key, {}).get('mean_p', 0.5)
        p_prior_arr[i] = prior_p
        if entry is not None:
            p_post_arr[i], conf_arr[i] = entry
        else:
            p_post_arr[i] = prior_p
            conf_arr[i]   = 0.0
    
    df['batter_p_swing'] = p_post_arr
    df['confidence']     = conf_arr
    df['league_p_swing'] = p_prior_arr   # per-bin prior (not XGBoost per-pitch)
    
    # Also store XGBoost per-pitch prediction for diagnostics (or use precomputed out-of-fold)
    if 'xgb_p_swing' not in df.columns:
        df['xgb_p_swing'] = league_model.predict(df)
    
    # P(Strike) — reference value for "should you swing?"
    p_strike = (df['lstm_p_strike'].values if 'lstm_p_strike' in df.columns
                else df['in_zone'].values.astype(float))
    
    batter_p = df['batter_p_swing'].values
    league_p = df['league_p_swing'].values
    xgb_p = df['xgb_p_swing'].values
    is_swing = df['is_swing'].values.astype(float)
    
    # ---- 1. Count-Dependent Expected Value DVA (EV-DVA) ----
    balls = df['balls'].values if 'balls' in df.columns else np.zeros(len(df), dtype=int)
    strikes = df['strikes'].values if 'strikes' in df.columns else np.zeros(len(df), dtype=int)
    
    ev_take = estimate_ev_take(p_strike, balls, strikes)
    ev_swing = estimate_ev_swing(p_strike, balls, strikes)
    
    ev_observed = is_swing * ev_swing + (1.0 - is_swing) * ev_take
    ev_league = xgb_p * ev_swing + (1.0 - xgb_p) * ev_take
    
    df['ev_take'] = ev_take
    df['ev_swing'] = ev_swing
    df['ev_dva'] = ev_observed - ev_league
    df['ev_dva_posthoc'] = df['ev_dva']
    
    # ---- 2. Component 1: Tendency DVA (posterior vs prior) ----
    batter_quality  = batter_p * p_strike + (1.0 - batter_p) * (1.0 - p_strike)
    league_quality  = league_p * p_strike + (1.0 - league_p) * (1.0 - p_strike)
    df['tendency_dva'] = batter_quality - league_quality
    
    # ---- 3. Component 2: Alignment DVA (symmetric strike alignment quality) ----
    actual_quality = is_swing * p_strike + (1.0 - is_swing) * (1.0 - p_strike)
    xgb_league_quality = xgb_p * p_strike + (1.0 - xgb_p) * (1.0 - p_strike)
    df['alignment_dva'] = actual_quality - xgb_league_quality
    df['alignment_dva_posthoc'] = df['alignment_dva']
    df['actual_dva']   = df['alignment_dva']  # Retain actual_dva as alias
    
    # Optional Decision-Time Baseline Evaluation
    if league_model_perceptual is not None:
        xgb_p_perc = league_model_perceptual.predict(df)
        df['xgb_p_swing_perceptual'] = xgb_p_perc
        df['deception_swing_oe'] = xgb_p_perc - xgb_p
        
        ev_league_perc = xgb_p_perc * ev_swing + (1.0 - xgb_p_perc) * ev_take
        df['ev_dva_perceptual'] = ev_observed - ev_league_perc
        
        xgb_qual_perc = xgb_p_perc * p_strike + (1.0 - xgb_p_perc) * (1.0 - p_strike)
        df['alignment_dva_perceptual'] = actual_quality - xgb_qual_perc

    # ---- 4. Blended & Primary DVA ----
    # Retain alignment_dva_raw for diagnostic comparison alongside ev_dva
    df['alignment_dva_raw'] = 0.5 * df['tendency_dva'] + 0.5 * df['actual_dva']
    df['dva_raw']  = df['ev_dva']
    df['dva']      = df['ev_dva'] * df['confidence']
    
    return df


def score_batters(df, league_model, bayesian_model, min_pitches=200, league_model_perceptual=None):
    """
    Produce final batter ratings using Empirical Bayes posterior mean EV-DVA
    and strike-alignment DVA.
    
    Principles:
    1. **EV-DVA**: Expected Run Value decision value added relative to baseline expectation.
       Shrinkage constant K* = sigma2_within / tau2_between is estimated directly from
       one-way random effects variance components on the pitch-level residuals.
    2. **Alignment DVA**: Retained alongside EV-DVA for direct baseline comparison of
       pure strike-zone recognition.
       
    Returns:
        DataFrame with one row per batter, sorted by swing_rating.
    """
    df = compute_dva(df, league_model, bayesian_model, league_model_perceptual=league_model_perceptual)
    
    # Estimate principled Empirical Bayes shrinkage constants K* from pitch-level residual variances
    k_ev, s2_w_ev, t2_b_ev = estimate_eb_shrinkage(df, residual_col='ev_dva', batter_col='batter')
    k_align, s2_w_al, t2_b_al = estimate_eb_shrinkage(df, residual_col='alignment_dva', batter_col='batter')
    print(f"[Empirical Bayes] EV-DVA variance components: sigma2_within={s2_w_ev:.6f}, tau2_between={t2_b_ev:.6f} -> K*={k_ev:.1f}")
    print(f"[Empirical Bayes] Alignment-DVA variance components: sigma2_within={s2_w_al:.6f}, tau2_between={t2_b_al:.6f} -> K*={k_align:.1f}")
    
    results = []
    for batter_id, batter_df in df.groupby('batter'):
        n = len(batter_df)
        if n < min_pitches:
            continue
        
        # Empirical Bayes Posterior Means: (N / (N + K*)) * mean_residual
        raw_ev = float(batter_df['ev_dva'].mean())
        eb_ev = float((n / (n + k_ev)) * raw_ev)
        
        raw_align = float(batter_df['alignment_dva'].mean())
        eb_align = float((n / (n + k_align)) * raw_align)
        
        total_confidence = float(batter_df['confidence'].sum())
        if total_confidence > 0:
            actual_dva   = float((batter_df['actual_dva']   * batter_df['confidence']).sum() / (total_confidence + 15.0))
            tendency_dva = float((batter_df['tendency_dva'] * batter_df['confidence']).sum() / total_confidence)
        else:
            actual_dva = tendency_dva = 0.0
            
        # Profile summary
        profile = bayesian_model.get_batter_profile(batter_id)
        if profile is not None and len(profile) > 0:
            weights  = profile['n_pitches'] * (1 - profile['shrinkage'])
            total_w  = weights.sum()
            weighted_dev = (profile['deviation'] * weights).sum() / max(total_w, 1)
            
            chase_bins = profile[profile['loc_bin'] == 'chase']
            if len(chase_bins) > 0:
                cw = chase_bins['n_pitches'] * (1 - chase_bins['shrinkage'])
                chase_dev = (chase_bins['deviation'] * cw).sum() / max(cw.sum(), 1)
            else:
                chase_dev = 0
            
            avg_confidence = (1 - profile['shrinkage']).mean()
        else:
            weighted_dev   = 0
            chase_dev      = 0
            avg_confidence = 0.0
            
        outside_zone = (batter_df['in_zone'] == 0)
        inside_zone  = (batter_df['in_zone'] == 1)
        chase_bin    = (batter_df['loc_bin'] == 'chase')
        
        rec = {
            'batter_id':             int(batter_id),
            'n_pitches':             n,
            'dva_mean':              eb_ev,
            'ev_dva':                raw_ev,
            'ev_dva_eb':             eb_ev,
            'alignment_dva':         raw_align,
            'alignment_dva_eb':      eb_align,
            'tendency_dva':          tendency_dva,
            'actual_dva':            actual_dva,
            'k_shrinkage_ev':        round(k_ev, 1),
            'k_shrinkage_align':     round(k_align, 1),
            'swing_rate':            float(batter_df['is_swing'].mean()),
            'chase_rate':            float(batter_df[outside_zone]['is_swing'].mean()) if outside_zone.sum() > 0 else 0.0,
            'restricted_chase_rate': float(batter_df[chase_bin]['is_swing'].mean()) if chase_bin.sum() > 0 else 0.0,
            'zone_swing_rate':       float(batter_df[inside_zone]['is_swing'].mean()) if inside_zone.sum() > 0 else 0.0,
            'deviation_from_league': weighted_dev,
            'chase_deviation':       chase_dev,
            'avg_confidence':        avg_confidence,
        }
        if 'deception_swing_oe' in batter_df.columns:
            rec['deception_swing_oe'] = float(batter_df['deception_swing_oe'].mean())
        if 'ev_dva_perceptual' in batter_df.columns:
            rec['ev_dva_perceptual'] = float(batter_df['ev_dva_perceptual'].mean())
        results.append(rec)
        
    lb = pd.DataFrame(results)
    if len(lb) == 0:
        return lb
        
    mu    = lb['dva_mean'].mean()
    sigma = lb['dva_mean'].std()
    if pd.isna(sigma) or sigma == 0:
        sigma = 1.0
        
    lb['z_score']      = (lb['dva_mean'] - mu) / sigma
    lb['swing_rating'] = (100 + lb['z_score'] * 10).round(1)
    lb = lb.sort_values('swing_rating', ascending=False).reset_index(drop=True)
    lb['rank'] = lb.index + 1
    
    return lb


def load_baseline_predictions_for_df(df, year=2025, data_dir=None, model_dir=None, require_oof=False):
    """
    Supplies baseline league P(Swing) for pitches in df with strict validation.
    
    1. Checks for persisted out-of-fold predictions at data/training/oof_predictions_{year}.parquet.
       - Validates that pitch keys (game_pk, at_bat_number, pitch_number) are unique (no duplicates).
       - Validates that 100% of pitches in df have corresponding predictions (no silent 0.5 fills).
       - Confirms out-of-fold provenance.
    2. If OOF predictions are not found and require_oof is False:
       - Loads trained XGBoost model from data/models/swing_model.xgb (honoring best_iteration).
       - Evaluates booster directly on df.
       - Emits clear notice that predictions are model-inferred rather than persisted OOF.
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'training')
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'models')
        
    oof_path = os.path.join(data_dir, f'oof_predictions_{year}.parquet')
    keys = ['game_pk', 'at_bat_number', 'pitch_number']
    
    # Check if all keys exist in df
    has_keys = all(k in df.columns for k in keys)
    
    if os.path.exists(oof_path) and has_keys:
        pred_df = pd.read_parquet(oof_path)
        
        # 1. Check uniqueness of pitch keys
        if pred_df[keys].duplicated().any():
            if require_oof:
                raise ValueError(f"Corrupt prediction file: Duplicate pitch keys found in {oof_path}!")
            print(f"[Notice] Duplicate pitch keys found in {oof_path}. Falling back to saved model.")
            pred_df = None
            
        # 2. Check prediction values and provenance
        if pred_df is not None:
            if 'xgb_p_swing' not in pred_df.columns:
                if require_oof:
                    raise ValueError(f"Prediction file {oof_path} missing required column 'xgb_p_swing'!")
                pred_df = None
            elif not np.all(np.isfinite(pred_df['xgb_p_swing'])) or (pred_df['xgb_p_swing'] < 0.0).any() or (pred_df['xgb_p_swing'] > 1.0).any():
                if require_oof:
                    raise ValueError(f"Prediction file {oof_path} contains non-finite or out-of-bounds [0, 1] probabilities!")
                pred_df = None
            elif 'is_oof' not in pred_df.columns or pred_df['is_oof'].isna().any() or not (pred_df['is_oof'] == True).all():
                if require_oof:
                    raise ValueError(f"Prediction file {oof_path} fails OOF provenance: 'is_oof' column missing, contains nulls, or contains False values!")
                print(f"[Notice] {oof_path} contains non-OOF predictions ('is_oof' != True). Falling back to saved booster.")
                pred_df = None
            elif 'oof_fold' not in pred_df.columns:
                if require_oof:
                    raise ValueError(f"Prediction file {oof_path} fails OOF provenance: missing required 'oof_fold' column!")
                print(f"[Notice] {oof_path} missing 'oof_fold' column. Falling back to saved booster.")
                pred_df = None
            elif pred_df['oof_fold'].isna().any() or (pred_df['oof_fold'] <= 0).any():
                if require_oof:
                    raise ValueError(f"Prediction file {oof_path} fails OOF provenance: 'oof_fold' contains non-positive or null fold IDs (fold 0 is in-sample)!")
                print(f"[Notice] {oof_path} contains in-sample fold IDs (<= 0). Falling back to saved booster.")
                pred_df = None
            elif pred_df['oof_fold'].nunique() < 2:
                if require_oof:
                    raise ValueError(f"Prediction file {oof_path} fails OOF provenance: 'oof_fold' has < 2 folds ({pred_df['oof_fold'].nunique()} found)!")
                print(f"[Notice] {oof_path} does not reflect multi-fold validation. Falling back to saved booster.")
                pred_df = None
            else:
                # Season provenance verification
                if 'season' in pred_df.columns:
                    if pred_df['season'].isna().any() or (pred_df['season'] != int(year)).any():
                        bad_seasons = pred_df['season'].dropna().unique().tolist()
                        if require_oof:
                            raise ValueError(
                                f"Prediction file {oof_path} fails season provenance: requested year {year}, "
                                f"but found season values {bad_seasons}!"
                            )
                        print(f"[Notice] {oof_path} season mismatch. Falling back to saved booster.")
                        pred_df = None

                # Batter grouping verification: all pitches for a batter must be in the exact same outer fold
                if pred_df is not None:
                    b_col_df = 'batter' if 'batter' in df.columns else ('batter_id' if 'batter_id' in df.columns else None)
                    b_col_pred = 'batter' if 'batter' in pred_df.columns else ('batter_id' if 'batter_id' in pred_df.columns else None)
                    
                    if b_col_df is not None and b_col_pred is not None:
                        # Both input data and artifact specify batter identity: verify reconciliation
                        chk = df[keys + [b_col_df]].merge(
                            pred_df[keys + [b_col_pred, 'oof_fold']], 
                            on=keys, 
                            how='inner',
                            suffixes=('_df', '_pred')
                        )
                        col_df = f"{b_col_df}_df" if b_col_df == b_col_pred else b_col_df
                        col_pred = f"{b_col_pred}_pred" if b_col_df == b_col_pred else b_col_pred
                        
                        if not (chk[col_df] == chk[col_pred]).all():
                            if require_oof:
                                raise ValueError(
                                    f"Prediction file {oof_path} fails provenance: batter IDs in artifact do not match input data!"
                                )
                            print(f"[Notice] {oof_path} batter IDs do not match input data. Falling back to saved booster.")
                            pred_df = None
                        elif chk.groupby(col_df)['oof_fold'].nunique().max() > 1:
                            if require_oof:
                                raise ValueError(
                                    f"Prediction file {oof_path} fails OOF provenance: batter grouping violated "
                                    f"(pitches for the same batter split across folds)!"
                                )
                            print(f"[Notice] {oof_path} violates batter grouping contract. Falling back to saved booster.")
                            pred_df = None
                    elif b_col_df is not None:
                        # Only input data specifies batter identity: verify against input grouping
                        chk = df[keys + [b_col_df]].merge(pred_df[keys + ['oof_fold']], on=keys, how='inner')
                        if chk.groupby(b_col_df)['oof_fold'].nunique().max() > 1:
                            if require_oof:
                                raise ValueError(
                                    f"Prediction file {oof_path} fails OOF provenance: batter grouping violated "
                                    f"(pitches for the same batter split across folds)!"
                                )
                            print(f"[Notice] {oof_path} violates batter grouping contract. Falling back to saved booster.")
                            pred_df = None
                    elif b_col_pred is not None:
                        # Only artifact specifies batter identity
                        if pred_df.groupby(b_col_pred)['oof_fold'].nunique().max() > 1:
                            if require_oof:
                                raise ValueError(
                                    f"Prediction file {oof_path} fails OOF provenance: batter grouping violated "
                                    f"(pitches for the same batter split across folds)!"
                                )
                            print(f"[Notice] {oof_path} violates batter grouping contract. Falling back to saved booster.")
                            pred_df = None
                
        # 3. Check coverage
        if pred_df is not None:
            merged = df[keys].merge(pred_df[keys + ['xgb_p_swing']], on=keys, how='left')
            if merged['xgb_p_swing'].isna().any():
                missing_count = merged['xgb_p_swing'].isna().sum()
                if require_oof:
                    raise ValueError(f"Prediction file {oof_path} is missing {missing_count:,} pitches! Incomplete OOF predictions rejected.")
                print(f"[Notice] {oof_path} is missing {missing_count:,} pitches. Falling back to saved booster.")
            else:
                df = df.copy()
                df['league_p_swing'] = merged['xgb_p_swing'].values
                print(f"Loaded verified out-of-fold baseline predictions ({len(df):,} pitches) from {oof_path}")
                return df
            
    if require_oof:
        raise FileNotFoundError(f"Verified out-of-fold predictions file not found or invalid: {oof_path}")
        
    # Model fallback: prefer year-specific model first
    year_model_path = os.path.join(model_dir, f'swing_model_{year}')
    generic_model_path = os.path.join(model_dir, 'swing_model')
    
    if os.path.exists(year_model_path + '.xgb'):
        model_path = year_model_path
    elif os.path.exists(generic_model_path + '.xgb'):
        model_path = generic_model_path
        meta_file = generic_model_path + '.meta'
        if os.path.exists(meta_file):
            meta = None
            try:
                import json
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
            except Exception:
                try:
                    import pickle
                    with open(meta_file, 'rb') as f:
                        meta = pickle.load(f)
                except Exception:
                    meta = None
            if isinstance(meta, dict):
                model_year = meta.get('year')
                if model_year is not None and model_year != year:
                    print(f"[Warning] Generic model {generic_model_path}.xgb was trained on year {model_year}, but year {year} was requested.")
                elif model_year is None:
                    print(f"[Notice] Generic model {generic_model_path}.xgb has no year metadata to verify against requested year {year}.")
    else:
        raise FileNotFoundError(
            f"Neither verified OOF predictions nor saved model found for year {year} in {model_dir} "
            f"(checked {year_model_path}.xgb and {generic_model_path}.xgb)"
        )
        
    league_model = LeagueSwingModel().load(model_path)
    df = df.copy()
    df['league_p_swing'] = league_model.predict(df)
    print(f"[Notice] Evaluated baseline P(Swing) using saved model {model_path}.xgb (best_iteration={getattr(league_model, 'best_iteration', None)})")
    return df
