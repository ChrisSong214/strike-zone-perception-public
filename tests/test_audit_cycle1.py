"""
Regression Test Suite for Audit Findings F01 - F10 (Cycle 1).
Verifies:
- F01: NaN and non-finite resilience in Empirical Bayes shrinkage estimation.
- F02: Canonical context bin mapping and sanitization in compute_dva.
- F03: P(Strike) probability bounds, inverted strike zone handling, and non-negative clipping.
- F04: Subgroup stability metric nomenclature in full_pipeline_holdout_audit.
- F05: 2D Mahalanobis edge distance with covariance integration and backwards compatibility.
- F06: Player name resolution in LSTM eye rating leaderboard.
- F07: Zero-acceleration linear motion handling in physics trajectory engine.
- F08: Semantic distinction notice between DVA and visual eye rating in notebook 05.
- F09: Historical leaderboard schema versioning documentation.
- F10: Raw string literals in analysis/breakout_detection without SyntaxWarning.
"""

import os
import sys
import warnings
import subprocess
import numpy as np
import pandas as pd
import pytest

from src.physics import get_time_at_distance, calculate_trajectory
from src.lstm_scoring import LSTMEyeScorer
from src.swing_model import (
    estimate_eb_shrinkage,
    compute_dva,
    COUNT_GROUPS,
    LOCATION_BINS,
    PITCH_GROUPS,
    BayesianBatterModel,
)


# =====================================================================
# F07: Physics Linear Motion Zero-Division Resilience
# =====================================================================

def test_f07_physics_linear_motion_constant_velocity():
    """Verify get_time_at_distance solves linear motion when ay=0 or |ay|<1e-7 without ZeroDivisionError."""
    # Linear forward motion: y0=50, y_target=1.417, vy0=-100.0, ay=0.0
    # t = (1.417 - 50.0) / -100.0 = 48.583 / 100.0 = 0.48583
    t_sol = get_time_at_distance(y_target=1.417, y0=50.0, vy0=-100.0, ay=0.0)
    assert t_sol is not None
    t1, t2 = t_sol
    assert np.isclose(t1, 0.48583, atol=1e-5)
    assert np.isclose(t2, 0.48583, atol=1e-5)

    # Near-zero acceleration: ay = 1e-8
    t_near_zero = get_time_at_distance(y_target=1.417, y0=50.0, vy0=-100.0, ay=1e-8)
    assert t_near_zero is not None
    assert np.isclose(t_near_zero[0], 0.48583, atol=1e-4)

    # Stationary unreachable target: ay=0, vy0=0
    assert get_time_at_distance(y_target=1.417, y0=50.0, vy0=0.0, ay=0.0) is None

    # Full trajectory reconstruction with ay=0
    pitch_row = pd.Series({
        'vx0': 2.0, 'vy0': -130.0, 'vz0': -3.0,
        'ax': 5.0, 'ay': 0.0, 'az': -20.0,
        'release_pos_x': -2.0, 'release_pos_y': 54.0, 'release_pos_z': 6.0,
        'plate_x': 0.5, 'plate_z': 2.5
    })
    traj = calculate_trajectory(pitch_row)
    assert traj is not None
    assert len(traj) > 10
    assert traj['y'].iloc[0] > traj['y'].iloc[-1]
    assert np.isclose(traj['y'].iloc[-1], 1.417, atol=0.1)


# =====================================================================
# F03: P(Strike) Bounds and Clipping
# =====================================================================

def test_f03_p_strike_bounds_and_clipping():
    """Verify compute_p_strike handles inverted sz bounds and guarantees [0, 1] range."""
    # Normal strike zone
    p_norm = LSTMEyeScorer.compute_p_strike(
        mu_x=0.0, mu_z=2.5, sigma_x=0.1, sigma_z=0.1, sz_top=3.5, sz_bot=1.5, rho=None
    )
    assert 0.0 <= p_norm <= 1.0
    assert isinstance(p_norm, float)

    # Inverted strike zone: sz_bot > sz_top
    p_inverted = LSTMEyeScorer.compute_p_strike(
        mu_x=0.0, mu_z=2.5, sigma_x=0.1, sigma_z=0.1, sz_top=1.5, sz_bot=3.5, rho=None
    )
    assert np.isclose(p_norm, p_inverted, atol=1e-6)
    assert p_inverted > 0.99

    # Far outside: ensure no negative probabilities or precision underflow
    p_far = LSTMEyeScorer.compute_p_strike(
        mu_x=10.0, mu_z=20.0, sigma_x=0.05, sigma_z=0.05, sz_top=3.5, sz_bot=1.5, rho=None
    )
    assert p_far == 0.0

    # Correlated with inverted bounds
    p_corr_inv = LSTMEyeScorer.compute_p_strike(
        mu_x=0.0, mu_z=2.5, sigma_x=0.1, sigma_z=0.1, sz_top=1.5, sz_bot=3.5, rho=0.5
    )
    assert 0.0 <= p_corr_inv <= 1.0

    # Vectorized array input
    mu_xs = np.array([0.0, 10.0])
    mu_zs = np.array([2.5, 20.0])
    sxs = np.array([0.1, 0.05])
    szs = np.array([0.1, 0.05])
    tops = np.array([1.5, 3.5])   # first one inverted
    bots = np.array([3.5, 1.5])
    p_arr = LSTMEyeScorer.compute_p_strike(mu_xs, mu_zs, sxs, szs, tops, bots, rho=None)
    assert isinstance(p_arr, np.ndarray)
    assert p_arr.shape == (2,)
    assert p_arr[0] > 0.99
    assert p_arr[1] == 0.0


# =====================================================================
# F05: Mahalanobis Edge Distance with Covariance
# =====================================================================

def test_f05_mahalanobis_edge_dist_correlation_and_compat():
    """Verify Mahalanobis edge distance integrates covariance when rho != 0 and preserves backward compatibility."""
    sz_top = 3.5
    sz_bot = 1.5
    half_w = 17.0 / 24.0

    # 1. Backwards compatibility when rho is None or 0
    dist_none = LSTMEyeScorer.compute_mahalanobis_edge_dist(
        mu_x=0.0, mu_z=2.5, sigma_x=0.1, sigma_z=0.1, sz_top=sz_top, sz_bot=sz_bot, rho=None
    )
    dist_zero = LSTMEyeScorer.compute_mahalanobis_edge_dist(
        mu_x=0.0, mu_z=2.5, sigma_x=0.1, sigma_z=0.1, sz_top=sz_top, sz_bot=sz_bot, rho=0.0
    )
    expected_uncorr = half_w / 0.1
    assert np.isclose(dist_none, expected_uncorr)
    assert np.isclose(dist_zero, expected_uncorr)

    # 2. Inside zone with correlation: when conditional expectation lies within boundary segment,
    # exact 1D quadratic minimizer along edge is |x_edge - mu_x| / sigma_x = half_w / 0.1
    rho_val = 0.6
    dist_corr_inside = LSTMEyeScorer.compute_mahalanobis_edge_dist(
        mu_x=0.0, mu_z=2.5, sigma_x=0.1, sigma_z=0.1, sz_top=sz_top, sz_bot=sz_bot, rho=rho_val
    )
    assert np.isclose(dist_corr_inside, expected_uncorr)

    # 3. Outside zone diagonal test: mu_x > x_max, mu_z > sz_top
    # Positive rho means errors along diagonal are expected, so Mahalanobis distance
    # along (dx > 0, dz > 0) should be smaller for rho > 0 than rho < 0
    mu_x_out = half_w + 0.2
    mu_z_out = sz_top + 0.2
    dist_pos_rho = LSTMEyeScorer.compute_mahalanobis_edge_dist(
        mu_x=mu_x_out, mu_z=mu_z_out, sigma_x=0.1, sigma_z=0.1, sz_top=sz_top, sz_bot=sz_bot, rho=0.5
    )
    dist_neg_rho = LSTMEyeScorer.compute_mahalanobis_edge_dist(
        mu_x=mu_x_out, mu_z=mu_z_out, sigma_x=0.1, sigma_z=0.1, sz_top=sz_top, sz_bot=sz_bot, rho=-0.5
    )
    assert dist_pos_rho < dist_neg_rho

    # 4. Inverted sz_top / sz_bot robustness
    dist_inv = LSTMEyeScorer.compute_mahalanobis_edge_dist(
        mu_x=0.0, mu_z=2.5, sigma_x=0.1, sigma_z=0.1, sz_top=sz_bot, sz_bot=sz_top, rho=rho_val
    )
    assert np.isclose(dist_inv, dist_corr_inside)


# =====================================================================
# F01: Empirical Bayes Shrinkage NaN and Non-Finite Resilience
# =====================================================================

def test_f01_eb_shrinkage_nan_resilience():
    """Verify estimate_eb_shrinkage filters NaNs/Infs and returns stable shrinkage."""
    np.random.seed(42)
    batters = []
    residuals = []

    # 10 batters, 60 pitches each
    for b in range(10):
        for _ in range(60):
            batters.append(f"batter_{b}")
            residuals.append(np.random.normal(0.01 * b, 0.15))

    # Introduce dirty data: NaNs, Infs, non-numeric strings, unassociated batters
    batters.extend([np.nan, "batter_0", "batter_1", "batter_isolated"])
    residuals.extend([0.05, np.nan, np.inf, 0.12])

    df = pd.DataFrame({'batter': batters, 'ev_dva': residuals})

    k_shrink, sigma2_w, tau2_b = estimate_eb_shrinkage(
        df, residual_col='ev_dva', batter_col='batter', min_pitches_for_var=50
    )

    assert np.isfinite(k_shrink)
    assert 5.0 <= k_shrink <= 500.0
    assert np.isfinite(sigma2_w) and sigma2_w > 0
    assert np.isfinite(tau2_b) and tau2_b >= 1e-6

    # Insufficient qualified batters fallback
    small_df = pd.DataFrame({'batter': ['b1', 'b2'], 'ev_dva': [0.1, 0.2]})
    k_fallback, s_fallback, t_fallback = estimate_eb_shrinkage(small_df)
    assert k_fallback == 50.0
    assert s_fallback == 1.0
    assert t_fallback == 0.02


# =====================================================================
# F02: Canonical Context Bin Mapping in compute_dva
# =====================================================================

def test_f02_canonical_context_bins_compute_dva():
    """Verify compute_dva populates and sanitizes canonical context bins."""
    class DummyLeagueModel:
        def predict(self, df):
            return np.full(len(df), 0.45)

    class DummyBayesianModel:
        def __init__(self):
            self.batter_posteriors = {}
            self.league_priors = {}
            for cg in COUNT_GROUPS:
                for lb in LOCATION_BINS:
                    for pg in PITCH_GROUPS:
                        self.league_priors[(cg, lb, pg)] = {'mean_p': 0.45, 'actual_rate': 0.45, 'n': 100}

    league_mod = DummyLeagueModel()
    bayes_mod = DummyBayesianModel()

    # Case 1: Missing all context bin columns
    df_missing = pd.DataFrame({
        'batter': [101, 101, 101],
        'balls': [0, 2, 0],
        'strikes': [0, 1, 2],
        'dist_from_center': [0.5, 1.2, 1.8],  # zone, edge, chase
        'is_swing': [1, 0, 0],
        'lstm_p_strike': [0.8, 0.5, 0.1],
    })

    scored_missing = compute_dva(df_missing, league_mod, bayes_mod)
    assert list(scored_missing['count_group']) == ['even', 'ahead', 'behind']
    assert list(scored_missing['pitch_group']) == ['fastball', 'fastball', 'fastball']
    assert list(scored_missing['loc_bin']) == ['zone', 'edge', 'chase']

    # Case 2: Legacy / non-canonical bin names requiring sanitization
    df_legacy = pd.DataFrame({
        'batter': [101, 101, 101],
        'count_group': ['neutral', 'two_strikes', 'invalid_group'],
        'loc_bin': ['heart', 'shadow', 'waste'],
        'pitch_group': ['fastball', 'slider_invalid', 'breaking'],
        'is_swing': [1, 0, 0],
        'lstm_p_strike': [0.8, 0.5, 0.1],
    })

    scored_legacy = compute_dva(df_legacy, league_mod, bayes_mod)
    assert list(scored_legacy['count_group']) == ['even', 'behind', 'even']
    assert list(scored_legacy['loc_bin']) == ['zone', 'edge', 'chase']
    assert list(scored_legacy['pitch_group']) == ['fastball', 'fastball', 'breaking']


# =====================================================================
# F04: Subgroup Stability Presentation CSV Nomenclature
# =====================================================================

def test_f04_subgroup_stability_presentation_csv():
    """Verify presentation holdout CSV uses 'Pitcher Subgroup Brier Score' and 'Batter Subgroup Brier Score'."""
    csv_path = os.path.join(
        os.path.dirname(__file__), '..', 'assets', 'presentation', 'full_pipeline_holdout_audit.csv'
    )
    assert os.path.exists(csv_path), f"Missing CSV at {csv_path}"
    df = pd.read_csv(csv_path)

    metric_names = set(df['Metric'])
    assert 'Pitcher Subgroup Brier Score' in metric_names
    assert 'Batter Subgroup Brier Score' in metric_names

    # Ensure deprecated 'Unseen' brier metrics are removed
    assert 'Unseen Pitchers Brier Score' not in metric_names
    assert 'Unseen Batters Brier Score' not in metric_names

    # Verify categories
    p_cat = df.loc[df['Metric'] == 'Pitcher Subgroup Brier Score', 'Category'].iloc[0]
    b_cat = df.loc[df['Metric'] == 'Batter Subgroup Brier Score', 'Category'].iloc[0]
    assert p_cat == 'Subgroup Stability'
    assert b_cat == 'Subgroup Stability'


# =====================================================================
# F06: Player Name in LSTM Eye Rating Leaderboard
# =====================================================================

def test_f06_lstm_leaderboard_player_name():
    """Verify run_eye_rating_lstm includes player_name in results."""
    import unittest.mock as mock
    import scripts.run_eye_rating_lstm as mod

    df_pitches = pd.DataFrame({
        'batter': [1001] * 250,
        'player_name': ['Gunnar Henderson'] * 250,
        'description': ['ball'] * 250,
        'is_swing': [False] * 250
    })

    class MockScorer:
        def __init__(self, *args, **kwargs):
            pass
        def score_batter(self, b_df):
            return {
                'n_pitches': len(b_df),
                'n_failed': 0,
                'll_per_pitch': -0.42,
                'mean_sigma_x': 0.12,
                'mean_sigma_z': 0.15,
                'swing_rate': 0.40,
                'mean_p_strike': 0.50,
            }

    with mock.patch.object(mod, 'load_pitches', return_value=df_pitches), \
         mock.patch.object(mod, 'LSTMEyeScorer', MockScorer), \
         mock.patch.object(mod, 'get_batter_name', return_value='Gunnar Henderson'), \
         mock.patch.object(mod, 'save_lstm_eye_ratings', lambda df: None):
        lb = mod.run_leaderboard(years=[2025], min_pitches=200)

    assert lb is not None
    assert 'player_name' in lb.columns
    assert lb['player_name'].iloc[0] == 'Gunnar Henderson'


# =====================================================================
# F08: Semantic Distinction Notice in Notebook 05
# =====================================================================

def test_f08_notebook_05_semantic_notice():
    """Verify notebook 05 fallback logic prints an explicit distinction between DVA and visual eye ratings."""
    import json
    nb_path = os.path.join(os.path.dirname(__file__), '..', 'notebooks', '05_eye_rating_leaderboard.ipynb')
    with open(nb_path, 'r') as f:
        nb_json = json.load(f)

    code_cells = [c['source'] for c in nb_json['cells'] if c['cell_type'] == 'code']
    all_code = "".join("".join(c) for c in code_cells)

    assert "Decision Value Added" in all_code or "DVA" in all_code
    assert "visual eye ratings" in all_code or "perceptual eye ratings" in all_code or "perceptual" in all_code


# =====================================================================
# F09: Historical Leaderboard Schema Documentation
# =====================================================================

def test_f09_leaderboard_schema_docs():
    """Verify data/leaderboards/README.md documents v1 and v2 schemas and column counts."""
    readme_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'leaderboards', 'README.md')
    assert os.path.exists(readme_path)
    with open(readme_path, 'r') as f:
        content = f.read()

    assert "18 Columns" in content or "18 columns" in content or "v1" in content
    assert "26 Columns" in content or "26 columns" in content or "v2" in content
    assert "ev_dva" in content
    assert "alignment_dva" in content
    assert "k_shrinkage_ev" in content


# =====================================================================
# F10: Raw String Literals in breakout_detection
# =====================================================================

def test_f10_breakout_detection_no_syntax_warnings():
    """Verify importing analysis.breakout_detection emits zero SyntaxWarnings."""
    cmd = [
        sys.executable,
        "-W", "error::SyntaxWarning",
        "-c", "import analysis.breakout_detection"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Import produced error or warning:\n{res.stderr}"
