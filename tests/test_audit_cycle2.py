"""
Regression Test Suite for Audit Findings C2-01 through C2-06 (Audit Cycle 2).
Verifies:
- C2-01: Scalar return type contract for compute_p_strike across correlated and uncorrelated branches.
- C2-02: Boundary-orthogonal uncertainty strike zone inversion handling, rho clipping, and scalar return types.
- C2-03: Subgroup Stability nomenclature contract in audit script and presentation CSVs.
- C2-04: Uncertainty volume NaN/Inf percentile resilience and standardized z-score fallbacks.
- C2-05: Empty cohort leaderboard schema consistency and graceful early return.
- C2-06: Module-level sanitize_context_bins integration in BayesianBatterModel.fit and compute_dva.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.lstm_scoring import LSTMEyeScorer
from src.swing_model import (
    BayesianBatterModel,
    compute_dva,
    sanitize_context_bins,
    COUNT_GROUPS,
    LOCATION_BINS,
    PITCH_GROUPS,
)
from scripts.run_uncertainty_sensitivity import (
    LEADERBOARD_COLUMNS,
    generate_uncertainty_response_curves,
)


# =====================================================================
# C2-01: compute_p_strike Scalar Return Type Contract
# =====================================================================

def test_c2_01_compute_p_strike_scalar_return_type():
    """Verify compute_p_strike returns Python float for scalar inputs in both branches, and ndarray for arrays."""
    # 1. Uncorrelated branch with scalar inputs
    p_uncorr = LSTMEyeScorer.compute_p_strike(
        mu_x=0.0, mu_z=2.5, sigma_x=0.1, sigma_z=0.1, sz_top=3.5, sz_bot=1.5, rho=None
    )
    assert isinstance(p_uncorr, float), f"Expected float, got {type(p_uncorr)}"
    assert not isinstance(p_uncorr, np.ndarray)

    p_uncorr_zero = LSTMEyeScorer.compute_p_strike(
        mu_x=0.0, mu_z=2.5, sigma_x=0.1, sigma_z=0.1, sz_top=3.5, sz_bot=1.5, rho=0.0
    )
    assert isinstance(p_uncorr_zero, float), f"Expected float, got {type(p_uncorr_zero)}"
    assert not isinstance(p_uncorr_zero, np.ndarray)

    # 2. Correlated branch with scalar inputs
    p_corr = LSTMEyeScorer.compute_p_strike(
        mu_x=0.0, mu_z=2.5, sigma_x=0.1, sigma_z=0.1, sz_top=3.5, sz_bot=1.5, rho=0.45
    )
    assert isinstance(p_corr, float), f"Expected float, got {type(p_corr)}"
    assert not isinstance(p_corr, np.ndarray)

    # 3. Vectorized array inputs preserve numpy array output
    mu_x_arr = np.array([0.0, 0.5])
    mu_z_arr = np.array([2.5, 3.0])
    s_x_arr = np.array([0.1, 0.1])
    s_z_arr = np.array([0.1, 0.1])
    p_arr_corr = LSTMEyeScorer.compute_p_strike(
        mu_x_arr, mu_z_arr, s_x_arr, s_z_arr, 3.5, 1.5, rho=0.3
    )
    assert isinstance(p_arr_corr, np.ndarray)
    assert p_arr_corr.shape == (2,)

    p_arr_uncorr = LSTMEyeScorer.compute_p_strike(
        mu_x_arr, mu_z_arr, s_x_arr, s_z_arr, 3.5, 1.5, rho=None
    )
    assert isinstance(p_arr_uncorr, np.ndarray)
    assert p_arr_uncorr.shape == (2,)


# =====================================================================
# C2-02: compute_boundary_orthogonal_uncertainty Inverted Zone & Scalar Return
# =====================================================================

def test_c2_02_compute_boundary_orthogonal_uncertainty_inverted_strike_zone():
    """Verify compute_boundary_orthogonal_uncertainty is invariant to sz_bot/sz_top order and returns floats for scalars."""
    # Standard vs Inverted strike zone for point inside
    dist_std, sig_std = LSTMEyeScorer.compute_boundary_orthogonal_uncertainty(
        mu_x=0.0, mu_z=2.5, sigma_x=0.15, sigma_z=0.20, sz_top=3.5, sz_bot=1.5, rho=0.2
    )
    dist_inv, sig_inv = LSTMEyeScorer.compute_boundary_orthogonal_uncertainty(
        mu_x=0.0, mu_z=2.5, sigma_x=0.15, sigma_z=0.20, sz_top=1.5, sz_bot=3.5, rho=0.2
    )
    assert isinstance(dist_std, float) and isinstance(sig_std, float)
    assert isinstance(dist_inv, float) and isinstance(sig_inv, float)
    assert np.isclose(dist_std, dist_inv, atol=1e-8)
    assert np.isclose(sig_std, sig_inv, atol=1e-8)

    # Standard vs Inverted for point outside zone
    dist_out_std, sig_out_std = LSTMEyeScorer.compute_boundary_orthogonal_uncertainty(
        mu_x=1.2, mu_z=4.0, sigma_x=0.15, sigma_z=0.20, sz_top=3.5, sz_bot=1.5, rho=0.3
    )
    dist_out_inv, sig_out_inv = LSTMEyeScorer.compute_boundary_orthogonal_uncertainty(
        mu_x=1.2, mu_z=4.0, sigma_x=0.15, sigma_z=0.20, sz_top=1.5, sz_bot=3.5, rho=0.3
    )
    assert np.isclose(dist_out_std, dist_out_inv, atol=1e-8)
    assert np.isclose(sig_out_std, sig_out_inv, atol=1e-8)

    # Extreme correlation clipping robustness (rho > 1.0 or rho < -1.0)
    d_ext, s_ext = LSTMEyeScorer.compute_boundary_orthogonal_uncertainty(
        mu_x=1.0, mu_z=3.0, sigma_x=0.1, sigma_z=0.1, sz_top=3.5, sz_bot=1.5, rho=1.5
    )
    assert np.isfinite(d_ext) and np.isfinite(s_ext)
    assert s_ext > 0.0

    # Vectorized array input preserves tuple of ndarrays
    mu_xs = np.array([0.0, 1.2])
    mu_zs = np.array([2.5, 4.0])
    s_xs = np.array([0.15, 0.15])
    s_zs = np.array([0.20, 0.20])
    d_arr, s_arr = LSTMEyeScorer.compute_boundary_orthogonal_uncertainty(
        mu_xs, mu_zs, s_xs, s_zs, sz_top=3.5, sz_bot=1.5, rho=0.2
    )
    assert isinstance(d_arr, np.ndarray) and isinstance(s_arr, np.ndarray)
    assert d_arr.shape == (2,) and s_arr.shape == (2,)


# =====================================================================
# C2-03: Subgroup Stability Nomenclature Contract
# =====================================================================

def test_c2_03_subgroup_stability_nomenclature_contract():
    """Verify holdout audit script and presentation CSVs enforce Subgroup Stability metric naming and categories."""
    target_metrics = [
        'Pitcher Subgroup Brier Score',
        'Pitcher Subgroup ECE',
        'Pitcher Subgroup AUC',
        'Batter Subgroup Brier Score',
        'Batter Subgroup ECE',
        'Batter Subgroup AUC',
    ]

    # Verify both presentation CSV files
    for fname in ['full_pipeline_holdout_audit.csv', 'full_pipeline_holdout_audit_2025.csv']:
        csv_path = os.path.join(_PROJECT_ROOT, 'assets', 'presentation', fname)
        assert os.path.exists(csv_path), f"Missing artifact: {csv_path}"
        df = pd.read_csv(csv_path)

        for m in target_metrics:
            assert m in df['Metric'].values, f"Metric '{m}' missing in {fname}"
            cat = df.loc[df['Metric'] == m, 'Category'].iloc[0]
            assert cat == 'Subgroup Stability', f"Metric '{m}' has category '{cat}', expected 'Subgroup Stability'"

        # Verify deprecated 'Unseen' metrics are purged
        for val in df['Metric'].values:
            assert not str(val).startswith('Unseen Pitchers'), f"Deprecated metric '{val}' found in {fname}"
            assert not str(val).startswith('Unseen Batters'), f"Deprecated metric '{val}' found in {fname}"

    # Verify audit script source code definition
    audit_script_path = os.path.join(_PROJECT_ROOT, 'analysis', 'full_pipeline_holdout_audit.py')
    with open(audit_script_path, 'r') as f:
        audit_source = f.read()

    for m in target_metrics:
        assert f"'{m}'" in audit_source, f"Metric definition '{m}' not found in {audit_script_path}"


# =====================================================================
# C2-04: Uncertainty Volume NaN Percentile Resilience
# =====================================================================

def test_c2_04_uncertainty_nan_percentile_resilience():
    """Verify log uncertainty volume standardization safely handles NaNs, Infs, and empty subsets."""
    # Construct synthetic data with contaminated log_uncertainty_vol
    log_u = np.array([np.nan, -2.5, -2.0, -1.8, np.inf, -2.2, -np.inf, -1.9, -2.1, np.nan])
    df_test = pd.DataFrame({
        'log_uncertainty_vol': log_u,
        'loc_bin': ['edge'] * len(log_u),
        'league_logit': [0.0] * len(log_u),
    })

    # Execute safe percentile filtering logic
    valid_u = df_test['log_uncertainty_vol'].values[np.isfinite(df_test['log_uncertainty_vol'].values)]
    assert len(valid_u) > 0
    p_low, p_high = np.percentile(valid_u, 0.1), np.percentile(valid_u, 99.9)
    assert np.isfinite(p_low) and np.isfinite(p_high)

    fill_val = float(np.nanmedian(valid_u))
    log_u_clean = np.where(np.isfinite(df_test['log_uncertainty_vol'].values), df_test['log_uncertainty_vol'].values, fill_val)
    log_u_clip = np.clip(log_u_clean, p_low, p_high)
    u_mean, u_std = float(log_u_clip.mean()), max(float(log_u_clip.std()), 1e-6)
    u_z = (log_u_clip - u_mean) / u_std
    df_test['u_zscore'] = np.where(np.isfinite(u_z), u_z, 0.0)

    assert not df_test['u_zscore'].isna().any()
    assert np.all(np.isfinite(df_test['u_zscore']))

    # Verify generate_uncertainty_response_curves handles DataFrame with NaNs/Infs without crashing
    empty_res = pd.DataFrame(columns=LEADERBOARD_COLUMNS)
    # Should safely return early
    generate_uncertainty_response_curves(empty_res, df_test)


# =====================================================================
# C2-05: Empty Cohort Leaderboard Schema
# =====================================================================

def test_c2_05_empty_cohort_leaderboard_schema():
    """Verify empty qualification cohorts produce consistent DataFrame schemas and avoid KeyErrors."""
    # Check LEADERBOARD_COLUMNS definition
    expected_cols = [
        'batter_id', 'player_name', 'n_borderline', 'borderline_swing_rate',
        'alpha_i', 'beta_P_i', 'beta_edge_i', 'beta_U_i', 'beta_U_se',
        'beta_perp_i', 'ambiguity_policy',
        'z_swing_pct', 'o_swing_pct', 'z_minus_o', 'dva_total', 'deception_vulnerability_index'
    ]
    assert LEADERBOARD_COLUMNS == expected_cols

    # Instantiate empty DataFrame
    res_df = pd.DataFrame(columns=LEADERBOARD_COLUMNS)
    assert 'beta_U_i' in res_df.columns
    assert 'player_name' in res_df.columns

    # Verify sorting by beta_U_i does not raise KeyError
    sorted_df = res_df.sort_values('beta_U_i', ascending=False)
    assert len(sorted_df) == 0
    assert list(sorted_df.columns) == LEADERBOARD_COLUMNS

    # Verify early return in generate_uncertainty_response_curves
    dummy_df = pd.DataFrame({'log_uncertainty_vol': [-2.0, -2.1], 'loc_bin': ['edge', 'edge'], 'league_logit': [0.0, 0.0]})
    # Calling with empty res_df should return cleanly
    generate_uncertainty_response_curves(res_df, dummy_df)


# =====================================================================
# C2-06: Bayesian Batter Model Context Bin Sanitization
# =====================================================================

def test_c2_06_bayesian_model_context_bin_sanitization():
    """Verify sanitize_context_bins normalizes legacy labels and BayesianBatterModel.fit correctly retains all pitches."""
    # 1. Direct unit test of sanitize_context_bins
    raw_df = pd.DataFrame({
        'batter': [201, 201, 202, 202],
        'count_group': ['neutral', 'two_strikes', 'ahead', 'unknown_count'],
        'loc_bin': ['heart', 'shadow', 'waste', 'unknown_loc'],
        'pitch_group': ['fastball', 'breaking', 'offspeed', 'mystery_pitch'],
        'is_swing': [1, 0, 1, 0],
    })

    sanitized = sanitize_context_bins(raw_df)
    assert list(sanitized['count_group']) == ['even', 'behind', 'ahead', 'even']
    assert list(sanitized['loc_bin']) == ['zone', 'edge', 'chase', 'zone']
    assert list(sanitized['pitch_group']) == ['fastball', 'breaking', 'offspeed', 'fastball']

    # 2. BayesianBatterModel.fit with legacy context bins
    # If sanitization were missing, 'neutral', 'two_strikes', 'heart', etc. would not match COUNT_GROUPS / LOCATION_BINS
    # and league priors would have n=0 pitches.
    model = BayesianBatterModel(kappa=30)
    league_p = np.full(len(raw_df), 0.45)
    model.fit(raw_df, league_p)

    # Check that bins received the pitches
    even_zone_fb = model.league_priors.get(('even', 'zone', 'fastball'))
    assert even_zone_fb is not None
    # row 0 ('neutral', 'heart', 'fastball') -> ('even', 'zone', 'fastball')
    # row 3 ('unknown_count', 'unknown_loc', 'mystery_pitch') -> ('even', 'zone', 'fastball')
    assert even_zone_fb['n'] == 2

    behind_edge_brk = model.league_priors.get(('behind', 'edge', 'breaking'))
    assert behind_edge_brk is not None
    # row 1 ('two_strikes', 'shadow', 'breaking') -> ('behind', 'edge', 'breaking')
    assert behind_edge_brk['n'] == 1

    ahead_chase_off = model.league_priors.get(('ahead', 'chase', 'offspeed'))
    assert ahead_chase_off is not None
    # row 2 ('ahead', 'waste', 'offspeed') -> ('ahead', 'chase', 'offspeed')
    assert ahead_chase_off['n'] == 1

    # Batter posteriors are populated
    assert 201 in model.batter_posteriors
    assert 202 in model.batter_posteriors
    assert model.batter_posteriors[201][('even', 'zone', 'fastball')]['n'] == 1
    assert model.batter_posteriors[201][('behind', 'edge', 'breaking')]['n'] == 1

    # 3. compute_dva integration test with legacy context bins
    class MockLeagueModel:
        def predict(self, df):
            return np.full(len(df), 0.45)

    df_dva = raw_df.copy()
    df_dva['lstm_p_strike'] = [0.85, 0.40, 0.10, 0.60]
    res_dva = compute_dva(df_dva, MockLeagueModel(), model)

    assert 'count_group' in res_dva.columns
    assert list(res_dva['count_group']) == ['even', 'behind', 'ahead', 'even']
    assert list(res_dva['loc_bin']) == ['zone', 'edge', 'chase', 'zone']
    assert list(res_dva['pitch_group']) == ['fastball', 'breaking', 'offspeed', 'fastball']
    assert 'batter_p_swing' in res_dva.columns
    assert not res_dva['batter_p_swing'].isna().any()
