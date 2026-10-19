import numpy as np
import pytest
from scipy.stats import norm, multivariate_normal
from src.lstm_scoring import LSTMEyeScorer


def test_bvn_cdf_zero_correlation():
    """Verify that BVN CDF with rho=0 equals product of univariate normal CDFs."""
    h = np.array([0.0, 1.0, -1.0, 1.96])
    k = np.array([0.0, -0.5, 1.5, 0.5])
    r = np.zeros_like(h)

    bvn_vals = LSTMEyeScorer._bvn_cdf(h, k, r)
    expected = norm.cdf(h) * norm.cdf(k)

    np.testing.assert_allclose(bvn_vals, expected, rtol=1e-5, atol=1e-5)


def test_bvn_cdf_correlated():
    """Verify that BVN CDF with correlation matches scipy.stats.multivariate_normal."""
    test_cases = [
        (0.5, 0.5, 0.5),
        (-0.5, 0.8, -0.4),
        (1.2, -0.3, 0.7),
        (0.0, 0.0, 0.3),
    ]

    for h_val, k_val, r_val in test_cases:
        calculated = LSTMEyeScorer._bvn_cdf(h_val, k_val, r_val)[0]
        cov = [[1.0, r_val], [r_val, 1.0]]
        expected = multivariate_normal.cdf([h_val, k_val], mean=[0.0, 0.0], cov=cov)
        np.testing.assert_allclose(calculated, expected, rtol=1e-4, atol=1e-4)


def test_bvn_cdf_boundary_limits():
    """Verify BVN CDF at extreme limits."""
    # Near -inf: should be 0
    val_neg = LSTMEyeScorer._bvn_cdf(-10.0, 0.0, 0.5)[0]
    assert np.isclose(val_neg, 0.0, atol=1e-6)

    # Near +inf: should approach 1
    val_pos = LSTMEyeScorer._bvn_cdf(10.0, 10.0, 0.5)[0]
    assert np.isclose(val_pos, 1.0, atol=1e-6)


def test_lstm_qualification_usable_pitch_threshold():
    """Verify that production run_leaderboard excludes batters whose surviving usable pitches after scoring are < min_pitches."""
    import unittest.mock as mock
    import pandas as pd
    import scripts.run_eye_rating_lstm as mod

    # Create mock data with 2 batters, each having 200 pitches
    df_pitches = pd.DataFrame({
        'batter': [100] * 200 + [200] * 200,
        'description': ['ball'] * 400,
        'is_swing': [False] * 400
    })

    saved_records = []
    def mock_save(df):
        saved_records.extend(df.to_dict('records'))

    class MockScorer:
        def __init__(self, *args, **kwargs):
            pass
        def score_batter(self, b_df):
            bid = b_df['batter'].iloc[0]
            if bid == 100:
                # 1 usable, 199 failed
                return {
                    'n_pitches': 1,
                    'n_failed': 199,
                    'll_per_pitch': -0.693,
                    'mean_sigma_x': 0.1,
                    'mean_sigma_z': 0.1,
                    'swing_rate': 0.5,
                    'mean_p_strike': 0.5
                }
            else:
                # 200 usable, 0 failed
                return {
                    'n_pitches': 200,
                    'n_failed': 0,
                    'll_per_pitch': -0.450,
                    'mean_sigma_x': 0.1,
                    'mean_sigma_z': 0.1,
                    'swing_rate': 0.48,
                    'mean_p_strike': 0.52
                }

    with mock.patch.object(mod, 'load_pitches', return_value=df_pitches), \
         mock.patch.object(mod, 'LSTMEyeScorer', MockScorer), \
         mock.patch.object(mod, 'save_lstm_eye_ratings', mock_save):
        lb = mod.run_leaderboard(years=[2025], min_pitches=200)

    # Verify that only batter 200 (with 200 usable pitches) reached the save recorder
    assert len(saved_records) == 1
    assert saved_records[0]['batter_id'] == 200
    assert saved_records[0]['n_pitches'] == 200


def test_compute_uncertainty_vol():
    """Verify uncertainty volume formula U = sigma_x * sigma_z * sqrt(max(1 - rho^2, 1e-6))."""
    # Uncorrelated
    u1 = LSTMEyeScorer.compute_uncertainty_vol(0.1, 0.2, 0.0)
    assert np.isclose(u1, 0.02)
    
    # Correlated rho = 0.6: sqrt(1 - 0.36) = 0.8 -> 0.1 * 0.2 * 0.8 = 0.016
    u2 = LSTMEyeScorer.compute_uncertainty_vol(0.1, 0.2, 0.6)
    assert np.isclose(u2, 0.016)
    
    # Vectorized
    sx = np.array([0.1, 0.2])
    sz = np.array([0.2, 0.3])
    r = np.array([0.0, 0.8])
    u_vec = LSTMEyeScorer.compute_uncertainty_vol(sx, sz, r)
    assert np.isclose(u_vec[0], 0.02)
    assert np.isclose(u_vec[1], 0.2 * 0.3 * 0.6)


def test_compute_mahalanobis_edge_dist():
    """Verify Mahalanobis distance to strike zone edges."""
    # Pitch right on center x=0, z=2.5, zone [-0.7083, 0.7083] x [1.5, 3.5]
    # Dist to horizontal edge = 0.7083 / 0.1 = 7.083
    # Dist to vertical edge = min(|2.5-1.5|, |2.5-3.5|) / 0.1 = 1.0 / 0.1 = 10.0
    dist = LSTMEyeScorer.compute_mahalanobis_edge_dist(
        mu_x=0.0, mu_z=2.5, sigma_x=0.1, sigma_z=0.1, sz_top=3.5, sz_bot=1.5
    )
    assert np.isclose(dist, 17.0 / 24.0 / 0.1)


