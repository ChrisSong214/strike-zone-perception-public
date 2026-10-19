"""
Regression Test Suite for Audit Findings C3-01, C3-02, and C3-03 (Audit Cycle 3).
Verifies:
- C3-01: Continuity of compute_mahalanobis_edge_dist across rho=0 and resilience to non-finite inputs.
- C3-02: Computational bounds clamping in _bvn_cdf and compute_p_strike non-finite evaluation.
- C3-03: Consistent 3-tuple return arity in detect_breakouts_and_declines across all early return paths.
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
from analysis.breakout_detection import detect_breakouts_and_declines


# =====================================================================
# C3-01: Mahalanobis Edge Dist Continuity at rho=0 & Non-Finite NaN Resilience
# =====================================================================

def test_c3_01_mahalanobis_edge_dist_continuity_at_rho_zero():
    """Verify that compute_mahalanobis_edge_dist is continuous across rho=0 for both inside and outside points."""
    sz_top = 3.5
    sz_bot = 1.5
    half_w = 17.0 / 24.0

    # 1. Point outside the strike zone (horizontal offset)
    mu_x_out = half_w + 0.3
    mu_z_out = 2.5
    sigma_x = 0.1
    sigma_z = 0.1

    dist_none = LSTMEyeScorer.compute_mahalanobis_edge_dist(
        mu_x=mu_x_out, mu_z=mu_z_out, sigma_x=sigma_x, sigma_z=sigma_z,
        sz_top=sz_top, sz_bot=sz_bot, rho=None
    )
    dist_zero = LSTMEyeScorer.compute_mahalanobis_edge_dist(
        mu_x=mu_x_out, mu_z=mu_z_out, sigma_x=sigma_x, sigma_z=sigma_z,
        sz_top=sz_top, sz_bot=sz_bot, rho=0.0
    )
    dist_pos_eps = LSTMEyeScorer.compute_mahalanobis_edge_dist(
        mu_x=mu_x_out, mu_z=mu_z_out, sigma_x=sigma_x, sigma_z=sigma_z,
        sz_top=sz_top, sz_bot=sz_bot, rho=1e-8
    )
    dist_neg_eps = LSTMEyeScorer.compute_mahalanobis_edge_dist(
        mu_x=mu_x_out, mu_z=mu_z_out, sigma_x=sigma_x, sigma_z=sigma_z,
        sz_top=sz_top, sz_bot=sz_bot, rho=-1e-8
    )

    # Expected distance: point is outside along x, inside along z (dz=0)
    # dx = (mu_x_out - half_w) / sigma_x = 0.3 / 0.1 = 3.0
    assert np.isclose(dist_none, 3.0, atol=1e-5)
    assert np.isclose(dist_zero, 3.0, atol=1e-5)
    assert np.isclose(dist_pos_eps, 3.0, atol=1e-5)
    assert np.isclose(dist_neg_eps, 3.0, atol=1e-5)
    assert np.isclose(dist_pos_eps, dist_neg_eps, atol=1e-6)

    # 2. Point inside the strike zone
    mu_x_in = 0.0
    mu_z_in = 2.5
    dist_in_none = LSTMEyeScorer.compute_mahalanobis_edge_dist(
        mu_x=mu_x_in, mu_z=mu_z_in, sigma_x=sigma_x, sigma_z=sigma_z,
        sz_top=sz_top, sz_bot=sz_bot, rho=None
    )
    dist_in_zero = LSTMEyeScorer.compute_mahalanobis_edge_dist(
        mu_x=mu_x_in, mu_z=mu_z_in, sigma_x=sigma_x, sigma_z=sigma_z,
        sz_top=sz_top, sz_bot=sz_bot, rho=0.0
    )
    dist_in_eps = LSTMEyeScorer.compute_mahalanobis_edge_dist(
        mu_x=mu_x_in, mu_z=mu_z_in, sigma_x=sigma_x, sigma_z=sigma_z,
        sz_top=sz_top, sz_bot=sz_bot, rho=1e-8
    )
    expected_in = half_w / 0.1
    assert np.isclose(dist_in_none, expected_in, atol=1e-5)
    assert np.isclose(dist_in_zero, expected_in, atol=1e-5)
    assert np.isclose(dist_in_eps, expected_in, atol=1e-5)

    # 3. Scalar return types verified
    assert isinstance(dist_none, float)
    assert isinstance(dist_zero, float)
    assert isinstance(dist_pos_eps, float)


def test_c3_01_mahalanobis_edge_dist_infinity_and_non_finite():
    """Verify that infinite coordinates result in np.inf and never produce NaN."""
    sz_top = 3.5
    sz_bot = 1.5

    # Scalar inf along one axis (with non-zero rho to exercise cross term)
    dist_inf_x = LSTMEyeScorer.compute_mahalanobis_edge_dist(
        mu_x=np.inf, mu_z=2.5, sigma_x=0.1, sigma_z=0.1,
        sz_top=sz_top, sz_bot=sz_bot, rho=0.5
    )
    assert np.isinf(dist_inf_x)
    assert dist_inf_x > 0
    assert not np.isnan(dist_inf_x)
    assert isinstance(dist_inf_x, float)

    # Scalar inf along both axes
    dist_both_inf = LSTMEyeScorer.compute_mahalanobis_edge_dist(
        mu_x=-np.inf, mu_z=np.inf, sigma_x=0.1, sigma_z=0.1,
        sz_top=sz_top, sz_bot=sz_bot, rho=-0.4
    )
    assert np.isinf(dist_both_inf)
    assert dist_both_inf > 0
    assert not np.isnan(dist_both_inf)

    # Vectorized array with mix of finite and infinite values
    mu_xs = np.array([0.0, np.inf, -np.inf, 0.5])
    mu_zs = np.array([2.5, 2.5, -np.inf, 2.5])
    sxs = np.array([0.1, 0.1, 0.1, 0.1])
    szs = np.array([0.1, 0.1, 0.1, 0.1])
    rhos = np.array([0.2, 0.5, -0.5, 0.0])

    dist_arr = LSTMEyeScorer.compute_mahalanobis_edge_dist(
        mu_xs, mu_zs, sxs, szs, sz_top, sz_bot, rho=rhos
    )
    assert isinstance(dist_arr, np.ndarray)
    assert dist_arr.shape == (4,)
    assert np.isfinite(dist_arr[0])
    assert np.isinf(dist_arr[1])
    assert np.isinf(dist_arr[2])
    assert np.isfinite(dist_arr[3])
    assert not np.any(np.isnan(dist_arr))


# =====================================================================
# C3-02: _bvn_cdf Bounds Clamping & compute_p_strike Non-Finite Resilience
# =====================================================================

def test_c3_02_bvn_cdf_non_finite_eval():
    """Verify _bvn_cdf clamps inputs safely to [-40.0, 40.0] and handles inf/-inf without NaN."""
    bvn = LSTMEyeScorer._bvn_cdf

    # Extreme bounds and infinities
    h = np.array([-np.inf, np.inf, 100.0, -100.0, 0.0])
    k = np.array([0.0, np.inf, 50.0, -50.0, 0.0])
    r = np.array([0.5, 0.5, -0.5, -0.5, 0.5])

    res = bvn(h, k, r)
    assert isinstance(res, np.ndarray)
    assert not np.any(np.isnan(res))
    assert np.all(res >= 0.0)
    assert np.all(res <= 1.0)

    # Limits verification
    assert np.isclose(res[0], 0.0, atol=1e-6)  # -inf -> 0
    assert np.isclose(res[1], 1.0, atol=1e-6)  # (+inf, +inf) -> 1
    assert np.isclose(res[2], 1.0, atol=1e-6)  # (+100, +50) -> 1
    assert np.isclose(res[3], 0.0, atol=1e-6)  # (-100, -50) -> 0


def test_c3_02_compute_p_strike_non_finite_bounds():
    """Verify compute_p_strike handles infinite/extreme pitch locations and bounds gracefully."""
    # Infinite mu_x produces 0.0 strike probability
    p_inf_scalar = LSTMEyeScorer.compute_p_strike(
        mu_x=np.inf, mu_z=2.5, sigma_x=0.1, sigma_z=0.1,
        sz_top=3.5, sz_bot=1.5, rho=0.5
    )
    assert isinstance(p_inf_scalar, float)
    assert p_inf_scalar == 0.0
    assert not np.isnan(p_inf_scalar)

    # Vectorized array with finite and infinite values
    mu_xs = np.array([0.0, np.inf, 0.0])
    mu_zs = np.array([2.5, 2.5, -np.inf])
    sxs = np.array([0.1, 0.1, 0.1])
    szs = np.array([0.1, 0.1, 0.1])
    p_arr = LSTMEyeScorer.compute_p_strike(
        mu_xs, mu_zs, sxs, szs, sz_top=3.5, sz_bot=1.5, rho=0.3
    )
    assert isinstance(p_arr, np.ndarray)
    assert p_arr.shape == (3,)
    assert p_arr[0] > 0.99
    assert p_arr[1] == 0.0
    assert p_arr[2] == 0.0
    assert not np.any(np.isnan(p_arr))


# =====================================================================
# C3-03: Consistent 3-tuple Arity in Breakout Detection
# =====================================================================

def test_c3_03_breakout_detection_consistent_tuple_arity():
    """Verify detect_breakouts_and_declines returns 3-tuple (None, None, None) on early returns."""
    # 1. Single season data: len(seasons) < 2
    df_single_season = pd.DataFrame({
        'season': [2024, 2024],
        'batter_id': [101, 102],
        'player_name': ['A', 'B'],
        'pa': [200, 200],
        'swing_rating': [0.1, -0.1],
        'woba': [0.320, 0.310],
        'bb_pct': [0.08, 0.09],
        'k_pct': [0.20, 0.22],
    })

    ret = detect_breakouts_and_declines(df_single_season, min_pa=150)
    assert isinstance(ret, tuple)
    assert len(ret) == 3
    breakouts, declines, all_pairs = ret
    assert breakouts is None
    assert declines is None
    assert all_pairs is None

    # 2. Multi-season data with no overlapping batters (not pairs)
    df_disjoint = pd.DataFrame({
        'season': [2023, 2024],
        'batter_id': [101, 102],
        'player_name': ['A', 'B'],
        'pa': [200, 200],
        'swing_rating': [0.1, -0.1],
        'woba': [0.320, 0.310],
        'bb_pct': [0.08, 0.09],
        'k_pct': [0.20, 0.22],
    })

    ret2 = detect_breakouts_and_declines(df_disjoint, min_pa=150)
    assert isinstance(ret2, tuple)
    assert len(ret2) == 3
    b2, d2, p2 = ret2
    assert b2 is None
    assert d2 is None
    assert p2 is None
