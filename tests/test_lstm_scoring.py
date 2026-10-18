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
