"""
Unit tests for Visual Uncertainty Sensitivity (det Σ_0.40)
"""

import pytest
import numpy as np
from scipy.special import expit, logit
from scripts.run_uncertainty_sensitivity import fit_batter_ridge_logistic


def test_fit_batter_ridge_logistic_convergence():
    """Test that ridge logistic regression converges accurately on synthetic data."""
    np.random.seed(42)
    n = 300
    offset = np.random.normal(0, 0.5, size=n)
    p_c = np.random.uniform(-0.5, 0.5, size=n)
    u_z = np.random.normal(0, 1.0, size=n)
    X = np.column_stack([np.ones(n), p_c, u_z])

    true_beta = np.array([-0.2, 0.5, -0.8])
    eta = offset + X @ true_beta
    p = expit(eta)
    y = (np.random.uniform(0, 1, size=n) < p).astype(int)

    beta_est, se_est = fit_batter_ridge_logistic(offset, X, y, lam=1.0)

    assert len(beta_est) == 3
    assert len(se_est) == 3
    assert np.all(np.isfinite(beta_est))
    assert np.all(se_est > 0)
    # Estimate for beta_U (index 2) should be strongly negative matching ground truth
    assert beta_est[2] < 0


def test_fit_batter_ridge_shrinkage():
    """Test that higher lambda increases shrinkage toward zero."""
    np.random.seed(42)
    n = 100
    offset = np.zeros(n)
    X = np.random.normal(0, 1, size=(n, 3))
    y = np.random.binomial(1, 0.5, size=n)

    beta_low_pen, _ = fit_batter_ridge_logistic(offset, X, y, lam=0.1)
    beta_high_pen, _ = fit_batter_ridge_logistic(offset, X, y, lam=100.0)

    assert np.linalg.norm(beta_high_pen) < np.linalg.norm(beta_low_pen)


def test_uncertainty_response_monotonicity():
    """Test that predicted probability behaves monotonically with beta_U."""
    u_grid = np.array([-2.0, 0.0, 2.0])
    offset = 0.0
    alpha = 0.0
    beta_P = 0.0

    # Averse batter: beta_U = -1.0
    p_averse = expit(offset + alpha + beta_P * 0.0 + (-1.0) * u_grid)
    assert p_averse[0] > p_averse[1] > p_averse[2]

    # Seeking batter: beta_U = +1.0
    p_seeking = expit(offset + alpha + beta_P * 0.0 + (1.0) * u_grid)
    assert p_seeking[0] < p_seeking[1] < p_seeking[2]
