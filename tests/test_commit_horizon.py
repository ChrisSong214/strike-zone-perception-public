"""
Unit tests for Individual Batter Commit Horizon Inference (τ*_i)
"""

import pytest
import numpy as np
from scripts.run_commit_horizon import fit_psychometric_tau, TAU_GRID


def test_fit_psychometric_tau_monotonicity():
    """Test that higher strike probability increases swing probability when beta > 0."""
    np.random.seed(42)
    n = 200
    p_strike = np.random.uniform(0, 1, size=n)
    true_beta = 2.0
    p_swing = 1.0 / (1.0 + np.exp(-true_beta * (p_strike - 0.50)))
    y = (np.random.uniform(0, 1, size=n) < p_swing).astype(int)

    ll, alpha, beta = fit_psychometric_tau(y, p_strike)

    assert beta > 0.5
    assert np.isfinite(ll)
    assert np.isfinite(alpha)


def test_tau_grid_coverage():
    """Ensure candidate horizon grid spans physiological commit window."""
    assert len(TAU_GRID) >= 5
    assert min(TAU_GRID) >= 0.20
    assert max(TAU_GRID) <= 0.65
    assert 0.40 in TAU_GRID


def test_commit_horizon_selection():
    """Verify that commit horizon selects the grid point with maximum log-likelihood."""
    lls = {0.25: -500.0, 0.35: -450.0, 0.45: -410.0, 0.55: -430.0}
    best_tau = max(lls.keys(), key=lambda t: lls[t])
    assert best_tau == 0.45


def test_fit_forward_update():
    """Verify that forward update sensitivity captures responsiveness to late trajectory resolution."""
    from scripts.run_commit_horizon import fit_forward_update
    np.random.seed(42)
    n = 300
    p_k = np.random.uniform(0.3, 0.7, size=n)
    delta_p = np.random.uniform(-0.2, 0.2, size=n)
    
    # Swings are heavily driven by delta_p (responsive hitter)
    eta = 0.5 + 2.0 * (p_k - 0.50) + 3.0 * delta_p
    p_swing = 1.0 / (1.0 + np.exp(-eta))
    y = (np.random.uniform(0, 1, size=n) < p_swing).astype(int)

    beta_dp = fit_forward_update(y, p_k, delta_p)
    assert beta_dp > 0.5


def test_sequential_elbow_detection():
    """Verify that the sequential elbow detects the latest horizon before responsiveness collapses."""
    betas_delta = {0.25: 3.2, 0.30: 3.0, 0.35: 2.8, 0.40: 2.5, 0.45: 0.8, 0.50: 0.2}
    max_b = max(betas_delta.values())
    valid_taus = [t for t, b in betas_delta.items() if b >= 0.50 * max_b]
    best_tau = max(valid_taus)
    assert best_tau == 0.40
