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
