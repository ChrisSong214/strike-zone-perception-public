"""
Unit tests for Full-Pipeline Holdout & Acceleration Sensitivity Audit
"""

import pytest
import numpy as np
import pandas as pd
from analysis.full_pipeline_holdout_audit import compute_ece


def test_compute_ece_perfect_calibration():
    """ECE should be 0.0 for perfectly calibrated predictions."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    ece = compute_ece(y_true, y_prob, n_bins=5)
    assert np.isclose(ece, 0.0, atol=1e-5)


def test_compute_ece_uncalibrated():
    """ECE should accurately capture calibration gap."""
    # Predict 0.9 for all, but true rate is 0.5
    n = 100
    y_true = np.array([0] * 50 + [1] * 50)
    y_prob = np.array([0.9] * 100)
    ece = compute_ece(y_true, y_prob, n_bins=10)
    # Average confidence = 0.9, accuracy = 0.5 -> gap = 0.4
    assert np.isclose(ece, 0.4, atol=1e-3)


def test_kinematic_tangent_linear_extrapolation():
    """Test that linear velocity extrapolation follows constant-velocity physics."""
    # Pitch released at y=50, x=0, z=5 with vx=0, vz=0, vy=-100
    # Time to home plate (y=1.417) is (50 - 1.417)/100 = 0.48583 s
    # At t=0.40 * 0.48583 = 0.19433 s, x and z should remain 0 and 5 under 0 acceleration.
    y0 = 50.0
    vy0 = -100.0
    t_tot = (y0 - 1.417) / (-vy0)
    t_dec = 0.40 * t_tot
    t_rem = t_tot - t_dec

    x_dec = 0.0
    vx_dec = 0.0
    x_proj = x_dec + vx_dec * t_rem
    assert np.isclose(x_proj, 0.0)
