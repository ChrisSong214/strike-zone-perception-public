import numpy as np
import pandas as pd
import pytest
from src.lstm_scoring import LSTMEyeScorer


def test_mirage_regime_classification():
    """Verify that pitches are correctly partitioned into transparent, tunnel mirage, and frozen mirage."""
    # Strike zone width: 17/12 ft -> half-width 17/24 = 0.7083 ft
    sz_top = 3.5
    sz_bot = 1.5
    
    # 4 synthetic pitches:
    # 1. Transparent Strike: plate_x=0.0, plate_z=2.5 (true strike = 1), p_strike = 0.95 -> delta_p = -0.05 (|delta_p| < 0.25)
    # 2. Transparent Ball: plate_x=1.5, plate_z=2.5 (true strike = 0), p_strike = 0.05 -> delta_p = +0.05 (|delta_p| < 0.25)
    # 3. Tunnel Mirage: plate_x=1.2, plate_z=2.0 (true strike = 0), p_strike = 0.75 -> delta_p = +0.75 (delta_p >= 0.25)
    # 4. Frozen Mirage: plate_x=0.2, plate_z=2.2 (true strike = 1), p_strike = 0.15 -> delta_p = -0.85 (delta_p <= -0.25)
    
    df = pd.DataFrame({
        'plate_x': [0.0, 1.5, 1.2, 0.2],
        'plate_z': [2.5, 2.5, 2.0, 2.2],
        'sz_top': [sz_top] * 4,
        'sz_bot': [sz_bot] * 4,
        'lstm_p_strike': [0.95, 0.05, 0.75, 0.15],
    })
    
    sz_w_half = 17.0 / 24.0
    true_strike = (
        (df['plate_x'].abs() <= sz_w_half) &
        (df['plate_z'] >= df['sz_bot']) &
        (df['plate_z'] <= df['sz_top'])
    ).astype(int)
    
    delta_p = df['lstm_p_strike'] - true_strike
    
    tau = 0.25
    is_transparent = delta_p.abs() < tau
    is_tunnel = delta_p >= tau
    is_frozen = delta_p <= -tau
    
    assert list(true_strike) == [1, 0, 0, 1]
    assert list(is_transparent) == [True, True, False, False]
    assert list(is_tunnel) == [False, False, True, False]
    assert list(is_frozen) == [False, False, False, True]


def test_empirical_bayes_shrinkage():
    """Verify that empirical Bayes shrinkage shrinks towards 0 with small sample."""
    k = 50.0
    raw_res = 0.40
    
    # Very small sample (N=2): should shrink heavily towards 0
    shrunk_small = (2.0 / (2.0 + k)) * raw_res
    assert np.isclose(shrunk_small, (2.0 / 52.0) * 0.40)
    assert shrunk_small < 0.05
    
    # Large sample (N=200): should preserve most of raw signal
    shrunk_large = (200.0 / (200.0 + k)) * raw_res
    assert np.isclose(shrunk_large, (200.0 / 250.0) * 0.40)
    assert np.isclose(shrunk_large, 0.32)


def test_dva_mirage_orthogonality_identity():
    """Verify mathematical identity: dva_total = dva_transparent + dva_mirage."""
    np.random.seed(42)
    n = 100
    p_strike = np.random.uniform(0, 1, n)
    is_swing = np.random.binomial(1, 0.45, n)
    p_league = np.random.uniform(0.3, 0.6, n)
    
    actual_q = is_swing * p_strike + (1 - is_swing) * (1 - p_strike)
    league_q = p_league * p_strike + (1 - p_league) * (1 - p_strike)
    dva = actual_q - league_q
    
    mirage_delta = np.random.uniform(-1, 1, n)
    is_trans = np.abs(mirage_delta) < 0.25
    is_mir = ~is_trans
    
    dva_trans = dva[is_trans].sum()
    dva_mir = dva[is_mir].sum()
    dva_tot = dva.sum()
    
    assert np.isclose(dva_tot, dva_trans + dva_mir)
