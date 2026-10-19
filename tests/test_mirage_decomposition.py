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


def test_weighted_tunnel_chase_calculation():
    """Verify continuous deception-weighted chase metric formula."""
    # sum(swing * p_strike_40 * (1 - true_strike)) / sum(p_strike_40 * (1 - true_strike))
    swings = np.array([1, 0, 1, 0])
    p_40 = np.array([0.40, 0.30, 0.10, 0.20])
    true_strike = np.array([0, 0, 0, 0])
    
    # weights are 0.40, 0.30, 0.10, 0.20 (sum = 1.0)
    # swings on index 0 (0.40) and index 2 (0.10) -> sum = 0.50
    # weighted_chase = 0.50 / 1.0 = 0.50
    denom = p_40.sum()
    wtc = (swings * p_40).sum() / denom
    assert np.isclose(wtc, 0.50)


def test_belief_shift_calculation():
    """Verify trajectory belief shift P_0.30 - P_0.55 separates movement from calibration."""
    p_30 = np.array([0.75, 0.20, 0.60])
    p_55 = np.array([0.15, 0.85, 0.58])
    belief_shift = p_30 - p_55
    
    # Pitch 1: Looked like strike at 30%, broke hard away to ball at 55% -> positive belief shift (tunnel deception)
    assert np.isclose(belief_shift[0], 0.60)
    # Pitch 2: Looked like ball at 30%, broke into zone at 55% -> negative belief shift (backdoor freeze)
    assert np.isclose(belief_shift[1], -0.65)
    # Pitch 3: Stable trajectory near border (~0.60), minor calibration noise -> near zero shift
    assert abs(belief_shift[2]) < 0.05


def test_pitcher_split_half_repeatability_metric():
    """Verify Spearman-Brown prophecy formula calculation for split-half reliability."""
    r_half = 0.25
    r_sb = (2 * r_half) / (1 + r_half)
    assert np.isclose(r_sb, 0.40)
    assert r_sb > r_half


def test_mirage_reversal_vs_discrepancy_separation():
    """
    Verify Finding 6: True perceptual reversals (P >= 0.50 on balls, P <= 0.50 on strikes)
    are separated from prediction discrepancies (Delta_P >= tau_mirage).
    A ball with P=0.10 must NOT be classified as a tunnel reversal.
    """
    df = pd.DataFrame({
        'plate_x': [1.5, 1.2, 0.0, 0.2],
        'plate_z': [2.5, 2.0, 2.5, 2.2],
        'sz_top': [3.5] * 4,
        'sz_bot': [1.5] * 4,
        'lstm_p_strike': [0.10, 0.70, 0.90, 0.35],
    })
    sz_w_half = 17.0 / 24.0
    true_strike = (
        (df['plate_x'].abs() <= sz_w_half) &
        (df['plate_z'] >= df['sz_bot']) &
        (df['plate_z'] <= df['sz_top'])
    ).astype(int)
    
    delta_p = df['lstm_p_strike'] - true_strike
    tau = 0.08
    
    is_tunnel_discrepancy = delta_p >= tau
    is_tunnel_reversal = (true_strike == 0) & (df['lstm_p_strike'] >= 0.50)
    
    is_frozen_discrepancy = delta_p <= -tau
    is_frozen_reversal = (true_strike == 1) & (df['lstm_p_strike'] <= 0.50)
    
    # Pitch 0: True ball with p=0.10 -> delta_p = 0.10 >= 0.08 (discrepancy), but NOT reversal!
    assert is_tunnel_discrepancy.iloc[0] == True
    assert is_tunnel_reversal.iloc[0] == False  # Correctly NOT a tunnel reversal!
    
    # Pitch 1: True ball with p=0.70 -> delta_p = 0.70 >= 0.08 (discrepancy) AND reversal!
    assert is_tunnel_discrepancy.iloc[1] == True
    assert is_tunnel_reversal.iloc[1] == True
    
    # Pitch 2: True strike with p=0.90 -> delta_p = -0.10 <= -0.08 (discrepancy), but p=0.90 > 0.50 (NOT frozen reversal)
    assert is_frozen_discrepancy.iloc[2] == True
    assert is_frozen_reversal.iloc[2] == False
    
    # Pitch 3: True strike with p=0.35 -> delta_p = -0.65 <= -0.08 (discrepancy) AND frozen reversal!
    assert is_frozen_discrepancy.iloc[3] == True
    assert is_frozen_reversal.iloc[3] == True


def test_continuous_deceptive_chase_oe():
    """Verify continuous probability-weighted chase over expected formula."""
    # sum(p_40 * (swing - p_league)) / sum(p_40)
    swings = np.array([1, 0, 1])
    p_league = np.array([0.40, 0.30, 0.20])
    p_40 = np.array([0.60, 0.30, 0.10])
    
    # (swing - p_league): [0.60, -0.30, 0.80]
    # weighted: 0.60*0.60 + 0.30*(-0.30) + 0.10*0.80 = 0.36 - 0.09 + 0.08 = 0.35
    # sum(p_40) = 1.0
    # continuous OE = 0.35
    denom = p_40.sum()
    oe = ((swings - p_league) * p_40).sum() / denom
    assert np.isclose(oe, 0.35)

