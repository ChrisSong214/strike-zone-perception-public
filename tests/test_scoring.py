import numpy as np
import pandas as pd
import pytest
from src.scoring import perceptual_strike_prob, batch_perceptual_strike_prob, FastEyeScorer


def test_perceptual_strike_prob_bounds():
    prob = perceptual_strike_prob(
        plate_x=0.0, plate_z=2.5,
        phys_sigma_x=0.1, phys_sigma_z=0.1,
        sz_top=3.5, sz_bot=1.5,
        visual_sigma_ft=0.2
    )
    assert 0.0 <= prob <= 1.0
    assert prob > 0.5  # Dead-center pitch should have high strike prob


def test_perceptual_strike_prob_correlated():
    """Verify correlated bivariate normal integration in perceptual_strike_prob (resolves A18)."""
    # Corner-centered pitch: plate_x = 17/24, plate_z = 3.5, phys_sigma = 0.2, visual_sigma = 0.0
    p_ind = perceptual_strike_prob(
        plate_x=17/24, plate_z=3.5,
        phys_sigma_x=0.2, phys_sigma_z=0.2,
        sz_top=3.5, sz_bot=1.5,
        visual_sigma_ft=0.0, rho=0.0
    )
    assert np.isclose(p_ind, 0.25, atol=1e-3)
    
    # Under rho = 0.8, correlated probability is ~0.397584
    p_corr = perceptual_strike_prob(
        plate_x=17/24, plate_z=3.5,
        phys_sigma_x=0.2, phys_sigma_z=0.2,
        sz_top=3.5, sz_bot=1.5,
        visual_sigma_ft=0.0, rho=0.8
    )
    assert np.isclose(p_corr, 0.397584, atol=1e-4)


def test_batch_perceptual_strike_prob():
    df = pd.DataFrame({
        'plate_x': [0.0, 3.0],
        'plate_z': [2.5, 6.0],
        'sz_top': [3.5, 3.5],
        'sz_bot': [1.5, 1.5],
        'phys_sigma_x': [0.1, 0.1],
        'phys_sigma_z': [0.1, 0.1]
    })
    probs = batch_perceptual_strike_prob(df, visual_sigma_ft=0.2)
    assert len(probs) == 2
    assert probs[0] > probs[1]  # Middle pitch much more likely strike than wild pitch


def test_fast_eye_scorer_filtering():
    df = pd.DataFrame({
        'plate_x': [0.0, np.nan, 0.2, np.inf],
        'plate_z': [2.5, 2.5, np.nan, 2.5],
        'sz_top': [3.5, 3.5, 3.5, 3.5],
        'sz_bot': [1.5, 1.5, 1.5, 1.5],
        'phys_sigma_x': [0.1, 0.1, 0.1, 0.1],
        'phys_sigma_z': [0.1, 0.1, 0.1, 0.1],
        'is_swing': [True, False, True, False]
    })
    scorer = FastEyeScorer(df)
    assert len(scorer.pitches) == 1
    assert scorer.is_swing[0] == True


def test_fast_eye_scorer_min_obs_guard():
    df = pd.DataFrame({
        'plate_x': [np.nan] * 50,
        'plate_z': [np.nan] * 50,
        'sz_top': [3.5] * 50,
        'sz_bot': [1.5] * 50,
        'phys_sigma_x': [0.1] * 50,
        'phys_sigma_z': [0.1] * 50,
        'is_swing': [False] * 50
    })
    scorer = FastEyeScorer(df)
    assert len(scorer.pitches) == 0
    with pytest.raises(ValueError, match="Cannot fit visual sigma with fewer than"):
        scorer.fit_sigma(min_obs=10)


def test_fast_eye_scorer_fit_success():
    # 20 pitches: center swings, outside takes
    df = pd.DataFrame({
        'plate_x': [0.0] * 10 + [2.5] * 10,
        'plate_z': [2.5] * 10 + [4.5] * 10,
        'sz_top': [3.5] * 20,
        'sz_bot': [1.5] * 20,
        'phys_sigma_x': [0.05] * 20,
        'phys_sigma_z': [0.05] * 20,
        'is_swing': [True] * 10 + [False] * 10
    })
    scorer = FastEyeScorer(df)
    sigma, loss = scorer.fit_sigma(min_obs=10)
    assert np.isfinite(sigma)
    assert np.isfinite(loss)
    assert sigma > 0.0


def test_count_run_values_coverage():
    """Verify all 12 (balls, strikes) states are defined in RE24 count transition table."""
    from src.swing_model import COUNT_RUN_VALUES
    assert len(COUNT_RUN_VALUES) == 12
    for b in range(4):
        for s in range(3):
            assert (b, s) in COUNT_RUN_VALUES
            entry = COUNT_RUN_VALUES[(b, s)]
            assert entry['rv_ball'] > 0  # Ball always adds positive run expectancy for batter
            assert entry['rv_called_strike'] < 0  # Strike always reduces run expectancy
            assert entry['rv_called_strike'] < entry['rv_ball']


def test_ev_take_3_0_vs_0_2_asymmetry():
    """Verify Issue 2 resolution: on 3-0 taking an 80% strike is superior to swinging, unlike 0-2."""
    from src.swing_model import estimate_ev_take, estimate_ev_swing
    
    # 3-0 count: 80% strike
    ev_take_30 = estimate_ev_take(p_strike=[0.80], balls=[3], strikes=[0])[0]
    ev_swing_30 = estimate_ev_swing(p_strike=[0.80], balls=[3], strikes=[0])[0]
    # On 3-0, taking an 80% strike yields higher expected run value than swinging
    assert ev_take_30 > ev_swing_30
    
    # 0-2 count: 45% strike
    ev_take_02 = estimate_ev_take(p_strike=[0.45], balls=[0], strikes=[2])[0]
    ev_swing_02 = estimate_ev_swing(p_strike=[0.45], balls=[0], strikes=[2])[0]
    # On 0-2, taking a 45% strike is catastrophic (high probability of strikeout)
    assert ev_take_02 < ev_swing_02


def test_estimate_eb_shrinkage():
    """Verify Empirical Bayes variance components estimation."""
    from src.swing_model import estimate_eb_shrinkage
    np.random.seed(42)
    # 20 batters, 100 pitches each
    batters = []
    residuals = []
    true_tau = 0.05
    true_sigma = 0.20
    
    for b in range(20):
        u_b = np.random.normal(0, true_tau)
        for _ in range(100):
            batters.append(b)
            residuals.append(u_b + np.random.normal(0, true_sigma))
            
    test_df = pd.DataFrame({'batter': batters, 'ev_dva': residuals})
    k_star, s2_w, t2_b = estimate_eb_shrinkage(test_df, residual_col='ev_dva', batter_col='batter')
    
    assert np.isfinite(k_star)
    assert k_star > 0
    assert np.isclose(s2_w, true_sigma**2, rtol=0.25)


def test_dual_baseline_feature_sets():
    """Verify dual baseline feature sets: final_location vs decision_time."""
    from src.swing_model import FINAL_LOCATION_FEATURES, DECISION_TIME_FEATURES, LeagueSwingModel

    # Decision time features must NOT contain final location features
    final_location_specific = ['plate_x', 'plate_z', 'loc_x_norm', 'loc_z_norm', 'dist_from_center', 'in_zone']
    for f in final_location_specific:
        assert f in FINAL_LOCATION_FEATURES
        assert f not in DECISION_TIME_FEATURES

    # Decision time features must contain early LSTM perceptual features
    early_features = ['lstm_mu_x', 'lstm_mu_z', 'lstm_sigma_x', 'lstm_sigma_z', 'lstm_rho', 'lstm_p_strike']
    for f in early_features:
        assert f in DECISION_TIME_FEATURES

    # Models instantiate with correct mode
    lm_posthoc = LeagueSwingModel(feature_mode='final_location')
    assert lm_posthoc.feature_mode == 'final_location'

    lm_perc = LeagueSwingModel(feature_mode='decision_time')
    assert lm_perc.feature_mode == 'decision_time'

