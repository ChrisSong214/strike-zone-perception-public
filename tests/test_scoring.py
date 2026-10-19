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
