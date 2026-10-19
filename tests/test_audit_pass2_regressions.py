import numpy as np
import pandas as pd
import pytest
import sqlite3

from scripts.run_uncertainty_sensitivity import compute_boundary_orthogonal_uncertainty
from scripts.backfill_physical_metrics import calculate_plate_metrics
from src.scoring import perceptual_strike_prob, batch_perceptual_strike_prob
from src.swing_model import compute_dva, score_batters, load_baseline_predictions_for_df


def test_r01_wrapper_accepts_rho():
    """R01: compute_boundary_orthogonal_uncertainty wrapper in scripts/run_uncertainty_sensitivity.py accepts rho."""
    mu_x = np.array([0.70, 0.0])
    mu_z = np.array([2.50, 2.50])
    sigma_x = np.array([0.20, 0.20])
    sigma_z = np.array([0.20, 0.20])
    sz_top = np.array([3.50, 3.50])
    sz_bot = np.array([1.50, 1.50])
    rho = np.array([0.50, -0.50])

    edge_dist, sigma_perp = compute_boundary_orthogonal_uncertainty(
        mu_x, mu_z, sigma_x, sigma_z, sz_top, sz_bot, rho=rho
    )
    assert len(edge_dist) == 2
    assert len(sigma_perp) == 2
    assert np.all(np.isfinite(edge_dist))
    assert np.all(np.isfinite(sigma_perp))


def test_r02_perceptual_oof_preserved():
    """R02: compute_dva does not overwrite pre-computed perceptual OOF predictions."""
    df = pd.DataFrame({
        'batter': [1, 1],
        'pitcher': [10, 10],
        'is_swing': [1, 0],
        'in_zone': [1, 0],
        'loc_bin': ['heart', 'chase'],
        'plate_x': [0.0, 1.2],
        'plate_z': [2.5, 3.8],
        'balls': [0, 1],
        'strikes': [0, 2],
        'lstm_p_strike': [0.90, 0.10],
        'xgb_p_swing': [0.40, 0.40],
        'xgb_p_swing_perceptual': [0.25, 0.35],
    })

    class MockModel:
        def predict(self, data):
            raise AssertionError("Standalone model should NOT be called when OOF predictions are present!")

    class MockBayes:
        batter_posteriors = {}
        def get_batter_profile(self, bid):
            return None
        def get_prior(self, loc_bin):
            return 0.50

    out = compute_dva(df, MockModel(), MockBayes(), league_model_perceptual=MockModel())
    # Verify precomputed OOF values are preserved
    assert np.allclose(out['xgb_p_swing_perceptual'].values, [0.25, 0.35])
    assert 'deception_swing_oe' in out.columns
    # deception = perceptual - objective = [0.25 - 0.40, 0.35 - 0.40] = [-0.15, -0.05]
    assert np.allclose(out['deception_swing_oe'].values, [-0.15, -0.05])


def test_r03_standardized_baseline_column_contract():
    """R03: load_baseline_predictions_for_df provides consistent columns (xgb_p_swing, league_p_swing, p_swing_league)."""
    df = pd.DataFrame({
        'game_pk': [101, 102],
        'at_bat_number': [1, 2],
        'pitch_number': [1, 1],
        'batter': [10, 20],
        'plate_x': [0.1, -0.2],
        'plate_z': [2.4, 3.1],
        'in_zone': [1, 0],
        'dist_from_center': [0.2, 0.8]
    })

    class FakeBooster:
        def predict(self, data):
            return np.array([0.45, 0.55])

    # Test that model fallback sets all three columns
    from src import swing_model
    original_cls = swing_model.LeagueSwingModel
    class FakeLeagueSwingModel:
        best_iteration = 10
        def load(self, path):
            return self
        def predict(self, data):
            return np.array([0.45, 0.55])

    swing_model.LeagueSwingModel = FakeLeagueSwingModel
    try:
        out = load_baseline_predictions_for_df(df, year=2099, require_oof=False)
        assert 'xgb_p_swing' in out.columns
        assert 'league_p_swing' in out.columns
        assert 'p_swing_league' in out.columns
        assert np.allclose(out['xgb_p_swing'], [0.45, 0.55])
        assert np.allclose(out['league_p_swing'], [0.45, 0.55])
        assert np.allclose(out['p_swing_league'], [0.45, 0.55])
    finally:
        swing_model.LeagueSwingModel = original_cls


def test_r04_calculate_plate_metrics_4tuple():
    """R04: calculate_plate_metrics always returns a 4-tuple on failure branches."""
    # Empty cloud
    res_empty = calculate_plate_metrics([], sz_top=3.5, sz_bot=1.5)
    assert len(res_empty) == 4
    assert all(np.isnan(v) for v in res_empty)

    # Non-finite endpoint
    bad_traj = [pd.DataFrame({'x': [0.0, np.nan], 'z': [2.5, 2.5]})]
    res_nan = calculate_plate_metrics(bad_traj, sz_top=3.5, sz_bot=1.5)
    assert len(res_nan) == 4
    assert all(np.isnan(v) for v in res_nan)

    # Valid cloud
    good_traj = [
        pd.DataFrame({'x': [0.0, 0.1], 'z': [2.5, 2.6]}),
        pd.DataFrame({'x': [0.0, -0.1], 'z': [2.5, 2.4]})
    ]
    p_str, sx, sz, rho = calculate_plate_metrics(good_traj, sz_top=3.5, sz_bot=1.5)
    assert np.isfinite(p_str)
    assert np.isfinite(sx)
    assert np.isfinite(sz)
    assert np.isfinite(rho)


def test_r08_additive_visual_noise_attenuates_correlation():
    """R08: Additive visual noise scales down physical correlation rho_tot = rho * (sx * sz) / (tot_sx * tot_sz)."""
    # Test case from audit: physical sx=0.2, sz=0.2, rho=0.8, visual sigma=0.2
    # tot_sx = sqrt(0.04 + 0.04) = sqrt(0.08), tot_sz = sqrt(0.08)
    # tot_rho = 0.8 * 0.04 / 0.08 = 0.40
    # For a corner-centered pitch, returns ~0.315495 (NOT 0.397583)
    plate_x = 17 / 24
    plate_z = 3.5
    prob = perceptual_strike_prob(
        plate_x=plate_x, plate_z=plate_z,
        phys_sigma_x=0.2, phys_sigma_z=0.2,
        sz_top=3.5, sz_bot=1.5,
        visual_sigma_ft=0.2,
        rho=0.8
    )
    assert np.isclose(prob, 0.315495, atol=1e-5)


def test_r10_score_batters_empty_typed_schema():
    """R10: score_batters with zero qualified batters returns typed DataFrame with batter_id."""
    df = pd.DataFrame({
        'batter': [1, 2],
        'is_swing': [1, 0],
        'in_zone': [1, 0],
        'loc_bin': ['heart', 'chase'],
        'plate_x': [0.0, 1.0],
        'plate_z': [2.5, 3.5],
        'balls': [0, 1],
        'strikes': [0, 2],
        'lstm_p_strike': [0.8, 0.2],
        'xgb_p_swing': [0.5, 0.4]
    })
    class MockLeague:
        def predict(self, d): return np.full(len(d), 0.5)
    class MockBayes:
        batter_posteriors = {}
        def get_batter_profile(self, bid): return None
        def get_prior(self, b): return 0.5

    # Request impossible pitch threshold
    lb = score_batters(df, MockLeague(), MockBayes(), min_pitches=1000)
    assert len(lb) == 0
    assert 'batter_id' in lb.columns
    assert 'dva_mean' in lb.columns
    assert 'ev_dva_eb' in lb.columns
    assert 'alignment_dva_eb' in lb.columns
