import os
import shutil
import tempfile
import json
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from src.swing_model import load_baseline_predictions_for_df, LeagueSwingModel


@pytest.fixture
def temp_dirs():
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)


def test_oof_provenance_checks(temp_dirs):
    df_pitches = pd.DataFrame({
        'game_pk': [1, 2],
        'at_bat_number': [1, 1],
        'pitch_number': [1, 2],
        'batter': [100, 100]
    })
    
    # 1. Missing oof_fold
    p1 = pd.DataFrame({
        'game_pk': [1, 2],
        'at_bat_number': [1, 1],
        'pitch_number': [1, 2],
        'xgb_p_swing': [0.5, 0.6],
        'is_oof': [True, True]
    })
    p1.to_parquet(os.path.join(temp_dirs, 'oof_predictions_2025.parquet'))
    with pytest.raises(ValueError, match="missing required 'oof_fold' column"):
        load_baseline_predictions_for_df(df_pitches, year=2025, data_dir=temp_dirs, model_dir=temp_dirs, require_oof=True)

    # 2. In-sample fold 0
    p2 = pd.DataFrame({
        'game_pk': [1, 2],
        'at_bat_number': [1, 1],
        'pitch_number': [1, 2],
        'xgb_p_swing': [0.5, 0.6],
        'is_oof': [True, True],
        'oof_fold': [0, 1]
    })
    p2.to_parquet(os.path.join(temp_dirs, 'oof_predictions_2025.parquet'))
    with pytest.raises(ValueError, match="non-positive or null fold IDs"):
        load_baseline_predictions_for_df(df_pitches, year=2025, data_dir=temp_dirs, model_dir=temp_dirs, require_oof=True)

    # 3. is_oof with None
    p3 = pd.DataFrame({
        'game_pk': [1, 2],
        'at_bat_number': [1, 1],
        'pitch_number': [1, 2],
        'xgb_p_swing': [0.5, 0.6],
        'is_oof': [True, None],
        'oof_fold': [1, 2]
    })
    p3.to_parquet(os.path.join(temp_dirs, 'oof_predictions_2025.parquet'))
    with pytest.raises(ValueError, match="column missing, contains nulls, or contains False"):
        load_baseline_predictions_for_df(df_pitches, year=2025, data_dir=temp_dirs, model_dir=temp_dirs, require_oof=True)

    # 4. Batter split across folds
    p4 = pd.DataFrame({
        'game_pk': [1, 2],
        'at_bat_number': [1, 1],
        'pitch_number': [1, 2],
        'batter': [100, 100],
        'xgb_p_swing': [0.5, 0.6],
        'is_oof': [True, True],
        'oof_fold': [1, 2]
    })
    p4.to_parquet(os.path.join(temp_dirs, 'oof_predictions_2025.parquet'))
    with pytest.raises(ValueError, match="batter grouping violated"):
        load_baseline_predictions_for_df(df_pitches, year=2025, data_dir=temp_dirs, model_dir=temp_dirs, require_oof=True)

    # 5. Mismatched batter identity reconciliation
    p5 = pd.DataFrame({
        'game_pk': [1, 2],
        'at_bat_number': [1, 1],
        'pitch_number': [1, 2],
        'batter': [100, 200],  # df has [100, 100]
        'xgb_p_swing': [0.5, 0.6],
        'is_oof': [True, True],
        'oof_fold': [1, 2]
    })
    p5.to_parquet(os.path.join(temp_dirs, 'oof_predictions_2025.parquet'))
    with pytest.raises(ValueError, match="batter IDs in artifact do not match input data"):
        load_baseline_predictions_for_df(df_pitches, year=2025, data_dir=temp_dirs, model_dir=temp_dirs, require_oof=True)

    # 6. Season provenance mismatch
    p6 = pd.DataFrame({
        'game_pk': [1, 2],
        'at_bat_number': [1, 1],
        'pitch_number': [1, 1],
        'batter': [100, 200],
        'season': [2024, 2024],  # Requested 2025
        'xgb_p_swing': [0.5, 0.6],
        'is_oof': [True, True],
        'oof_fold': [1, 2]
    })
    p6.to_parquet(os.path.join(temp_dirs, 'oof_predictions_2025.parquet'))
    with pytest.raises(ValueError, match="fails season provenance"):
        load_baseline_predictions_for_df(df_pitches, year=2025, data_dir=temp_dirs, model_dir=temp_dirs, require_oof=True)

    # 7. Valid OOF with season and reconciled batter matching
    df_valid = pd.DataFrame({
        'game_pk': [1, 2],
        'at_bat_number': [1, 1],
        'pitch_number': [1, 1],
        'batter': [100, 200]
    })
    p_valid = pd.DataFrame({
        'game_pk': [1, 2],
        'at_bat_number': [1, 1],
        'pitch_number': [1, 1],
        'batter': [100, 200],
        'season': [2025, 2025],
        'xgb_p_swing': [0.5, 0.6],
        'is_oof': [True, True],
        'oof_fold': [1, 2]
    })
    p_valid.to_parquet(os.path.join(temp_dirs, 'oof_predictions_2025.parquet'))
    res = load_baseline_predictions_for_df(df_valid, year=2025, data_dir=temp_dirs, model_dir=temp_dirs, require_oof=True)
    assert len(res) == 2
    assert 'league_p_swing' in res.columns


def test_generic_model_metadata_year_warning(temp_dirs, capsys):
    df_pitches = pd.DataFrame({
        'game_pk': [1],
        'at_bat_number': [1],
        'pitch_number': [1],
        'batter': [100],
        'f_0': [0.5]
    })
    
    # Train tiny dummy model
    X = np.ones((5, 1), dtype=np.float32)
    y = np.array([0, 1, 0, 1, 0])
    dtrain = xgb.DMatrix(X, label=y, feature_names=['f_0'])
    bst = xgb.train({'max_depth': 1, 'objective': 'binary:logistic'}, dtrain, num_boost_round=1)
    
    bst.save_model(os.path.join(temp_dirs, 'swing_model.xgb'))
    with open(os.path.join(temp_dirs, 'swing_model.meta'), 'w') as f:
        json.dump({'year': 2023, 'feature_names': ['f_0'], 'best_iteration': 0}, f)
        
    res = load_baseline_predictions_for_df(df_pitches, year=2024, data_dir=temp_dirs, model_dir=temp_dirs, require_oof=False)
    out = capsys.readouterr().out
    assert "[Warning] Generic model" in out
    assert "trained on year 2023, but year 2024 was requested" in out
    assert len(res) == 1
    assert 'league_p_swing' in res.columns
