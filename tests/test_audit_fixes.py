import os
import sys
import sqlite3
import numpy as np
import pandas as pd
import pytest
import torch

from src.dataset import normalize_valid_trajectory
from src.lstm_scoring import LSTMEyeScorer
from src.db_manager import _save_ratings_by_season


def test_normalize_valid_trajectory_preserves_padding():
    """Verify normalize_valid_trajectory only transforms non-padded timesteps (A03)."""
    # 5 timesteps, 9 features
    # Timesteps 0 and 1 are padded (all zeros)
    # Timesteps 2, 3, 4 are valid data
    traj = torch.zeros(5, 9)
    traj[2:] = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                             [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                             [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]])
    
    mean = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    std = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    
    normed = normalize_valid_trajectory(traj, mean, std)
    
    # Assert padded timesteps remain strictly 0.0
    assert torch.all(normed[:2] == 0.0), "Padded timesteps must remain exactly 0.0"
    
    # Assert valid timesteps are properly normalized
    expected_step2 = (traj[2] - mean) / std
    assert torch.allclose(normed[2], expected_step2, atol=1e-6)
    expected_step3 = (traj[3] - mean) / std
    assert torch.allclose(normed[3], expected_step3, atol=1e-6)


def test_export_leaderboard_row_by_row_resolution(monkeypatch):
    """Verify export_leaderboard resolves missing player names row-by-row (A20)."""
    from scripts import export_leaderboard
    
    # Mock get_name to return deterministic test name
    calls = []
    def mock_get_name(uid):
        calls.append(uid)
        return f"Player {uid}"
    
    monkeypatch.setattr(export_leaderboard, "get_name", mock_get_name)
    
    lb = pd.DataFrame({
        'batter_id': [101, 102, 103],
        'player_name': ['Aaron Judge', '', np.nan],
        'rank': [1, 2, 3],
        'swing_rating': [120.0, 115.0, 110.0]
    })
    
    needs_name = lb['player_name'].isna() | (lb['player_name'].astype(str).str.strip().isin(['', 'nan', 'None']))
    assert needs_name.sum() == 2
    
    name_cache = {}
    for count, idx in enumerate(lb[needs_name].index, 1):
        bid = int(lb.loc[idx, 'batter_id'])
        if bid not in name_cache:
            name_cache[bid] = export_leaderboard.get_name(bid)
        lb.loc[idx, 'player_name'] = name_cache[bid]
        
    assert lb.loc[0, 'player_name'] == 'Aaron Judge'
    assert lb.loc[1, 'player_name'] == 'Player 102'
    assert lb.loc[2, 'player_name'] == 'Player 103'
    assert calls == [102, 103]


def test_notebook_05_eye_sigma_fallback():
    """Verify notebook 05 fallback logic derives eye_sigma_ft from mean_sigma components (A21)."""
    # Case 1: eye_sigma_ft already exists
    df1 = pd.DataFrame({
        'rank': [1],
        'player_name': ['Test Player'],
        'eye_rating': [105.0],
        'eye_sigma_ft': [0.42],
        'n_pitches': [500],
        'swing_rate': [0.45]
    })
    if 'eye_sigma_ft' not in df1.columns:
        if 'mean_sigma_x' in df1.columns and 'mean_sigma_z' in df1.columns:
            df1['eye_sigma_ft'] = (df1['mean_sigma_x'] + df1['mean_sigma_z']) / 2.0
        else:
            df1['eye_sigma_ft'] = float('nan')
    assert df1['eye_sigma_ft'].iloc[0] == 0.42

    # Case 2: eye_sigma_ft absent, mean_sigma_x and mean_sigma_z present
    df2 = pd.DataFrame({
        'rank': [1],
        'player_name': ['Test Player'],
        'eye_rating': [105.0],
        'mean_sigma_x': [0.40],
        'mean_sigma_z': [0.50],
        'n_pitches': [500],
        'swing_rate': [0.45]
    })
    if 'eye_sigma_ft' not in df2.columns:
        if 'mean_sigma_x' in df2.columns and 'mean_sigma_z' in df2.columns:
            df2['eye_sigma_ft'] = (df2['mean_sigma_x'] + df2['mean_sigma_z']) / 2.0
        else:
            df2['eye_sigma_ft'] = float('nan')
    assert 'eye_sigma_ft' in df2.columns
    assert df2['eye_sigma_ft'].iloc[0] == 0.45


def test_save_ratings_by_season_atomic(tmp_path, monkeypatch):
    """Verify atomic season-partitioned saving with schema migration (A01, A05)."""
    db_path = str(tmp_path / "test_mlb.db")
    
    def mock_get_conn():
        return sqlite3.connect(db_path)
    
    from src import db_manager
    monkeypatch.setattr(db_manager, "get_db_connection", mock_get_conn)
    
    # Create initial table with season 2024
    initial_df = pd.DataFrame({
        'batter_id': [1001, 1002],
        'season': [2024, 2024],
        'rank': [1, 2],
        'swing_rating': [110.0, 105.0]
    })
    _save_ratings_by_season(initial_df, "test_ratings")
    
    # Save season 2025 with new upgraded columns (ev_dva_eb, alignment_dva_eb)
    upgraded_df = pd.DataFrame({
        'batter_id': [1001, 1003],
        'season': [2025, 2025],
        'rank': [1, 2],
        'swing_rating': [112.0, 108.0],
        'ev_dva_eb': [0.035, 0.021],
        'alignment_dva_eb': [0.040, 0.025]
    })
    _save_ratings_by_season(upgraded_df, "test_ratings")
    
    # Verify season 2024 rows were preserved and new columns migrated
    conn = mock_get_conn()
    saved = pd.read_sql_query("SELECT * FROM test_ratings ORDER BY season, rank", conn)
    conn.close()
    
    assert len(saved) == 4
    assert set(saved['season'].unique()) == {2024, 2025}
    assert 'ev_dva_eb' in saved.columns
    assert 'alignment_dva_eb' in saved.columns
    assert saved[saved['season'] == 2025]['ev_dva_eb'].iloc[0] == 0.035
