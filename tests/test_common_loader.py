import pytest
import pandas as pd
import numpy as np
from analysis.common_loader import (
    load_all_swing_ratings,
    validate_schema,
    REQUIRED_COLUMNS
)


def test_validate_schema_success():
    """Verify schema validation with all required columns."""
    df = pd.DataFrame({
        'batter_id': ['123', '456'],
        'season': ['2024', '2025'],
        'n_pitches': [500, 600],
        'swing_rating': [105.0, 98.5],
        'name': ['Player A', 'Player B']
    })
    validated = validate_schema(df, source_name='test')
    assert 'player_name' in validated.columns
    assert validated['batter_id'].dtype == int or validated['batter_id'].dtype == np.int64
    assert validated['season'].dtype == int or validated['season'].dtype == np.int64
    assert validated['player_name'].iloc[0] == 'Player A'


def test_validate_schema_warning_on_missing():
    """Verify warning issued when required columns are missing."""
    df = pd.DataFrame({
        'batter_id': [123],
        # Missing season, n_pitches, swing_rating
    })
    with pytest.warns(UserWarning, match="Missing required columns"):
        validate_schema(df, source_name='test_missing')


def test_load_all_swing_ratings_csv_source():
    """Verify loading from canonical CSV files in data/leaderboards/."""
    df = load_all_swing_ratings(min_pitches=300, source='csv')
    assert len(df) > 0
    assert 'batter_id' in df.columns
    assert 'season' in df.columns
    assert 'swing_rating' in df.columns
    assert (df['n_pitches'] >= 300).all()
    # Confirm no duplicate (batter_id, season) pairs
    assert not df.duplicated(subset=['batter_id', 'season']).any()


def test_load_all_swing_ratings_invalid_source():
    """Verify that an invalid source parameter raises ValueError."""
    with pytest.raises(ValueError, match="Invalid source 'invalid_src'"):
        load_all_swing_ratings(source='invalid_src')
