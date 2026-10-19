import os
import sys
import glob
import sqlite3
import logging
import warnings
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

REQUIRED_COLUMNS = ['batter_id', 'season', 'n_pitches', 'swing_rating']


def validate_schema(df: pd.DataFrame, source_name: str, required_cols=None) -> pd.DataFrame:
    """
    Validates and standardizes the schema of loaded swing ratings DataFrame.
    """
    if required_cols is None:
        required_cols = REQUIRED_COLUMNS
        
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        msg = f"Schema validation warning for source '{source_name}': Missing required columns {missing}."
        logger.warning(msg)
        warnings.warn(msg, UserWarning)
        
    df = df.copy()
    if 'player_name' not in df.columns and 'name' in df.columns:
        df['player_name'] = df['name']
        
    if 'batter_id' in df.columns:
        df['batter_id'] = pd.to_numeric(df['batter_id'], errors='coerce').fillna(0).astype(int)
    if 'season' in df.columns:
        df['season'] = pd.to_numeric(df['season'], errors='coerce').fillna(0).astype(int)
        
    return df


def load_all_swing_ratings(min_pitches=300, source='csv', validate=True, metric_filter=None):
    """
    Loads swing ratings across all available seasons with explicit source precedence.
    
    Parameters:
    -----------
    min_pitches : int, default=300
        Minimum pitches threshold to include a batter-season.
    source : str, default='csv'
        Data source precedence:
        - 'csv': Load exclusively from canonical CSV leaderboards in data/leaderboards/
        - 'db': Load exclusively from SQLite DB swing_ratings table in data/mlb.db
        - 'auto': Prefer CSV leaderboards; fallback to SQLite DB if no CSVs are found.
    validate : bool, default=True
        Whether to perform schema validation on loaded data.
        
    Returns:
    --------
    pd.DataFrame
        Unified DataFrame keyed on (batter_id, season).
    """
    valid_sources = ('csv', 'db', 'auto')
    if source not in valid_sources:
        raise ValueError(f"Invalid source '{source}'. Expected one of {valid_sources}.")

    records = []
    
    # 1. Canonical CSV loading
    if source in ('csv', 'auto'):
        csv_dir = os.path.join(_PROJECT_ROOT, 'data', 'leaderboards')
        csv_files = sorted(glob.glob(os.path.join(csv_dir, "*_dva_leaderboard.csv")))
        
        for csv_file in csv_files:
            try:
                c_df = pd.read_csv(csv_file)
                if 'season' not in c_df.columns:
                    base = os.path.basename(csv_file)
                    yr = int(base.split('_')[0])
                    c_df['season'] = yr
                if 'metric_definition' not in c_df.columns:
                    yr = int(c_df['season'].iloc[0]) if len(c_df) > 0 and 'season' in c_df.columns else 2025
                    c_df['metric_definition'] = 'ev_dva' if (yr == 2025 or 'ev_dva_eb' in c_df.columns) else 'alignment_dva_legacy'
                if validate:
                    c_df = validate_schema(c_df, source_name=os.path.basename(csv_file))
                records.append(c_df)
            except Exception as e:
                logger.error(f"Error loading leaderboard CSV {csv_file}: {e}")
                warnings.warn(f"Failed to read leaderboard CSV {csv_file}: {e}", RuntimeWarning)

    # 2. SQLite DB loading (either requested or fallback)
    if (source == 'db') or (source == 'auto' and not records):
        db_path = os.path.join(_PROJECT_ROOT, 'data', 'mlb.db')
        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='swing_ratings';")
                if cursor.fetchone():
                    db_df = pd.read_sql_query("SELECT * FROM swing_ratings", conn)
                    if len(db_df) > 0 and 'season' in db_df.columns:
                        if 'metric_definition' not in db_df.columns:
                            db_df['metric_definition'] = np.where(
                                (db_df['season'] == 2025) | ('ev_dva_eb' in db_df.columns),
                                'ev_dva', 'alignment_dva_legacy'
                            )
                        if validate:
                            db_df = validate_schema(db_df, source_name='sqlite:swing_ratings')
                        records.append(db_df)
                conn.close()
            except Exception as e:
                logger.error(f"Error reading SQLite table swing_ratings from {db_path}: {e}")
                warnings.warn(f"Failed to read SQLite table swing_ratings: {e}", RuntimeWarning)

    if not records:
        raise FileNotFoundError(
            f"No swing ratings found using source='{source}'. "
            f"Checked {'data/leaderboards/' if source != 'db' else ''} "
            f"{'and ' if source == 'auto' else ''}{'data/mlb.db' if source != 'csv' else ''}."
        )

    all_df = pd.concat(records, ignore_index=True)

    # Check for duplicate batter-season records and report conflicts
    dups = all_df.duplicated(subset=['batter_id', 'season'], keep=False)
    if dups.any():
        n_dups = dups.sum()
        dup_keys = all_df.loc[dups, ['batter_id', 'season']].drop_duplicates().values.tolist()
        logger.warning(
            f"Detected {n_dups} duplicate records across (batter_id, season) in loaded data. "
            f"Resolving conflicts by retaining highest pitch sample. Conflict keys: {dup_keys[:5]}..."
        )
        all_df = all_df.sort_values('n_pitches', ascending=False)
        all_df = all_df.drop_duplicates(subset=['batter_id', 'season']).reset_index(drop=True)
    else:
        all_df = all_df.reset_index(drop=True)

    if metric_filter is not None and 'metric_definition' in all_df.columns:
        all_df = all_df[all_df['metric_definition'] == metric_filter].reset_index(drop=True)

    if 'zone_swing_rate' in all_df.columns and 'chase_rate' in all_df.columns:
        all_df['z_minus_o'] = all_df['zone_swing_rate'] - all_df['chase_rate']

    if min_pitches > 0 and 'n_pitches' in all_df.columns:
        all_df = all_df[all_df['n_pitches'] >= min_pitches].reset_index(drop=True)

    return all_df


def load_batter_outcomes():
    """Loads seasonal offensive outcomes (wOBA, K%, BB%, PA)."""
    parquet_path = os.path.join(_PROJECT_ROOT, 'data', 'batter_season_outcomes.parquet')
    if os.path.exists(parquet_path):
        try:
            return pd.read_parquet(parquet_path)
        except Exception as e:
            logger.warning(f"Failed to read cached parquet {parquet_path}: {e}. Recomputing from SQLite.")

    # Compute from SQLite statcast table
    db_path = os.path.join(_PROJECT_ROOT, 'data', 'mlb.db')
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path} to compute batter outcomes.")

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query('''
        SELECT 
            batter as batter_id, 
            game_year as season,
            COUNT(*) as pa,
            SUM(CASE WHEN events LIKE "%strikeout%" THEN 1 ELSE 0 END) as k_count,
            SUM(CASE WHEN events = "walk" THEN 1 ELSE 0 END) as bb_count,
            SUM(woba_value) as sum_woba,
            SUM(woba_denom) as sum_woba_denom
        FROM statcast
        WHERE events IS NOT NULL AND events != ''
        GROUP BY batter, game_year;
        ''', conn)
    finally:
        conn.close()

    df['k_pct'] = df['k_count'] / df['pa'].clip(lower=1)
    df['bb_pct'] = df['bb_count'] / df['pa'].clip(lower=1)
    df['woba'] = df['sum_woba'] / df['sum_woba_denom'].clip(lower=1)
    
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception as e:
        logger.warning(f"Could not cache batter outcomes to {parquet_path}: {e}")

    return df


def load_merged_data(min_pitches=300, min_pa=100, source='csv'):
    """Merges DVA ratings with real seasonal offensive outcomes."""
    dva = load_all_swing_ratings(min_pitches=min_pitches, source=source)
    outcomes = load_batter_outcomes()

    outcomes = outcomes[outcomes['pa'] >= min_pa]
    merged = pd.merge(dva, outcomes, on=['batter_id', 'season'], how='inner')
    return merged

