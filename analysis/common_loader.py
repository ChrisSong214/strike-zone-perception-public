import os
import sys
import glob
import sqlite3
import pandas as pd
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def load_all_swing_ratings(min_pitches=300):
    """
    Loads swing ratings across all available seasons from SQLite DB or CSV leaderboards.
    Returns a unified DataFrame keyed on (batter_id, season).
    """
    records = []
    
    # 1. Try loading from SQLite DB swing_ratings table
    db_path = os.path.join(_PROJECT_ROOT, 'data', 'mlb.db')
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='swing_ratings';")
            if cursor.fetchone():
                db_df = pd.read_sql_query("SELECT * FROM swing_ratings", conn)
                if len(db_df) > 0 and 'season' in db_df.columns:
                    records.append(db_df)
            conn.close()
        except Exception as e:
            pass
            
    # 2. Try loading from CSV leaderboards in data/leaderboards/
    csv_dir = os.path.join(_PROJECT_ROOT, 'data', 'leaderboards')
    for csv_file in glob.glob(os.path.join(csv_dir, "*_dva_leaderboard.csv")):
        try:
            c_df = pd.read_csv(csv_file)
            if 'season' not in c_df.columns:
                # Infer season from filename
                base = os.path.basename(csv_file)
                yr = int(base.split('_')[0])
                c_df['season'] = yr
            records.append(c_df)
        except Exception:
            pass
            
    if not records:
        raise FileNotFoundError("No swing ratings found in data/mlb.db or data/leaderboards/")
        
    all_df = pd.concat(records, ignore_index=True)
    
    # Deduplicate on (batter_id, season)
    if 'player_name' not in all_df.columns and 'name' in all_df.columns:
        all_df['player_name'] = all_df['name']
        
    all_df = all_df.sort_values('n_pitches', ascending=False)
    all_df = all_df.drop_duplicates(subset=['batter_id', 'season']).reset_index(drop=True)
    
    if min_pitches > 0:
        all_df = all_df[all_df['n_pitches'] >= min_pitches].reset_index(drop=True)
        
    return all_df


def load_batter_outcomes():
    """Loads seasonal offensive outcomes (wOBA, K%, BB%, PA)."""
    parquet_path = os.path.join(_PROJECT_ROOT, 'data', 'batter_season_outcomes.parquet')
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
        
    # If not cached, compute from SQLite
    db_path = os.path.join(_PROJECT_ROOT, 'data', 'mlb.db')
    conn = sqlite3.connect(db_path)
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
    conn.close()
    
    df['k_pct'] = df['k_count'] / df['pa'].clip(lower=1)
    df['bb_pct'] = df['bb_count'] / df['pa'].clip(lower=1)
    df['woba'] = df['sum_woba'] / df['sum_woba_denom'].clip(lower=1)
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception:
        pass
    return df


def load_merged_data(min_pitches=300, min_pa=100):
    """Merges DVA ratings with real seasonal offensive outcomes."""
    dva = load_all_swing_ratings(min_pitches=min_pitches)
    outcomes = load_batter_outcomes()
    
    outcomes = outcomes[outcomes['pa'] >= min_pa]
    merged = pd.merge(dva, outcomes, on=['batter_id', 'season'], how='inner')
    return merged
