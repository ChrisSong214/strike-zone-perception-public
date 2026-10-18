import sqlite3
import pandas as pd
import os
import sys

# Ensure imports work if run as script
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DB_PATH = os.path.join(_PROJECT_ROOT, "data", "mlb.db")

def get_db_connection(db_path=DB_PATH):
    """
    Returns a connection to the SQLite database.
    Creates the directory if it doesn't exist.
    """
    if db_path == DB_PATH and (not os.path.exists(db_path) or os.path.getsize(db_path) == 0):
        alt_path = os.path.join(_PROJECT_ROOT, "notebooks", "data", "mlb.db")
        if os.path.exists(alt_path) and os.path.getsize(alt_path) > 0:
            db_path = alt_path
            
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    return conn

def sync_feather_to_sql(years=None):
    """
    Reads Feather files using StatcastDataHandler and writes them to the SQLite DB.
    Replaces existing data to ensure sync.
    """
    if years is None:
        years = range(2020, 2026)
    
    from src.data_handler import StatcastDataHandler
    handler = StatcastDataHandler()
    conn = get_db_connection()
    
    # We might want to drop the table and recreate to ensure clean sync
    # Or append. Given row-level editing, dropping might lose edits?
    # actually, the request is "rewriting the 2020-2025 feather files into a sql database that allows row level editing"
    # This implies the SQL DB is the initialized state. If we re-sync, we overwrite edits.
    # For now, let's assume we are initializing or resetting the DB from source.
    # Use 'replace' for the first chunk, 'append' for others.
    
    first_chunk = True
    total_rows = 0
    
    print(f"Syncing years {list(years)} to {DB_PATH}...")
    
    for year in years:
        print(f"Loading {year} from Feather...")
        # Use fetch_statcast which loads from local feather
        df = handler.fetch_statcast(start_year=year, end_year=year)
        
        if not df.empty:
            # Ensure types are SQL friendly (e.g. datetime)
            # pandas to_sql usually handles this, but let's be safe
            
            print(f"Writing {len(df)} rows to SQL...")
            if first_chunk:
                df.to_sql('statcast', conn, if_exists='replace', index=False)
                first_chunk = False
            else:
                df.to_sql('statcast', conn, if_exists='append', index=False)
            
            total_rows += len(df)
        else:
            print(f"No data for {year}")

    if total_rows > 0:
        print("Creating indexes...")
        # Create useful indexes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_game_pk ON statcast (game_pk)",
            "CREATE INDEX IF NOT EXISTS idx_game_date ON statcast (game_date)",
            "CREATE INDEX IF NOT EXISTS idx_pitcher ON statcast (pitcher)",
            "CREATE INDEX IF NOT EXISTS idx_batter ON statcast (batter)",
            "CREATE INDEX IF NOT EXISTS idx_pitch_type ON statcast (pitch_type)"
        ]
        for idx_sql in indexes:
            conn.execute(idx_sql)
            
    conn.close()
    print(f"Sync complete. Total rows: {total_rows}")

def _get_sql_type(series):
    """Determine SQLite column type from pandas Series."""
    if pd.api.types.is_integer_dtype(series):
        return "INTEGER"
    elif pd.api.types.is_float_dtype(series):
        return "REAL"
    elif pd.api.types.is_bool_dtype(series):
        return "INTEGER"
    elif pd.api.types.is_datetime64_any_dtype(series):
        return "TIMESTAMP"
    else:
        return "TEXT"


def _save_ratings_by_season(df_ratings, table_name):
    """Saves ratings to table atomically and non-destructively by season."""
    if 'season' not in df_ratings.columns or df_ratings['season'].isna().any():
        raise ValueError(
            f"Cannot save to {table_name}: 'season' column is required and must not contain NULLs "
            f"in df_ratings to preserve multi-season history and prevent accidental table erasure."
        )

    conn = get_db_connection()
    df_ratings = df_ratings.copy()
    df_ratings['season'] = df_ratings['season'].astype(int)
    df_ratings['updated_at'] = pd.Timestamp.now()
    
    cursor = conn.cursor()
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
    table_exists = cursor.fetchone() is not None
    
    if table_exists:
        # Schema migration: ensure existing table has all columns present in df_ratings
        cursor.execute(f"PRAGMA table_info({table_name})")
        existing_cols = {r[1] for r in cursor.fetchall()}
        for col in df_ratings.columns:
            if col not in existing_cols:
                col_type = _get_sql_type(df_ratings[col])
                conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {col} {col_type}")
                existing_cols.add(col)
                
        # Re-fetch columns after migration
        cursor.execute(f"PRAGMA table_info({table_name})")
        cols = {r[1] for r in cursor.fetchall()}
        
        # Atomic transaction: delete matching season rows and append new season rows
        with conn:
            seasons_to_replace = df_ratings['season'].dropna().unique().tolist()
            for s in seasons_to_replace:
                conn.execute(f"DELETE FROM {table_name} WHERE season = ?", (int(s),))
            df_ratings.to_sql(table_name, conn, if_exists='append', index=False)
    else:
        with conn:
            df_ratings.to_sql(table_name, conn, if_exists='replace', index=False)
        
    try:
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_batter ON {table_name} (batter_id)")
        if 'season' in df_ratings.columns:
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_season ON {table_name} (season)")
        conn.commit()
    except Exception:
        pass
        
    conn.close()
    print(f"Saved {len(df_ratings)} ratings to DB table: {table_name}")


def save_eye_ratings(df_ratings, table_name='eye_ratings'):
    """Saves eye ratings to specified table (default 'eye_ratings')."""
    _save_ratings_by_season(df_ratings, table_name)


def save_physical_eye_ratings(df_ratings):
    """
    Saves FastEyeScorer physical ratings (with eye_sigma_ft) to physical_eye_ratings
    and updates eye_ratings for notebook compatibility.
    """
    _save_ratings_by_season(df_ratings, 'physical_eye_ratings')
    _save_ratings_by_season(df_ratings, 'eye_ratings')


def save_lstm_eye_ratings(df_ratings):
    """
    Saves LSTM streaming perception ratings (with mean_sigma_x, mean_sigma_z)
    to lstm_eye_ratings without overwriting physical eye_ratings.
    """
    _save_ratings_by_season(df_ratings, 'lstm_eye_ratings')


def save_swing_ratings(df_ratings):
    """Saves Bayesian DVA swing ratings to swing_ratings table non-destructively by season."""
    _save_ratings_by_season(df_ratings, 'swing_ratings')

if __name__ == "__main__":
    # Sync 2020-2025
    sync_feather_to_sql(range(2020, 2026))
