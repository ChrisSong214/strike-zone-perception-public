
import sys
import os
import pandas as pd
import numpy as np
import time
import sqlite3

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_handler import StatcastDataHandler
from src.uncertainty import generate_pitch_cloud
from src.db_manager import get_db_connection

def upsert_metrics(conn, records):
    """Upsert metrics batch avoiding primary key collision."""
    if not records:
        return
    c = conn.cursor()
    c.executemany('''
        INSERT INTO physical_pitch_metrics (
            game_pk, at_bat_number, pitch_number,
            phys_prob_strike, phys_sigma_x, phys_sigma_z, phys_rho, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(game_pk, at_bat_number, pitch_number) DO UPDATE SET
            phys_prob_strike = excluded.phys_prob_strike,
            phys_sigma_x = excluded.phys_sigma_x,
            phys_sigma_z = excluded.phys_sigma_z,
            phys_rho = excluded.phys_rho,
            updated_at = excluded.updated_at
    ''', [
        (
            r['game_pk'], r['at_bat_number'], r['pitch_number'],
            r['phys_prob_strike'], r['phys_sigma_x'], r['phys_sigma_z'],
            r.get('phys_rho', 0.0), str(r['updated_at'])
        ) for r in records
    ])
    conn.commit()


def calculate_plate_metrics(cloud_trajs, sz_top, sz_bot, sz_width=17/12):
    """
    Calculates the 'Universal Constants' for the pitch based on its cloud.
    """
    if not cloud_trajs or len(cloud_trajs) == 0:
        return np.nan, np.nan, np.nan
        
    final_x = []
    final_z = []
    in_zone = 0
    
    # Plate limits
    x_min = -sz_width / 2
    x_max = sz_width / 2
    
    for traj in cloud_trajs:
        if traj is None or len(traj) == 0:
            continue
        fx = traj['x'].iloc[-1]
        fz = traj['z'].iloc[-1]
        if not np.isfinite(fx) or not np.isfinite(fz):
            return np.nan, np.nan, np.nan
        final_x.append(fx)
        final_z.append(fz)
        
        # Strike check
        if (x_min <= fx <= x_max) and (sz_bot <= fz <= sz_top):
            in_zone += 1
            
    if not final_x:
        return np.nan, np.nan, np.nan, np.nan
        
    prob_strike = in_zone / len(final_x)
    sigma_x = float(np.std(final_x))
    sigma_z = float(np.std(final_z))
    if len(final_x) > 1 and sigma_x > 1e-6 and sigma_z > 1e-6:
        corr_mat = np.corrcoef(final_x, final_z)
        rho = float(corr_mat[0, 1])
        if not np.isfinite(rho):
            rho = 0.0
    else:
        rho = 0.0
    
    return prob_strike, sigma_x, sigma_z, rho


def backfill_constants(years=None, batch_size=1000, resume=True, seed=42):
    if years is None:
        years = [2025]
        
    np.random.seed(seed)
    print(f"--- Backfilling Universal Constants for Years: {years} (seed={seed}, resume={resume}) ---")
    
    # Prepare DB (Once)
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS physical_pitch_metrics (
            game_pk INTEGER,
            at_bat_number INTEGER,
            pitch_number INTEGER,
            phys_prob_strike REAL,
            phys_sigma_x REAL,
            phys_sigma_z REAL,
            phys_rho REAL,
            updated_at TIMESTAMP,
            PRIMARY KEY (game_pk, at_bat_number, pitch_number)
        )
    ''')
    # Schema migration if phys_rho is missing
    c.execute("PRAGMA table_info(physical_pitch_metrics)")
    cols = [col[1] for col in c.fetchall()]
    if 'phys_rho' not in cols:
        try:
            c.execute("ALTER TABLE physical_pitch_metrics ADD COLUMN phys_rho REAL")
        except Exception:
            pass
    conn.commit()
    
    existing_keys = set()
    if resume:
        # Only treat pitches with valid non-null calculations as completed, allowing retries of failed rows
        existing_keys = set(conn.execute("""
            SELECT game_pk, at_bat_number, pitch_number 
            FROM physical_pitch_metrics 
            WHERE phys_prob_strike IS NOT NULL 
              AND phys_sigma_x IS NOT NULL 
              AND phys_sigma_z IS NOT NULL
        """).fetchall())
        print(f"Found {len(existing_keys):,} valid previously computed pitches. Resuming...")
    conn.close()
    
    handler = StatcastDataHandler()
    
    for year in years:
        print(f"\nProcessing Year: {year}...")
        try:
            df = handler.fetch_statcast(start_year=year, end_year=year)
        except Exception as e:
            print(f"Error fetching data for {year}: {e}")
            continue

        print(f"Loaded {len(df)} pitches for {year}.")
        conn = get_db_connection()
        
        results = []
        start_time = time.time()
        
        for i, row in df.iterrows():
            if i % 500 == 0 and i > 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                print(f"[{year}] Processed {i}/{len(df)} pitches ({rate:.1f} p/s)...")
                
            pitch_key = (int(row['game_pk']), int(row['at_bat_number']), int(row['pitch_number']))
            if resume and pitch_key in existing_keys:
                continue
                
            # Optimize: Only process if we have mechanics
            if pd.isna(row['release_speed']) or pd.isna(row['pfx_x']):
                continue
            if pd.isna(row.get('release_pos_x')) or pd.isna(row.get('release_pos_z')):
                continue
                
            # Defaults for zone
            sz_top = row.get('sz_top', 3.5)
            sz_bot = row.get('sz_bot', 1.5)
            if pd.isna(sz_top): sz_top = 3.5
            if pd.isna(sz_bot): sz_bot = 1.5
                
            # Deterministic per-pitch RNG ensures identical trajectories across runs and restarts
            pitch_seed = int((seed * 31 + pitch_key[0] * 10007 + pitch_key[1] * 101 + pitch_key[2]) % (2**32 - 1))
            pitch_rng = np.random.RandomState(pitch_seed)
            cloud = generate_pitch_cloud(row, n_samples=25, rng=pitch_rng)
            prob, sig_x, sig_z, rho = calculate_plate_metrics(cloud, sz_top, sz_bot)
            
            results.append({
                'game_pk': pitch_key[0],
                'at_bat_number': pitch_key[1],
                'pitch_number': pitch_key[2],
                'phys_prob_strike': prob,
                'phys_sigma_x': sig_x,
                'phys_sigma_z': sig_z,
                'phys_rho': rho,
                'updated_at': pd.Timestamp.now()
            })
            if not pd.isna(prob):
                existing_keys.add(pitch_key)
            
            if len(results) >= batch_size:
                upsert_metrics(conn, results)
                results = []
                
        if results:
            upsert_metrics(conn, results)
            
        conn.close()
        print(f"Backfill Complete for {year}.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Backfill physical pitch metrics")
    parser.add_argument("years", nargs="*", type=int, default=[2025])
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--no-resume", action="store_true", help="Overwrite existing pitches")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    backfill_constants(years=args.years, batch_size=args.batch_size, resume=not args.no_resume, seed=args.seed)
