"""
Eye Rating Leaderboard Generator (v2 - Fast, DB-backed)

Uses pre-computed physical pitch metrics from physical_pitch_metrics table.
Joins with statcast to get swing decisions and zone dimensions.
Runs vectorized Gaussian CDF optimization per batter.

Usage:
    python3 scripts/run_eye_rating.py           # 2025 only
    python3 scripts/run_eye_rating.py 2024 2025  # Multiple years
"""

import sys
import os
import pandas as pd
import numpy as np
import time
import sqlite3

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.scoring import FastEyeScorer
from src.db_manager import get_db_connection, save_physical_eye_ratings


def load_joined_data(year=2025):
    """
    Loads statcast + physical_pitch_metrics via SQL JOIN.
    Returns a DataFrame with all columns needed for scoring.
    """
    conn = get_db_connection()
    
    query = """
    SELECT 
        s.game_pk,
        s.at_bat_number,
        s.pitch_number,
        s.batter,
        s.player_name,
        s.plate_x,
        s.plate_z,
        s.sz_top,
        s.sz_bot,
        s.description,
        s.game_year,
        p.phys_prob_strike,
        p.phys_sigma_x,
        p.phys_sigma_z,
        p.phys_rho
    FROM statcast s
    INNER JOIN physical_pitch_metrics p
        ON s.game_pk = p.game_pk
        AND s.at_bat_number = p.at_bat_number
        AND s.pitch_number = p.pitch_number
    WHERE s.game_year = ?
      AND p.phys_sigma_x IS NOT NULL
      AND p.phys_sigma_z IS NOT NULL
      AND s.plate_x IS NOT NULL
      AND s.plate_z IS NOT NULL
      AND s.sz_top IS NOT NULL
      AND s.sz_bot IS NOT NULL
    """
    
    print(f"Loading joined data for {year}...")
    df = pd.read_sql_query(query, conn, params=(int(year),))
    conn.close()
    
    # Create is_swing
    swing_events = [
        'hit_into_play', 'foul', 'swinging_strike', 'swinging_strike_blocked', 
        'foul_tip', 'foul_bunt', 'missed_bunt', 'bunt_foul_tip'
    ]
    df['is_swing'] = df['description'].isin(swing_events)
    
    print(f"Loaded {len(df)} pitches with physical metrics for {year}.")
    return df


def run_leaderboard(years=None, min_pitches=200):
    """
    Main execution: generates Eye Rating for every qualified batter.
    """
    if years is None:
        years = [2025]
    
    all_results = []
    
    for year in years:
        print(f"\n{'='*60}")
        print(f"  EYE RATING ANALYSIS - {year}")
        print(f"{'='*60}")
        
        df = load_joined_data(year)
        
        if len(df) == 0:
            print(f"No data for {year}, skipping.")
            continue
        
        # Filter for valid finite rows required by FastEyeScorer before qualification
        required_cols = ['plate_x', 'plate_z', 'sz_top', 'sz_bot', 'phys_sigma_x', 'phys_sigma_z', 'is_swing']
        valid_subset = [c for c in required_cols if c in df.columns]
        if valid_subset:
            valid_mask = df[valid_subset].notna().all(axis=1)
            numeric_cols = [c for c in ['plate_x', 'plate_z', 'sz_top', 'sz_bot', 'phys_sigma_x', 'phys_sigma_z'] if c in df.columns]
            for col in numeric_cols:
                num = pd.to_numeric(df[col], errors='coerce')
                valid_mask &= np.isfinite(num)
            df = df[valid_mask].copy()
        
        # Find qualified batters on usable observations
        pitch_counts = df.groupby('batter').size()
        qualified = pitch_counts[pitch_counts >= min_pitches].index.tolist()
        print(f"Qualified batters (>= {min_pitches} valid pitches): {len(qualified)}")
        
        # Build name lookup
        # player_name in statcast is the PITCHER. We'll use batter ID for now.
        # TODO: Add batter name lookup table
        
        start_time = time.time()
        
        for i, batter_id in enumerate(qualified):
            if i % 25 == 0:
                elapsed = time.time() - start_time
                print(f"[{year}] Processing batter {i}/{len(qualified)}... ({elapsed:.1f}s)")
            
            batter_df = df[df['batter'] == batter_id]
            
            try:
                scorer = FastEyeScorer(batter_df)
                est_sigma, min_loss = scorer.fit_sigma(bounds=(0.01, 2.0), min_obs=min_pitches)
                
                all_results.append({
                    'batter_id': int(batter_id),
                    'season': year,
                    'n_pitches': len(scorer.pitches),
                    'eye_sigma_ft': est_sigma,
                    'loss': min_loss,
                    'swing_rate': scorer.pitches['is_swing'].mean()
                })
            except Exception as e:
                print(f"  Error on batter {batter_id}: {e}")
        
        elapsed = time.time() - start_time
        print(f"[{year}] Done. {len(qualified)} batters in {elapsed:.1f}s")
    
    if not all_results:
        print("No results generated.")
        return
    
    # Build Leaderboard
    lb = pd.DataFrame(all_results)
    
    # Grading Scale: 100 = Avg, +/-10 = StdDev
    # Standardize within each season so rating scales are season-specific
    standardized_dfs = []
    for s_val in sorted(lb['season'].unique()):
        s_df = lb[lb['season'] == s_val].copy()
        s_mu = s_df['eye_sigma_ft'].mean()
        s_sigma = s_df['eye_sigma_ft'].std()
        if pd.isna(s_sigma) or s_sigma == 0:
            s_sigma = 1.0
        s_df['z_score'] = (s_df['eye_sigma_ft'] - s_mu) / s_sigma
        s_df['eye_rating'] = (100 - (s_df['z_score'] * 10)).round(1)
        s_df = s_df.sort_values('eye_rating', ascending=False).reset_index(drop=True)
        s_df['rank'] = s_df.index + 1
        s_df['season'] = int(s_val)
        standardized_dfs.append(s_df)
        
    lb = pd.concat(standardized_dfs, ignore_index=True)
    
    # Save to DB (physical_eye_ratings and compatible eye_ratings)
    save_physical_eye_ratings(lb)
    
    # Print Results
    for s_val in sorted(lb['season'].unique()):
        s_sub = lb[lb['season'] == s_val]
        s_mu = s_sub['eye_sigma_ft'].mean()
        s_sigma = s_sub['eye_sigma_ft'].std()
        print(f"\n{'='*60}")
        print(f"  SEASON {s_val} LEADERBOARD (Avg Sigma: {s_mu:.4f} ft, Std: {s_sigma:.4f} ft)")
        print(f"{'='*60}")
        display_cols = ['rank', 'batter_id', 'eye_rating', 'eye_sigma_ft', 'n_pitches', 'swing_rate']
        display_cols = [c for c in display_cols if c in s_sub.columns]
        print(f"\n--- TOP 10 (Eagle Eyes) [{s_val}] ---")
        print(s_sub[display_cols].head(10).to_string(index=False))
        print(f"\n--- BOTTOM 10 (The Gamblers) [{s_val}] ---")
        print(s_sub[display_cols].tail(10).to_string(index=False))
        
    return lb


if __name__ == "__main__":
    # Parse command line args for years
    if len(sys.argv) > 1:
        years = [int(y) for y in sys.argv[1:]]
    else:
        years = [2025]
    
    run_leaderboard(years=years, min_pitches=200)
