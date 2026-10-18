"""
LSTM Eye Rating Leaderboard Generator

Uses the trained LSTM+MDN streaming perception model to score batters.

Instead of the old approach (fit visual_sigma per batter via MLE),
this directly uses the model's predicted uncertainty to compute P(Strike)
and scores batters on how well their swings align with model predictions.

Usage:
    python3 scripts/run_eye_rating_lstm.py              # 2025 only
    python3 scripts/run_eye_rating_lstm.py 2024 2025     # Multiple years
    python3 scripts/run_eye_rating_lstm.py --min-pitches 500
"""

import sys
import os
import time
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np

from src.lstm_scoring import LSTMEyeScorer
from src.db_manager import get_db_connection, save_lstm_eye_ratings


SWING_EVENTS = {
    'hit_into_play', 'foul', 'swinging_strike', 'swinging_strike_blocked',
    'foul_tip', 'foul_bunt', 'missed_bunt', 'bunt_foul_tip'
}


def load_pitches(year):
    """Load pitches with all columns needed for trajectory + labels."""
    conn = get_db_connection()
    
    query = f"""
    SELECT 
        game_pk, at_bat_number, pitch_number,
        batter, player_name,
        vx0, vy0, vz0, ax, ay, az,
        release_pos_x, release_pos_y, release_pos_z, release_extension,
        plate_x, plate_z, sz_top, sz_bot,
        description, pitch_type, p_throws, stand,
        release_speed, game_year
    FROM statcast
    WHERE game_year = {year}
      AND vx0 IS NOT NULL
      AND release_pos_x IS NOT NULL
      AND plate_x IS NOT NULL
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['is_swing'] = df['description'].isin(SWING_EVENTS)
    print(f"Loaded {len(df):,} pitches for {year}")
    return df


def run_leaderboard(years, min_pitches=200, device='cpu', decision_frac=0.40):
    """Generate LSTM-based eye rating leaderboard."""
    
    # Load model
    scorer = LSTMEyeScorer(device=device, decision_frac=decision_frac)
    
    all_results = []
    
    for year in years:
        print(f"\n{'='*60}")
        print(f"  LSTM EYE RATING — {year}")
        print(f"{'='*60}")
        
        df = load_pitches(year)
        
        if len(df) == 0:
            print(f"No data for {year}, skipping.")
            continue
        
        # Find qualified batters
        pitch_counts = df.groupby('batter').size()
        qualified = pitch_counts[pitch_counts >= min_pitches].index.tolist()
        print(f"Qualified batters (>= {min_pitches} pitches): {len(qualified)}")
        
        start_time = time.time()
        
        for i, batter_id in enumerate(qualified):
            if i % 25 == 0:
                elapsed = time.time() - start_time
                if i > 0:
                    eta = elapsed / i * (len(qualified) - i)
                    print(f"  [{year}] {i}/{len(qualified)} ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")
                else:
                    print(f"  [{year}] {i}/{len(qualified)} starting...")
            
            batter_df = df[df['batter'] == batter_id]
            
            try:
                result = scorer.score_batter(batter_df)
                
                if result is None:
                    continue
                
                all_results.append({
                    'batter_id': int(batter_id),
                    'season': year,
                    'n_pitches': result['n_pitches'],
                    'll_per_pitch': result['ll_per_pitch'],
                    'mean_sigma_x': result['mean_sigma_x'],
                    'mean_sigma_z': result['mean_sigma_z'],
                    'swing_rate': result['swing_rate'],
                    'mean_p_strike': result['mean_p_strike'],
                })
            except Exception as e:
                print(f"  Error on batter {batter_id}: {e}")
        
        elapsed = time.time() - start_time
        print(f"  [{year}] Done. {len(qualified)} batters in {elapsed:.1f}s")
    
    if not all_results:
        print("No results generated.")
        return None
    
    # Build leaderboard
    lb = pd.DataFrame(all_results)
    
    # Grading scale: higher log-likelihood per pitch = better eye
    # Standardize within each season so rating scales are season-specific
    standardized_dfs = []
    for s_val in sorted(lb['season'].unique()):
        s_df = lb[lb['season'] == s_val].copy()
        s_mu = s_df['ll_per_pitch'].mean()
        s_sigma = s_df['ll_per_pitch'].std()
        if pd.isna(s_sigma) or s_sigma == 0:
            s_sigma = 1.0
        s_df['z_score'] = (s_df['ll_per_pitch'] - s_mu) / s_sigma
        s_df['eye_rating'] = (100 + (s_df['z_score'] * 10)).round(1)
        s_df = s_df.sort_values('eye_rating', ascending=False).reset_index(drop=True)
        s_df['rank'] = s_df.index + 1
        s_df['season'] = int(s_val)
        standardized_dfs.append(s_df)
        
    lb = pd.concat(standardized_dfs, ignore_index=True)
    
    # Save to distinct lstm_eye_ratings table
    save_lstm_eye_ratings(lb)
    
    # Print
    for s_val in sorted(lb['season'].unique()):
        s_sub = lb[lb['season'] == s_val]
        s_mu = s_sub['ll_per_pitch'].mean()
        s_sigma = s_sub['ll_per_pitch'].std()
        print(f"\n{'='*60}")
        print(f"  SEASON {s_val} LEADERBOARD (LL/pitch avg: {s_mu:.4f}, std: {s_sigma:.4f})")
        print(f"{'='*60}")
        display_cols = ['rank', 'batter_id', 'eye_rating', 'll_per_pitch', 
                        'mean_sigma_x', 'mean_sigma_z', 'n_pitches', 'swing_rate']
        display_cols = [c for c in display_cols if c in s_sub.columns]
        print(f"\n--- TOP 15 (Eagle Eyes) [{s_val}] ---")
        top = s_sub[display_cols].head(15)
        print(top.to_string(index=False))
        print(f"\n--- BOTTOM 15 (Hackers) [{s_val}] ---")
        bot = s_sub[display_cols].tail(15)
        print(bot.to_string(index=False))
    
    # Summary stats
    print(f"\n--- Summary ---")
    print(f"Total batters scored: {len(lb)}")
    print(f"Mean eye rating: {lb['eye_rating'].mean():.1f}")
    print(f"Rating range: {lb['eye_rating'].min():.1f} — {lb['eye_rating'].max():.1f}")
    print(f"Mean σ(x): {lb['mean_sigma_x'].mean():.4f} ft ({lb['mean_sigma_x'].mean()*12:.2f} in)")
    print(f"Mean σ(z): {lb['mean_sigma_z'].mean():.4f} ft ({lb['mean_sigma_z'].mean()*12:.2f} in)")
    
    return lb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LSTM Eye Rating Leaderboard')
    parser.add_argument('years', nargs='*', type=int, default=[2025])
    parser.add_argument('--min-pitches', type=int, default=200)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--decision-frac', type=float, default=0.40,
                        help='Fraction of pitch flight for decision point (default: 0.40)')
    args = parser.parse_args()
    
    run_leaderboard(args.years, min_pitches=args.min_pitches, device=args.device, decision_frac=args.decision_frac)
