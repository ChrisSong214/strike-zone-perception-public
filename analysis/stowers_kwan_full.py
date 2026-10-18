import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import argparse
import sqlite3
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb

parser = argparse.ArgumentParser(description="Full comparison of Stowers vs Kwan")
parser.add_argument("--year", type=int, default=2025, help="Season year (default: 2025)")
args, _ = parser.parse_known_args()

df = pd.read_parquet(f'data/training/swing_features_{args.year}.parquet')

conn = sqlite3.connect('data/mlb.db')
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(swing_ratings)")
cols = {r[1] for r in cursor.fetchall()}
if "season" not in cols:
    conn.close()
    raise ValueError(f"Table 'swing_ratings' lacks 'season' column; cannot filter for season {args.year}.")

lb = pd.read_sql_query(f"SELECT * FROM swing_ratings WHERE season = {args.year}", conn)
conn.close()

stowers_id = 669065
kwan_id = 680757

from src.swing_model import load_baseline_predictions_for_df
df = load_baseline_predictions_for_df(df, year=args.year)

is_swing = df['is_swing'].values.astype(float)
p_strike = df['lstm_p_strike']
actual_quality = is_swing * p_strike + (1 - is_swing) * (1 - p_strike)
league_quality = df['league_p_swing'] * p_strike + (1 - df['league_p_swing']) * (1 - p_strike)
df['actual_dva'] = actual_quality - league_quality

def full_stats(batter_id, name):
    bdf = df[df['batter'] == batter_id]
    r = lb[lb['batter_id'] == batter_id].iloc[0]
    
    # Filter by location bin
    zone = bdf[bdf['loc_bin'] == 'zone']
    edge = bdf[bdf['loc_bin'] == 'edge']
    chase = bdf[bdf['loc_bin'] == 'chase']
    
    balls = bdf[bdf['in_zone'] == 0]
    strikes = bdf[bdf['in_zone'] == 1]
    takes_on_balls = bdf[(bdf['in_zone'] == 0) & (bdf['is_swing'] == 0)]
    
    elite = (bdf['actual_dva'] > 0.15).sum()
    terrible = (bdf['actual_dva'] < -0.15).sum()
    
    print(f"{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  DVA Rating:        {r['swing_rating']:.1f}")
    print(f"  Tendency DVA:      {r['tendency_dva']:+.4f}")
    print(f"  Actual DVA:        {r['actual_dva']:+.4f}")
    print(f"  N Pitches:         {int(r['n_pitches'])}")
    print()
    print(f"  --- Traditional Metrics ---")
    print(f"  Overall Swing%:    {bdf['is_swing'].mean()*100:.1f}%")
    print(f"  Chase Rate:        {r['chase_rate']*100:.1f}%")
    print(f"  Zone Swing Rate:   {r['zone_swing_rate']*100:.1f}%")
    print(f"  Edge Swing Rate:   {edge['is_swing'].mean()*100:.1f}%")
    print()
    print(f"  --- LSTM Perception (40% Flight) ---")
    print(f"  Avg P(Strike) all pitches:     {bdf['lstm_p_strike'].mean()*100:.1f}%")
    print(f"  Avg P(Strike) on taken balls:  {takes_on_balls['lstm_p_strike'].mean()*100:.1f}%")
    print(f"  Avg Sigma X:       {bdf['lstm_sigma_x'].mean():.3f} ft")
    print(f"  Avg Sigma Z:       {bdf['lstm_sigma_z'].mean():.3f} ft")
    print()
    print(f"  --- Decision Quality ---")
    print(f"  Elite Decisions (>+0.15 DVA):   {elite} ({elite/len(bdf)*100:.1f}%)")
    print(f"  Terrible Decisions (<-0.15):    {terrible} ({terrible/len(bdf)*100:.1f}%)")
    print(f"  Avg DVA per pitch:              {bdf['actual_dva'].mean():+.4f}")
    print()
    print(f"  --- Zone Breakdown ---")
    print(f"  Zone pitches:  {len(zone):>4} | Swing%: {zone['is_swing'].mean()*100:.1f}%")
    print(f"  Edge pitches:  {len(edge):>4} | Swing%: {edge['is_swing'].mean()*100:.1f}%")
    print(f"  Chase pitches: {len(chase):>4} | Swing%: {chase['is_swing'].mean()*100:.1f}%")
    print()

full_stats(stowers_id, "Kyle Stowers")
full_stats(kwan_id, "Steven Kwan")
