import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import argparse
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import sqlite3

parser = argparse.ArgumentParser(description="Compare Bregman vs Ramirez")
parser.add_argument("--year", type=int, default=2025, help="Season year (default: 2025)")
args, _ = parser.parse_known_args()

df = pd.read_parquet(f'data/training/swing_features_{args.year}.parquet')

# Retrieve IDs from db by name lookup if naive IDs are unknown,
# but we know 608324 is Bregman and 608070 is JRam. Let's just lookup safely:

conn = sqlite3.connect('data/mlb.db')
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(swing_ratings)")
cols = {r[1] for r in cursor.fetchall()}
if "season" not in cols:
    conn.close()
    raise ValueError(f"Table 'swing_ratings' lacks 'season' column; cannot filter for season {args.year}.")

lb = pd.read_sql_query(f"SELECT batter_id, rank, swing_rating, chase_rate, zone_swing_rate, n_pitches FROM swing_ratings WHERE season = {args.year}", conn)
conn.close()

bregman_id = 608324
ramirez_id = 608070

from src.swing_model import load_baseline_predictions_for_df
df = load_baseline_predictions_for_df(df, year=args.year)

is_swing = df['is_swing'].values.astype(float)
p_strike = df['lstm_p_strike']
actual_quality = is_swing * p_strike + (1 - is_swing) * (1 - p_strike)
league_quality = df['league_p_swing'] * p_strike + (1 - df['league_p_swing']) * (1 - p_strike)

df['actual_dva'] = actual_quality - league_quality

def get_stats(batter_id, name):
    bdf = df[df['batter'] == batter_id]
    r_info = lb[lb['batter_id'] == batter_id]
    if len(r_info) == 0:
        print(f"Skipping {name}, not in ratings")
        return
        
    r_info = r_info.iloc[0]
    
    elite_decisions = (bdf['actual_dva'] > 0.15).sum()
    terrible_decisions = (bdf['actual_dva'] < -0.15).sum()
    
    takes_on_balls = bdf[(bdf['in_zone'] == 0) & (bdf['is_swing'] == 0)]
    avg_p_strike = takes_on_balls['lstm_p_strike'].mean() * 100
    nasty_takes = (takes_on_balls['lstm_p_strike'] > 0.4).mean() * 100
    
    print(f"--- {name} ---")
    print(f"Model Rank: #{r_info['rank']} (Rating {r_info['swing_rating']:.1f})")
    print(f"Chase Rate: {r_info['chase_rate']*100:.1f}% | Zone Swing Rate: {r_info['zone_swing_rate']*100:.1f}%")
    print(f"Elite Decisions (> +0.15 DVA): {elite_decisions} ({elite_decisions/len(bdf)*100:.1f}%)")
    print(f"Terrible Decisions (< -0.15): {terrible_decisions} ({terrible_decisions/len(bdf)*100:.1f}%)")
    print(f"Avg P(Strike) on taken balls: {avg_p_strike:.1f}%")
    print()

get_stats(bregman_id, "Alex Bregman")
get_stats(ramirez_id, "Jose Ramirez")

