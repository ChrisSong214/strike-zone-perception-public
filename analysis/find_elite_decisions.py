import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description="Find elite decisions")
parser.add_argument("--year", type=int, default=2025, help="Season year (default: 2025)")
args, _ = parser.parse_known_args()

df = pd.read_parquet(f'data/training/swing_features_{args.year}.parquet')

from src.swing_model import load_baseline_predictions_for_df
df = load_baseline_predictions_for_df(df, year=args.year)

is_swing = df['is_swing'].values.astype(float)
p_strike = df['lstm_p_strike']
actual_quality = is_swing * p_strike + (1 - is_swing) * (1 - p_strike)
league_quality = df['league_p_swing'] * p_strike + (1 - df['league_p_swing']) * (1 - p_strike)

df['actual_dva'] = actual_quality - league_quality

trout_id = 545361
kwan_id = 680757

# Filter for the batters
trout_df = df[df['batter'] == trout_id]
kwan_df = df[df['batter'] == kwan_id]

# Elite decisions (DVA > 0.15 indicates a very difficult decision handled correctly)
trout_elite = (trout_df['actual_dva'] > 0.15).sum()
kwan_elite = (kwan_df['actual_dva'] > 0.15).sum()
# Terrible decisions (DVA < -0.15)
trout_terrible = (trout_df['actual_dva'] < -0.15).sum()
kwan_terrible = (kwan_df['actual_dva'] < -0.15).sum()

print("--- Elite vs Terrible Decisions ---")
print(f"Mike Trout  | Elite Decisions (> +0.15 DVA): {trout_elite:<4} ({trout_elite/len(trout_df)*100:.1f}%) | Terrible Decisions (< -0.15 DVA): {trout_terrible:<4} ({trout_terrible/len(trout_df)*100:.1f}%)")
print(f"Steven Kwan | Elite Decisions (> +0.15 DVA): {kwan_elite:<4} ({kwan_elite/len(kwan_df)*100:.1f}%) | Terrible Decisions (< -0.15 DVA): {kwan_terrible:<4} ({kwan_terrible/len(kwan_df)*100:.1f}%)")

