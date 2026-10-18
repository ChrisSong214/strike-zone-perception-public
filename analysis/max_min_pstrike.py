import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import pandas as pd
import numpy as np

# Load features
df = pd.read_parquet('data/training/swing_features_2025.parquet')

# Filter only pitches that were BALLS and the batter didn't swing
takes_on_balls = df[(df['in_zone'] == 0) & (df['is_swing'] == 0)]

# Calculate average P(Strike) per batter
stats = takes_on_balls.groupby('batter').agg(
    avg_p_strike=('lstm_p_strike', 'mean'),
    n_takes=('is_swing', 'count')
).reset_index()

# Filter for batters who had enough takes to be a reliable sample (e.g. > 200 takes on balls)
stats = stats[stats['n_takes'] > 200].copy()
stats['avg_p_strike'] = stats['avg_p_strike'] * 100

stats = stats.sort_values('avg_p_strike', ascending=False)

import urllib.request, json
import time

def get_name(uid):
    url = f"https://statsapi.mlb.com/api/v1/people/{uid}"
    try:
        req = urllib.request.urlopen(url)
        data = json.loads(req.read())
        return data['people'][0]['fullName']
    except:
        return str(uid)

print("--- HIGHEST Avg P(Strike) on Taken Balls (Hardest Pitches) ---")
for idx, row in stats.head(5).iterrows():
    print(f"Name: {get_name(int(row['batter'])):<20} | Avg P(Strike): {row['avg_p_strike']:.1f}% | Takes: {int(row['n_takes'])}")

print("\n--- LOWEST Avg P(Strike) on Taken Balls (Easiest Pitches) ---")
for idx, row in stats.tail(5).iterrows():
    print(f"Name: {get_name(int(row['batter'])):<20} | Avg P(Strike): {row['avg_p_strike']:.1f}% | Takes: {int(row['n_takes'])}")

# Context for Trout and Kwan
trout_val = stats[stats['batter'] == 545361]['avg_p_strike'].values
kwan_val = stats[stats['batter'] == 680757]['avg_p_strike'].values

print("\n--- CONTEXT ---")
print(f"Total Batters Evaluated: {len(stats)}")
print(f"League Maximum: {stats['avg_p_strike'].max():.1f}%")
print(f"League Minimum: {stats['avg_p_strike'].min():.1f}%")
print(f"Mike Trout:  {trout_val[0]:.1f}%" if len(trout_val)>0 else "Mike Trout not found")
print(f"Steven Kwan: {kwan_val[0]:.1f}%" if len(kwan_val)>0 else "Steven Kwan not found")

