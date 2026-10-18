import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import argparse
import sqlite3
import pandas as pd
import urllib.request, json
import itertools

def get_name(uid):
    url = f"https://statsapi.mlb.com/api/v1/people/{uid}"
    try:
        req = urllib.request.urlopen(url, timeout=5)
        data = json.loads(req.read())
        return data['people'][0]['fullName']
    except:
        return str(uid)

parser = argparse.ArgumentParser(description="Find player twins with similar traditional metrics")
parser.add_argument("--year", type=int, default=2025, help="Season year (default: 2025)")
args, _ = parser.parse_known_args()

conn = sqlite3.connect('data/mlb.db')
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(swing_ratings)")
cols = {r[1] for r in cursor.fetchall()}
if "season" not in cols:
    conn.close()
    raise ValueError(f"Table 'swing_ratings' lacks 'season' column; cannot filter for season {args.year}.")

lb = pd.read_sql_query(f"SELECT batter_id, rank, swing_rating, chase_rate, zone_swing_rate, n_pitches FROM swing_ratings WHERE season = {args.year} AND n_pitches > 1000", conn)
conn.close()

if len(lb) == 0:
    print(f"No records found in swing_ratings for season {args.year}.")
    exit(0)

best_pair = None
max_rating_diff = 0

for i, rowA in lb.iterrows():
    for j, rowB in lb.iterrows():
        if i >= j: continue
        
        c_diff = abs(rowA['chase_rate'] - rowB['chase_rate'])
        z_diff = abs(rowA['zone_swing_rate'] - rowB['zone_swing_rate'])
        
        # Must have nearly identical traditional metrics (within 1%)
        if c_diff < 0.015 and z_diff < 0.015:
            r_diff = abs(rowA['swing_rating'] - rowB['swing_rating'])
            if r_diff > max_rating_diff:
                max_rating_diff = r_diff
                best_pair = (rowA, rowB)

if best_pair:
    rA, rB = best_pair
    # Make A the better player
    if rB['swing_rating'] > rA['swing_rating']:
        rA, rB = rB, rA
        
    print("--- TRADITIONAL METRICS ARE IDENTICAL ---")
    print(f"Player A: {get_name(int(rA['batter_id']))} | Rank {rA['rank']}")
    print(f"  Chase Rate:      {rA['chase_rate']*100:.1f}%")
    print(f"  Zone Swing Rate: {rA['zone_swing_rate']*100:.1f}%")
    print(f"  --> MODEL RATING: {rA['swing_rating']:.1f}")
    
    print(f"\nPlayer B: {get_name(int(rB['batter_id']))} | Rank {rB['rank']}")
    print(f"  Chase Rate:      {rB['chase_rate']*100:.1f}%")
    print(f"  Zone Swing Rate: {rB['zone_swing_rate']*100:.1f}%")
    print(f"  --> MODEL RATING: {rB['swing_rating']:.1f}")
    print(f"\nRating Difference: {max_rating_diff:.1f} points!")

