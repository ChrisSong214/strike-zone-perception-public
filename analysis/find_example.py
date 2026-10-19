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

parser = argparse.ArgumentParser(description="Find low chase rate but lower DVA rating players")
parser.add_argument("--year", type=int, default=2025, help="Season year (default: 2025)")
args, _ = parser.parse_known_args()

# Load swing ratings
conn = sqlite3.connect('data/mlb.db')
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(swing_ratings)")
cols = {r[1] for r in cursor.fetchall()}
if "season" not in cols:
    conn.close()
    raise ValueError(f"Table 'swing_ratings' lacks 'season' column; cannot filter for season {args.year}.")

lb = pd.read_sql_query(f"SELECT * FROM swing_ratings WHERE season = {args.year} AND n_pitches > 800", conn)
conn.close()

if len(lb) == 0:
    print(f"No records found in swing_ratings for season {args.year}.")
    exit(0)

# Find players with elite chase rates (bottom 20% of league)
# But average or low swing ratings
elite_chase_threshold = 0.20
yordan_rating = lb[lb['batter_id'] == 670541]['swing_rating'].values
if len(yordan_rating) > 0:
    y_rating = yordan_rating[0]
else:
    y_rating = 127.4

# Filter for Low Chase Rate but significantly lower DVA rating than Yordan
candidates = lb[
    (lb['chase_rate'] < 0.18) & 
    (lb['swing_rating'] < 105)
].copy()

candidates = candidates.sort_values('chase_rate', ascending=True)

import urllib.request, json
def get_name(uid):
    url = f"https://statsapi.mlb.com/api/v1/people/{uid}"
    try:
        req = urllib.request.urlopen(url)
        data = json.loads(req.read())
        return data['people'][0]['fullName']
    except:
        return str(uid)

print("--- CANDIDATES: Elite Chase Rate (< 18%), But Average/Low Rating (< 105) ---")
for idx, row in candidates.head(10).iterrows():
    name = get_name(int(row['batter_id']))
    print(f"Name: {name:<20} | Rank: {row['rank']:<3} | Rating: {row['swing_rating']:.1f} | Chase Rate: {row['chase_rate']:.3f} | Pitches: {row['n_pitches']}")

