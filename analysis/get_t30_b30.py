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
import time

def get_name(uid):
    url = f"https://statsapi.mlb.com/api/v1/people/{uid}"
    try:
        req = urllib.request.urlopen(url, timeout=5)
        data = json.loads(req.read())
        return data['people'][0]['fullName']
    except:
        return str(uid)

parser = argparse.ArgumentParser(description="View top and bottom 30 swing decisions")
parser.add_argument("--year", type=int, default=2025, help="Season year (default: 2025)")
args, _ = parser.parse_known_args()

conn = sqlite3.connect('data/mlb.db')
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(swing_ratings)")
cols = {r[1] for r in cursor.fetchall()}
if "season" not in cols:
    conn.close()
    raise ValueError(f"Table 'swing_ratings' lacks 'season' column; cannot filter for season {args.year}.")

# Filtering to n_pitches > 500 to keep it relevant to MLB regulars
lb = pd.read_sql_query(f"SELECT batter_id, rank, swing_rating, chase_rate, zone_swing_rate, n_pitches FROM swing_ratings WHERE season = {args.year} AND n_pitches > 500 ORDER BY rank ASC", conn)
conn.close()

if len(lb) == 0:
    print(f"No records found in swing_ratings for season {args.year}.")
    exit(0)

print(f"--- TOP 30 (Best Decisions, >500 Pitches, {args.year}) ---")
for idx, row in lb.head(30).iterrows():
    name = get_name(int(row['batter_id']))
    print(f"{idx+1:>2}. {name:<20} | Rating: {row['swing_rating']:.1f} | Chase: {row['chase_rate']*100:>4.1f}% | Zone Swing: {row['zone_swing_rate']*100:>4.1f}% | Pitches: {int(row['n_pitches'])}")

print("\n--- BOTTOM 30 (Worst Decisions, >500 Pitches) ---")
# Reset index for bottom 30 to count correctly, though it's easiest to just iterate tail
bottom_30 = lb.tail(30).iloc[::-1] # Reverse so absolute worst is at the top of this list? No, let's keep ascending or descending neatly.
# Let's print absolute worst at bottom
count = len(lb) - 29
for idx, row in lb.tail(30).iterrows():
    name = get_name(int(row['batter_id']))
    print(f"{count:>3}. {name:<20} | Rating: {row['swing_rating']:.1f} | Chase: {row['chase_rate']*100:>4.1f}% | Zone Swing: {row['zone_swing_rate']*100:>4.1f}% | Pitches: {int(row['n_pitches'])}")
    count += 1
