import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import argparse
import json
import sqlite3
import urllib.request
import pandas as pd


def get_name(uid):
    url = f"https://statsapi.mlb.com/api/v1/people/{uid}"
    try:
        req = urllib.request.urlopen(url, timeout=5)
        data = json.loads(req.read())
        return data['people'][0]['fullName']
    except Exception:
        return str(uid)


def main():
    parser = argparse.ArgumentParser(description="View top swing decision leaderboard")
    parser.add_argument("--year", type=int, default=2025, help="Season year (default: 2025)")
    parser.add_argument("--limit", type=int, default=15, help="Number of batters to display (default: 15)")
    args = parser.parse_args()

    conn = sqlite3.connect('data/mlb.db')
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(swing_ratings)")
    cols = {r[1] for r in cursor.fetchall()}

    if "season" not in cols:
        conn.close()
        raise ValueError(f"Table 'swing_ratings' lacks 'season' column; cannot filter for season {args.year}.")

    query = f"SELECT * FROM swing_ratings WHERE season = {args.year} ORDER BY rank ASC LIMIT {args.limit}"
    lb = pd.read_sql_query(query, conn)
    conn.close()

    if len(lb) == 0:
        print(f"No records found in swing_ratings for season {args.year}.")
        return

    print(f"--- TOP {len(lb)} SWING DECISION LEADERBOARD ({args.year}) ---")
    for idx, row in lb.iterrows():
        name = row.get('player_name') or row.get('name')
        if pd.isna(name) or name == str(row['batter_id']) or not name:
            name = get_name(int(row['batter_id']))
        chase = row.get('chase_rate', 0.0) * 100
        z_swing = row.get('zone_swing_rate', 0.0) * 100
        print(f"{int(row['rank']):>2}. {name:<20} | Rating: {row['swing_rating']:.1f} | Chase: {chase:.1f}% | Zone Swing: {z_swing:.1f}% | Pitches: {int(row['n_pitches'])}")


if __name__ == "__main__":
    main()
    
