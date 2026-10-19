import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import argparse
import pandas as pd
import numpy as np
import sqlite3
import urllib.request, json

def get_name(uid):
    url = f"https://statsapi.mlb.com/api/v1/people/{uid}"
    try:
        req = urllib.request.urlopen(url, timeout=5)
        data = json.loads(req.read())
        return data['people'][0]['fullName']
    except:
        return str(uid)

parser = argparse.ArgumentParser(description="Find paper tiger examples")
parser.add_argument("--year", type=int, default=2025, help="Season year (default: 2025)")
args, _ = parser.parse_known_args()

print(f"Loading data for season {args.year}...")
# Load features
df = pd.read_parquet(f'data/training/swing_features_{args.year}.parquet')

# Deception
df['deception'] = np.abs(df['lstm_p_strike'] - df['in_zone'])

conn = sqlite3.connect('data/mlb.db')
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(swing_ratings)")
cols = {r[1] for r in cursor.fetchall()}
if "season" not in cols:
    conn.close()
    raise ValueError(f"Table 'swing_ratings' lacks 'season' column; cannot filter for season {args.year}.")

lb = pd.read_sql_query(f"SELECT batter_id, rank, swing_rating, chase_rate, n_pitches FROM swing_ratings WHERE season = {args.year}", conn)
conn.close()

# Let's find players with <18% chase rate
candidates = lb[(lb['chase_rate'] < 0.18) & (lb['n_pitches'] > 500)].copy()

results = []
for idx, row in candidates.iterrows():
    bid = int(row['batter_id'])
    bdf = df[df['batter'] == bid]
    edge_pct = (bdf['loc_bin'] == 'edge').mean() * 100
    chase_pitches = (bdf['loc_bin'] == 'chase').mean() * 100
    high_deception = (bdf['deception'] > 0.4).mean() * 100
    ambiguous = ((bdf['lstm_p_strike'] >= 0.25) & (bdf['lstm_p_strike'] <= 0.75)).mean() * 100
    
    results.append({
        'batter_id': bid,
        'rank': row['rank'],
        'chase_rate': row['chase_rate'] * 100,
        'high_deception': high_deception,
        'chase_seen_pct': chase_pitches,
        'edge_pct': edge_pct,
        'ambiguous': ambiguous
    })

res_df = pd.DataFrame(results)

# Best candidates: Lowest high_deception, lowest chase_seen_pct compared to Yordan
res_df['deception_gap'] = 13.0 - res_df['high_deception'] # Yordan is 13.0
res_df['chase_gap'] = 41.7 - res_df['chase_seen_pct']

res_df = res_df.sort_values(['deception_gap', 'chase_gap'], ascending=False)

print("\n--- TOP PAPER TIGERS ---")
for idx, row in res_df.head(5).iterrows():
    name = get_name(int(row['batter_id']))
    print(f"{name} (Rank {row['rank']})")
    print(f"  Chase Rate: {row['chase_rate']:.1f}%")
    print(f"  High Deception: {row['high_deception']:.1f}% (Yordan: 13.0%) -> Diff: {13.0 - row['high_deception']:.1f}%")
    print(f"  Chase Seen: {row['chase_seen_pct']:.1f}% (Yordan: 41.7%) -> Diff: {41.7 - row['chase_seen_pct']:.1f}%")
    print()

