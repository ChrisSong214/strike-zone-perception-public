import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import argparse
import pandas as pd
import numpy as np
import urllib.request, json
import sqlite3

parser = argparse.ArgumentParser(description="Compare Call vs Alvarez")
parser.add_argument("--year", type=int, default=2025, help="Season year (default: 2025)")
args, _ = parser.parse_known_args()

# Load features
df = pd.read_parquet(f'data/training/swing_features_{args.year}.parquet')

# Add deception metrics
df['certainty'] = np.abs(df['lstm_p_strike'] - 0.5)
df['deception'] = np.abs(df['lstm_p_strike'] - df['in_zone'])

conn = sqlite3.connect('data/mlb.db')
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(swing_ratings)")
cols = {r[1] for r in cursor.fetchall()}
if "season" not in cols:
    conn.close()
    raise ValueError(f"Table 'swing_ratings' lacks 'season' column; cannot filter for season {args.year}.")

lb = pd.read_sql_query(f"SELECT batter_id, rank, swing_rating, chase_rate FROM swing_ratings WHERE season = {args.year}", conn)
conn.close()

def get_id_by_name(name_str):
    url = f"https://statsapi.mlb.com/api/v1/sports/1/players"
    # Actually just matching ID directly from our candidate list
    return None

# Alex Call ID:
# Let's find his ID from the DF since I didn't save it. I'll just check all batters and find the one that hits the API as Alex Call, or just get it via Pybaseball if it wasn't broken.
# Actually I can just look up the names for the candidates list again and save the ID.
candidates = lb[(lb['chase_rate'] < 0.18) & (lb['swing_rating'] < 105)].sort_values('chase_rate')
target_id = None
for idx, row in candidates.iterrows():
    uid = int(row['batter_id'])
    url = f"https://statsapi.mlb.com/api/v1/people/{uid}"
    try:
        req = urllib.request.urlopen(url)
        data = json.loads(req.read())
        name = data['people'][0]['fullName']
        if name == 'Alex Call':
            target_id = uid
            break
    except:
        continue

yordan_id = 670541

def get_stats(batter_id, name):
    bdf = df[df['batter'] == batter_id]
    r_info = lb[lb['batter_id'] == batter_id].iloc[0]
    
    n_pitches = len(bdf)
    edge_pct = (bdf['loc_bin'] == 'edge').mean() * 100
    chase_pitches = (bdf['loc_bin'] == 'chase').mean() * 100
    high_deception_pct = (bdf['deception'] > 0.4).mean() * 100
    ambiguous_pct = ((bdf['lstm_p_strike'] >= 0.25) & (bdf['lstm_p_strike'] <= 0.75)).mean() * 100
    
    print(f"--- {name} ---")
    print(f"Model Rank:            #{r_info['rank']}")
    print(f"Chase Rate:            {r_info['chase_rate']*100:.1f}%")
    print(f"% Pitches on Edge:     {edge_pct:.1f}%")
    print(f"% Chase Pitches Seen:  {chase_pitches:.1f}%")
    print(f"% High Deception:      {high_deception_pct:.1f}%")
    print()

get_stats(yordan_id, "Yordan Alvarez")
get_stats(target_id, "Alex Call")

