import os
import sys
import argparse
import json
import urllib.request
import pandas as pd
import numpy as np
script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..')) if os.path.basename(script_dir) == 'scripts' else script_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.db_manager import get_db_connection


def get_name(uid):
    url = f"https://statsapi.mlb.com/api/v1/people/{uid}"
    try:
        req = urllib.request.urlopen(url, timeout=5)
        data = json.loads(req.read())
        return data['people'][0]['fullName']
    except Exception:
        return str(uid)


def main():
    parser = argparse.ArgumentParser(description="Export swing decision leaderboard to CSV")
    parser.add_argument("--year", type=int, default=2025, help="Season year (default: 2025)")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    default_dir = os.path.join(project_root, 'data', 'leaderboards')
    os.makedirs(default_dir, exist_ok=True)
    output_path = args.output or os.path.join(default_dir, f"{args.year}_dva_leaderboard.csv")
    print(f"Exporting leaderboard to {output_path}...")

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(swing_ratings)")
    cols = [r[1] for r in cursor.fetchall()]

    if "season" not in cols:
        print("Error: 'swing_ratings' table lacks a 'season' column. Cannot verify season provenance.")
        conn.close()
        return

    query = f"SELECT * FROM swing_ratings WHERE season = {args.year} ORDER BY rank ASC"
    lb = pd.read_sql_query(query, conn)
    conn.close()

    if len(lb) == 0:
        print(f"No records found in swing_ratings table for season {args.year}.")
        return

    # Canonicalize player name into a single column
    has_player_name = 'player_name' in lb.columns
    has_name = 'name' in lb.columns

    if has_player_name and has_name:
        # Coalesce: take non-empty value from name or player_name
        canonical_name = lb['player_name'].fillna('').astype(str).replace('', np.nan)
        canonical_name = canonical_name.combine_first(lb['name'].fillna('').astype(str).replace('', np.nan))
        lb['player_name'] = canonical_name
        lb.drop(columns=['name'], inplace=True)
    elif has_name and not has_player_name:
        lb.rename(columns={'name': 'player_name'}, inplace=True)
    elif not has_player_name and not has_name:
        lb['player_name'] = np.nan

    # If player names are missing or empty, fetch from MLB stats API row-by-row
    needs_name = lb['player_name'].isna() | (lb['player_name'].astype(str).str.strip().isin(['', 'nan', 'None']))
    if needs_name.any():
        n_missing = int(needs_name.sum())
        print(f"Fetching {n_missing} missing names from MLB stats API...")
        name_cache = {}
        for count, idx in enumerate(lb[needs_name].index, 1):
            bid = int(lb.loc[idx, 'batter_id'])
            if bid not in name_cache:
                name_cache[bid] = get_name(bid)
            lb.loc[idx, 'player_name'] = name_cache[bid]
            if count % 50 == 0:
                print(f"  Fetched {count}/{n_missing} names...")

    # Ensure player_name is the second column (after batter_id or rank)
    cols = [c for c in lb.columns if c != 'player_name']
    if 'rank' in cols:
        insert_idx = cols.index('rank') + 1
    elif 'batter_id' in cols:
        insert_idx = cols.index('batter_id') + 1
    else:
        insert_idx = 1
    cols.insert(insert_idx, 'player_name')
    lb = lb[cols]

    lb.to_csv(output_path, index=False)
    print(f"Export complete: {output_path} ({len(lb)} batters)")


if __name__ == "__main__":
    main()
