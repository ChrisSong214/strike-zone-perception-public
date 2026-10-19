import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import urllib.request, json
import numpy as np

matplotlib.rcParams['font.family'] = 'sans-serif'

def get_name(uid, conn=None):
    if conn is not None:
        try:
            cur = conn.cursor()
            cur.execute("SELECT name FROM player_names WHERE batter_id = ?", (uid,))
            row = cur.fetchone()
            if row and row[0] and row[0] != str(uid):
                return row[0]
        except Exception:
            pass
        try:
            cur = conn.cursor()
            cur.execute("SELECT batter_name FROM statcast WHERE batter = ? AND batter_name IS NOT NULL AND batter_name != '' LIMIT 1", (uid,))
            row = cur.fetchone()
            if row and row[0]:
                return row[0]
        except Exception:
            pass

    try:
        from pybaseball import playerid_reverse_lookup
        res = playerid_reverse_lookup([uid], key_type='mlbam')
        if len(res) > 0 and 'name_first' in res.columns:
            first = str(res.iloc[0]['name_first']).strip().title()
            last = str(res.iloc[0]['name_last']).strip().title()
            return f"{first} {last}"
    except Exception:
        pass

    url = f"https://statsapi.mlb.com/api/v1/people/{uid}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read())
            return data['people'][0]['fullName']
    except Exception:
        return str(uid)

import argparse
parser = argparse.ArgumentParser(description="Generate leaderboard image")
parser.add_argument("--year", type=int, default=2025)
args, _ = parser.parse_known_args()

conn = sqlite3.connect('data/mlb.db')
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(swing_ratings)")
cols = {r[1] for r in cursor.fetchall()}
if "season" not in cols:
    conn.close()
    raise ValueError(f"Table 'swing_ratings' lacks 'season' column; cannot verify season {args.year}.")

season_filter = f"AND s.season = {args.year}"
name_clause = "COALESCE(p.name, s.player_name, s.name)"

# Check if player_names table exists
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='player_names'")
has_pn_table = cursor.fetchone() is not None

join_clause = "LEFT JOIN player_names p ON s.batter_id = p.batter_id" if has_pn_table else ""
name_select = "COALESCE(p.name, s.player_name, s.name)" if has_pn_table else ("s.player_name" if "player_name" in cols else "s.name")

lb = pd.read_sql_query(f"""
    SELECT s.batter_id, s.swing_rating, s.tendency_dva, s.actual_dva, s.n_pitches, {name_select} as name 
    FROM swing_ratings s
    {join_clause}
    WHERE s.n_pitches > 500 {season_filter}
    ORDER BY s.swing_rating DESC
""", conn)

if len(lb) == 0:
    conn.close()
    print(f"No records found for season {args.year}.")
    exit(0)

lb = lb.reset_index(drop=True)
lb['rank'] = lb.index + 1

top10 = lb.head(10).copy()
bot10 = lb.tail(10).copy()
bot10['rank'] = range(len(lb) - 9, len(lb) + 1)

for df in [top10, bot10]:
    names = []
    for _, r in df.iterrows():
        n = r.get('name')
        if pd.isna(n) or not n or str(n).strip() == str(r['batter_id']):
            n = get_name(int(r['batter_id']), conn=conn)
        names.append(n)
    df['name'] = names

conn.close()

def make_table(df, title, filename, color_accent):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    ax.text(0.5, 0.95, title, ha='center', va='top', fontsize=18, fontweight='bold', color='#1a1a2e')
    
    headers = ['Rank', 'Player', 'Rating', 'Tendency DVA', 'Actual DVA', 'Pitches']
    col_x = [0.06, 0.27, 0.48, 0.62, 0.78, 0.93]
    
    header_y = 0.84
    for i, (h, x) in enumerate(zip(headers, col_x)):
        ax.text(x, header_y, h, ha='center', va='center', fontsize=10, fontweight='bold', color='#555555')
    
    ax.plot([0.02, 0.98], [header_y - 0.03, header_y - 0.03], color='#cccccc', linewidth=1)
    
    row_height = 0.065
    start_y = header_y - 0.08
    
    for i, (_, row) in enumerate(df.iterrows()):
        y = start_y - i * row_height
        
        if i % 2 == 0:
            rect = plt.Rectangle((0.02, y - 0.025), 0.96, row_height, 
                                  facecolor='#f8f9fa', edgecolor='none', transform=ax.transAxes)
            ax.add_patch(rect)
        
        ax.text(col_x[0], y, str(int(row['rank'])), ha='center', va='center', fontsize=10, color='#333333')
        ax.text(col_x[1], y, row['name'], ha='center', va='center', fontsize=10, fontweight='bold', color='#1a1a2e')
        ax.text(col_x[2], y, f"{row['swing_rating']:.1f}", ha='center', va='center', fontsize=11, fontweight='bold', color=color_accent)
        
        tend_color = '#2d6a4f' if row['tendency_dva'] > 0 else '#ae2012'
        act_color = '#2d6a4f' if row['actual_dva'] > 0 else '#ae2012'
        ax.text(col_x[3], y, f"{row['tendency_dva']:+.4f}", ha='center', va='center', fontsize=10, color=tend_color)
        ax.text(col_x[4], y, f"{row['actual_dva']:+.4f}", ha='center', va='center', fontsize=10, color=act_color)
        ax.text(col_x[5], y, str(int(row['n_pitches'])), ha='center', va='center', fontsize=10, color='#666666')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved {filename}")

suffix = f"_{args.year}" if args.year != 2025 else ""
assets_dir = os.path.join(_PROJECT_ROOT, "assets")
os.makedirs(assets_dir, exist_ok=True)
top_file = os.path.join(assets_dir, f"dva_top10{suffix}.png")
bot_file = os.path.join(assets_dir, f"dva_bottom10{suffix}.png")
make_table(top10, f'{args.year} DVA Leaders — Top 10', top_file, '#2d6a4f')
make_table(bot10, f'{args.year} DVA Leaders — Bottom 10', bot_file, '#ae2012')

if args.year == 2025:
    top_file_2025 = os.path.join(assets_dir, "dva_top10_2025.png")
    bot_file_2025 = os.path.join(assets_dir, "dva_bottom10_2025.png")
    make_table(top10, f'{args.year} DVA Leaders — Top 10', top_file_2025, '#2d6a4f')
    make_table(bot10, f'{args.year} DVA Leaders — Bottom 10', bot_file_2025, '#ae2012')


