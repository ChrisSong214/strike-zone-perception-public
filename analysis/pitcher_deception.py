import os
import sys
import glob
import urllib.request
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)


def get_pitcher_names(pitcher_ids):
    """Fetches pitcher names from MLB Stats API."""
    names = {}
    print(f"Fetching names for {len(pitcher_ids)} top pitchers from MLB Stats API...")
    for idx, pid in enumerate(pitcher_ids):
        url = f"https://statsapi.mlb.com/api/v1/people/{pid}"
        try:
            req = urllib.request.urlopen(url, timeout=5)
            data = json.loads(req.read())
            names[pid] = data['people'][0]['fullName']
        except Exception:
            names[pid] = f"ID: {pid}"
        if (idx + 1) % 10 == 0:
            print(f"  Fetched {idx+1}/{len(pitcher_ids)} names...")
    return names


def evaluate_pitcher_deception(parquet_path, min_pitches=500):
    """Evaluates how effectively each pitcher induces negative decision value in batters."""
    print(f"Loading pitch features from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    # Filter qualified pitchers
    counts = df.groupby('pitcher').size()
    qualified = counts[counts >= min_pitches].index.values
    print(f"Qualified pitchers (>= {min_pitches} pitches): {len(qualified)}")
    
    sub = df[df['pitcher'].isin(qualified)].copy()
    
    # Batter decision value per pitch:
    # swing on strike (p > 0.5) = positive, take on ball (p <= 0.5) = positive
    # swing on ball (p <= 0.5) = negative, take on strike (p > 0.5) = negative
    sub['batter_dva_pitch'] = np.where(
        sub['is_swing'] == 1,
        sub['lstm_p_strike'] - 0.5,
        0.5 - sub['lstm_p_strike']
    )
    
    # Aggregation per pitcher
    grouped = sub.groupby('pitcher').agg(
        n_pitches=('pitcher', 'count'),
        mean_batter_dva=('batter_dva_pitch', 'mean'),
        chase_rate=('is_swing', lambda s: s[sub.loc[s.index, 'in_zone'] == 0].mean()),
        frozen_strike_rate=('is_swing', lambda s: (1 - s[sub.loc[s.index, 'in_zone'] == 1]).mean()),
        mean_sigma_x=('lstm_sigma_x', 'mean'),
        mean_sigma_z=('lstm_sigma_z', 'mean')
    ).reset_index()
    
    # The lower the mean_batter_dva, the more deceptive the pitcher is!
    # Let Deception Index = -mean_batter_dva * 100
    grouped['deception_index'] = -grouped['mean_batter_dva'] * 100
    grouped = grouped.sort_values('deception_index', ascending=False).reset_index(drop=True)
    grouped['rank'] = grouped.index + 1
    
    # Get names for top 20
    top20_ids = grouped.head(20)['pitcher'].tolist()
    name_map = get_pitcher_names(top20_ids)
    grouped['pitcher_name'] = grouped['pitcher'].map(name_map).fillna(grouped['pitcher'].astype(str))
    
    return grouped


def plot_pitcher_deception(grouped, save_path, year=2025):
    """Plots top 15 most deceptive pitchers table chart."""
    top15 = grouped.head(15).copy()
    
    fig, ax = plt.subplots(figsize=(12, 8), dpi=200)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.96, f"{year} MLB Pitcher Deception Leaders", 
            ha='center', va='center', fontsize=16, fontweight='bold', color='#1d3557')
    ax.text(0.5, 0.92, "Pitchers who force the highest rate of batter decision errors at the 40% flight mark (~175 ms)",
            ha='center', va='center', fontsize=10, style='italic', color='#6c757d')
            
    headers = ['Rank', 'Pitcher Name', 'Deception Score', 'Chase % Induced', 'Frozen Strike %', 'Pitches']
    col_x = [0.06, 0.28, 0.52, 0.68, 0.83, 0.94]
    
    # Draw Header
    ax.axhline(0.87, color='#1d3557', linewidth=2)
    for x_pos, h in zip(col_x, headers):
        ax.text(x_pos, 0.885, h, ha='center', va='center', fontsize=10.5, fontweight='bold', color='#1d3557')
    ax.axhline(0.86, color='#dee2e6', linewidth=1)
    
    # Rows
    y_start = 0.82
    row_height = 0.05
    for i, (_, row) in enumerate(top15.iterrows()):
        y = y_start - i * row_height
        
        # Alternating background
        if i % 2 == 0:
            rect = plt.Rectangle((0.02, y - 0.022), 0.96, row_height,
                                 facecolor='#f8f9fa', edgecolor='none', transform=ax.transAxes)
            ax.add_patch(rect)
            
        ax.text(col_x[0], y, str(int(row['rank'])), ha='center', va='center', fontsize=10, color='#333333')
        ax.text(col_x[1], y, row['pitcher_name'], ha='center', va='center', fontsize=10, fontweight='bold', color='#1a1a2e')
        ax.text(col_x[2], y, f"+{row['deception_index']:.2f}", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#2a9d8f')
        ax.text(col_x[3], y, f"{row['chase_rate']*100:.1f}%", ha='center', va='center', fontsize=10, color='#333333')
        ax.text(col_x[4], y, f"{row['frozen_strike_rate']*100:.1f}%", ha='center', va='center', fontsize=10, color='#333333')
        ax.text(col_x[5], y, f"{int(row['n_pitches']):,}", ha='center', va='center', fontsize=9.5, color='#6c757d')
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved pitcher deception table to {save_path}")


def main():
    print("=" * 70)
    print("  ANALYSIS 6: Pitcher Deception & Induced Bad Decision Index")
    print("=" * 70)
    
    feature_files = glob.glob(os.path.join(_PROJECT_ROOT, 'data', 'training', 'swing_features_*.parquet'))
    if not feature_files:
        print("No swing features found in data/training/")
        return
        
    feature_files.sort()
    target_file = feature_files[-1]
    
    grouped = evaluate_pitcher_deception(target_file, min_pitches=500)
    
    print("\n--- TOP 10 MOST DECEPTIVE PITCHERS (Induced Batter Error Rate) ---")
    display_cols = ['rank', 'pitcher_name', 'deception_index', 'chase_rate', 'frozen_strike_rate', 'n_pitches']
    print(grouped[display_cols].head(10).to_string(index=False))
    
    out_plot = os.path.join(_PROJECT_ROOT, 'assets', 'presentation', 'pitcher_deception.png')
    plot_pitcher_deception(grouped, out_plot, year=2025)
    
    out_csv = os.path.join(_PROJECT_ROOT, 'assets', 'presentation', 'pitcher_deception_leaders.csv')
    grouped.to_csv(out_csv, index=False)
    print(f"Saved pitcher deception rankings to {out_csv}")


if __name__ == "__main__":
    main()
