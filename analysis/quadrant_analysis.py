import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    from adjustText import adjust_text
except ImportError:
    adjust_text = None


_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

from analysis.common_loader import load_all_swing_ratings


def plot_quadrants(df, year=2025, save_path=None):
    """Generates the 2D Tendency vs. Execution talent decomposition quadrant chart."""
    sub = df[df['season'] == year].copy() if 'season' in df.columns else df.copy()
    
    if len(sub) == 0:
        sub = df.copy()
        
    sub = sub[sub['n_pitches'] >= 500].copy()
    print(f"Plotting quadrant archetypes for {len(sub)} qualified batters (season {year})...")
    
    fig, ax = plt.subplots(figsize=(13, 9))
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    x = sub['tendency_dva']
    y = sub['actual_dva']
    colors = sub['swing_rating']
    
    # Midpoints for quadrants (medians)
    x_mid = x.median()
    y_mid = y.median()
    
    scatter = ax.scatter(x, y, c=colors, cmap='viridis', s=sub['n_pitches'] / 35,
                         alpha=0.65, edgecolors='none')
                         
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Overall DVA Rating (100 = League Avg)', fontsize=11, fontweight='bold')
    
    # Quadrant dividing lines
    ax.axvline(x_mid, color='#495057', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axhline(y_mid, color='#495057', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Quadrant Labels
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    bbox_props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#cccccc', alpha=0.9)
    
    ax.text(x_max * 0.95, y_max * 0.95, 'QUADRANT I: THE ELITES\nHigh Macro Discipline + Elite Pitch Execution\n(Rarely Chase, Punish Strikes)',
            ha='right', va='top', fontsize=10, fontweight='bold', color='#1b4332', bbox=bbox_props)
            
    ax.text(x_min * 0.95, y_max * 0.95, 'QUADRANT II: THE PASSIVE SNIPERS\nLow Swing Tendency + Elite Pitch Execution\n(High Take Quality, Overly Passive)',
            ha='left', va='top', fontsize=10, fontweight='bold', color='#0077b6', bbox=bbox_props)
            
    ax.text(x_max * 0.95, y_min * 0.95, 'QUADRANT III: THE AMBUSH SLUGGERS\nHigh Macro Aggression + Foiled by Movement\n(Aggressive Plan, Fooled by Spin)',
            ha='right', va='bottom', fontsize=10, fontweight='bold', color='#d00000', bbox=bbox_props)
            
    ax.text(x_min * 0.95, y_min * 0.95, 'QUADRANT IV: THE OVERMATCHED\nPoor Count Discipline + Fooled on Execution\n(High Chase, Low Contact Value)',
            ha='left', va='bottom', fontsize=10, fontweight='bold', color='#6a040f', bbox=bbox_props)
            
    # Highlight notable stars / extreme batters
    notable = []
    
    # Top 5 and Bottom 5 DVA
    top_dva = sub.sort_values('swing_rating', ascending=False).head(8)
    bot_dva = sub.sort_values('swing_rating', ascending=True).head(5)
    
    # Extreme execution vs tendency
    top_snipers = sub[(sub['actual_dva'] > sub['actual_dva'].quantile(0.85)) & (sub['tendency_dva'] < sub['tendency_dva'].quantile(0.35))].head(3)
    
    tagged = pd.concat([top_dva, bot_dva, top_snipers]).drop_duplicates(subset=['batter_id'])
    
    texts = []
    for _, r in tagged.iterrows():
        pname = r['player_name'] if pd.notna(r.get('player_name')) else f"ID:{r['batter_id']}"
        t = ax.text(r['tendency_dva'], r['actual_dva'], pname, fontsize=9.5, fontweight='bold', color='#111111')
        texts.append(t)
        
    if adjust_text is not None:
        try:
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='->', color='#333333', lw=1.0))
        except Exception:
            pass

        
    ax.set_title(f'{year} MLB Batter Talent Decomposition: Macro Tendency vs. Micro Execution DVA',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Macro Tendency DVA (Prior Strike Zone Awareness & Count Leverage)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Micro Actual DVA (Borderline Pitch Decision Execution & Visual Precision)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved quadrant chart to {save_path}")


def main():
    print("=" * 70)
    print("  ANALYSIS 4: 2D Talent Decomposition (Tendency vs. Execution Quadrants)")
    print("=" * 70)
    
    df = load_all_swing_ratings(min_pitches=300)
    out_plot = os.path.join(_PROJECT_ROOT, 'assets', 'presentation', 'quadrant_archetypes.png')
    plot_quadrants(df, year=2025, save_path=out_plot)


if __name__ == "__main__":
    main()
