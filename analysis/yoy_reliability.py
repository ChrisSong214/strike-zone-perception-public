import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

from analysis.common_loader import load_merged_data, load_all_swing_ratings


def compute_yoy_correlations(df, metrics=None):
    """Computes consecutive year-over-year correlations across paired seasons."""
    if metrics is None:
        metrics = ['swing_rating', 'tendency_dva', 'actual_dva', 'chase_rate', 'zone_swing_rate', 'woba']
        
    seasons = sorted(df['season'].unique())
    pairs = []
    
    for i in range(len(seasons) - 1):
        s1, s2 = seasons[i], seasons[i+1]
        df1 = df[df['season'] == s1].copy()
        df2 = df[df['season'] == s2].copy()
        
        m = pd.merge(df1, df2, on='batter_id', suffixes=('_s1', '_s2'))
        if len(m) >= 25:
            m['pair'] = f"{s1}→{s2}"
            pairs.append(m)
            
    if not pairs:
        return None, None
        
    paired_df = pd.concat(pairs, ignore_index=True)
    
    results = []
    for metric in metrics:
        col1 = f"{metric}_s1"
        col2 = f"{metric}_s2"
        if col1 in paired_df.columns and col2 in paired_df.columns:
            valid = paired_df[[col1, col2]].dropna()
            r, p = stats.pearsonr(valid[col1], valid[col2])
            rho, _ = stats.spearmanr(valid[col1], valid[col2])
            results.append({
                'Metric': metric,
                'N_Pairs': len(valid),
                'Pearson_r': round(r, 4),
                'R_squared': round(r**2, 4),
                'Spearman_rho': round(rho, 4),
                'p_value': p
            })
            
    res_df = pd.DataFrame(results).sort_values('Pearson_r', ascending=False)
    return res_df, paired_df


def plot_yoy_grid(paired_df, save_path):
    """Plots a 4-panel comparison of Year-over-Year stability."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    panels = [
        ('swing_rating', 'Overall DVA Rating (Year N vs N+1)', '#1d3557', axes[0, 0]),
        ('tendency_dva', 'Macro Tendency DVA (Year N vs N+1)', '#2a9d8f', axes[0, 1]),
        ('chase_rate', 'Traditional Chase Rate (Year N vs N+1)', '#e76f51', axes[1, 0]),
        ('woba', 'Offensive Production wOBA (Year N vs N+1)', '#457b9d', axes[1, 1]),
    ]
    
    for metric, title, color, ax in panels:
        c1, c2 = f"{metric}_s1", f"{metric}_s2"
        if c1 in paired_df.columns and c2 in paired_df.columns:
            v = paired_df[[c1, c2]].dropna()
            r, _ = stats.pearsonr(v[c1], v[c2])
            
            sns.regplot(data=v, x=c1, y=c2, ax=ax,
                        scatter_kws={'alpha': 0.45, 'color': color, 's': 28},
                        line_kws={'color': '#111111', 'linewidth': 2, 'linestyle': '--'})
            
            ax.set_title(f"{title}\n$r = {r:.3f}$  ($R^2 = {r**2:.3f}$, N={len(v)})",
                         fontsize=12, fontweight='bold', pad=10)
            ax.set_xlabel(f"Season N", fontsize=11, fontweight='bold')
            ax.set_ylabel(f"Season N+1", fontsize=11, fontweight='bold')
        else:
            ax.text(0.5, 0.5, f"Metric {metric} not available", ha='center', va='center')
            
    plt.suptitle("Year-over-Year Metric Reliability (YoY Persistence: True Talent vs. Noise)",
                 fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved YoY plot to {save_path}")


def main():
    print("=" * 70)
    print("  ANALYSIS 1: Year-over-Year Metric Reliability & True Talent Stickiness")
    print("=" * 70)
    
    try:
        df = load_merged_data(min_pitches=300, min_pa=100)
    except Exception as e:
        print(f"Loading merged data failed ({e}); falling back to swing ratings only...")
        df = load_all_swing_ratings(min_pitches=300)
        
    seasons = sorted(df['season'].unique())
    print(f"Available seasons in dataset: {seasons} (Total player-seasons: {len(df):,})")
    
    if len(seasons) < 2:
        print(f"\n[Note] Only {len(seasons)} season ({seasons[0]}) detected in local environment.")
        print("To compute multi-year YoY correlations (2020-2025), run this script on the UCSF VM")
        print("where all 6 season leaderboards are generated.")
        return
        
    res_df, paired_df = compute_yoy_correlations(df)
    
    if res_df is not None:
        print("\n--- YEAR-OVER-YEAR AUTOCORRELATION TABLE ---")
        print(res_df.to_string(index=False))
        
        out_plot = os.path.join(_PROJECT_ROOT, 'assets', 'presentation', 'yoy_reliability.png')
        plot_yoy_grid(paired_df, out_plot)
        
        # Save summary CSV
        csv_out = os.path.join(_PROJECT_ROOT, 'assets', 'presentation', 'yoy_reliability_summary.csv')
        res_df.to_csv(csv_out, index=False)
        print(f"Saved summary metrics to {csv_out}")
    else:
        print("Insufficient overlapping batter pairs across consecutive seasons.")


if __name__ == "__main__":
    main()
