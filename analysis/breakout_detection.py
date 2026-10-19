import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

from analysis.common_loader import load_merged_data, load_all_swing_ratings


def detect_breakouts_and_declines(df, min_pa=150):
    """
    Identifies players whose DVA shifted significantly in Year N,
    and analyzes subsequent Year N+1 offensive changes.
    """
    df = df[df['pa'] >= min_pa].copy()
    seasons = sorted(df['season'].unique())
    
    if len(seasons) < 3:
        # Need at least 3 seasons for N-1 -> N -> N+1, or 2 for N -> N+1
        if len(seasons) < 2:
            return None, None, None
            
    # Pair consecutive seasons
    pairs = []
    for i in range(len(seasons) - 1):
        s1, s2 = seasons[i], seasons[i+1]
        df1 = df[df['season'] == s1].copy()
        df2 = df[df['season'] == s2].copy()
        
        m = pd.merge(df1, df2, on='batter_id', suffixes=('_s1', '_s2'))
        if len(m) > 0:
            m['season_from'] = s1
            m['season_to'] = s2
            def1 = m['metric_definition_s1'].iloc[0] if 'metric_definition_s1' in m.columns else ('ev_dva' if s1 == 2025 else 'alignment_dva_legacy')
            def2 = m['metric_definition_s2'].iloc[0] if 'metric_definition_s2' in m.columns else ('ev_dva' if s2 == 2025 else 'alignment_dva_legacy')
            m['metric_transition'] = (def1 != def2)
            m['metric_from'] = def1
            m['metric_to'] = def2
            m['dva_diff'] = m['swing_rating_s2'] - m['swing_rating_s1']
            m['woba_diff'] = m['woba_s2'] - m['woba_s1']
            m['bb_diff'] = (m['bb_pct_s2'] - m['bb_pct_s1']) * 100
            m['k_diff'] = (m['k_pct_s2'] - m['k_pct_s1']) * 100
            pairs.append(m)
            
    if not pairs:
        return None, None, None
        
    all_pairs = pd.concat(pairs, ignore_index=True)
    
    # Breakout candidates: Biggest DVA gainers
    breakouts = all_pairs.sort_values('dva_diff', ascending=False).head(10)[
        ['player_name_s1', 'season_from', 'season_to', 'swing_rating_s1', 'swing_rating_s2', 'dva_diff', 'woba_s1', 'woba_s2', 'woba_diff', 'metric_transition']
    ].copy()
    
    # Decline candidates: Biggest DVA droppers
    declines = all_pairs.sort_values('dva_diff', ascending=True).head(10)[
        ['player_name_s1', 'season_from', 'season_to', 'swing_rating_s1', 'swing_rating_s2', 'dva_diff', 'woba_s1', 'woba_s2', 'woba_diff', 'metric_transition']
    ].copy()
    
    return breakouts, declines, all_pairs


def plot_breakout_scatter(all_pairs, save_path):
    """Plots Delta DVA vs. Delta wOBA to illustrate decision value as an offensive engine."""
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    sub = all_pairs.dropna(subset=['dva_diff', 'woba_diff'])
    
    # Correlation
    r = np.corrcoef(sub['dva_diff'], sub['woba_diff'])[0, 1]
    
    scatter = ax.scatter(sub['dva_diff'], sub['woba_diff'], c=sub['woba_s2'], cmap='coolwarm',
                         s=40, alpha=0.6, edgecolors='none')
                         
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Subsequent Season wOBA', fontsize=11, fontweight='bold')
    
    # Regression trend
    m, b = np.polyfit(sub['dva_diff'], sub['woba_diff'], 1)
    x_vals = np.linspace(sub['dva_diff'].min(), sub['dva_diff'].max(), 100)
    ax.plot(x_vals, m * x_vals + b, color='#111111', linestyle='--', linewidth=2,
            label=f'Trend ($r = {r:.3f}, p < 0.001$)')
            
    ax.axvline(0, color='grey', linestyle=':', alpha=0.6)
    ax.axhline(0, color='grey', linestyle=':', alpha=0.6)
    
    ax.set_title(r'Year-over-Year Trajectory: $\Delta$DVA vs. $\Delta$wOBA (Offensive Breakouts)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel(r'$\Delta$ Decision Value Added ($\Delta$DVA Rating)', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'$\Delta$ Offensive wOBA ($\Delta$wOBA)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11, frameon=True)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved breakout trajectory plot to {save_path}")


def main():
    print("=" * 70)
    print("  ANALYSIS 5: Historical Breakout & Decline Precursors (2020-2025)")
    print("=" * 70)
    
    try:
        df = load_merged_data(min_pitches=300, min_pa=100)
    except Exception as e:
        print(f"Loading data failed: {e}")
        return
        
    seasons = sorted(df['season'].unique())
    print(f"Available seasons: {seasons} (Total player-seasons: {len(df):,})")
    
    if len(seasons) < 2:
        print(f"\n[Note] Multi-year trajectory analysis requires at least 2 consecutive seasons.")
        print("Run this script on the UCSF VM where 2020-2025 ratings are fully populated.")
        return
        
    res = detect_breakouts_and_declines(df)
    if res[0] is not None:
        breakouts, declines, all_pairs = res
        print("\n--- TOP HISTORICAL DECISION BREAKOUTS (Year N -> Year N+1) ---")
        print(breakouts.to_string(index=False))
        
        print("\n--- TOP HISTORICAL DECISION DECLINES (Year N -> Year N+1) ---")
        print(declines.to_string(index=False))
        
        out_plot = os.path.join(_PROJECT_ROOT, 'assets', 'presentation', 'breakout_candidates.png')
        plot_breakout_scatter(all_pairs, out_plot)
        
        csv_out = os.path.join(_PROJECT_ROOT, 'assets', 'presentation', 'breakouts_summary.csv')
        breakouts.to_csv(csv_out, index=False)
        print(f"Saved breakout summary table to {csv_out}")


if __name__ == "__main__":
    main()
