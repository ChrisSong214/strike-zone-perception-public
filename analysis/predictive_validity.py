import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

from analysis.common_loader import load_merged_data, load_all_swing_ratings


def evaluate_predictive_power(df, target_cols=['woba', 'bb_pct', 'k_pct']):
    """
    Fits regressions comparing Baseline (Outcome_N + ChaseRate_N)
    vs Extended (Outcome_N + ChaseRate_N + DVA_N) predicting Outcome_{N+1}.
    """
    seasons = sorted(df['season'].unique())
    pairs = []
    
    for i in range(len(seasons) - 1):
        s1, s2 = seasons[i], seasons[i+1]
        df1 = df[df['season'] == s1].copy()
        df2 = df[df['season'] == s2].copy()
        
        m = pd.merge(df1, df2, on='batter_id', suffixes=('_s1', '_s2'))
        if len(m) >= 20:
            pairs.append(m)
            
    if not pairs:
        return None, None
        
    paired_df = pd.concat(pairs, ignore_index=True)
    
    records = []
    
    for target in target_cols:
        t_s1 = f"{target}_s1"
        t_s2 = f"{target}_s2"
        chase_s1 = "chase_rate_s1"
        dva_s1 = "swing_rating_s1"
        
        required = [t_s1, t_s2, chase_s1, dva_s1]
        if 'z_minus_o_s1' in paired_df.columns:
            required.append('z_minus_o_s1')
        if not all(c in paired_df.columns for c in required):
            continue
            
        sub = paired_df[required].dropna()
        n = len(sub)
        if n < 30:
            continue
            
        y = sub[t_s2].values
        
        # Model 1: Baseline (Prior Outcome + Chase Rate)
        X1 = sub[[t_s1, chase_s1]].values
        reg1 = LinearRegression().fit(X1, y)
        r2_base = reg1.score(X1, y)
        
        # Model 1b: Baseline with Z-O Swing% (Prior Outcome + Z-O)
        has_zo = 'z_minus_o_s1' in sub.columns
        if has_zo:
            X_zo = sub[[t_s1, 'z_minus_o_s1']].values
            reg_zo = LinearRegression().fit(X_zo, y)
            r2_zo = reg_zo.score(X_zo, y)
        else:
            r2_zo = r2_base
        
        # Model 2: Extended (Prior Outcome + Chase Rate + DVA)
        X2 = sub[[t_s1, chase_s1, dva_s1]].values
        reg2 = LinearRegression().fit(X2, y)
        r2_full = reg2.score(X2, y)
        
        delta_r2 = max(0.0, r2_full - r2_base)
        
        # F-test for incremental variance
        p1 = X1.shape[1]
        p2 = X2.shape[1]
        f_stat = ((r2_full - r2_base) / (p2 - p1)) / ((1 - r2_full) / (n - p2 - 1)) if r2_full < 1.0 else 0
        f_pval = 1.0 - stats.f.cdf(f_stat, p2 - p1, n - p2 - 1)
        
        records.append({
            'Target': target.upper(),
            'N_Batters': n,
            'R2_Baseline': round(r2_base, 4),
            'R2_Baseline_Chase': round(r2_base, 4),
            'R2_Baseline_ZO': round(r2_zo, 4),
            'R2_With_DVA': round(r2_full, 4),
            'Delta_R2': round(delta_r2, 4),
            'Pct_Improvement': round((delta_r2 / max(r2_base, 1e-4)) * 100, 1),
            'F_Stat': round(f_stat, 2),
            'p_value': f_pval
        })
        
    return pd.DataFrame(records), paired_df


def plot_predictive_power(results_df, save_path):
    """Plots incremental R^2 bar chart showing how DVA enhances forecasting vs Chase Rate and Z-O Swing%."""
    fig, ax = plt.subplots(figsize=(11, 6.5))
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    x = np.arange(len(results_df))
    width = 0.26
    
    has_zo = 'R2_Baseline_ZO' in results_df.columns
    
    if has_zo:
        r1 = ax.bar(x - width, results_df['R2_Baseline_Chase'], width, label='Baseline: Stat + Chase Rate', color='#6c757d', alpha=0.85)
        r2 = ax.bar(x, results_df['R2_Baseline_ZO'], width, label='Baseline: Stat + Z-O Swing%', color='#b07d62', alpha=0.85)
        r3 = ax.bar(x + width, results_df['R2_With_DVA'], width, label='Extended: + DVA Decision Engine', color='#2a9d8f', alpha=0.95)
    else:
        r1 = ax.bar(x - width/2, results_df['R2_Baseline'], width, label='Baseline: Stat + Chase Rate', color='#6c757d', alpha=0.85)
        r3 = ax.bar(x + width/2, results_df['R2_With_DVA'], width, label='Extended: + DVA Decision Engine', color='#2a9d8f', alpha=0.95)
    
    ax.set_ylabel('Out-of-Sample Predictive $R^2$ ($N \to N+1$)', fontsize=12, fontweight='bold')
    ax.set_title('Forecasting Future Offensive Production: DVA vs. Traditional Plate Discipline', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['Target'], fontsize=12, fontweight='bold')
    ax.legend(fontsize=10.5, frameon=True)
    ax.set_ylim(0, max(results_df['R2_With_DVA'].max() * 1.35, 0.35))
    
    offset = width if has_zo else width/2
    for i, row in results_df.iterrows():
        gain = f"+{row['Delta_R2']:.3f}\n(+{row['Pct_Improvement']}%)"
        ax.text(i + offset, row['R2_With_DVA'] + 0.008, gain, ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#1d3557')
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved predictive power plot to {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Out-of-Sample Predictive Validity")
    parser.add_argument("--min-pitches", type=int, default=750, help="Minimum pitches seen in season (default: 750)")
    args = parser.parse_args()

    print("=" * 70)
    print(f"  ANALYSIS 2: Out-of-Sample Predictive Validity (Min Pitches >= {args.min_pitches})")
    print("=" * 70)
    
    min_pa = max(50, int(args.min_pitches / 3.8))
    try:
        df = load_merged_data(min_pitches=args.min_pitches, min_pa=min_pa)
    except Exception as e:
        print(f"Loading merged data failed ({e}).")
        return
        
    seasons = sorted(df['season'].unique())
    print(f"Available seasons: {seasons} (Total player-seasons: {len(df):,})")

    
    if len(seasons) < 2:
        print(f"\n[Note] Multi-year forecasting requires at least 2 consecutive seasons.")
        print("Run this script on the UCSF VM to evaluate across all 2020-2025 seasons.")
        return
        
    results_df, _ = evaluate_predictive_power(df)
    
    if results_df is not None and len(results_df) > 0:
        print("\n--- OUT-OF-SAMPLE INCREMENTAL PREDICTION (YEAR N -> N+1) ---")
        print(results_df.to_string(index=False))
        
        out_plot = os.path.join(_PROJECT_ROOT, 'assets', 'presentation', 'predictive_power.png')
        plot_predictive_power(results_df, out_plot)
        
        csv_out = os.path.join(_PROJECT_ROOT, 'assets', 'presentation', 'predictive_power_summary.csv')
        results_df.to_csv(csv_out, index=False)
        print(f"Saved summary metrics to {csv_out}")
    else:
        print("Insufficient multi-year data available locally.")


if __name__ == "__main__":
    main()
