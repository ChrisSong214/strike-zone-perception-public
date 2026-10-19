import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score

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
        
        required = ['batter_id', t_s1, t_s2, chase_s1, dva_s1]
        if 'z_minus_o_s1' in paired_df.columns:
            required.append('z_minus_o_s1')
        if not all(c in paired_df.columns for c in required):
            continue
            
        sub = paired_df[required].dropna()
        n = len(sub)
        if n < 30:
            continue
            
        y = sub[t_s2].values
        groups = sub['batter_id'].values
        n_groups = len(np.unique(groups))
        
        # GroupKFold by batter_id to prevent leaking across multi-year pairs (Finding 4)
        from sklearn.model_selection import GroupKFold
        n_splits = min(5, n_groups)
        gkf = GroupKFold(n_splits=n_splits)
        
        # Model 1: Baseline (Prior Outcome + Chase Rate)
        X1 = sub[[t_s1, chase_s1]].values
        y_pred1 = cross_val_predict(LinearRegression(), X1, y, cv=gkf, groups=groups)
        r2_base = float(r2_score(y, y_pred1))
        
        # Model 1b: Baseline with Z-O Swing% (Prior Outcome + Z-O)
        has_zo = 'z_minus_o_s1' in sub.columns
        if has_zo:
            X_zo = sub[[t_s1, 'z_minus_o_s1']].values
            y_pred_zo = cross_val_predict(LinearRegression(), X_zo, y, cv=gkf, groups=groups)
            r2_zo = float(r2_score(y, y_pred_zo))
        else:
            r2_zo = r2_base
        
        # Model 2: Extended (Prior Outcome + Chase Rate + DVA)
        X2 = sub[[t_s1, chase_s1, dva_s1]].values
        y_pred2 = cross_val_predict(LinearRegression(), X2, y, cv=gkf, groups=groups)
        r2_full = float(r2_score(y, y_pred2))
        
        # Uncensored out-of-sample change (Finding 3: do NOT clamp at 0)
        delta_r2 = float(r2_full - r2_base)
        
        # In-sample nested OLS fit for classical F-test (Finding 5)
        reg1_full = LinearRegression().fit(X1, y)
        reg2_full = LinearRegression().fit(X2, y)
        r2_in1 = reg1_full.score(X1, y)
        r2_in2 = reg2_full.score(X2, y)
        p1 = X1.shape[1]
        p2 = X2.shape[1]
        in_sample_f = ((r2_in2 - r2_in1) / (p2 - p1)) / ((1 - r2_in2) / (n - p2 - 1)) if r2_in2 < 1.0 else 0.0
        in_sample_p = float(1.0 - stats.f.cdf(in_sample_f, p2 - p1, n - p2 - 1))
        
        # Out-of-sample batter-clustered bootstrap for 95% CI (Finding 5)
        unique_batters = np.unique(groups)
        rng = np.random.RandomState(42)
        boot_deltas = []
        n_boot = min(500, len(unique_batters) * 5)
        
        batter_idx_map = {b: np.where(groups == b)[0] for b in unique_batters}
        for _ in range(n_boot):
            boot_b = rng.choice(unique_batters, size=len(unique_batters), replace=True)
            boot_indices = np.concatenate([batter_idx_map[b] for b in boot_b])
            y_b = y[boot_indices]
            r2_b1 = r2_score(y_b, y_pred1[boot_indices])
            r2_b2 = r2_score(y_b, y_pred2[boot_indices])
            boot_deltas.append(r2_b2 - r2_b1)
            
        boot_deltas = np.array(boot_deltas)
        ci_lower = float(np.percentile(boot_deltas, 2.5))
        ci_upper = float(np.percentile(boot_deltas, 97.5))
        boot_pval = float(np.mean(boot_deltas <= 0.0))
        
        records.append({
            'Target': target.upper(),
            'N_Batters': n,
            'R2_Baseline': round(r2_base, 4),
            'R2_Baseline_Chase': round(r2_base, 4),
            'R2_Baseline_ZO': round(r2_zo, 4),
            'R2_With_DVA': round(r2_full, 4),
            'Delta_R2': round(delta_r2, 4),
            'Delta_R2_CI_Lower': round(ci_lower, 4),
            'Delta_R2_CI_Upper': round(ci_upper, 4),
            'Boot_p_value': round(boot_pval, 4),
            'Pct_Improvement': round((delta_r2 / max(abs(r2_base), 1e-4)) * 100, 1),
            'In_Sample_F': round(in_sample_f, 2),
            'In_Sample_p': in_sample_p,
            'F_Stat': round(in_sample_f, 2),  # Backwards compatibility alias
            'p_value': in_sample_p            # Backwards compatibility alias
        })
        
    return pd.DataFrame(records), paired_df


def evaluate_rolling_temporal_holdout(df, target_cols=['woba', 'bb_pct', 'k_pct']):
    """
    Evaluates strictly chronological holdout pairs (train on pairs up to Y-1, test on Y).
    """
    seasons = sorted(df['season'].unique())
    if len(seasons) < 3:
        return None
        
    pairs = []
    for i in range(len(seasons) - 1):
        s1, s2 = seasons[i], seasons[i+1]
        df1 = df[df['season'] == s1].copy()
        df2 = df[df['season'] == s2].copy()
        m = pd.merge(df1, df2, on='batter_id', suffixes=('_s1', '_s2'))
        if len(m) >= 20:
            m['test_season'] = s2
            pairs.append(m)
            
    if not pairs:
        return None
        
    paired_df = pd.concat(pairs, ignore_index=True)
    test_seasons = sorted(paired_df['test_season'].unique())[1:]  # Needs at least one prior pair
    
    records = []
    for test_yr in test_seasons:
        train_df = paired_df[paired_df['test_season'] < test_yr]
        test_df = paired_df[paired_df['test_season'] == test_yr]
        
        for target in target_cols:
            t_s1, t_s2 = f"{target}_s1", f"{target}_s2"
            chase_s1, dva_s1 = "chase_rate_s1", "swing_rating_s1"
            req = [t_s1, t_s2, chase_s1, dva_s1]
            if not all(c in train_df.columns and c in test_df.columns for c in req):
                continue
                
            tr = train_df[req].dropna()
            te = test_df[req].dropna()
            if len(tr) < 30 or len(te) < 20:
                continue
                
            m_base = LinearRegression().fit(tr[[t_s1, chase_s1]], tr[t_s2])
            m_full = LinearRegression().fit(tr[[t_s1, chase_s1, dva_s1]], tr[t_s2])
            
            r2_base = r2_score(te[t_s2], m_base.predict(te[[t_s1, chase_s1]]))
            r2_full = r2_score(te[t_s2], m_full.predict(te[[t_s1, chase_s1, dva_s1]]))
            
            records.append({
                'Test_Year': test_yr,
                'Target': target.upper(),
                'N_Test': len(te),
                'R2_Base': round(r2_base, 4),
                'R2_With_DVA': round(r2_full, 4),
                'Delta_R2': round(r2_full - r2_base, 4)
            })
            
    return pd.DataFrame(records) if records else None


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
