import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)


from src.swing_model import (
    LeagueSwingModel, BayesianBatterModel, 
    compute_dva, estimate_eb_shrinkage
)


def compute_split_half_curve(parquet_path, sample_sizes=[50, 100, 200, 300, 500, 750, 1000]):
    """Computes split-half reliability curves for production DVA vs traditional chase rate."""
    print(f"Loading pitch-level features from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    # Check necessary columns
    required = ['batter', 'is_swing', 'in_zone', 'lstm_p_strike']
    if not all(c in df.columns for c in required):
        raise ValueError(f"Missing required columns in {parquet_path}: {[c for c in required if c not in df.columns]}")
        
    print(f"Loaded {len(df):,} pitches across {df['batter'].nunique():,} unique batters.")
    
    # Filter batters with at least max(sample_sizes) pitches
    counts = df.groupby('batter').size()
    qualified_batters = counts[counts >= max(sample_sizes)].index.values
    print(f"Qualified batters with >= {max(sample_sizes)} pitches: {len(qualified_batters)}")
    
    if len(qualified_batters) < 25:
        # Fall back to lower max threshold if needed
        max_possible = int(counts.quantile(0.75))
        sample_sizes = [s for s in sample_sizes if s <= max_possible]
        qualified_batters = counts[counts >= max(sample_sizes)].index.values
        print(f"Adjusted sample sizes: {sample_sizes} (Batters: {len(qualified_batters)})")
        
    sub_df = df[df['batter'].isin(qualified_batters)].copy()
    
    # Sort chronologically by game and pitch
    sort_cols = [c for c in ['game_pk', 'at_bat_number', 'pitch_number'] if c in sub_df.columns]
    if sort_cols:
        sub_df = sub_df.sort_values(sort_cols)
        
    # Ensure production DVA metrics (EV-DVA & Alignment DVA) are computed
    if 'ev_dva' not in sub_df.columns:
        print("Scoring pitches with production LeagueSwingModel and BayesianBatterModel...")
        model_path = os.path.join(_PROJECT_ROOT, 'data', 'models', 'swing_model_2025')
        if not os.path.exists(model_path + '.xgb'):
            model_path = os.path.join(_PROJECT_ROOT, 'data', 'models', 'swing_model')
        league_model = LeagueSwingModel()
        league_model.load(model_path)
        bayesian_model = BayesianBatterModel(kappa=30.0)
        bayesian_model.fit(sub_df, league_model.predict(sub_df))
        sub_df = compute_dva(sub_df, league_model, bayesian_model)
        
    k_star_ev, _, _ = estimate_eb_shrinkage(sub_df, residual_col='ev_dva', batter_col='batter')
    k_star_align, _, _ = estimate_eb_shrinkage(sub_df, residual_col='alignment_dva', batter_col='batter')
    print(f"Empirical Bayes shrinkage constants: K*(EV-DVA) = {k_star_ev:.1f}, K*(Alignment) = {k_star_align:.1f}")

    results = []
    
    for N in sample_sizes:
        print(f"  Evaluating stabilization at N = {N} pitches...")
        batters_ev_odd, batters_ev_even = [], []
        batters_align_odd, batters_align_even = [], []
        batters_chase_odd, batters_chase_even = [], []
        
        for b_id, b_pitches in sub_df.groupby('batter'):
            sample = b_pitches.iloc[:N]
            
            # Split odd and even pitches
            odd = sample.iloc[0::2]
            even = sample.iloc[1::2]
            n_o, n_e = len(odd), len(even)
            
            # Production EV-DVA with Empirical Bayes shrinkage
            raw_ev_odd = odd['ev_dva'].mean()
            raw_ev_even = even['ev_dva'].mean()
            dva_ev_odd = (n_o / (n_o + k_star_ev)) * raw_ev_odd
            dva_ev_even = (n_e / (n_e + k_star_ev)) * raw_ev_even
            
            # Production Alignment DVA with Empirical Bayes shrinkage
            raw_align_odd = odd['alignment_dva'].mean()
            raw_align_even = even['alignment_dva'].mean()
            dva_align_odd = (n_o / (n_o + k_star_align)) * raw_align_odd
            dva_align_even = (n_e / (n_e + k_star_align)) * raw_align_even
            
            # Chase rate on out-of-zone pitches
            oz_odd = odd[odd['in_zone'] == 0]
            oz_even = even[even['in_zone'] == 0]
            
            chase_odd = oz_odd['is_swing'].mean() if len(oz_odd) > 5 else np.nan
            chase_even = oz_even['is_swing'].mean() if len(oz_even) > 5 else np.nan
            
            batters_ev_odd.append(dva_ev_odd)
            batters_ev_even.append(dva_ev_even)
            batters_align_odd.append(dva_align_odd)
            batters_align_even.append(dva_align_even)
            batters_chase_odd.append(chase_odd)
            batters_chase_even.append(chase_even)
            
        # Compute Pearson r and Spearman-Brown corrected reliability
        r_ev, _ = stats.pearsonr(batters_ev_odd, batters_ev_even)
        sb_ev = (2 * r_ev) / (1 + r_ev)
        
        r_align, _ = stats.pearsonr(batters_align_odd, batters_align_even)
        sb_align = (2 * r_align) / (1 + r_align)
        
        # Filter valid chase rate pairs
        chase_pairs = [(o, e) for o, e in zip(batters_chase_odd, batters_chase_even) if np.isfinite(o) and np.isfinite(e)]
        if len(chase_pairs) >= 20:
            c_odd, c_even = zip(*chase_pairs)
            r_chase, _ = stats.pearsonr(c_odd, c_even)
            sb_chase = (2 * r_chase) / (1 + r_chase)
        else:
            sb_chase = np.nan
            
        results.append({
            'N_Pitches': N,
            'DVA_SplitHalf_r': round(sb_ev, 4),
            'Alignment_DVA_SplitHalf_r': round(sb_align, 4),
            'Chase_SplitHalf_r': round(sb_chase, 4) if np.isfinite(sb_chase) else None
        })
        
    return pd.DataFrame(results)


def plot_stabilization_curve(res_df, save_path):
    """Plots the stabilization curve comparing production DVA vs Chase Rate."""
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    ax.plot(res_df['N_Pitches'], res_df['DVA_SplitHalf_r'], marker='o', linewidth=2.8, markersize=8,
            color='#2a9d8f', label='Expected Value DVA (EV-DVA)')
            
    if 'Alignment_DVA_SplitHalf_r' in res_df.columns and res_df['Alignment_DVA_SplitHalf_r'].dropna().any():
        ax.plot(res_df['N_Pitches'], res_df['Alignment_DVA_SplitHalf_r'], marker='^', linewidth=2.2, markersize=7,
                color='#1d3557', linestyle='-.', label='Alignment DVA (Zone Recognition)')
            
    if 'Chase_SplitHalf_r' in res_df.columns and res_df['Chase_SplitHalf_r'].dropna().any():
        ax.plot(res_df['N_Pitches'], res_df['Chase_SplitHalf_r'], marker='s', linewidth=2.5, markersize=7,
                color='#e76f51', linestyle='--', label='Traditional Chase Rate (O-Swing%)')
                
    # 0.70 Reliability Threshold
    ax.axhline(0.70, color='#d62828', linestyle=':', linewidth=2, label='Reliability Threshold ($r = 0.70$)')
    ax.text(res_df['N_Pitches'].min() + 10, 0.71, 'Standard Sabermetric Reliability Threshold (r = 0.70)',
            fontsize=10, fontweight='bold', color='#d62828')
            
    ax.set_title('Metric Stabilization Rate: How Quickly Does True Talent Emerge?', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Sample Size (Number of Pitches Observed)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Split-Half Reliability (Spearman-Brown Corrected $r$)', fontsize=12, fontweight='bold')
    ax.set_ylim(0.2, 1.0)
    ax.legend(fontsize=11, frameon=True, loc='lower right')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved stabilization curve to {save_path}")


def main():
    print("=" * 70)
    print("  ANALYSIS 3: Sample Size Stabilization & Split-Half Reliability")
    print("=" * 70)
    
    feature_files = glob.glob(os.path.join(_PROJECT_ROOT, 'data', 'training', 'swing_features_*.parquet'))
    if not feature_files:
        print("No swing features parquet files found in data/training/")
        return
        
    # Prefer 2025 or latest
    feature_files.sort()
    target_file = feature_files[-1]
    
    res_df = compute_split_half_curve(target_file)
    print("\n--- STABILIZATION CURVE RESULTS ---")
    print(res_df.to_string(index=False))
    
    out_plot = os.path.join(_PROJECT_ROOT, 'assets', 'presentation', 'stabilization_curve.png')
    plot_stabilization_curve(res_df, out_plot)
    
    out_csv = os.path.join(_PROJECT_ROOT, 'assets', 'presentation', 'stabilization_summary.csv')
    res_df.to_csv(out_csv, index=False)
    print(f"Saved summary metrics to {out_csv}")


if __name__ == "__main__":
    main()
