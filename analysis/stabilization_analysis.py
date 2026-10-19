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


def compute_split_half_curve(parquet_path, sample_sizes=[50, 100, 200, 300, 500, 750, 1000]):
    """Computes split-half reliability curves for DVA vs traditional chase rate."""
    print(f"Loading pitch-level features from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    # Check necessary columns
    required = ['batter', 'is_swing', 'in_zone', 'lstm_p_strike']
    if not all(c in df.columns for c in required):
        raise ValueError(f"Missing required columns in {parquet_path}: {[c for c in required if c not in df.columns]}")
        
    print(f"Loaded {len(df):,} pitches across {df['batter'].nunique():,} unique batters.")
    
    # Filter batters with at least 1,000 pitches
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
        
    results = []
    
    for N in sample_sizes:
        print(f"  Evaluating stabilization at N = {N} pitches...")
        batters_dva_odd = []
        batters_dva_even = []
        batters_chase_odd = []
        batters_chase_even = []
        
        for b_id, b_pitches in sub_df.groupby('batter'):
            sample = b_pitches.iloc[:N]
            
            # Split odd and even pitches
            odd = sample.iloc[0::2]
            even = sample.iloc[1::2]
            
            # Decision quality metric (strike agreement: swing when p_strike > 0.5, take when p_strike <= 0.5)
            # Alignment: swing * (p_strike - 0.5) + take * (0.5 - p_strike)
            dva_odd = ((odd['is_swing'] * (odd['lstm_p_strike'] - 0.5)) + ((1 - odd['is_swing']) * (0.5 - odd['lstm_p_strike']))).mean()
            dva_even = ((even['is_swing'] * (even['lstm_p_strike'] - 0.5)) + ((1 - even['is_swing']) * (0.5 - even['lstm_p_strike']))).mean()
            
            # Chase rate on out-of-zone pitches
            oz_odd = odd[odd['in_zone'] == 0]
            oz_even = even[even['in_zone'] == 0]
            
            chase_odd = oz_odd['is_swing'].mean() if len(oz_odd) > 5 else np.nan
            chase_even = oz_even['is_swing'].mean() if len(oz_even) > 5 else np.nan
            
            batters_dva_odd.append(dva_odd)
            batters_dva_even.append(dva_even)
            batters_chase_odd.append(chase_odd)
            batters_chase_even.append(chase_even)
            
        # Compute Pearson r and Spearman-Brown corrected reliability
        r_dva, _ = stats.pearsonr(batters_dva_odd, batters_dva_even)
        sb_dva = (2 * r_dva) / (1 + r_dva)
        
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
            'DVA_SplitHalf_r': round(sb_dva, 4),
            'Chase_SplitHalf_r': round(sb_chase, 4) if np.isfinite(sb_chase) else None
        })
        
    return pd.DataFrame(results)


def plot_stabilization_curve(res_df, save_path):
    """Plots the stabilization curve comparing DVA vs Chase Rate."""
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    ax.plot(res_df['N_Pitches'], res_df['DVA_SplitHalf_r'], marker='o', linewidth=2.8, markersize=8,
            color='#2a9d8f', label='Perceptual Decision Value (DVA)')
            
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
