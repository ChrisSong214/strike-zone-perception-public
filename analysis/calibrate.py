import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import pandas as pd
import numpy as np
from src.data_handler import StatcastDataHandler

def calibrate_noise_model():
    print("Loading data for calibration...")
    handler = StatcastDataHandler()
    df = handler.fetch_statcast(start_year=2024, end_year=2025) # Use recent data
    
    if df.empty:
        print("No data found for 2024-2025. Trying extended range 2020-2025.")
        df = handler.fetch_statcast(start_year=2020, end_year=2025)
    
    if df.empty:
        print("Still no data. Cannot calibrate.")
        return

    # Select a top pitcher with enough sampling
    top_pitcher_id = df['pitcher'].mode()[0]
    pitcher_df = df[df['pitcher'] == top_pitcher_id]
    pitcher_name = pitcher_df['player_name'].iloc[0]
    
    print(f"Calibrating for Pitcher: {pitcher_name} (n={len(pitcher_df)})")
    
    # We want to find the standard deviation within a game-pitch_type cluster.
    # But for a simple 'global' uncertainty model for now, we can look at the overall standard deviation 
    # of release parameters for a specific pitch type.
    # Note: This is an overestimate because it includes fatigue, intended location changes, etc.
    # A better approximation of "motor noise" is the std dev per game per pitch type.
    
    stats = {}
    
    for ptype in pitcher_df['pitch_type'].unique():
        if not isinstance(ptype, str): continue
        
        subset = pitcher_df[pitcher_df['pitch_type'] == ptype]
        if len(subset) < 10: continue
        
        # Calculate std dev of release parameters
        # release_speed, release_pos_x, release_pos_z, release_extension
        # pfx_x, pfx_z (movement consistency)
        
        std_speed = subset['release_speed'].std()
        std_rel_x = subset['release_pos_x'].std()
        std_rel_z = subset['release_pos_z'].std()
        std_pfx_x = subset['pfx_x'].std()
        std_pfx_z = subset['pfx_z'].std()
        
        stats[ptype] = {
            'std_speed': std_speed,
            'std_rel_x': std_rel_x,
            'std_rel_z': std_rel_z,
            'std_pfx_x': std_pfx_x, # already in feet in db? no, verify. pfx is usually inches? 
            # In database, pfx_x is usually feet? 
            # Let's check magnitude. If it's ~0.5 it's feet. If it's ~6 it's inches.
            # Usually pfx_x/z in Statcast CSVs are in feet? No, Savant CSVs are usually feet.
            # But let's just calc std.
            'std_pfx_z': std_pfx_z
        }
    
    print("\nCalibration Results (Standard Deviations):")
    for ptype, s in stats.items():
        print(f"[{ptype}] Speed: {s['std_speed']:.2f}, RelX: {s['std_rel_x']:.3f}, RelZ: {s['std_rel_z']:.3f}, PfxX: {s['std_pfx_x']:.3f}, PfxZ: {s['std_pfx_z']:.3f}")

    # Use the Fastball (FF) as the baseline if available, or average across types
    key = None
    if 'FF' in stats:
        ref = stats['FF']
        key = 'FF'
    elif len(stats) > 0:
        key = list(stats.keys())[0]
        ref = stats[key]
        
    # Get the actual subset dataframe for the reference pitch
    ref_subset = pitcher_df[pitcher_df['pitch_type'] == (key if key else 'FF')]
    
    if ref_subset.empty and len(stats) > 0:
         # Fallback if logic above was fuzzy
         key = list(stats.keys())[0]
         ref_subset = pitcher_df[pitcher_df['pitch_type'] == key]
    
    if ref_subset.empty:
        print("No stats derived (subset empty).")
        return

    # Calculate Covariance Matrix for the 5 key parameters
    # We want to capture correlations (e.g. if you throw harder, does it break less?)
    features = ['release_speed', 'release_pos_x', 'release_pos_z', 'pfx_x', 'pfx_z']
    
    print(f"\n--- Covariance Matrix (Correlation Analysis) for {key} ---")
    
    # Drop NaNs
    subset_clean = ref_subset[features].dropna()
    
    # Calculate Covariance
    cov_matrix = subset_clean.cov()
    print("Covariance Matrix:")
    print(cov_matrix)
    
    # Calculate Correlation Matrix (easier to read)
    corr_matrix = subset_clean.corr()
    print("\nCorrelation Matrix (easier to interpret):")
    print(corr_matrix.round(2))

    # Output likely helpful code snippet for src/uncertainty.py
    print("\n[COPY THIS] Recommended Covariance setup for uncertainty.py:")
    print("mean_vector = [row[f] for f in ['release_speed', 'release_pos_x', 'release_pos_z', 'pfx_x', 'pfx_z']]")
    print(f"cov_matrix = np.array({cov_matrix.values.tolist()})")

if __name__ == "__main__":
    calibrate_noise_model()
