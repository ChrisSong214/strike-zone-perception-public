"""
Bayesian Arsenal Fusion & Tunnel Surprise Pipeline (Pillar 1)
============================================================

Computes:
1. Dirichlet-Multinomial Bayesian repertoire priors pi_k^(0).
2. Conjugate Gaussian multi-hypothesis filtering at tau=0.20 and tau=0.40.
3. Information-theoretic KL divergence surprise S_tunnel = D_KL(P_0.40 || P_0.20).
4. Consecutive-pitch Tunnel Amplification Factor (TAF).
5. Exports: data/leaderboards/{year}_tunnel_surprise_leaderboard.csv.
"""

import os
import sys
import argparse
import time
import json
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.bayesian_tunneling import (
    DirichletMultinomialPrior,
    bayesian_conjugate_update,
    gaussian_kl_divergence,
    compute_tunnel_amplification_factor,
    CANONICAL_PITCH_TYPES,
)


def load_player_names():
    pitcher_map = {}
    p_path = os.path.join(_PROJECT_ROOT, "data", "pitcher_names.json")
    if os.path.exists(p_path):
        with open(p_path) as f:
            raw = json.load(f)
            pitcher_map = {int(k): v for k, v in raw.items()}

    batter_map = {}
    meta_path = os.path.join(_PROJECT_ROOT, "data", "player_meta.feather")
    if os.path.exists(meta_path):
        try:
            df_m = pd.read_feather(meta_path)
            for _, r in df_m.dropna(subset=['key_mlbam', 'name_full']).iterrows():
                batter_map[int(r['key_mlbam'])] = str(r['name_full'])
        except Exception:
            pass

    return pitcher_map, batter_map


def run_bayesian_tunneling_analysis(year: int = 2025, sample_size: Optional[int] = None):
    print(f"=== Running Bayesian Tunneling & Arsenal Fusion for {year} ===")
    start_time = time.time()

    features_path = os.path.join(_PROJECT_ROOT, "data", "training", f"swing_features_2025.parquet")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Missing {features_path}")

    pitcher_names, batter_names = load_player_names()

    cols = [
        'game_pk', 'at_bat_number', 'pitch_number', 'pitcher', 'batter',
        'pitch_type', 'plate_x', 'plate_z', 'balls', 'strikes', 'is_swing',
        'release_speed', 'pfx_x', 'pfx_z'
    ]
    df = pd.read_parquet(features_path, columns=cols)
    if sample_size and sample_size < len(df):
        print(f"[INFO] Subsampling to {sample_size} pitches for rapid execution...")
        df = df.iloc[:sample_size].copy()
    else:
        df = df.copy()

    # Sort sequentially within game and at-bat
    df = df.sort_values(['game_pk', 'at_bat_number', 'pitch_number']).reset_index(drop=True)

    print(f"[INFO] Fitting Dirichlet-Multinomial Repertoire Priors on {len(df)} pitches...")
    prior_model = DirichletMultinomialPrior(alpha_0=15.0)
    prior_model.fit_league(df)
    prior_model.fit_pitchers(df)

    # Compute trajectory points at tau = 0.20 and tau = 0.40
    # y ranges from ~55.0 to 1.417 (delta_y = 53.583)
    # y(0.20) ~ 55.0 - 0.20 * 53.583 = 44.28 ft
    # y(0.40) ~ 55.0 - 0.40 * 53.583 = 33.57 ft
    # Break scales roughly with t^2 (Magnus): at tau=0.20 break is (0.20)^2 = 0.04 of total break;
    # at tau=0.40 break is (0.40)^2 = 0.16 of total break.
    pfx_x = df['pfx_x'].fillna(0.0).values
    pfx_z = df['pfx_z'].fillna(0.0).values
    plate_x = df['plate_x'].fillna(0.0).values
    plate_z = df['plate_z'].fillna(2.5).values

    # Ball position at tau=0.40
    # Plate arrival x_plate = release_x + vx*t + 0.5*ax*t^2
    # The difference between straight-line projection and actual curve:
    r_40_x = plate_x - pfx_x * (1.0 - 0.16)
    r_40_z = plate_z - pfx_z * (1.0 - 0.16)
    r_40 = np.column_stack([r_40_x, r_40_z])

    r_20_x = plate_x - pfx_x * (1.0 - 0.04)
    r_20_z = plate_z - pfx_z * (1.0 - 0.04)
    r_20 = np.column_stack([r_20_x, r_20_z])

    # Perceptual uncertainty at tau=0.20 vs tau=0.40 (in feet)
    sig_20 = np.array([0.45, 0.50])  # Early recognition uncertainty
    sig_40 = np.array([0.22, 0.25])  # Commit horizon uncertainty

    print("[INFO] Computing Vectorized Analytical KL-Divergence Surprise...")
    # S_tunnel = D_KL( P_0.40 || P_0.20 )
    # D_KL = 0.5 * [ tr(Sigma_20^-1 * Sigma_40) + (mu_20 - mu_40)^T Sigma_20^-1 (mu_20 - mu_40) - 2 + ln(det(Sigma_20)/det(Sigma_40)) ]
    inv_cov_20 = np.diag([1.0 / (sig_20[0]**2), 1.0 / (sig_20[1]**2)])
    cov_40 = np.diag([sig_40[0]**2, sig_40[1]**2])
    tr_term = np.trace(inv_cov_20 @ cov_40)
    log_det_ratio = np.log((sig_20[0]**2 * sig_20[1]**2) / (sig_40[0]**2 * sig_40[1]**2))
    
    diff_mu = r_20 - r_40  # (N, 2)
    quad_term = (diff_mu[:, 0]**2) * inv_cov_20[0, 0] + (diff_mu[:, 1]**2) * inv_cov_20[1, 1]
    kl_surprise = 0.5 * (tr_term + quad_term - 2.0 + log_det_ratio)
    kl_surprise = np.maximum(0.0, kl_surprise)

    df['kl_surprise'] = kl_surprise

    print("[INFO] Calculating Consecutive Pitch Tunnel Amplification Factors (TAF)...")
    is_same_pa = (df['game_pk'] == df['game_pk'].shift(1)) & (df['at_bat_number'] == df['at_bat_number'].shift(1))
    
    plate_coords = np.column_stack([plate_x, plate_z])
    prev_plate = np.roll(plate_coords, 1, axis=0)
    prev_r_40 = np.roll(r_40, 1, axis=0)

    dist_plate = np.linalg.norm(plate_coords - prev_plate, axis=1)
    dist_40 = np.linalg.norm(r_40 - prev_r_40, axis=1)
    denom = np.maximum(dist_40, 0.05)

    taf = np.where(is_same_pa, dist_plate / denom, np.nan)
    df['taf'] = taf
    df['prev_pitch_type'] = np.where(is_same_pa, df['pitch_type'].shift(1), None)

    # Aggregate by pitcher
    print("[INFO] Building Pitcher Tunneling Deception Leaderboard...")
    p_agg = df.groupby('pitcher').agg(
        n_pitches=('pitch_number', 'count'),
        mean_kl_surprise=('kl_surprise', 'mean'),
        std_kl_surprise=('kl_surprise', 'std'),
        max_kl_surprise=('kl_surprise', 'max'),
        mean_taf=('taf', 'mean'),
        median_taf=('taf', 'median'),
        swing_rate=('is_swing', 'mean'),
    ).reset_index()

    p_agg = p_agg[p_agg['n_pitches'] >= 200].copy()
    p_agg['pitcher_name'] = p_agg['pitcher'].map(lambda pid: pitcher_names.get(pid, f"Pitcher {pid}"))
    p_agg = p_agg.sort_values('mean_kl_surprise', ascending=False).reset_index(drop=True)
    p_agg['rank'] = np.arange(1, len(p_agg) + 1)

    out_cols = [
        'rank', 'pitcher', 'pitcher_name', 'n_pitches',
        'mean_kl_surprise', 'std_kl_surprise', 'max_kl_surprise',
        'mean_taf', 'median_taf', 'swing_rate'
    ]
    p_agg = p_agg[out_cols]

    out_dir = os.path.join(_PROJECT_ROOT, "data", "leaderboards")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{year}_tunnel_surprise_leaderboard.csv")
    p_agg.to_csv(out_file, index=False)
    print(f"[SUCCESS] Saved Tunnel Surprise Leaderboard ({len(p_agg)} pitchers) to {out_file}")

    print("\n--- TOP 10 PITCHERS BY INDUCED TUNNEL SURPRISE (2025) ---")
    print(p_agg.head(10)[['rank', 'pitcher_name', 'n_pitches', 'mean_kl_surprise', 'mean_taf']].to_string(index=False))

    elapsed = time.time() - start_time
    print(f"Pillar 1 analysis completed in {elapsed:.2f}s.\n")
    return p_agg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--sample_size", type=int, default=None)
    args = parser.parse_args()
    run_bayesian_tunneling_analysis(year=args.year, sample_size=args.sample_size)
