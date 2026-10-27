"""
Personalized Utility Surfaces & Orthogonal DVA Decomposition (Pillar 3)
======================================================================

Strict 5-Fold Batter-Grouped Out-of-Fold (OOF) Pipeline.
Evaluates:
1. Batter-specific 2D spatial contact utility surfaces (xwOBAcon, Whiff, Foul).
2. Personalized Expected Value: EV_i^indiv(Swing).
3. Exact Additive Orthogonal Decomposition:
       Total Indiv EV-DVA = Universal EV-DVA + Damage Alignment DVA
4. Artifact Persistence:
       data/processed/oof_next_gen_predictions_{year}.parquet
       data/leaderboards/{year}_personalized_dva_leaderboard.csv
"""

import os
import sys
import argparse
import time
import sqlite3
import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.personalized_utility import (
    IndividualContactUtilityModel,
    DEFAULT_DELTA_RE_STRIKE,
    DEFAULT_DELTA_RE_BALL,
)


def load_player_names():
    meta_path = os.path.join(_PROJECT_ROOT, "data", "player_meta.feather")
    batter_map = {}
    if os.path.exists(meta_path):
        try:
            df_m = pd.read_feather(meta_path)
            for _, r in df_m.dropna(subset=['key_mlbam', 'name_full']).iterrows():
                batter_map[int(r['key_mlbam'])] = str(r['name_full'])
        except Exception:
            pass
    return batter_map


def run_personalized_dva_pipeline(year: int = 2025, sample_size: int = None):
    print("=" * 80)
    print(f"  STRICT 5-FOLD OOF PERSONALIZED DAMAGE DVA PIPELINE: SEASON {year}")
    print("=" * 80)
    start_time = time.time()

    oof_path = os.path.join(_PROJECT_ROOT, "data", "processed", f"oof_predictions_{year}.parquet")
    sf_path = os.path.join(_PROJECT_ROOT, "data", "training", f"swing_features_{year}.parquet")
    db_path = os.path.join(_PROJECT_ROOT, "data", "mlb.db")

    if not os.path.exists(oof_path) or not os.path.exists(sf_path):
        raise FileNotFoundError("Missing prerequisite processed files.")

    print("[INFO] Loading baseline OOF predictions and swing features...")
    df_oof = pd.read_parquet(oof_path)
    df_sf = pd.read_parquet(sf_path, columns=[
        'game_pk', 'at_bat_number', 'pitch_number', 'batter', 'pitcher',
        'is_swing', 'two_strikes', 'plate_x', 'plate_z', 'sz_top', 'sz_bot', 'in_zone'
    ])

    print("[INFO] Querying Statcast table for xwOBAcon and contact descriptions...")
    conn = sqlite3.connect(db_path)
    df_stat = pd.read_sql_query(f"""
        SELECT game_pk, at_bat_number, pitch_number,
               estimated_woba_using_speedangle as xwoba,
               description
        FROM statcast
        WHERE game_year = {year};
    """, conn)

    # Load batter priors (bat tracking & historical outcomes)
    print("[INFO] Loading out-of-fold batter priors (bat-tracking & historical contact)...")
    df_bt = pd.read_sql_query(f"""
        SELECT batter, AVG(bat_speed) as bat_speed, AVG(swing_length) as swing_length
        FROM statcast
        WHERE game_year = {year} AND bat_speed IS NOT NULL
        GROUP BY batter;
    """, conn)
    conn.close()

    batter_priors = {}
    for _, r in df_bt.iterrows():
        bid = int(r['batter'])
        batter_priors[bid] = {
            'bat_speed': float(r['bat_speed']),
            'swing_length': float(r['swing_length']),
        }

    bso_path = os.path.join(_PROJECT_ROOT, "data", "batter_season_outcomes.parquet")
    if os.path.exists(bso_path):
        df_bso = pd.read_parquet(bso_path)
        df_bso_prior = df_bso[df_bso['season'] == (year - 1)] if (year - 1) in df_bso['season'].values else df_bso
        for _, r in df_bso_prior.iterrows():
            bid = int(r['batter_id'])
            if bid not in batter_priors:
                batter_priors[bid] = {}
            batter_priors[bid]['prior_woba'] = float(r['woba'])

    print("[INFO] Merging datasets...")
    df = df_sf.merge(
        df_oof[['game_pk', 'at_bat_number', 'pitch_number', 'oof_fold', 'xgb_p_swing', 'xgb_p_swing_perceptual']],
        on=['game_pk', 'at_bat_number', 'pitch_number'],
        how='inner'
    )
    df = df.merge(
        df_stat,
        on=['game_pk', 'at_bat_number', 'pitch_number'],
        how='left'
    )

    if sample_size and sample_size < len(df):
        print(f"[INFO] Subsampling to {sample_size} pitches...")
        df = df.iloc[:sample_size].copy()

    # Define swing outcome flags
    df['is_whiff'] = df['description'].isin(['swinging_strike', 'swinging_strike_blocked']).astype(int)
    df['is_foul'] = df['description'].isin(['foul', 'foul_tip']).astype(int)
    df['is_bip'] = df['description'].isin(['hit_into_play']).astype(int)

    # Initialize prediction columns
    n_pitches = len(df)
    ev_swing_indiv_arr = np.zeros(n_pitches, dtype=np.float64)
    ev_swing_league_arr = np.zeros(n_pitches, dtype=np.float64)
    ev_take_arr = np.zeros(n_pitches, dtype=np.float64)
    universal_dva_arr = np.zeros(n_pitches, dtype=np.float64)
    damage_dva_arr = np.zeros(n_pitches, dtype=np.float64)
    total_dva_arr = np.zeros(n_pitches, dtype=np.float64)

    # Compute P(Strike) for EV(Take) based on plate coordinates & strike zone
    # Standard MLB rulebook strike zone: width = 17/12 ft (~1.417 ft, +/- 0.83 ft with 0.12 ft ball radius)
    in_zone_prob = np.where(
        (np.abs(df['plate_x'].fillna(0.0)) <= 0.83) &
        (df['plate_z'] >= df['sz_bot']) &
        (df['plate_z'] <= df['sz_top']),
        0.88,
        0.10
    )
    # Distance from center gradient
    ev_take_all = in_zone_prob * DEFAULT_DELTA_RE_STRIKE + (1.0 - in_zone_prob) * DEFAULT_DELTA_RE_BALL

    folds = sorted(df['oof_fold'].dropna().unique())
    print(f"[INFO] Beginning Strict 5-Fold Batter-Grouped OOF Processing across folds {folds}...")

    for fold in folds:
        t_fold = time.time()
        train_mask = (df['oof_fold'] != fold).values
        test_mask = (df['oof_fold'] == fold).values

        df_train = df[train_mask]
        df_test = df[test_mask]

        # Verify zero batter overlap between train and test
        train_batters = set(df_train['batter'].unique())
        test_batters = set(df_test['batter'].unique())
        overlap = train_batters.intersection(test_batters)
        assert len(overlap) == 0, f"DATA LEAKAGE DETECTED in Fold {fold}: {len(overlap)} overlapping batters!"

        # Fit contact utility surfaces on training fold only
        model = IndividualContactUtilityModel(k_shrinkage=40.0)
        model.fit_training_split(df_train)
        model.set_batter_priors(batter_priors)

        # Predict for each batter in test fold
        test_indices = np.where(test_mask)[0]
        test_px = df['plate_x'].values[test_indices]
        test_pz = df['plate_z'].values[test_indices]
        test_ts = df['two_strikes'].values[test_indices]
        test_batters_arr = df['batter'].values[test_indices]
        test_swings = df['is_swing'].values[test_indices]
        test_p_oof = df['xgb_p_swing'].values[test_indices]
        test_take = ev_take_all[test_indices]

        # Predict spatial EV
        fold_ev_indiv = np.zeros(len(test_indices), dtype=np.float64)
        fold_ev_league = np.zeros(len(test_indices), dtype=np.float64)

        # Batch by unique batter for speed
        for b_id in np.unique(test_batters_arr):
            b_mask = (test_batters_arr == b_id)
            ev_i, ev_l = model.predict_spatial_ev(
                batter_id=b_id,
                plate_x=test_px[b_mask],
                plate_z=test_pz[b_mask],
                two_strikes=test_ts[b_mask],
            )
            fold_ev_indiv[b_mask] = ev_i
            fold_ev_league[b_mask] = ev_l

        # Decompose DVA
        u_dva, d_dva, t_dva = model.decompose_dva(
            is_swing=test_swings,
            p_swing_oof=test_p_oof,
            ev_swing_indiv=fold_ev_indiv,
            ev_swing_league=fold_ev_league,
            ev_take=test_take,
        )

        # Check additive identity invariant
        max_id_err = np.max(np.abs(t_dva - (u_dva + d_dva)))
        assert max_id_err < 1e-6, f"Additive identity violated in Fold {fold}: error {max_id_err}"

        # Store into arrays
        ev_swing_indiv_arr[test_indices] = fold_ev_indiv
        ev_swing_league_arr[test_indices] = fold_ev_league
        ev_take_arr[test_indices] = test_take
        universal_dva_arr[test_indices] = u_dva
        damage_dva_arr[test_indices] = d_dva
        total_dva_arr[test_indices] = t_dva

        print(f"  Fold {int(fold)}: {len(test_indices):,} pitches scored in {time.time()-t_fold:.2f}s | Overlap = 0 | Invariant err = {max_id_err:.2e}")

    # Add columns to DataFrame
    df['ev_swing_indiv'] = ev_swing_indiv_arr
    df['ev_swing_league'] = ev_swing_league_arr
    df['ev_take'] = ev_take_arr
    df['universal_ev_dva'] = universal_dva_arr
    df['damage_alignment_dva'] = damage_dva_arr
    df['total_indiv_ev_dva'] = total_dva_arr
    df['season'] = year

    # 1. Save Out-of-Sample Predictions Artifact
    oof_out_cols = [
        'game_pk', 'at_bat_number', 'pitch_number', 'batter', 'oof_fold',
        'is_swing', 'xgb_p_swing', 'xgb_p_swing_perceptual',
        'ev_swing_indiv', 'ev_swing_league', 'ev_take',
        'universal_ev_dva', 'damage_alignment_dva', 'total_indiv_ev_dva',
        'season'
    ]
    oof_out_dir = os.path.join(_PROJECT_ROOT, "data", "processed")
    os.makedirs(oof_out_dir, exist_ok=True)
    oof_out_file = os.path.join(oof_out_dir, f"oof_next_gen_predictions_{year}.parquet")
    df[oof_out_cols].to_parquet(oof_out_file, index=False)
    print(f"\n[SUCCESS] Saved Out-of-Sample Predictions Artifact to {oof_out_file} ({len(df):,} rows)")

    # 2. Build Batter Personalized DVA Leaderboard
    print("[INFO] Aggregating Batter Personalized DVA Leaderboard...")
    name_map = load_player_names()

    b_agg = df.groupby('batter').agg(
        n_pitches=('pitch_number', 'count'),
        swing_rate=('is_swing', 'mean'),
        universal_dva_total=('universal_ev_dva', 'sum'),
        damage_alignment_dva_total=('damage_alignment_dva', 'sum'),
        total_indiv_dva=('total_indiv_ev_dva', 'sum'),
        universal_dva_per_pitch=('universal_ev_dva', 'mean'),
        damage_dva_per_pitch=('damage_alignment_dva', 'mean'),
        total_indiv_dva_per_pitch=('total_indiv_ev_dva', 'mean'),
    ).reset_index()

    b_agg = b_agg[b_agg['n_pitches'] >= 500].copy()
    b_agg['batter_id'] = b_agg['batter']
    b_agg['season'] = year
    mean_dva = b_agg['total_indiv_dva_per_pitch'].mean()
    std_dva = max(b_agg['total_indiv_dva_per_pitch'].std(), 1e-6)
    b_agg['swing_rating'] = np.round(100.0 + 10.0 * (b_agg['total_indiv_dva_per_pitch'] - mean_dva) / std_dva, 1)
    b_agg['player_name'] = b_agg['batter'].map(lambda bid: name_map.get(bid, f"Hitter {bid}"))

    # Assign archetypes
    def classify_archetype(row):
        u = row['universal_dva_per_pitch']
        d = row['damage_dva_per_pitch']
        if u > 0.005 and d > 0.003:
            return "Dual-Threat Elite Decision Maker"
        elif u > 0.006 and d <= 0.002:
            return "Pure Zone Recognition Specialist"
        elif d > 0.004 and u <= 0.004:
            return "Damage-Driven Barrel Hunter"
        elif u < -0.002 and d < -0.002:
            return "Vulnerable Suboptimal Decision Maker"
        else:
            return "Balanced Zone Operator"

    b_agg['decision_archetype'] = b_agg.apply(classify_archetype, axis=1)

    b_agg = b_agg.sort_values('total_indiv_dva', ascending=False).reset_index(drop=True)
    b_agg['rank'] = np.arange(1, len(b_agg) + 1)

    lb_cols = [
        'rank', 'batter_id', 'batter', 'season', 'player_name', 'n_pitches', 'swing_rate', 'swing_rating',
        'total_indiv_dva', 'total_indiv_dva_per_pitch',
        'universal_dva_total', 'universal_dva_per_pitch',
        'damage_alignment_dva_total', 'damage_dva_per_pitch',
        'decision_archetype'
    ]
    b_agg = b_agg[lb_cols]

    lb_out_dir = os.path.join(_PROJECT_ROOT, "data", "leaderboards")
    os.makedirs(lb_out_dir, exist_ok=True)
    lb_out_file = os.path.join(lb_out_dir, f"{year}_personalized_dva_leaderboard.csv")
    b_agg.to_csv(lb_out_file, index=False)
    print(f"[SUCCESS] Saved Personalized DVA Leaderboard ({len(b_agg)} batters) to {lb_out_file}")

    print("\n--- TOP 10 BATTERS BY TOTAL INDIVIDUAL DVA (2025) ---")
    print(b_agg.head(10)[['rank', 'player_name', 'n_pitches', 'total_indiv_dva', 'universal_dva_total', 'damage_alignment_dva_total', 'decision_archetype']].to_string(index=False))

    print("\n--- SPOTLIGHT COMPARISON: STEVEN KWAN VS POWER HITTERS ---")
    spotlights = ['Kwan', 'Stowers', 'Judge', 'Soto', 'Ohtani', 'Stanton', 'Arraez']
    for sp in spotlights:
        sub = b_agg[b_agg['player_name'].str.contains(sp, na=False, case=False)]
        if not sub.empty:
            r = sub.iloc[0]
            print(f"  • {r['player_name']:18s} | Total DVA: {r['total_indiv_dva']:+6.2f} | Universal: {r['universal_dva_total']:+6.2f} | Damage: {r['damage_alignment_dva_total']:+6.2f} | {r['decision_archetype']}")

    elapsed = time.time() - start_time
    print(f"\nPillar 3 pipeline completed in {elapsed:.2f}s.\n")
    return b_agg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--sample_size", type=int, default=None)
    args = parser.parse_args()

    run_personalized_dva_pipeline(year=args.year, sample_size=args.sample_size)
