"""
Train Swing Decision Model + Generate Bayesian Leaderboard

Pipeline:
    1. Load swing feature matrix (from build_swing_features.py)
    2. Train XGBoost league model
    3. Compute hierarchical Bayesian posteriors per batter
    4. Score batters with DVA (Decision Value Added)
    5. Output leaderboard

Usage:
    python3 scripts/train_swing_model.py
    python3 scripts/train_swing_model.py --kappa 50    # Stronger prior
"""

import sys
import os
import argparse
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

from src.swing_model import (
    LeagueSwingModel, BayesianBatterModel, 
    score_batters, XGBOOST_FEATURES
)
from src.db_manager import get_db_connection, save_swing_ratings

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'training')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'models')


def main():
    parser = argparse.ArgumentParser(description='Train swing decision model')
    parser.add_argument('--year', type=int, default=2025)
    parser.add_argument('--kappa', type=float, default=30,
                        help='Bayesian prior strength (equiv sample size)')
    parser.add_argument('--min-pitches', type=int, default=200)
    parser.add_argument('--oof-splits', type=int, default=5,
                        help='Number of out-of-fold batter splits for leak-free baseline scoring (default: 5)')
    parser.add_argument('--train-perceptual', action='store_true', default=False,
                        help='Train decision-time perceptual baseline model in addition to final-location model')
    args = parser.parse_args()
    
    print(f"{'='*60}")
    print(f"  Swing Decision Model — {args.year}")
    print(f"  κ = {args.kappa} (prior strength)")
    print(f"{'='*60}\n")
    
    # --- 1. Load features ---
    feature_path = os.path.join(DATA_DIR, f'swing_features_{args.year}.parquet')
    if not os.path.exists(feature_path):
        print(f"Feature file not found: {feature_path}")
        print(f"Run: python3 scripts/build_swing_features.py {args.year}")
        return
    
    df = pd.read_parquet(feature_path)
    print(f"Loaded {len(df):,} pitches with {df.shape[1]} features")
    print(f"Swing rate: {df['is_swing'].mean():.3f}")
    
    # --- 2. Train XGBoost ---
    print(f"\n--- XGBoost League Model ---")
    
    # Split by batter (no data leakage between train/val batters)
    batters = df['batter'].unique()
    train_batters, val_batters = train_test_split(
        batters, test_size=0.2, random_state=42
    )
    
    train_df = df[df['batter'].isin(train_batters)].copy()
    val_df = df[df['batter'].isin(val_batters)].copy()
    print(f"Train: {len(train_df):,} pitches ({len(train_batters)} batters)")
    print(f"Val:   {len(val_df):,} pitches ({len(val_batters)} batters)")
    
    league_model = LeagueSwingModel()
    league_model.train(train_df, val_df)
    
    # Evaluate
    val_pred = league_model.predict(val_df)
    auc = roc_auc_score(val_df['is_swing'], val_pred)
    acc = accuracy_score(val_df['is_swing'], (val_pred > 0.5).astype(int))
    ll = log_loss(val_df['is_swing'], val_pred)
    print(f"\nValidation:")
    print(f"  AUC:      {auc:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Log Loss: {ll:.4f}")
    
    # Calibration check
    print(f"\n  Calibration (predicted vs actual swing rate):")
    bins = np.linspace(0, 1, 11)
    for i in range(len(bins) - 1):
        mask = (val_pred >= bins[i]) & (val_pred < bins[i+1])
        if mask.sum() > 0:
            pred_mean = val_pred[mask].mean()
            actual_mean = val_df['is_swing'].values[mask].mean()
            n = mask.sum()
            print(f"    P(Swing) {bins[i]:.1f}-{bins[i+1]:.1f}: "
                  f"predicted={pred_mean:.3f}, actual={actual_mean:.3f} (n={n:,})")
    
    # Feature importances
    print(f"\n  Top 10 Feature Importances (gain):")
    fi = league_model.feature_importances_
    if fi:
        sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)
        for name, gain in sorted_fi[:10]:
            print(f"    {name:25s}: {gain:.1f}")
    
    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f'swing_model_{args.year}')
    league_model.save(model_path, year=args.year)
    # Also save as default swing_model
    league_model.save(os.path.join(MODEL_DIR, 'swing_model'), year=args.year)
    print(f"\n  Model saved: {model_path}.xgb")
    
    league_model_perceptual = None
    if args.train_perceptual:
        print(f"\n--- Training Perceptual Decision-Time Baseline Model ---")
        league_model_perceptual = LeagueSwingModel(feature_mode='decision_time')
        league_model_perceptual.train(train_df, val_df)
        perc_path = os.path.join(MODEL_DIR, f'swing_model_{args.year}_perceptual')
        league_model_perceptual.save(perc_path, year=args.year)
        league_model_perceptual.save(os.path.join(MODEL_DIR, 'swing_model_perceptual'), year=args.year)
        print(f"  Perceptual baseline model saved: {perc_path}.xgb")

    # --- 3. Out-Of-Fold Baseline Predictions for Fair Scoring ---
    print(f"\n--- Out-of-Fold Baseline Generation ({args.oof_splits}-fold by Batter) ---")
    if args.oof_splits > 1:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=args.oof_splits, shuffle=True, random_state=42)
        oof_preds = np.zeros(len(df), dtype=np.float32)
        oof_preds_perc = np.zeros(len(df), dtype=np.float32) if args.train_perceptual else None
        fold_assignments = np.zeros(len(df), dtype=np.int32)
        
        for fold, (train_b_idx, val_b_idx) in enumerate(kf.split(batters)):
            fold_val_batters = set(batters[val_b_idx])
            fold_val_mask = df['batter'].isin(fold_val_batters)
            fold_assignments[fold_val_mask] = fold + 1
            
            # To prevent outer validation labels from influencing early stopping,
            # split the outer training batters into inner_train and inner_val.
            fold_train_batters = batters[train_b_idx]
            n_inner_val = max(1, int(len(fold_train_batters) * 0.15))
            rng = np.random.RandomState(42 + fold)
            shuffled_train_b = rng.permutation(fold_train_batters)
            inner_val_b = set(shuffled_train_b[:n_inner_val])
            inner_train_b = set(shuffled_train_b[n_inner_val:])
            
            inner_train_mask = df['batter'].isin(inner_train_b)
            inner_val_mask = df['batter'].isin(inner_val_b)
            
            fold_model = LeagueSwingModel(feature_mode='final_location')
            fold_model.train(df[inner_train_mask], df[inner_val_mask])
            oof_preds[fold_val_mask] = fold_model.predict(df[fold_val_mask])

            if args.train_perceptual:
                fold_model_perc = LeagueSwingModel(feature_mode='decision_time')
                fold_model_perc.train(df[inner_train_mask], df[inner_val_mask])
                oof_preds_perc[fold_val_mask] = fold_model_perc.predict(df[fold_val_mask])

            print(f"  Fold {fold+1}/{args.oof_splits} complete ({fold_val_mask.sum():,} pitches, untouched outer holdout)")
            
        df['xgb_p_swing'] = oof_preds
        if args.train_perceptual:
            df['xgb_p_swing_perceptual'] = oof_preds_perc
        df['oof_fold'] = fold_assignments
        auc_label = "Out-of-Fold"
    else:
        df['xgb_p_swing'] = league_model.predict(df)
        if league_model_perceptual is not None:
            df['xgb_p_swing_perceptual'] = league_model_perceptual.predict(df)
        df['oof_fold'] = 0
        auc_label = "In-Sample"
        
    baseline_auc = roc_auc_score(df['is_swing'], df['xgb_p_swing'])
    print(f"  Overall {auc_label} Baseline AUC: {baseline_auc:.4f}")
    
    # Persist baseline predictions for downstream consumers with explicit provenance
    oof_cols = ['game_pk', 'at_bat_number', 'pitch_number', 'batter', 'oof_fold', 'xgb_p_swing']
    if 'xgb_p_swing_perceptual' in df.columns:
        oof_cols.append('xgb_p_swing_perceptual')
    available_oof = [c for c in oof_cols if c in df.columns]
    oof_df = df[available_oof].copy()
    oof_df['is_oof'] = (args.oof_splits > 1)
    oof_df['season'] = int(args.year)
    
    pred_file = f'oof_predictions_{args.year}.parquet' if args.oof_splits > 1 else f'insample_predictions_{args.year}.parquet'
    pred_path = os.path.join(DATA_DIR, pred_file)
    oof_df.to_parquet(pred_path, index=False)
    print(f"  Saved {'out-of-fold' if args.oof_splits > 1 else 'in-sample'} baseline predictions: {pred_path}")
    
    # Also save to data/processed for consumer compatibility
    processed_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    oof_df.to_parquet(os.path.join(processed_dir, pred_file), index=False)
    
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='swing_pitch_predictions'")
        table_exists = cursor.fetchone() is not None
        if table_exists:
            cursor.execute("PRAGMA table_info(swing_pitch_predictions)")
            cols = [col[1] for col in cursor.fetchall()]
            if 'season' in cols:
                conn.execute("DELETE FROM swing_pitch_predictions WHERE season = ?", (int(args.year),))
            else:
                conn.execute("DROP TABLE swing_pitch_predictions")
        
        oof_df.to_sql('swing_pitch_predictions', conn, if_exists='append', index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pitch_pred ON swing_pitch_predictions (game_pk, at_bat_number, pitch_number)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pitch_pred_season ON swing_pitch_predictions (season)")
        conn.commit()
    except Exception as e:
        print(f"  Note: Could not save predictions to DB: {e}")
    finally:
        conn.close()

    # --- 4. Bayesian Batter Posteriors ---
    print(f"\n--- Hierarchical Bayesian Model (κ={args.kappa}) ---")
    
    bayesian_model = BayesianBatterModel(kappa=args.kappa)
    bayesian_model.fit(df, df['xgb_p_swing'].values)
    
    # Show example batter profile
    example_batter = df['batter'].value_counts().index[0]  # batter with most pitches
    profile = bayesian_model.get_batter_profile(example_batter)
    if profile is not None:
        print(f"\n  Example profile (batter {example_batter}, {len(df[df['batter']==example_batter])} pitches):")
        print(f"  {'Context':<30s} {'n':>5s} {'Prior':>7s} {'Post':>7s} {'Dev':>7s} {'Shrink':>7s}")
        for _, row in profile.iterrows():
            ctx = f"{row['count_group']}/{row['loc_bin']}/{row['pitch_group']}"
            print(f"  {ctx:<30s} {row['n_pitches']:5.0f} {row['p_prior']:7.3f} {row['p_posterior']:7.3f} "
                  f"{row['deviation']:+7.3f} {row['shrinkage']:7.3f}")
    
    # --- 5. Score Batters ---
    print(f"\n--- Scoring Batters (DVA) ---")
    
    t0 = time.time()
    lb = score_batters(df, league_model, bayesian_model, min_pitches=args.min_pitches, league_model_perceptual=league_model_perceptual)
    elapsed = time.time() - t0
    
    if len(lb) == 0:
        print("No batters scored.")
        return
        
    lb['season'] = args.year
    
    # Look up names (first check local database, fallback to pybaseball)
    name_map = {}
    try:
        _c = get_db_connection()
        _df_pn = pd.read_sql_query("SELECT batter_id, name FROM player_names", _c)
        name_map = dict(zip(_df_pn['batter_id'], _df_pn['name']))
        _c.close()
    except Exception:
        pass

    missing_ids = [b for b in lb['batter_id'].tolist() if b not in name_map]
    if missing_ids:
        try:
            from pybaseball import playerid_reverse_lookup
            names = playerid_reverse_lookup(missing_ids, key_type='mlbam')
            for _, r in names.iterrows():
                first = str(r['name_first']).strip().title()
                last = str(r['name_last']).strip().title()
                name_map[int(r['key_mlbam'])] = f"{first} {last}"
        except Exception:
            pass

    lb['name'] = lb['batter_id'].map(name_map).fillna(lb['batter_id'].astype(str))
    lb['player_name'] = lb['name']
    
    print(f"\nScored {len(lb)} batters in {elapsed:.1f}s")
    
    # Save to DB non-destructively by season with atomic schema migration
    save_swing_ratings(lb)
    
    # Print leaderboard
    display_cols = ['rank', 'name', 'swing_rating', 'dva_mean', 'tendency_dva',
                    'actual_dva', 'chase_rate', 'zone_swing_rate', 'n_pitches']
    
    print(f"\n{'='*60}")
    print(f"  SWING DECISION LEADERBOARD (Bayesian-Integrated)")
    print(f"{'='*60}")
    
    print("\n--- TOP 15 (Best Decisions) ---")
    print(lb[display_cols].head(15).to_string(index=False))
    
    print("\n--- BOTTOM 15 (Worst Decisions) ---")
    print(lb[display_cols].tail(15).to_string(index=False))
    
    # Stats
    print(f"\n--- Summary ---")
    print(f"  Total batters: {len(lb)}")
    print(f"  Rating range: {lb['swing_rating'].min():.1f} — {lb['swing_rating'].max():.1f}")
    print(f"  DVA blended mean: {lb['dva_mean'].mean():.6f}")
    print(f"  DVA blended std:  {lb['dva_mean'].std():.6f}")
    print(f"  Tendency DVA std: {lb['tendency_dva'].std():.6f}")
    print(f"  Actual DVA std:   {lb['actual_dva'].std():.6f}")
    print(f"\n  Correlation with chase rate: {lb['swing_rating'].corr(lb['chase_rate']):.3f}")
    print(f"  Correlation with zone swing: {lb['swing_rating'].corr(lb['zone_swing_rate']):.3f}")
    
    # Bayesian shrinkage diagnostics
    print(f"\n--- Bayesian Diagnostics ---")
    print(f"  Avg confidence:    {lb['avg_confidence'].mean():.3f}")
    print(f"  Tendency vs Actual DVA correlation: {lb['tendency_dva'].corr(lb['actual_dva']):.3f}")
    
    # Small vs large sample comparison
    small = lb[lb['n_pitches'] < 400]
    large = lb[lb['n_pitches'] > 1500]
    if len(small) > 0 and len(large) > 0:
        print(f"\n  Small-sample (<400 pitches, n={len(small)}):")
        print(f"    Rating spread: {small['swing_rating'].std():.2f}")
        print(f"    Avg confidence: {small['avg_confidence'].mean():.3f}")
        print(f"  Large-sample (>1500 pitches, n={len(large)}):")
        print(f"    Rating spread: {large['swing_rating'].std():.2f}")
        print(f"    Avg confidence: {large['avg_confidence'].mean():.3f}")


if __name__ == "__main__":
    main()
