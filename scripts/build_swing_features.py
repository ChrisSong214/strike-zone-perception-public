"""
Build Swing Decision Feature Matrix

For each pitch in statcast, extracts:
    - Context features (count, handedness, pitch type, game state)
    - LSTM perception features (σx, σz, P(Strike) from trained model)
    - Derived features (platoon, in_zone, context bin assignments)

Saves to data/training/swing_features_{year}.parquet

Usage:
    python3 scripts/build_swing_features.py              # 2025
    python3 scripts/build_swing_features.py 2024 2025     # Multiple years
"""

import sys
import os
import time
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import torch

from src.lstm_scoring import LSTMEyeScorer
from src.dataset import normalize_valid_trajectory
from src.db_manager import get_db_connection

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'training')

SWING_EVENTS = {
    'hit_into_play', 'foul', 'swinging_strike', 'swinging_strike_blocked',
    'foul_tip', 'foul_bunt', 'missed_bunt', 'bunt_foul_tip'
}

# Pitch type groupings
FASTBALL_TYPES = {'FF', 'SI', 'FC', 'FA'}
BREAKING_TYPES = {'SL', 'CU', 'KC', 'SV', 'CS', 'SC', 'ST', 'KN'}
OFFSPEED_TYPES = {'CH', 'FS', 'FO', 'EP'}


def load_pitches(year):
    """Load all pitches with required columns."""
    conn = get_db_connection()
    query = f"""
    SELECT 
        game_pk, at_bat_number, pitch_number,
        batter, pitcher, player_name,
        -- Trajectory inputs for LSTM
        vx0, vy0, vz0, ax, ay, az,
        release_pos_x, release_pos_y, release_pos_z, release_extension,
        -- Context features
        balls, strikes, stand, p_throws, pitch_type, pitch_name,
        zone, plate_x, plate_z, sz_top, sz_bot,
        release_speed, pfx_x, pfx_z,
        outs_when_up, inning,
        on_1b, on_2b, on_3b,
        bat_score, fld_score,
        -- Target
        description, type, events,
        game_year
    FROM statcast
    WHERE game_year = {year}
      AND vx0 IS NOT NULL
      AND release_pos_x IS NOT NULL
      AND plate_x IS NOT NULL
      AND pitch_type IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"Loaded {len(df):,} pitches for {year}")
    return df


def derive_features(df):
    """Add derived context features."""
    # Target
    df['is_swing'] = df['description'].isin(SWING_EVENTS).astype(int)
    
    # Platoon (same-side = 1, opposite = 0)
    df['platoon'] = (df['stand'] == df['p_throws']).astype(int)
    
    # Encode handedness
    df['stand_R'] = (df['stand'] == 'R').astype(int)
    df['p_throws_R'] = (df['p_throws'] == 'R').astype(int)
    
    # Pitch type group
    df['pitch_group'] = 'other'
    df.loc[df['pitch_type'].isin(FASTBALL_TYPES), 'pitch_group'] = 'fastball'
    df.loc[df['pitch_type'].isin(BREAKING_TYPES), 'pitch_group'] = 'breaking'
    df.loc[df['pitch_type'].isin(OFFSPEED_TYPES), 'pitch_group'] = 'offspeed'
    
    # Location relative to zone
    sz_mid_z = (df['sz_top'] + df['sz_bot']) / 2
    sz_height = df['sz_top'] - df['sz_bot']
    sz_width = 17 / 12  # feet
    
    # Normalized location (0 = center, 1 = edge of zone)
    df['loc_x_norm'] = df['plate_x'] / (sz_width / 2)
    df['loc_z_norm'] = (df['plate_z'] - sz_mid_z) / (sz_height / 2)
    
    # Chebyshev distance (L_infinity norm) for rectangular strike zone:
    # max(|loc_x_norm|, |loc_z_norm|) <= 1.0 defines the exact rectangular strike zone,
    # ensuring all four corners are properly recognized as "zone" rather than rounded off.
    df['dist_from_center'] = np.maximum(df['loc_x_norm'].abs(), df['loc_z_norm'].abs())
    
    # Location bin: zone / edge / chase
    df['loc_bin'] = 'chase'
    df.loc[df['dist_from_center'] <= 1.0, 'loc_bin'] = 'zone'
    df.loc[(df['dist_from_center'] > 1.0) & (df['dist_from_center'] <= 1.5), 'loc_bin'] = 'edge'
    
    # In-zone: defined consistently by the Chebyshev rectangular strike zone (L_infinity <= 1.0)
    # matching loc_bin == 'zone'. Preserve statcast_zone for reference.
    df['statcast_zone'] = df['zone']
    df['in_zone'] = (df['dist_from_center'] <= 1.0).astype(int)
    
    # Count group
    df['count_group'] = 'even'  # default
    ahead_counts = {(1, 0), (2, 0), (3, 0), (3, 1), (2, 1)}
    behind_counts = {(0, 1), (0, 2), (1, 2)}
    for b, s in ahead_counts:
        df.loc[(df['balls'] == b) & (df['strikes'] == s), 'count_group'] = 'ahead'
    for b, s in behind_counts:
        df.loc[(df['balls'] == b) & (df['strikes'] == s), 'count_group'] = 'behind'
    
    # Two-strike indicator (important: batters expand zone)
    df['two_strikes'] = (df['strikes'] == 2).astype(int)
    
    # Runners
    df['runners_on'] = (df['on_1b'].notna().astype(int) + 
                        df['on_2b'].notna().astype(int) + 
                        df['on_3b'].notna().astype(int))
    
    # Score differential
    df['score_diff'] = df['bat_score'] - df['fld_score']
    
    # Count as single int (for features)
    df['count_code'] = df['balls'] * 10 + df['strikes']
    
    # True strike ground truth based on rulebook plate arrival zone
    # Geometrically identical to in_zone (|plate_x| <= 17/24, sz_bot <= plate_z <= sz_top)
    df['true_strike'] = df['in_zone'].copy()
    
    return df


def add_lstm_features(df, scorer, chunk_size=10000, year=None):
    """Run LSTM inference in memory-bounded chunks and add perception features at the decision point."""
    print(f"Running LSTM inference on {len(df):,} pitches (chunk size: {chunk_size:,})...", flush=True)
    print(f"  Evaluating at {scorer.decision_frac:.0%} through flight (decision point)", flush=True)
    t0 = time.time()
    
    traj_path = os.path.join(OUTPUT_DIR, f'trajectories_{year}.pt') if year else None
    use_cached_traj = False
    data = None
    if traj_path and os.path.exists(traj_path):
        try:
            print(f"  Inspecting pre-computed trajectory dataset: {traj_path}...", flush=True)
            data = torch.load(traj_path, weights_only=False, map_location='cpu')
            labels = data.get('labels', {})
            required_keys = ['game_pk', 'at_bat_number', 'pitch_number']
            if all(k in labels for k in required_keys):
                use_cached_traj = True
            else:
                print(f"  [Notice] Cached trajectory file {traj_path} lacks required keys {required_keys}. "
                      f"Found keys: {list(labels.keys())}. Falling back to chunked extraction.", flush=True)
        except Exception as e:
            print(f"  [Warning] Failed to inspect cached trajectory dataset {traj_path}: {e}. Falling back to chunked extraction.", flush=True)

    if use_cached_traj and data is not None:
        print(f"  Using pre-computed trajectory dataset: {traj_path}...", flush=True)
        seqs = data['sequences']
        vlens = data['valid_lengths']
        labels = data['labels']
        
        batch_size = 4096
        all_decision = []
        with torch.no_grad():
            for i in range(0, len(seqs), batch_size):
                b_s = seqs[i:i+batch_size].to(scorer.device)
                b_vlens = vlens[i:i+batch_size].to(scorer.device)
                
                # Apply normalization strictly to non-padded timesteps using unified helper
                b_seqs = normalize_valid_trajectory(b_s, scorer.norm_mean, scorer.norm_std, valid_len=b_vlens)
                decision_params = scorer.model.predict_at_fraction(
                    b_seqs, b_vlens, fraction=scorer.decision_frac
                )
                all_decision.append(decision_params.cpu())
                if (i // batch_size) % 10 == 0 or i + batch_size >= len(seqs):
                    print(f"  Inference progress: {min(i + batch_size, len(seqs)):,}/{len(seqs):,} pitches", flush=True)
        
        decision_params = torch.cat(all_decision, dim=0).numpy()
        preds_df = pd.DataFrame({
            'game_pk': labels['game_pk'].numpy(),
            'at_bat_number': labels['at_bat_number'].numpy(),
            'pitch_number': labels['pitch_number'].numpy(),
            'lstm_mu_x': decision_params[:, 0],
            'lstm_mu_z': decision_params[:, 1],
            'lstm_sigma_x': decision_params[:, 2],
            'lstm_sigma_z': decision_params[:, 3],
            'lstm_rho': decision_params[:, 4],
        })
        
        orig_len = len(df)
        df = df.merge(preds_df, on=['game_pk', 'at_bat_number', 'pitch_number'], how='inner')
        elapsed = time.time() - t0
        print(f"  LSTM inference complete: {len(df):,} succeeded in {elapsed:.1f}s", flush=True)
    else:
        pred_dfs = []
        n_chunks = (len(df) + chunk_size - 1) // chunk_size
        for i in range(0, len(df), chunk_size):
            df_chunk = df.iloc[i:i + chunk_size]
            chunk_preds = scorer.batch_predict(df_chunk)
            if not chunk_preds.empty:
                pred_dfs.append(chunk_preds)
            chunk_num = i // chunk_size + 1
            if chunk_num % 5 == 0 or chunk_num == n_chunks:
                print(f"  Processed chunk {chunk_num}/{n_chunks} ({min(i + chunk_size, len(df)):,}/{len(df):,} pitches)", flush=True)
        
        if not pred_dfs:
            predictions = pd.DataFrame()
        else:
            predictions = pd.concat(pred_dfs)
        
        elapsed = time.time() - t0
        n_success = len(predictions)
        n_fail = len(df) - n_success
        print(f"  LSTM inference: {n_success:,} succeeded, {n_fail:,} failed in {elapsed:.1f}s", flush=True)
        
        df = df.loc[predictions.index].copy()
        df['lstm_mu_x'] = predictions['mu_x']
        df['lstm_mu_z'] = predictions['mu_z']
        df['lstm_sigma_x'] = predictions['sigma_x']
        df['lstm_sigma_z'] = predictions['sigma_z']
        df['lstm_rho'] = predictions['rho']
    
    # Compute LSTM P(Strike) incorporating bivariate correlation rho
    df['lstm_p_strike'] = LSTMEyeScorer.compute_p_strike(
        df['lstm_mu_x'].values, df['lstm_mu_z'].values,
        df['lstm_sigma_x'].values, df['lstm_sigma_z'].values,
        df['sz_top'].values, df['sz_bot'].values,
        rho=df['lstm_rho'].values
    )
    
    # Uncertainty volume & Mirage delta
    u_vol = LSTMEyeScorer.compute_uncertainty_vol(
        df['lstm_sigma_x'].values, df['lstm_sigma_z'].values, df['lstm_rho'].values
    )
    df['uncertainty_vol'] = u_vol
    df['log_uncertainty_vol'] = np.log(u_vol + 1e-6)
    df['mirage_delta'] = df['lstm_p_strike'] - df['true_strike']
    
    # Diagnostic: show that the decision point has meaningful uncertainty
    print(f"\n  Decision-point perception diagnostics:", flush=True)
    print(f"    Mean \u03c3x: {df['lstm_sigma_x'].mean():.4f} ft ({df['lstm_sigma_x'].mean()*12:.2f} in)", flush=True)
    print(f"    Mean \u03c3z: {df['lstm_sigma_z'].mean():.4f} ft ({df['lstm_sigma_z'].mean()*12:.2f} in)", flush=True)
    print(f"    Mean Uncertainty Vol: {df['uncertainty_vol'].mean():.6f}", flush=True)
    print(f"    P(Strike) range: [{df['lstm_p_strike'].quantile(0.05):.3f}, {df['lstm_p_strike'].quantile(0.95):.3f}]", flush=True)
    print(f"    P(Strike) mean:  {df['lstm_p_strike'].mean():.3f}", flush=True)
    print(f"    Mirage Pitches (|ΔP| >= 0.25): {(df['mirage_delta'].abs() >= 0.25).mean():.1%}", flush=True)
    
    # Verify uncertainty is not near-zero (which would mean hindsight)
    mean_sigma = (df['lstm_sigma_x'].mean() + df['lstm_sigma_z'].mean()) / 2
    if mean_sigma < 0.02:
        print(f"    \u26a0\ufe0f  WARNING: \u03c3 is very small ({mean_sigma:.4f} ft) — model may be\n"
              f"       evaluating too close to the plate. Consider a lower decision_frac.", flush=True)
    else:
        print(f"    \u2705 Meaningful uncertainty at decision point (\u03c3 avg = {mean_sigma:.4f} ft)", flush=True)
    
    return df


def parse_years(years_arg, positional_years):
    tokens = []
    if positional_years:
        tokens.extend(positional_years)
    if years_arg:
        tokens.append(years_arg)
    
    if not tokens:
        return [2025]
        
    years = []
    for token in tokens:
        token = str(token).strip()
        for part in token.split(','):
            part = part.strip()
            if not part:
                continue
            if '-' in part:
                start, end = part.split('-', 1)
                years.extend(range(int(start), int(end) + 1))
            else:
                years.append(int(part))
    
    return sorted(list(dict.fromkeys(years)))


def main():
    parser = argparse.ArgumentParser(description='Build swing decision features')
    parser.add_argument('positional_years', nargs='*', type=str, default=None,
                        help='Positional years (e.g. 2024 2025)')
    parser.add_argument('--years', type=str, default=None,
                        help='Years to process (e.g., "2025", "2020-2025", or "2020,2021")')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--decision-frac', type=float, default=0.40,
                        help='Fraction of flight at which to evaluate perception '
                             '(0=release, 1=plate, default=0.40 ~ 175ms after release, decision point)')
    parser.add_argument('--skip-lstm', action='store_true',
                        help='Skip LSTM inference (use only context features)')
    parser.add_argument('--output', type=str, default=None,
                        help='Optional custom output parquet path (default: data/training/swing_features_{year}.parquet)')
    args = parser.parse_args()
    
    years = parse_years(args.years, args.positional_years)
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Note: CUDA requested but torch.cuda.is_available() is False. Using CPU.", flush=True)
        args.device = 'cpu'

    # Load LSTM model
    scorer = None
    if not args.skip_lstm:
        scorer = LSTMEyeScorer(device=args.device, decision_frac=args.decision_frac)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for year in years:
        print(f"\n{'='*60}", flush=True)
        print(f"  Building swing features for {year}", flush=True)
        print(f"{'='*60}", flush=True)
        
        df = load_pitches(year)
        df = derive_features(df)
        
        if scorer is not None:
            df = add_lstm_features(df, scorer, year=year)
        
        # Select final columns
        feature_cols = [
            # IDs
            'game_pk', 'at_bat_number', 'pitch_number', 'batter', 'pitcher',
            # Target
            'is_swing',
            # Count
            'balls', 'strikes', 'count_code', 'count_group', 'two_strikes',
            # Handedness
            'stand', 'p_throws', 'stand_R', 'p_throws_R', 'platoon',
            # Pitch
            'pitch_type', 'pitch_group', 'release_speed', 'pfx_x', 'pfx_z',
            # Location
            'plate_x', 'plate_z', 'sz_top', 'sz_bot', 'zone', 'statcast_zone',
            'loc_x_norm', 'loc_z_norm', 'dist_from_center',
            'in_zone', 'loc_bin', 'true_strike',
            # Game state
            'outs_when_up', 'inning', 'runners_on', 'score_diff',
            # LSTM features (if available)
        ]
        
        if scorer is not None:
            feature_cols += [
                'lstm_mu_x', 'lstm_mu_z', 'lstm_sigma_x', 'lstm_sigma_z',
                'lstm_rho', 'lstm_p_strike', 'uncertainty_vol', 'log_uncertainty_vol',
                'mirage_delta',
            ]
        
        # Keep only existing columns
        feature_cols = [c for c in feature_cols if c in df.columns]
        out_df = df[feature_cols].copy()
        
        outpath = args.output if args.output else os.path.join(OUTPUT_DIR, f'swing_features_{year}.parquet')
        out_df.to_parquet(outpath, index=False)
        
        print(f"\n  Saved: {outpath}")
        print(f"  Shape: {out_df.shape}")
        print(f"  Swing rate: {out_df['is_swing'].mean():.3f}")
        print(f"  Features: {len(feature_cols)}")
        
        # Quick stats
        print(f"\n  Count group distribution:")
        for g in ['ahead', 'even', 'behind']:
            sub = out_df[out_df['count_group'] == g]
            print(f"    {g:8s}: {len(sub):>8,} pitches, swing rate = {sub['is_swing'].mean():.3f}")
        
        print(f"\n  Location bin distribution:")
        for b in ['zone', 'edge', 'chase']:
            sub = out_df[out_df['loc_bin'] == b]
            print(f"    {b:8s}: {len(sub):>8,} pitches, swing rate = {sub['is_swing'].mean():.3f}")


if __name__ == "__main__":
    main()
