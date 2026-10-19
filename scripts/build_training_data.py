"""
Build Training Data for LSTM+MDN Pitch Perception Model

For each pitch, runs calculate_trajectory() to produce the full 9-feature
state vector [x, y, z, vx, vy, vz, ax, ay, az] at each timestep.

Sequences are padded/truncated to a fixed length and saved as .pt files
for fast loading during training.

Usage:
    python3 scripts/build_training_data.py           # 2025 only
    python3 scripts/build_training_data.py 2024 2025  # Multiple years
"""

import sys
import os
import time
import numpy as np
import sqlite3

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.physics import calculate_trajectory
from src.db_manager import get_db_connection

# Lazy import torch — fail fast with helpful message
try:
    import torch
except ImportError:
    print("ERROR: PyTorch not installed. Run: pip install torch")
    sys.exit(1)


# --- Configuration ---
SEQ_LEN = 100          # Fixed sequence length (pad/truncate to this)
FEATURES = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az']
N_FEATURES = len(FEATURES)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'training')

# Swing event descriptions (same as run_eye_rating.py)
SWING_EVENTS = {
    'hit_into_play', 'foul', 'swinging_strike', 'swinging_strike_blocked',
    'foul_tip', 'foul_bunt', 'missed_bunt', 'bunt_foul_tip'
}


def load_pitches(year):
    """Load only the columns needed for trajectory + labels."""
    conn = get_db_connection()
    
    query = f"""
    SELECT 
        game_pk, at_bat_number, pitch_number,
        vx0, vy0, vz0, ax, ay, az,
        release_pos_x, release_pos_y, release_pos_z, release_extension,
        plate_x, plate_z, sz_top, sz_bot,
        description, pitch_type, p_throws, stand, batter,
        release_speed
    FROM statcast
    WHERE game_year = {year}
      AND vx0 IS NOT NULL
      AND release_pos_x IS NOT NULL
      AND plate_x IS NOT NULL
    """
    
    import pandas as pd
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"Loaded {len(df)} pitches with complete physics for {year}", flush=True)
    return df


def trajectory_to_tensor(row):
    """
    Run calculate_trajectory on a pitch row, extract the 9-feature sequence,
    and pad/truncate to SEQ_LEN.
    
    Returns:
        seq (np.ndarray): Shape (SEQ_LEN, N_FEATURES), float32
        valid_len (int): Actual number of timesteps before padding
        success (bool): Whether trajectory was computed successfully
    """
    traj = calculate_trajectory(row)
    
    if traj is None or len(traj) == 0:
        return np.zeros((SEQ_LEN, N_FEATURES), dtype=np.float32), 0, False
    
    # Extract feature matrix
    feat_matrix = traj[FEATURES].values.astype(np.float32)
    actual_len = len(feat_matrix)
    
    # Pad or truncate
    seq = np.zeros((SEQ_LEN, N_FEATURES), dtype=np.float32)
    
    if actual_len > SEQ_LEN:
        # Resample uniformly across the entire flight from release to plate to preserve full trajectory
        indices = np.linspace(0, actual_len - 1, SEQ_LEN).round().astype(int)
        seq = feat_matrix[indices]
        valid_len = SEQ_LEN
    else:
        # Pad: put actual data at the END, pad beginning with zeros
        # This way the final timestep (at plate) is always the last element
        seq[-actual_len:] = feat_matrix
        valid_len = actual_len
    
    return seq, valid_len, True


def build_training_data(year):
    """Generate training data for a single year."""
    
    df = load_pitches(year)
    n_pitches = len(df)
    
    # Pre-allocate arrays
    sequences = np.zeros((n_pitches, SEQ_LEN, N_FEATURES), dtype=np.float32)
    valid_lengths = np.zeros(n_pitches, dtype=np.int32)
    
    # Labels and identifiers
    plate_x = np.zeros(n_pitches, dtype=np.float32)
    plate_z = np.zeros(n_pitches, dtype=np.float32)
    sz_top = np.zeros(n_pitches, dtype=np.float32)
    sz_bot = np.zeros(n_pitches, dtype=np.float32)
    is_swing = np.zeros(n_pitches, dtype=np.bool_)
    batter_ids = np.zeros(n_pitches, dtype=np.int64)
    game_pks = np.zeros(n_pitches, dtype=np.int64)
    at_bat_numbers = np.zeros(n_pitches, dtype=np.int32)
    pitch_numbers = np.zeros(n_pitches, dtype=np.int32)
    
    # Process each pitch
    success_count = 0
    fail_count = 0
    start_time = time.time()
    
    for i, (idx, row) in enumerate(df.iterrows()):
        if i % 10000 == 0 and i > 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            eta = (n_pitches - i) / rate
            print(f"  [{year}] {i:,}/{n_pitches:,} ({rate:,.0f} pitches/s, ETA: {eta:.0f}s)", flush=True)
        
        seq, vlen, ok = trajectory_to_tensor(row)
        
        if ok:
            sequences[success_count] = seq
            valid_lengths[success_count] = vlen
            plate_x[success_count] = row['plate_x']
            plate_z[success_count] = row['plate_z']
            sz_top[success_count] = row.get('sz_top', 3.5) or 3.5
            sz_bot[success_count] = row.get('sz_bot', 1.5) or 1.5
            is_swing[success_count] = row['description'] in SWING_EVENTS
            batter_ids[success_count] = int(row['batter'])
            game_pks[success_count] = int(row['game_pk'])
            at_bat_numbers[success_count] = int(row['at_bat_number'])
            pitch_numbers[success_count] = int(row['pitch_number'])
            success_count += 1
        else:
            fail_count += 1
    
    elapsed = time.time() - start_time
    print(f"  [{year}] Done: {success_count:,} succeeded, {fail_count:,} failed in {elapsed:.1f}s", flush=True)
    
    # Trim to actual count (remove pre-allocated but unused slots)
    sequences = sequences[:success_count]
    valid_lengths = valid_lengths[:success_count]
    plate_x = plate_x[:success_count]
    plate_z = plate_z[:success_count]
    sz_top = sz_top[:success_count]
    sz_bot = sz_bot[:success_count]
    is_swing = is_swing[:success_count]
    batter_ids = batter_ids[:success_count]
    game_pks = game_pks[:success_count]
    at_bat_numbers = at_bat_numbers[:success_count]
    pitch_numbers = pitch_numbers[:success_count]
    
    # Convert to tensors and save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f'trajectories_{year}.pt')
    
    data = {
        'sequences': torch.from_numpy(sequences),       # (N, 100, 9)
        'valid_lengths': torch.from_numpy(valid_lengths), # (N,)
        'labels': {
            'plate_x': torch.from_numpy(plate_x),       # (N,)
            'plate_z': torch.from_numpy(plate_z),       # (N,)
            'sz_top': torch.from_numpy(sz_top),         # (N,)
            'sz_bot': torch.from_numpy(sz_bot),         # (N,)
            'is_swing': torch.from_numpy(is_swing),     # (N,)
            'batter_id': torch.from_numpy(batter_ids),  # (N,)
            'game_pk': torch.from_numpy(game_pks),      # (N,)
            'at_bat_number': torch.from_numpy(at_bat_numbers), # (N,)
            'pitch_number': torch.from_numpy(pitch_numbers),   # (N,)
        },
        'metadata': {
            'year': year,
            'seq_len': SEQ_LEN,
            'features': FEATURES,
            'n_pitches': success_count,
        }
    }
    
    torch.save(data, output_path)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  [{year}] Saved {output_path} ({file_size_mb:.1f} MB)", flush=True)
    
    return success_count


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


if __name__ == "__main__":
    import argparse
    from concurrent.futures import ProcessPoolExecutor, as_completed

    parser = argparse.ArgumentParser(description="Build Training Data for LSTM+MDN Pitch Perception Model")
    parser.add_argument('positional_years', nargs='*', type=str, default=None,
                        help='Positional years (e.g. 2024 2025)')
    parser.add_argument('--years', type=str, default=None,
                        help='Years to process (e.g., "2025", "2020-2025", or "2020,2021")')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers to process years (default: min(num_years, cpu_count))')
    args = parser.parse_args()

    years = parse_years(args.years, args.positional_years)
    
    print(f"Building training data for years: {years}", flush=True)
    print(f"Sequence length: {SEQ_LEN}, Features: {N_FEATURES}", flush=True)
    print(f"Output directory: {OUTPUT_DIR}", flush=True)
    print(flush=True)
    
    total = 0
    if len(years) > 1 and (args.workers is None or args.workers > 1):
        try:
            available_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            available_cpus = os.cpu_count() or 1
        
        max_workers = args.workers if args.workers is not None else min(len(years), available_cpus)
        print(f"Processing {len(years)} years across {max_workers} parallel workers...\n", flush=True)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_year = {executor.submit(build_training_data, y): y for y in years}
            for future in as_completed(future_to_year):
                y = future_to_year[future]
                try:
                    n = future.result()
                    total += n
                except Exception as exc:
                    print(f"Year {y} generated an exception: {exc}", flush=True)
                    raise exc
    else:
        for year in years:
            print(f"{'='*50}", flush=True)
            print(f"  Processing {year}", flush=True)
            print(f"{'='*50}", flush=True)
            n = build_training_data(year)
            total += n
    
    print(f"\nTotal pitches processed: {total:,}", flush=True)
