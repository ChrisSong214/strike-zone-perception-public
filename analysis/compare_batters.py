import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import pandas as pd
import numpy as np

# Load features
df = pd.read_parquet('data/training/swing_features_2025.parquet')

# Add deception metrics
# 1. Uncertainty: how close is the out-of-hand read to 50/50? (Lower is harder)
df['certainty'] = np.abs(df['lstm_p_strike'] - 0.5)

# 2. Deception gap: how much does the out-of-hand read differ from the final location (0 or 1)?
df['deception'] = np.abs(df['lstm_p_strike'] - df['in_zone'])

# IDs
yordan_id = 670541
gleyber_id = 650402

def get_stats(batter_id, name):
    bdf = df[df['batter'] == batter_id]
    
    n_pitches = len(bdf)
    edge_pct = (bdf['loc_bin'] == 'edge').mean() * 100
    chase_pitches = (bdf['loc_bin'] == 'chase').mean() * 100
    avg_deception = bdf['deception'].mean()
    high_deception_pct = (bdf['deception'] > 0.4).mean() * 100
    
    # Uncertainty: % of pitches that are highly ambiguous out of the hand (p_strike between 0.25 and 0.75)
    ambiguous_pct = ((bdf['lstm_p_strike'] >= 0.25) & (bdf['lstm_p_strike'] <= 0.75)).mean() * 100
    
    print(f"--- {name} ---")
    print(f"Pitches Seen:          {n_pitches}")
    print(f"% Pitches on Edge:     {edge_pct:.1f}%")
    print(f"% Chase Pitches Seen:  {chase_pitches:.1f}%")
    print(f"Avg Deception Gap:     {avg_deception:.3f}")
    print(f"% High Deception:      {high_deception_pct:.1f}% (Pitch looks like strike but is ball, or vice versa)")
    print(f"% Ambiguous out of hand: {ambiguous_pct:.1f}% (lstm_p_strike between 25% and 75%)")
    print()

get_stats(yordan_id, "Yordan Alvarez")
get_stats(gleyber_id, "Gleyber Torres")

