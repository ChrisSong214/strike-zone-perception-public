import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import pandas as pd
import numpy as np

df = pd.read_parquet('data/training/swing_features_2025.parquet')
df['deception'] = np.abs(df['lstm_p_strike'] - df['in_zone'])

trout_id = 545361
kwan_id = 680757

def get_stats(batter_id, name):
    bdf = df[df['batter'] == batter_id]
    
    # Let's look at the "Takes" specifically on balls (the good decisions)
    takes_on_balls = bdf[(bdf['in_zone'] == 0) & (bdf['is_swing'] == 0)]
    
    # How nasty were the balls they successfully took?
    avg_p_strike = takes_on_balls['lstm_p_strike'].mean() * 100
    nasty_takes = (takes_on_balls['lstm_p_strike'] > 0.4).mean() * 100
    
    high_deception = (bdf['deception'] > 0.4).mean() * 100
    
    print(f"--- {name} ---")
    print(f"High Deception Pitches Seen: {high_deception:.1f}%")
    print(f"When successfully taking a ball, Avg P(Strike) out of hand: {avg_p_strike:.1f}%")
    print(f"% of successful takes that were 'Nasty' (Looked >40% like a strike): {nasty_takes:.1f}%")
    print()

get_stats(trout_id, "Mike Trout")
get_stats(kwan_id, "Steven Kwan")
