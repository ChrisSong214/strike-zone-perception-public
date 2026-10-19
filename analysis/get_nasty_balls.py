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

# Filter only pitches that were BALLS (out of the zone, the ones you are supposed to take)
balls_df = df[df['in_zone'] == 0]

def get_nasty_balls(batter_id, name):
    bdf = balls_df[balls_df['batter'] == batter_id]
    
    avg_p_strike = bdf['lstm_p_strike'].mean() * 100
    median_p_strike = bdf['lstm_p_strike'].median() * 100
    
    # % of balls that looked > 40% like a strike
    nasty_pct = (bdf['lstm_p_strike'] > 0.4).mean() * 100
    
    print(f"{name}:")
    print(f"  Avg P(Strike) on Balls: {avg_p_strike:.1f}%")
    print(f"  % 'Nasty' Balls (>40% strike prob): {nasty_pct:.1f}%")

get_nasty_balls(670541, "Yordan Alvarez (#1)")
get_nasty_balls(669743, "Alex Call (Elite Chase Rate)")
get_nasty_balls(641343, "Jake Bauers")
get_nasty_balls(663757, "Trent Grisham")

