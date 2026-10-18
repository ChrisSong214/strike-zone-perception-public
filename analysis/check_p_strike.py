import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import pandas as pd
import numpy as np

df = pd.read_parquet('data/training/swing_features_2025.parquet')

print(f"Total pitches: {len(df)}")
print(f"Overall league strike %: {df['in_zone'].mean()*100:.1f}%")
print(f"Overall model avg lstm_p_strike: {df['lstm_p_strike'].mean()*100:.1f}%")

balls = df[df['in_zone'] == 0]
strikes = df[df['in_zone'] == 1]

print(f"\nFor ACTUAL BALLS (N={len(balls)}):")
print(f"  Avg lstm_p_strike: {balls['lstm_p_strike'].mean()*100:.1f}%")
print(f"  Median lstm_p_strike: {balls['lstm_p_strike'].median()*100:.1f}%")
print(f"  % > 40% (looked like strikes): {(balls['lstm_p_strike'] > 0.4).mean()*100:.1f}%")

print(f"\nFor ACTUAL STRIKES (N={len(strikes)}):")
print(f"  Avg lstm_p_strike: {strikes['lstm_p_strike'].mean()*100:.1f}%")
print(f"  Median lstm_p_strike: {strikes['lstm_p_strike'].median()*100:.1f}%")
print(f"  % < 40% (looked like balls): {(strikes['lstm_p_strike'] < 0.4).mean()*100:.1f}%")

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(df['in_zone'], df['lstm_p_strike'])
print(f"\nModel AUC at 0.0 fraction: {auc:.3f}")
