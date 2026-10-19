"""
Visualize LSTM Streaming Perception — The "Living Cloud"

Shows how the model's belief about plate-crossing location evolves
over time as the pitch approaches the plate.

Usage:
    python3 scripts/visualize_perception.py              # Random 2025 pitch
    python3 scripts/visualize_perception.py --batter 660271  # Specific batter
    python3 scripts/visualize_perception.py --n 6            # Show 6 pitches
"""

import sys
import os
import argparse
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse

from src.lstm_scoring import LSTMEyeScorer
from src.db_manager import get_db_connection

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')


def draw_zone(ax, sz_top, sz_bot, sz_width=17/12):
    """Draw the strike zone rectangle."""
    x_min = -sz_width / 2
    rect = patches.Rectangle(
        (x_min, sz_bot), sz_width, sz_top - sz_bot,
        linewidth=2, edgecolor='white', facecolor='none',
        linestyle='--', alpha=0.7
    )
    ax.add_patch(rect)


def draw_perception_ellipse(ax, mu_x, mu_z, sigma_x, sigma_z, rho=0, 
                             color='cyan', alpha=0.3, label=None):
    """Draw a 2σ confidence ellipse for the predicted Gaussian."""
    # Eigenvalue decomposition of covariance matrix
    cov = np.array([
        [sigma_x**2, rho * sigma_x * sigma_z],
        [rho * sigma_x * sigma_z, sigma_z**2]
    ])
    
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    
    # 2σ ellipse
    width = 2 * 2 * np.sqrt(abs(eigenvalues[0]))
    height = 2 * 2 * np.sqrt(abs(eigenvalues[1]))
    
    ellipse = Ellipse(
        (mu_x, mu_z), width, height, angle=angle,
        facecolor=color, edgecolor=color, alpha=alpha, label=label
    )
    ax.add_patch(ellipse)


def visualize_pitch_perception(scorer, pitch_row, ax=None, show_evolution=True):
    """
    Visualize the model's streaming perception for a single pitch.
    
    Shows snapshot ellipses at several points along the trajectory,
    demonstrating how the cloud shrinks from release to plate.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 8))
    
    result = scorer.predict_perception(pitch_row)
    
    if result is None:
        ax.text(0.5, 0.5, 'Trajectory generation failed', 
                ha='center', va='center', transform=ax.transAxes, color='white')
        return
    
    params = result['params']  # (100, 5)
    vlen = result['valid_len']
    start_idx = 100 - vlen
    
    plate_x = pitch_row['plate_x']
    plate_z = pitch_row['plate_z']
    sz_top = pitch_row.get('sz_top', 3.5) or 3.5
    sz_bot = pitch_row.get('sz_bot', 1.5) or 1.5
    
    # Draw zone
    draw_zone(ax, sz_top, sz_bot)
    
    if show_evolution:
        # Show snapshots at 20%, 40%, 60%, 80%, 100% of flight
        fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
        colors = ['#ff6b6b', '#ffa502', '#ffd93d', '#6bcb77', '#4d96ff']
        alphas = [0.12, 0.15, 0.18, 0.22, 0.35]
        
        for frac, color, alpha in zip(fractions, colors, alphas):
            idx = start_idx + int(frac * (vlen - 1))
            idx = min(idx, 99)
            
            mu_x_t = params[idx, 0]
            mu_z_t = params[idx, 1]
            sigma_x_t = params[idx, 2]
            sigma_z_t = params[idx, 3]
            rho_t = params[idx, 4]
            
            pct = int(frac * 100)
            label = f"{pct}% ({sigma_x_t:.2f}, {sigma_z_t:.2f})"
            draw_perception_ellipse(
                ax, mu_x_t, mu_z_t, sigma_x_t, sigma_z_t, rho_t,
                color=color, alpha=alpha, label=label
            )
    
    # Final prediction marker
    final = result['final']
    ax.plot(final['mu_x'], final['mu_z'], 'o', color='#4d96ff', 
            markersize=8, markeredgecolor='white', markeredgewidth=1.5,
            zorder=10, label=f"Prediction")
    
    # Actual location
    ax.plot(plate_x, plate_z, 'x', color='#ff4757', markersize=12,
            markeredgewidth=3, zorder=11, label=f"Actual")
    
    # Formatting
    is_swing = pitch_row['description'] in {
        'hit_into_play', 'foul', 'swinging_strike', 'swinging_strike_blocked',
        'foul_tip', 'foul_bunt', 'missed_bunt', 'bunt_foul_tip'
    }
    action = "SWING" if is_swing else "TAKE"
    pitch_type = pitch_row.get('pitch_type', '?')
    speed = pitch_row.get('release_speed', 0)
    
    ax.set_title(
        f"{pitch_type} @ {speed:.0f} mph — {action}\n"
        f"σ: ({final['sigma_x']:.3f}, {final['sigma_z']:.3f}) ft",
        color='white', fontsize=10
    )
    
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(0, 5)
    ax.set_aspect('equal')
    ax.set_facecolor('#1a1a2e')
    ax.tick_params(colors='white')
    ax.set_xlabel('X (ft)', color='white')
    ax.set_ylabel('Z (ft)', color='white')
    ax.legend(fontsize=7, loc='upper right', framealpha=0.3,
              labelcolor='white', edgecolor='gray')
    
    for spine in ax.spines.values():
        spine.set_color('gray')


def visualize_sigma_over_time(scorer, pitch_row, ax=None):
    """
    Plot σx and σz over time to show how uncertainty changes.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    
    result = scorer.predict_perception(pitch_row)
    if result is None:
        return
    
    params = result['params']
    vlen = result['valid_len']
    start_idx = 100 - vlen
    
    valid_params = params[start_idx:]
    t = np.arange(len(valid_params)) * 0.005  # 5ms timesteps
    
    ax.plot(t, valid_params[:, 2], color='#4d96ff', linewidth=2, label='σ_x')
    ax.plot(t, valid_params[:, 3], color='#ff6b6b', linewidth=2, label='σ_z')
    
    ax.set_xlabel('Time (s)', color='white')
    ax.set_ylabel('Uncertainty (ft)', color='white')
    ax.set_title('Perception Uncertainty Over Time', color='white')
    ax.set_facecolor('#1a1a2e')
    ax.tick_params(colors='white')
    ax.legend(fontsize=9, framealpha=0.3, labelcolor='white')
    ax.grid(True, alpha=0.2)
    for spine in ax.spines.values():
        spine.set_color('gray')


def main():
    parser = argparse.ArgumentParser(description='Visualize LSTM Perception')
    parser.add_argument('--year', type=int, default=2025)
    parser.add_argument('--batter', type=int, default=None)
    parser.add_argument('--n', type=int, default=4, help='Number of pitches to show')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Load model
    scorer = LSTMEyeScorer(device=args.device)
    
    # Load some pitches
    conn = get_db_connection()
    if args.batter:
        query = f"""
        SELECT * FROM statcast 
        WHERE game_year = {args.year} AND batter = {args.batter}
          AND vx0 IS NOT NULL AND release_pos_x IS NOT NULL AND plate_x IS NOT NULL
        LIMIT 100
        """
    else:
        query = f"""
        SELECT * FROM statcast 
        WHERE game_year = {args.year}
          AND vx0 IS NOT NULL AND release_pos_x IS NOT NULL AND plate_x IS NOT NULL
        ORDER BY RANDOM()
        LIMIT 100
        """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if len(df) == 0:
        print("No pitches found.")
        return
    
    # Pick n pitches
    sample = df.sample(min(args.n, len(df)), random_state=args.seed)
    
    # Create figure with cloud views and sigma-over-time
    n = len(sample)
    fig = plt.figure(figsize=(5 * n, 12), facecolor='#0f0f23')
    
    for i, (idx, row) in enumerate(sample.iterrows()):
        # Top: perception cloud
        ax_cloud = fig.add_subplot(2, n, i + 1)
        visualize_pitch_perception(scorer, row, ax=ax_cloud)
        
        # Bottom: sigma over time
        ax_sigma = fig.add_subplot(2, n, n + i + 1)
        visualize_sigma_over_time(scorer, row, ax=ax_sigma)
    
    plt.tight_layout()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outpath = os.path.join(OUTPUT_DIR, 'perception_visualization.png')
    fig.savefig(outpath, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
    print(f"\nSaved: {outpath}")
    plt.close()


if __name__ == "__main__":
    main()
