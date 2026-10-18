import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import sys, os
sys.path.append(os.path.abspath('.'))

import pandas as pd
import numpy as np
import sqlite3
from src.physics import calculate_trajectory
from src.uncertainty import generate_pitch_cloud
from src.perception import PerceptualCloud
from src.visualization import visualize_pitch_cloud

# Load a real pitch from the database
conn = sqlite3.connect('data/mlb.db')
# Pick a nasty slider from a known pitcher
row_df = pd.read_sql_query("""
    SELECT * FROM statcast 
    WHERE pitch_type = 'SL' 
      AND vx0 IS NOT NULL 
      AND release_pos_x IS NOT NULL
      AND plate_x IS NOT NULL
      AND game_year = 2024
    ORDER BY RANDOM()
    LIMIT 1
""", conn)
conn.close()

row = row_df.iloc[0]
pitcher_name = row.get('player_name', 'Pitcher')
pitch_type = row.get('pitch_type', 'SL')

print(f"Selected: {pitcher_name} {pitch_type}")

# Generate clouds
print("Generating Physical Cloud...")
cloud_trajs = generate_pitch_cloud(row, n_samples=200)
actual_traj = calculate_trajectory(row)

print("Generating Perceptual Cloud...")
eye_model = PerceptualCloud()
perceptual_trajs = eye_model.calculate_posterior_cloud(cloud_trajs, actual_traj)

print("Rendering...")
fig = visualize_pitch_cloud(
    actual_traj, 
    cloud_trajs, 
    pitch_type=f"{pitcher_name} {pitch_type} (Perceptual Only)", 
    perceptual_trajs=perceptual_trajs,
    show_paths=False,
    show_physical=False
)

assets_dir = os.path.join(_PROJECT_ROOT, "assets")
os.makedirs(assets_dir, exist_ok=True)
png_path = os.path.join(assets_dir, "perceptual_cloud_3d.png")
html_path = os.path.join(assets_dir, "perceptual_cloud_3d.html")

# Try to export as static image
try:
    fig.write_image(png_path, width=1400, height=800, scale=2)
    print(f"Saved {png_path}")
except Exception as e:
    print(f"write_image failed ({e}), trying HTML export...")
    fig.write_html(html_path)
    print(f"Saved {html_path} (open in browser and screenshot)")


