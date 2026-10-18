import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import multivariate_normal

matplotlib.rcParams['font.family'] = 'sans-serif'

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Strike zone dimensions (in feet, centered at 0)
zone_left = -0.708
zone_right = 0.708
zone_bot = 1.5
zone_top = 3.5

# Create grid
x = np.linspace(-2.0, 2.0, 300)
z = np.linspace(0.0, 5.0, 300)
X, Z = np.meshgrid(x, z)
pos = np.dstack((X, Z))

# Custom colormap: transparent white to orange to red
colors = [(1, 1, 1, 0), (1, 0.6, 0, 0.4), (1, 0.2, 0, 0.7), (0.8, 0, 0, 0.9)]
cmap = LinearSegmentedColormap.from_list('cloud', colors, N=256)

def draw_zone(ax):
    zone = Rectangle((zone_left, zone_bot), zone_right - zone_left, zone_top - zone_bot,
                      linewidth=2.5, edgecolor='#333333', facecolor='none', linestyle='-')
    ax.add_patch(zone)
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(0.0, 5.0)
    ax.set_aspect('equal')
    ax.set_xlabel('Horizontal (ft)', fontsize=11, color='#555555')
    ax.set_ylabel('Vertical (ft)', fontsize=11, color='#555555')
    ax.tick_params(colors='#888888')
    for spine in ax.spines.values():
        spine.set_color('#cccccc')

# --- LEFT: Deceptive Pitch (High Uncertainty Schematic) ---
ax1 = axes[0]
mu1 = [0.5, 2.0]
cov1 = [[0.35, -0.15], [-0.15, 0.45]]
rv1 = multivariate_normal(mu1, cov1)
p1_exact = rv1.cdf([zone_right, zone_top], lower_limit=[zone_left, zone_bot])
ax1.set_title(f'Deceptive Slider at 40% Flight (Schematic)\nP(Strike) = {p1_exact*100:.1f}%', fontsize=14, fontweight='bold', color='#1a1a2e', pad=15)

density1 = rv1.pdf(pos)
ax1.contourf(X, Z, density1, levels=15, cmap=cmap)
ax1.contour(X, Z, density1, levels=5, colors='#cc3300', alpha=0.3, linewidths=0.8)
draw_zone(ax1)

# Mark the actual final location (outside zone)
ax1.plot(1.1, 1.2, 'x', color='#0066cc', markersize=14, markeredgewidth=3)
ax1.annotate('Actual\nLocation', xy=(1.1, 1.2), xytext=(1.5, 0.7),
             fontsize=9, color='#0066cc', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#0066cc', lw=1.5))

# Mark center of cloud
ax1.plot(mu1[0], mu1[1], '+', color='white', markersize=16, markeredgewidth=2.5)

# --- RIGHT: Obvious Fastball (Low Uncertainty Schematic) ---
ax2 = axes[1]
mu2 = [0.05, 2.6]
cov2 = [[0.06, 0.01], [0.01, 0.08]]
rv2 = multivariate_normal(mu2, cov2)
p2_exact = rv2.cdf([zone_right, zone_top], lower_limit=[zone_left, zone_bot])
ax2.set_title(f'Center-Cut Fastball at 40% Flight (Schematic)\nP(Strike) = {p2_exact*100:.1f}%', fontsize=14, fontweight='bold', color='#1a1a2e', pad=15)

density2 = rv2.pdf(pos)
ax2.contourf(X, Z, density2, levels=15, cmap=cmap)
ax2.contour(X, Z, density2, levels=5, colors='#cc3300', alpha=0.3, linewidths=0.8)
draw_zone(ax2)

# Actual location (center of zone)
ax2.plot(0.1, 2.5, 'x', color='#0066cc', markersize=14, markeredgewidth=3)
ax2.annotate('Actual\nLocation', xy=(0.1, 2.5), xytext=(0.9, 1.5),
             fontsize=9, color='#0066cc', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#0066cc', lw=1.5))

ax2.plot(mu2[0], mu2[1], '+', color='white', markersize=16, markeredgewidth=2.5)

plt.suptitle('Schematic Representation: The Probability Cloud at the Biological Decision Point',
             fontsize=16, fontweight='bold', color='#1a1a2e', y=1.02)
plt.tight_layout()
out_path = os.path.join(_PROJECT_ROOT, 'assets', 'probability_cloud.png')
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print(f"Saved {out_path}")

