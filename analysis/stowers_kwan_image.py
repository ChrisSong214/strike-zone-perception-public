import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'sans-serif'

fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

ax.text(0.5, 0.96, 'Stowers vs. Kwan — The Passivity Penalty', 
        ha='center', va='top', fontsize=20, fontweight='bold', color='#1a1a2e')

left_x = 0.25
right_x = 0.75
label_x = 0.50

ax.text(left_x, 0.88, 'Kyle Stowers', ha='center', va='center', fontsize=16, fontweight='bold', color='#2d6a4f')
ax.text(right_x, 0.88, 'Steven Kwan', ha='center', va='center', fontsize=16, fontweight='bold', color='#ae2012')

ax.plot([0.50, 0.50], [0.05, 0.85], color='#cccccc', linewidth=1.5, linestyle='--')
ax.plot([0.05, 0.95], [0.84, 0.84], color='#cccccc', linewidth=1)

# Using Savant/FanGraphs figures for Chase% and Z-Swing%
rows = [
    ('DVA Rating',           '114.8',     '97.1',      True),
    ('Tendency DVA',         '+0.0347',   '-0.0105',   True),
    ('Actual DVA',           '+0.0536',   '+0.0006',   True),
    ('Pitches Seen',         '1,712',     '2,829',     False),
    ('',                     '',          '',          False),
    ('Overall Swing%',       '54.0%',     '42.0%',     False),
    ('Chase% (Savant)',      '29.0%',     '22.8%',     False),
    ('Z-Swing% (Savant)',    '77.5%',     '56.9%',     True),
    ('Edge Swing Rate',      '63.9%',     '46.8%',     False),
    ('',                     '',          '',          False),
    ('Elite Decisions',      '35.9%',     '31.8%',     True),
    ('Terrible Decisions',   '24.4%',     '28.0%',     True),
    ('Avg DVA / Pitch',      '+0.0296',   '-0.0074',   True),
]

start_y = 0.80
row_h = 0.055

for i, (label, left_val, right_val, highlight) in enumerate(rows):
    y = start_y - i * row_h
    if label == '':
        continue
    if i % 2 == 0:
        rect = plt.Rectangle((0.05, y - 0.02), 0.90, row_h, 
                              facecolor='#f8f9fa', edgecolor='none', transform=ax.transAxes)
        ax.add_patch(rect)
    label_weight = 'bold' if highlight else 'normal'
    label_size = 11 if highlight else 10
    ax.text(label_x, y, label, ha='center', va='center', fontsize=label_size, 
            fontweight=label_weight, color='#333333')
    val_size = 13 if highlight else 11
    val_weight = 'bold' if highlight else 'normal'
    if highlight:
        left_color = '#2d6a4f'
        right_color = '#ae2012'
    else:
        left_color = '#444444'
        right_color = '#444444'
    ax.text(left_x, y, left_val, ha='center', va='center', fontsize=val_size, 
            fontweight=val_weight, color=left_color)
    ax.text(right_x, y, right_val, ha='center', va='center', fontsize=val_size, 
            fontweight=val_weight, color=right_color)

plt.tight_layout()
out_path = os.path.join(_PROJECT_ROOT, 'assets', 'stowers_vs_kwan.png')
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print(f"Saved {out_path}")

