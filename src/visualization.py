import plotly.graph_objects as go
import numpy as np

def visualize_pitch_cloud(actual_traj, cloud_trajs, pitch_type="Pitch", pitch_hand=None, perceptual_trajs=None, show_paths=True, show_physical=True, sz_top=None, sz_bot=None, invert_x=True):
    """
    Visualizes the pitch cloud using Plotly 3D Mesh (Blob).
    
    Args:
        actual_traj (pd.DataFrame): The ground truth trajectory.
        cloud_trajs (list): List of pd.DataFrames for perturbed trajectories (Physical).
        pitch_type (str): Label for the plot.
        pitch_hand (str): 'L' or 'R' (optional) for annotation.
        perceptual_trajs (list): Optional second cloud (Perceptual/Green).
        show_paths (bool): If True, shows individual sample lines. If False, shows only Cloud + Actual.
        show_physical (bool): If True, shows the Cyan Physical Cloud mesh.
        sz_top (float): Top of strike zone in feet. Defaults to 3.5.
        sz_bot (float): Bottom of strike zone in feet. Defaults to 1.5.
        invert_x (bool): If True, inverts X-axis so LHP is on negative X (default: True).
    """
    
    # Collect points for the Mesh (Blob)
    all_x, all_y, all_z = [], [], []
    final_x, final_z = [], []
    
    # Determine handedness from data if not provided
    if pitch_hand is None and len(actual_traj) > 0:
        start_x = actual_traj.iloc[0]['x']
        pitch_hand = 'L' if start_x > 0 else 'R'
    
    for traj in cloud_trajs:
        # Downsample trajectory points to keep mesh light but representative
        # Take every 5th point
        sub = traj.iloc[::5] 
        all_x.extend(sub['x'].values)
        all_y.extend(sub['y'].values)
        all_z.extend(sub['z'].values)
        
        # Capture endpoint
        final_x.append(traj['x'].iloc[-1])
        final_z.append(traj['z'].iloc[-1])

    # Coordinate transform: invert X if user prefers LHP on left of screen
    x_mult = -1.0 if invert_x else 1.0
    all_x = [x_mult * x for x in all_x]
    final_x = [x_mult * x for x in final_x]
    actual_traj_x = [x_mult * x for x in actual_traj['x'].values]
    
    hand_desc = "Left-Handed" if pitch_hand == 'L' else "Right-Handed"
    
    fig = go.Figure()

    # --- A. The "Physical Cloud" (Cyan) ---
    if show_physical:
        fig.add_trace(go.Mesh3d(
            x=all_x, 
            y=all_y, 
            z=all_z, 
            alphahull=6, 
            opacity=0.10,
            color='cyan',
            name='Physical Cloud (Physics)'
        ))
    
    # --- A2. The "Perceptual Cloud" (Green) - OPTIONAL ---
    if perceptual_trajs:
        perc_x, perc_y, perc_z = [], [], []
        perc_final_x, perc_final_z = [], []
        
        for traj in perceptual_trajs:
            sub = traj.iloc[::5]
            perc_x.extend([x_mult * x for x in sub['x'].values])
            perc_y.extend(sub['y'].values)
            perc_z.extend(sub['z'].values)
            
            perc_final_x.append(x_mult * traj['x'].iloc[-1])
            perc_final_z.append(traj['z'].iloc[-1])
            
        fig.add_trace(go.Mesh3d(
            x=perc_x, 
            y=perc_y, 
            z=perc_z, 
            alphahull=6, 
            opacity=0.20,
            color='lime',  # Green for Perception
            name='Perceptual Cloud (The Eye)'
        ))
        
        # Add Perceptual "Splat" (Green)
        fig.add_trace(go.Scatter3d(
            x=perc_final_x,
            y=[1.417]*len(perc_final_x),
            z=perc_final_z,
            mode='markers',
            marker=dict(size=3, color='lime', opacity=0.8),
            name='Perceptual Probabilities'
        ))

    # --- B. The Individual Trajectories (optional, fainter) ---
    if show_paths:
        for i, traj in enumerate(cloud_trajs[:20]): 
            trace_x = [x_mult * x for x in traj['x'].values]
            fig.add_trace(go.Scatter3d(
                x=trace_x, y=traj['y'], z=traj['z'],
                mode='lines',
                line=dict(color='rgba(0, 255, 255, 0.3)', width=1),
                showlegend=False,
                name='Path'
            ))

    # --- C. The "Actual" Path ---
    fig.add_trace(go.Scatter3d(
        x=actual_traj_x, y=actual_traj['y'], z=actual_traj['z'],
        mode='lines',
        line=dict(color='white', width=4),
        name='Actual Pitch'
    ))
    
    # --- D. The Strike Zone (Home Plate) ---
    if sz_top is None: sz_top = 3.5
    if sz_bot is None: sz_bot = 1.5
    sz_width = 17/12 # 17 inches
    plate_y = 1.417
    
    fig.add_trace(go.Scatter3d(
        x=[-sz_width/2, sz_width/2, sz_width/2, -sz_width/2, -sz_width/2],
        y=[plate_y, plate_y, plate_y, plate_y, plate_y],
        z=[sz_bot, sz_bot, sz_top, sz_top, sz_bot],
        mode='lines',
        line=dict(color='red', width=3),
        name='Strike Zone'
    ))
    
    # --- E. The "Splat" (Endpoints) ---
    if show_paths:
        fig.add_trace(go.Scatter3d(
            x=final_x,
            y=[plate_y]*len(final_x),
            z=final_z,
            mode='markers',
            marker=dict(size=4, color='cyan', opacity=0.5), # Reduced opacity to show Green
            name='Physical Probabilities'
        ))

    # Layout
    fig.update_layout(
        title=f"The Probability Clouds: {pitch_type}<br><sup>View: Catcher's Perspective (Inverted X) | Pitcher: {hand_desc}</sup>",
        scene=dict(
            xaxis_title='X (ft) [Inverted]',
            yaxis_title='Distance (ft)',
            zaxis_title='Height (ft)',
            aspectmode='manual',
            aspectratio=dict(x=1, y=4, z=1), 
            camera=dict(
                eye=dict(x=2.5, y=0.5, z=0.5) 
            ),
            # Invert Y to show from mound perspective
            yaxis=dict(autorange='reversed'),
            # Ensure X axis direction is standard (Left=Neg, Right=Pos)
            xaxis=dict(autorange=True) 
        ),
        template="plotly_dark",
        margin=dict(l=0, r=0, b=0, t=60)
    )
    
    # Add explicit annotation for X-axis side
    # LHP is now Negative X (Left Side)
    # RHP is now Positive X (Right Side)
    annot_x_pos = 0.1 if pitch_hand == 'L' else 0.9
    
    fig.add_annotation(
        text=f"{hand_desc}",
        xref="paper", yref="paper",
        x=annot_x_pos, 
        y=0.9, showarrow=False,
        font=dict(color="yellow", size=14),
        bgcolor="rgba(0,0,0,0.5)"
    )

    return fig
