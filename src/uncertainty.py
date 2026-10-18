import numpy as np
import pandas as pd
from src.physics import calculate_trajectory

def generate_pitch_cloud(row, n_samples=50, cov_matrix=None, rng=None):
    """
    Generates a 'cloud' of possible trajectories for a given pitch by adding 
    Gaussian noise to the initial conditions.

    Args:
        row (pd.Series): A single pitch row containing physics parameters 
                         (release_speed, release_pos_x, etc.)
        n_samples (int): Number of trajectory variations to generate.
        cov_matrix (np.ndarray, optional): 5x5 covariance matrix.
        rng (np.random.RandomState, optional): Random number generator for deterministic sampling.

    Returns:
        list of pd.DataFrame: A list where each element is a DataFrame representing 
                              one sampled trajectory (x, y, z, t).
    """
    
    # Default standard deviations (uncertainties)
    # Calibrated from 2024-2025 Rodón (Representative Ace) data:
    # Speed: ~1.5 mph, Pos: ~0.2 ft, Mov: ~0.2 ft
    # MULTIVARIATE: We now use a covariance matrix to capture correlations.
    
    features = ['release_speed', 'release_pos_x', 'release_pos_z', 'pfx_x', 'pfx_z']
    
    if cov_matrix is None:
        # Default Covariance Matrix (Calibrated from Rodón 2024-25 via calibrate.py)
        # Features: [release_speed, release_pos_x, release_pos_z, pfx_x, pfx_z]
        cov_matrix = np.array([
            [ 2.277, -0.159, -0.023, -0.050, -0.010],
            [-0.159,  0.041, -0.002,  0.010, -0.002],
            [-0.023, -0.002,  0.010,  0.001,  0.001],
            [-0.050,  0.010,  0.001,  0.035, -0.004],
            [-0.010, -0.002,  0.001, -0.004,  0.017]
        ])
    
    # 1. Generate N samples of perturbations at once
    # Result is shape (n_samples, 5)
    if rng is not None:
        noise = rng.multivariate_normal(mean=[0]*5, cov=cov_matrix, size=n_samples)
    else:
        noise = np.random.multivariate_normal(mean=[0]*5, cov=cov_matrix, size=n_samples)
    
    trajectories = []
    
    for i in range(n_samples):
        # Create a perturbed copy of the row
        sim_row = row.copy()
        
        # Apply noise
        # features = ['release_speed', 'release_pos_x', 'release_pos_z', 'pfx_x', 'pfx_z']
        if 'release_speed' in sim_row: sim_row['release_speed'] += noise[i, 0]
        if 'release_pos_x' in sim_row: sim_row['release_pos_x'] += noise[i, 1]
        if 'release_pos_z' in sim_row: sim_row['release_pos_z'] += noise[i, 2]
        if 'pfx_x' in sim_row:         sim_row['pfx_x']         += noise[i, 3]
        if 'pfx_z' in sim_row:         sim_row['pfx_z']         += noise[i, 4]

        # Propagate speed perturbation to 3D velocity vectors (vx0, vy0, vz0)
        if 'vx0' in sim_row and 'vy0' in sim_row and 'vz0' in sim_row:
            v_mag = np.sqrt(sim_row['vx0']**2 + sim_row['vy0']**2 + sim_row['vz0']**2)
            if v_mag > 0:
                v_new = v_mag + noise[i, 0] * (5280.0 / 3600.0)
                v_ratio = max(v_new / v_mag, 0.1)
                sim_row['vx0'] *= v_ratio
                sim_row['vy0'] *= v_ratio
                sim_row['vz0'] *= v_ratio

        # Propagate pfx movement perturbations to acceleration (ft/s^2)
        # In Statcast / calibrate.py, pfx_x and pfx_z are stored in FEET (e.g. ~0.5 ft).
        # Distance d (ft) = 0.5 * delta_a * t^2  => delta_a = 2 * delta_pfx_ft / t^2
        t_est = 0.40
        y_rel = sim_row.get('release_pos_y')
        if pd.isna(y_rel):
            ext = sim_row.get('release_extension')
            y_rel = 60.5 - ext if pd.notna(ext) else 54.5
            
        if 'vy0' in sim_row and sim_row['vy0'] != 0:
            t_est = abs((y_rel - 1.417) / sim_row['vy0'])
        
        # noise[i, 3] and noise[i, 4] are delta_pfx in feet from calibrate.py
        delta_ax = 2.0 * noise[i, 3] / max(t_est ** 2, 0.01)
        delta_az = 2.0 * noise[i, 4] / max(t_est ** 2, 0.01)
        
        if 'ax' in sim_row: sim_row['ax'] += delta_ax
        if 'az' in sim_row: sim_row['az'] += delta_az

        # Generate trajectory for this variation
        traj_df = calculate_trajectory(sim_row)
        
        if traj_df is not None and len(traj_df) > 1:
            if np.isfinite(traj_df[['x', 'y', 'z', 'vx', 'vy', 'vz']]).all().all():
                # Add an identifier for the sample index
                traj_df['sample_id'] = i
                trajectories.append(traj_df)
            
    return trajectories
