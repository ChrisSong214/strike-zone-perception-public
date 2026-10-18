import pandas as pd
import numpy as np
import pytest
from src.physics import calculate_trajectory
from src.uncertainty import generate_pitch_cloud
from src.perception import PerceptualCloud


def test_perceptual_cloud_fusion():
    row = pd.Series({
        'release_speed': 96.0,
        'release_pos_x': -2.5,
        'release_pos_z': 6.0,
        'vx0': 5.0,
        'vy0': -130.0,
        'vz0': -5.0,
        'ax': -15.0,
        'ay': 25.0,
        'az': -15.0,
        'release_extension': 6.0,
        'pfx_x': -10.0,
        'pfx_z': 10.0
    })

    cloud_trajs = generate_pitch_cloud(row, n_samples=10)
    actual_traj = calculate_trajectory(row)

    perc_model = PerceptualCloud()
    perc_trajs = perc_model.calculate_posterior_cloud(cloud_trajs, actual_traj)

    assert len(perc_trajs) == 10
    assert len(perc_trajs[0]) > 0
    # Check that all coordinates are finite (no NaNs)
    assert np.all(np.isfinite(perc_trajs[0]['x']))
    assert np.all(np.isfinite(perc_trajs[0]['y']))
    assert np.all(np.isfinite(perc_trajs[0]['z']))


def test_single_particle_cloud_finite():
    """Verify that a 1-sample cloud doesn't cause NaNs in posterior calculation."""
    row = pd.Series({
        'release_speed': 96.0,
        'release_pos_x': -2.5,
        'release_pos_z': 6.0,
        'vx0': 5.0,
        'vy0': -130.0,
        'vz0': -5.0,
        'ax': -15.0,
        'ay': 25.0,
        'az': -15.0,
        'release_extension': 6.0,
        'pfx_x': -10.0,
        'pfx_z': 10.0
    })

    cloud_trajs = generate_pitch_cloud(row, n_samples=1)
    actual_traj = calculate_trajectory(row)

    perc_model = PerceptualCloud()
    perc_trajs = perc_model.calculate_posterior_cloud(cloud_trajs, actual_traj)

    assert len(perc_trajs) == 1
    assert np.all(np.isfinite(perc_trajs[0]['x']))
    assert np.all(np.isfinite(perc_trajs[0]['z']))
