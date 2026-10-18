import pandas as pd
import numpy as np
import pytest
from src.uncertainty import generate_pitch_cloud


def test_cloud_generation():
    mock_row = pd.Series({
        'release_speed': 95.0,
        'release_pos_x': -2.0,
        'release_pos_z': 6.0,
        'release_extension': 6.0,
        'pfx_x': 5.0,
        'pfx_z': 10.0,
        'vx0': 5.0,
        'vy0': -130.0,
        'vz0': -3.0,
        'ax': 10.0,
        'ay': 25.0,
        'az': -20.0,
    })

    cloud = generate_pitch_cloud(mock_row, n_samples=10)
    assert len(cloud) == 10, f"Expected 10 trajectories, got {len(cloud)}"
    assert cloud[0].shape[0] > 0, "Trajectory is empty"
    assert 'x' in cloud[0].columns and 'z' in cloud[0].columns and 'y' in cloud[0].columns

    # Verify perturbation occurred across samples
    diff = np.abs(cloud[0]['x'].iloc[-1] - cloud[1]['x'].iloc[-1])
    assert diff > 0, "Trajectories should be perturbed and non-identical"


if __name__ == "__main__":
    test_cloud_generation()
    print("test_cloud_generation passed.")
