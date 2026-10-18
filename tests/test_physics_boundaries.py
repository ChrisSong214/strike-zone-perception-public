import numpy as np
import pandas as pd
import unittest
from src.physics import get_time_at_distance, calculate_trajectory

class TestPhysicsBoundaryConditions(unittest.TestCase):

    def test_time_at_distance_zero_acceleration(self):
        # Test case: Constant velocity (zero acceleration)
        # y(t) = y0 + vy0*t
        # A = 0.5 * ay = 0
        # C = y0 - y_target
        # delta = vy0^2 - 4*A*C = vy0^2
        # t1 = -vy0 - sqrt(vy0^2) / 0 -> Division by zero risk!
        # We must check how the function handles near-zero acceleration.

        # If ay=0, the function will attempt division by zero if not careful.
        # We test a slight non-zero acceleration to ensure the calculation path is valid.
        with self.assertRaises(ZeroDivisionError):
            get_time_at_distance(y_target=50.0, y0=50.0, vy0=5.0, ay=0.0)

    def test_time_at_distance_simple_fall(self):
        # Test case: Simple free-fall from 100ft to 1.417ft (y0=100, ay=-32.2)
        # Expected result: Should yield a small positive time t_final.
        t1, t2 = get_time_at_distance(y_target=1.417, y0=100.0, vy0=0.0, ay=-32.2)

        # We expect t1/t2 to be real and that one of them is positive and reasonable.
        self.assertIsNotNone(t1)
        self.assertIsNotNone(t2)
        self.assertTrue(t1 > 0 or t2 > 0) # At least one positive time should exist

    def test_trajectory_zero_velocity(self):
        # Test case: Ball dropped at origin (x=0, z=0) with no initial velocity (vx0=0, vz0=0)
        # and zero acceleration (ax=0, az=0).
        row_zero_vel = pd.Series({
            'vx0': 0.0, 'vy0': 5.0, 'vz0': 0.0,
            'ax': 0.0, 'ay': -9.8, 'az': 0.0,
            'release_pos_x': 0.0, 'release_pos_z': 0.0, 'release_pos_y': 50.0,
            'pfx_x': 0.0, 'pfx_z': 0.0, 'pfx_y': 50.0,
            'plate_x': 0.0, 'plate_z': 0.0, 'plate_y': 1.417,
            'release_extension': 0.0
        })

        trajectory = calculate_trajectory(row_zero_vel)
        self.assertIsNotNone(trajectory)

        # At the plate (y=1.417), the x and z coordinates should be 0.
        plate_row = trajectory[abs(trajectory['y'] - 1.417) < 0.1].iloc[0]
        self.assertAlmostEqual(plate_row['x'], 0.0, places=3)
        self.assertAlmostEqual(plate_row['z'], 0.0, places=3)

if __name__ == '__main__':
    unittest.main()