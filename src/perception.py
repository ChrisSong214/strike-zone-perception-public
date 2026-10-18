import numpy as np
import pandas as pd

class VisualNoiseModel:
    """
    Models the human visual system's uncertainty about a moving object.
    
    The core principle is 'Angular Uncertainty':
    - The eye has a fixed angular resolution (e.g. 0.1 degrees).
    - As the object gets closer, the same angular error corresponds to a smaller linear error.
    - Thus, positional uncertainty shrinks linearly with distance.
    """
    
    def __init__(self, angular_std_dev_deg=0.2, base_noise_ft=0.05):
        """
        Args:
            angular_std_dev_deg (float): Visual acuity in degrees. 
                                         0.1 is roughly 20/20 vision dynamic.
            base_noise_ft (float): Minimum processing error (brain noise) even at 0 distance.
        """
        self.angular_std = np.radians(angular_std_dev_deg)
        self.base_noise = base_noise_ft

    def get_uncertainty_at_distance(self, distance_ft):
        """
        Returns the standard deviation of positional error (in feet) at a given distance.
        sigma = distance * tan(theta) + base
        """
        # Distance cannot be negative
        dist = np.maximum(distance_ft, 0)
        
        # Linear spread due to angular resolution
        # For small angles, tan(theta) ~ theta
        visual_sigma = dist * np.tan(self.angular_std)
        
        return visual_sigma + self.base_noise

class PerceptualCloud:
    """
    Generates the 'Perceptual Cloud' (Green Blob) by combining:
    1. PRIOR: The Physical Cloud (Where the ball actually could be based on physics)
    2. LIKELIHOOD: The Eye's observation (Where the batter sees it, with noise)
    
    Result: POSTERIOR (Where the batter *believes* the ball is).
    """
    
    def __init__(self, visual_model=None):
        if visual_model is None:
            self.visual_model = VisualNoiseModel()
        else:
            self.visual_model = visual_model

    def calculate_posterior_cloud(self, physical_trajs, actual_traj):
        """
        Calculates the Perceptual Cloud at each timestep.
        
        Args:
            physical_trajs (list of pd.DataFrame): The Cyan Cloud (Physical variations).
                                                   Used to estimate the PRIOR Covariance.
            actual_traj (pd.DataFrame): The ground truth path (The Likelihood center).
            
        Returns:
            list of pd.DataFrame: Approximated trajectories representing the Perceptual Cloud.
        """
        if actual_traj is None or len(actual_traj) == 0:
            return []
        if not np.isfinite(actual_traj[['x', 'y', 'z']]).all().all():
            return []
            
        # Filter physical trajectories for valid, finite states
        valid_trajs = []
        for traj in physical_trajs:
            if traj is not None and len(traj) > 1:
                if np.isfinite(traj[['x', 'y', 'z']]).all().all():
                    valid_trajs.append(traj)
                    
        if not valid_trajs:
            return []
            
        n_samples = len(valid_trajs)
        new_xs = [[] for _ in range(n_samples)]
        new_ys = [[] for _ in range(n_samples)]
        new_zs = [[] for _ in range(n_samples)]
        
        # Distance steps from actual trajectory (monotonic from release to plate at y=1.417)
        y_steps = actual_traj['y'].values
        x_actual = actual_traj['x'].values
        z_actual = actual_traj['z'].values
        
        # Pre-prepare reversed physical arrays for np.interp (np.interp requires ascending x)
        interp_trajs = []
        for traj in valid_trajs:
            ty = traj['y'].values[::-1]
            tx = traj['x'].values[::-1]
            tz = traj['z'].values[::-1]
            interp_trajs.append((ty, tx, tz))
        
        print("Scoring: Computing Bayesian Perception...")
        
        for i, y_current in enumerate(y_steps):
            dist_to_plate = np.maximum(y_current - 1.417, 0)
            
            # 1. PRIOR (Physical Uncertainty interpolated at the exact y_current plane)
            phys_xs = np.empty(n_samples, dtype=np.float64)
            phys_zs = np.empty(n_samples, dtype=np.float64)
            for idx, (ty, tx, tz) in enumerate(interp_trajs):
                phys_xs[idx] = np.interp(y_current, ty, tx)
                phys_zs[idx] = np.interp(y_current, ty, tz)
            
            mu_prior = np.array([np.mean(phys_xs), np.mean(phys_zs)], dtype=np.float64)
            
            # Prior covariance matrix at this distance plane
            if n_samples >= 2:
                Sigma_prior = np.cov(phys_xs, phys_zs)
                if Sigma_prior.shape != (2, 2) or not np.all(np.isfinite(Sigma_prior)):
                    var_x = max(float(np.var(phys_xs)), 1e-6)
                    var_z = max(float(np.var(phys_zs)), 1e-6)
                    Sigma_prior = np.diag([var_x, var_z])
                else:
                    Sigma_prior = Sigma_prior + np.eye(2) * 1e-6
            else:
                # With a single particle, sample covariance is undefined (N - 1 = 0).
                # Fall back to a default isotropic prior variance.
                Sigma_prior = np.diag([1e-4, 1e-4])
            
            # 2. LIKELIHOOD (Visual Uncertainty)
            sigma_vis = self.visual_model.get_uncertainty_at_distance(dist_to_plate)
            var_vis = max(float(sigma_vis ** 2), 1e-6)
            Sigma_vis = np.eye(2) * var_vis
            
            # 3. POSTERIOR (Bivariate Gaussian Bayesian Fusion)
            # Kalman gain: K = Sigma_prior @ inv(Sigma_prior + Sigma_vis)
            S = Sigma_prior + Sigma_vis
            K = Sigma_prior @ np.linalg.inv(S)
            
            # Posterior mean shifts toward observation (actual ball location):
            actual_pt = np.array([x_actual[i], z_actual[i]], dtype=np.float64)
            mu_post = mu_prior + K @ (actual_pt - mu_prior)
            
            # Scale ensemble along principal axes to match theoretical posterior covariance
            # Since Sigma_vis is isotropic (var_vis * I), Sigma_prior and Sigma_vis share eigenvectors
            eigvals_prior, eigvecs = np.linalg.eigh(Sigma_prior)
            eigvals_prior = np.maximum(eigvals_prior, 1e-8)
            eigvals_post = (eigvals_prior * var_vis) / (eigvals_prior + var_vis)
            scales = np.sqrt(eigvals_post / eigvals_prior)
            M = eigvecs @ np.diag(scales) @ eigvecs.T
            
            # Transform deviations from prior mean
            D_prior = np.vstack([phys_xs - mu_prior[0], phys_zs - mu_prior[1]])
            D_post = M @ D_prior
            
            new_phys_xs = mu_post[0] + D_post[0, :]
            new_phys_zs = mu_post[1] + D_post[1, :]
            
            if not np.all(np.isfinite(new_phys_xs)) or not np.all(np.isfinite(new_phys_zs)):
                new_phys_xs = np.full(n_samples, actual_pt[0])
                new_phys_zs = np.full(n_samples, actual_pt[1])
            
            for k in range(n_samples):
                new_xs[k].append(new_phys_xs[k])
                new_ys[k].append(y_current)
                new_zs[k].append(new_phys_zs[k])
                
        # Reconstruct DataFrames
        final_trajs = []
        for idx in range(n_samples):
            df = pd.DataFrame({
                'x': new_xs[idx],
                'y': new_ys[idx],
                'z': new_zs[idx]
            })
            final_trajs.append(df)
            
        return final_trajs
