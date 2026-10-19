"""
Eye Rating Scoring Engine (v2 - Fast, DB-backed)

Uses pre-computed physical pitch metrics from the database instead of 
running physics simulations on-the-fly.

The key insight: we model the batter's perception as a Gaussian centered
on the actual pitch location with variance = physical_variance + visual_variance.
P(Strike) is then a closed-form integral of this Gaussian over the strike zone.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm


def perceptual_strike_prob(plate_x, plate_z, phys_sigma_x, phys_sigma_z,
                           sz_top, sz_bot, visual_sigma_ft,
                           sz_width=17/12, rho=None):
    """
    Closed-form probability that the batter PERCEIVES the pitch as a strike,
    supporting correlated cross-axis bivariate Gaussian integration.
    
    The batter's belief about pitch location is a 2D Gaussian:
        [X, Z] ~ N([plate_x, plate_z], Sigma_total)
    where:
        sigma_total_x = sqrt(phys_sigma_x^2 + visual_sigma_ft^2)
        sigma_total_z = sqrt(phys_sigma_z^2 + visual_sigma_ft^2)
        rho is the cross-axis correlation coefficient (defaulting to 0.0).
    
    Integrates the bivariate Gaussian over the rectangular strike zone:
        [-sz_width/2, sz_width/2] x [sz_bot, sz_top].
    """
    total_sigma_x = np.sqrt(phys_sigma_x**2 + visual_sigma_ft**2)
    total_sigma_z = np.sqrt(phys_sigma_z**2 + visual_sigma_ft**2)
    
    if rho is not None:
        denom = np.maximum(total_sigma_x * total_sigma_z, 1e-12)
        total_rho = np.clip((rho * phys_sigma_x * phys_sigma_z) / denom, -0.999, 0.999)
    else:
        total_rho = None
    
    from src.lstm_scoring import LSTMEyeScorer
    res = LSTMEyeScorer.compute_p_strike(
        mu_x=plate_x, mu_z=plate_z,
        sigma_x=total_sigma_x, sigma_z=total_sigma_z,
        sz_top=sz_top, sz_bot=sz_bot,
        rho=total_rho, sz_width=sz_width
    )
    if isinstance(res, np.ndarray) and res.ndim > 0 and len(res) == 1 and np.isscalar(plate_x):
        return float(res[0])
    return res


def batch_perceptual_strike_prob(df, visual_sigma_ft):
    """
    Vectorized version of perceptual_strike_prob for a DataFrame of pitches.
    Returns array of P(Strike) for each pitch, honoring attenuated cross-axis correlation.
    """
    phys_sx = df['phys_sigma_x'].values
    phys_sz = df['phys_sigma_z'].values
    total_sigma_x = np.sqrt(phys_sx**2 + visual_sigma_ft**2)
    total_sigma_z = np.sqrt(phys_sz**2 + visual_sigma_ft**2)
    
    if 'phys_rho' in df.columns:
        rho_phys = df['phys_rho'].fillna(0.0).values
        denom = np.maximum(total_sigma_x * total_sigma_z, 1e-12)
        total_rho = np.clip((rho_phys * phys_sx * phys_sz) / denom, -0.999, 0.999)
    else:
        total_rho = None
    
    from src.lstm_scoring import LSTMEyeScorer
    sz_width = 17 / 12
    return LSTMEyeScorer.compute_p_strike(
        mu_x=df['plate_x'].values, mu_z=df['plate_z'].values,
        sigma_x=total_sigma_x, sigma_z=total_sigma_z,
        sz_top=df['sz_top'].values, sz_bot=df['sz_bot'].values,
        rho=total_rho, sz_width=sz_width
    )


class FastEyeScorer:
    """
    Grades a batter's vision using pre-computed physical metrics.
    
    Instead of running physics simulations, uses stored sigma_x/sigma_z
    from the backfill and calculates P(Strike) analytically via Gaussian CDF.
    
    This turns a 2-hour job into a 30-second job.
    """
    
    def __init__(self, df_pitches):
        """
        Args:
            df_pitches: DataFrame with columns:
                plate_x, plate_z, sz_top, sz_bot,
                phys_sigma_x, phys_sigma_z, is_swing
        """
        # Drop rows with missing or non-finite data
        required = ['plate_x', 'plate_z', 'sz_top', 'sz_bot', 
                     'phys_sigma_x', 'phys_sigma_z', 'is_swing']
        if not all(c in df_pitches.columns for c in required):
            missing = [c for c in required if c not in df_pitches.columns]
            raise KeyError(f"Missing required columns for FastEyeScorer: {missing}")
            
        valid_mask = df_pitches[required].notna().all(axis=1)
        numeric_cols = ['plate_x', 'plate_z', 'sz_top', 'sz_bot', 'phys_sigma_x', 'phys_sigma_z']
        for col in numeric_cols:
            num = pd.to_numeric(df_pitches[col], errors='coerce')
            valid_mask &= np.isfinite(num)
        self.pitches = df_pitches[valid_mask].copy()
        self.is_swing = self.pitches['is_swing'].values.astype(bool)
        
    def loss_function(self, visual_sigma_ft):
        """
        Negative Log Likelihood of swing decisions given the visual sigma.
        
        Vectorized for speed.
        """
        if len(self.pitches) == 0:
            return 0.0
            
        p_strike = batch_perceptual_strike_prob(self.pitches, visual_sigma_ft)
        
        # Clip for numerical stability
        p_strike = np.clip(p_strike, 0.01, 0.99)
        
        # Log Loss
        # swung => loss -= log(p_strike)      [expected to swing at strikes]
        # took  => loss -= log(1 - p_strike)  [expected to take on balls]
        log_loss = -np.sum(
            self.is_swing * np.log(p_strike) + 
            (~self.is_swing) * np.log(1 - p_strike)
        )
        
        return log_loss

    def fit_sigma(self, bounds=(0.01, 2.0), min_obs=10):
        """
        Finds the optimal Visual Sigma (in feet) that explains swing decisions.
        Raises ValueError if fewer than min_obs usable observations are present.
        """
        if len(self.pitches) < min_obs:
            raise ValueError(
                f"Cannot fit visual sigma with fewer than {min_obs} usable observations "
                f"(got {len(self.pitches)})"
            )
        result = minimize_scalar(
            self.loss_function, 
            bounds=bounds, 
            method='bounded'
        )
        return result.x, result.fun
