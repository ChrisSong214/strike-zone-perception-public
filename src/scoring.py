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
                           sz_width=17/12):
    """
    Closed-form probability that the batter PERCEIVES the pitch as a strike.
    
    The batter's belief about pitch location is a 2D Gaussian:
        X ~ N(plate_x, sigma_total_x^2)
        Z ~ N(plate_z, sigma_total_z^2)
    
    where sigma_total = sqrt(sigma_physical^2 + sigma_visual^2)
    
    P(Strike) = P(X in zone) * P(Z in zone)  [independent axes]
    """
    # Total uncertainty = physical + visual (in quadrature)
    total_sigma_x = np.sqrt(phys_sigma_x**2 + visual_sigma_ft**2)
    total_sigma_z = np.sqrt(phys_sigma_z**2 + visual_sigma_ft**2)
    
    # Avoid division by zero
    total_sigma_x = np.maximum(total_sigma_x, 1e-6)
    total_sigma_z = np.maximum(total_sigma_z, 1e-6)
    
    # P(X in zone) = CDF(x_max) - CDF(x_min)
    x_min = -sz_width / 2
    x_max = sz_width / 2
    
    p_x = norm.cdf(x_max, loc=plate_x, scale=total_sigma_x) - \
          norm.cdf(x_min, loc=plate_x, scale=total_sigma_x)
    
    # P(Z in zone) = CDF(sz_top) - CDF(sz_bot)
    p_z = norm.cdf(sz_top, loc=plate_z, scale=total_sigma_z) - \
          norm.cdf(sz_bot, loc=plate_z, scale=total_sigma_z)
    
    return p_x * p_z


def batch_perceptual_strike_prob(df, visual_sigma_ft):
    """
    Vectorized version of perceptual_strike_prob for a DataFrame of pitches.
    Returns array of P(Strike) for each pitch.
    """
    total_sigma_x = np.sqrt(df['phys_sigma_x'].values**2 + visual_sigma_ft**2)
    total_sigma_z = np.sqrt(df['phys_sigma_z'].values**2 + visual_sigma_ft**2)
    
    total_sigma_x = np.maximum(total_sigma_x, 1e-6)
    total_sigma_z = np.maximum(total_sigma_z, 1e-6)
    
    sz_width = 17 / 12
    x_min = -sz_width / 2
    x_max = sz_width / 2
    
    p_x = norm.cdf(x_max, loc=df['plate_x'].values, scale=total_sigma_x) - \
          norm.cdf(x_min, loc=df['plate_x'].values, scale=total_sigma_x)
    
    p_z = norm.cdf(df['sz_top'].values, loc=df['plate_z'].values, scale=total_sigma_z) - \
          norm.cdf(df['sz_bot'].values, loc=df['plate_z'].values, scale=total_sigma_z)
    
    return p_x * p_z


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
        # Drop rows with missing data
        required = ['plate_x', 'plate_z', 'sz_top', 'sz_bot', 
                     'phys_sigma_x', 'phys_sigma_z', 'is_swing']
        self.pitches = df_pitches.dropna(subset=required).copy()
        self.is_swing = self.pitches['is_swing'].values.astype(bool)
        
    def loss_function(self, visual_sigma_ft):
        """
        Negative Log Likelihood of swing decisions given the visual sigma.
        
        Vectorized for speed.
        """
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

    def fit_sigma(self, bounds=(0.01, 2.0)):
        """
        Finds the optimal Visual Sigma (in feet) that explains swing decisions.
        """
        result = minimize_scalar(
            self.loss_function, 
            bounds=bounds, 
            method='bounded'
        )
        return result.x, result.fun
