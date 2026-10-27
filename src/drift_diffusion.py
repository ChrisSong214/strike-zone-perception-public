"""
Continuous-Time Optimal Stopping & Drift-Diffusion Control (Pillar 2)
=====================================================================

Implements a continuous-time Drift-Diffusion Model (DDM) with Dynamic Collapsing
Boundaries, biomechanically anchored by Statcast bat-tracking metrics:
    bat_speed (mph) and swing_length (ft).

    T_swing = 2 * swing_length / (bat_speed * 1.46667)
    T_deadline = T_plate - T_swing
    B(t) = +/- B0 * max(0, 1 - kappa * (t / T_deadline))^gamma

Evidence accumulates as:
    dE(t) = nu * u(t) dt + sigma_sensory dW_t
where momentary drift u(t) is driven by precision-weighted strike utility.
"""

import math
from typing import Optional, Tuple, Union, Dict, Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

# Conversion factor: 1 mph = 1.46667 ft/s
MPH_TO_FT_PER_SEC = 1.46667
DEFAULT_T_PLATE = 0.42  # ~420 ms typical fastball flight duration


def compute_swing_duration(bat_speed_mph: float, swing_length_ft: float) -> float:
    """
    Computes biomechanical swing duration T_swing from bat tracking:
        T_swing = 2 * swing_length / (bat_speed * 1.46667)
    
    Assumes approximately constant acceleration from swing start to impact.
    Returns duration in seconds.
    """
    v_fps = max(float(bat_speed_mph), 40.0) * MPH_TO_FT_PER_SEC
    length = max(float(swing_length_ft), 3.0)
    t_swing = (2.0 * length) / v_fps
    # Clip to realistic human swing window [0.10, 0.25] s (100 - 250 ms)
    return float(np.clip(t_swing, 0.10, 0.25))


def compute_motor_deadline(t_plate: float, t_swing: float) -> float:
    """
    Computes latest motor commit deadline before which swing must be initiated:
        T_deadline = T_plate - T_swing
    """
    return float(max(0.15, t_plate - t_swing))


class CollapsingBoundDDM:
    """
    Drift-Diffusion Model with Biomechanically Collapsing Boundaries.
    """

    def __init__(self, dt: float = 0.005, sigma_sensory: float = 1.0):
        self.dt = dt
        self.sigma_sensory = sigma_sensory

    def boundary(
        self,
        t: Union[float, np.ndarray],
        b0: float,
        kappa: float,
        t_deadline: float,
        gamma: float = 1.0,
    ) -> Union[float, np.ndarray]:
        """
        Evaluates collapsing decision boundary at time t:
            B(t) = B0 * [ max(0, 1 - kappa * (t / t_deadline)) ]^gamma
            
        Properties:
        - B(0) = B0
        - Monotonically non-increasing for t in [0, t_deadline]
        - For kappa = 1.0, B(t_deadline) = 0.0
        """
        t_arr = np.asarray(t, dtype=np.float64)
        scaled_t = t_arr / max(t_deadline, 1e-4)
        decay = np.maximum(0.0, 1.0 - kappa * scaled_t)
        b_val = b0 * (decay ** gamma)
        return float(b_val) if np.isscalar(t) else b_val

    def simulate_first_passage(
        self,
        v: float,
        b0: float,
        kappa: float,
        t_deadline: float,
        gamma: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[bool, float]:
        """
        Simulates single-trial evidence accumulation via Euler-Maruyama integration.
        
        Args:
            v: Net drift rate (nu * u)
            b0: Initial boundary height (> 0)
            kappa: Collapse rate [0, 1]
            t_deadline: Motor deadline in seconds
            gamma: Collapse curvature
            rng: NumPy random generator
            
        Returns:
            (is_swing, commit_time)
        """
        if rng is None:
            rng = np.random.default_rng()

        n_steps = max(2, int(math.ceil(t_deadline / self.dt)))
        time_steps = np.linspace(0.0, t_deadline, n_steps)
        boundary = self.boundary(time_steps, b0=b0, kappa=kappa, t_deadline=t_deadline, gamma=gamma)

        # Brownian noise increments
        noise = rng.normal(0.0, np.sqrt(self.dt) * self.sigma_sensory, size=n_steps)
        evidence = np.cumsum(v * self.dt + noise)

        swing_cross = np.where(evidence >= boundary)[0]
        take_cross = np.where(evidence <= -boundary)[0]

        first_swing = swing_cross[0] if len(swing_cross) > 0 else np.inf
        first_take = take_cross[0] if len(take_cross) > 0 else np.inf

        if first_swing < first_take:
            return True, float(first_swing * self.dt)
        elif first_take < first_swing:
            return False, float(first_take * self.dt)
        else:
            # Reached deadline without explicit crossing
            final_ev = evidence[-1]
            return bool(final_ev >= 0.0), float(t_deadline)

    def simulate_batch(
        self,
        v: float,
        b0: float,
        kappa: float,
        t_deadline: float,
        n_trials: int = 500,
        gamma: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[float, float]:
        """
        Simulates a batch of trials to estimate P(Swing) and mean commit time.
        
        Returns:
            (p_swing, mean_commit_time)
        """
        if rng is None:
            rng = np.random.default_rng()

        n_steps = max(2, int(math.ceil(t_deadline / self.dt)))
        time_steps = np.linspace(0.0, t_deadline, n_steps)
        boundary = self.boundary(time_steps, b0=b0, kappa=kappa, t_deadline=t_deadline, gamma=gamma)

        # Batch Brownian paths: shape (n_trials, n_steps)
        noise = rng.normal(0.0, np.sqrt(self.dt) * self.sigma_sensory, size=(n_trials, n_steps))
        evidence = np.cumsum(v * self.dt + noise, axis=1)

        # Find crossings
        above = evidence >= boundary[None, :]
        below = evidence <= -boundary[None, :]

        # First crossing index
        has_swing = np.any(above, axis=1)
        has_take = np.any(below, axis=1)

        idx_swing = np.where(has_swing, np.argmax(above, axis=1), n_steps + 1)
        idx_take = np.where(has_take, np.argmax(below, axis=1), n_steps + 1)

        # Swing condition: swing crossing before take, or if neither, positive end evidence
        swing_won = (idx_swing < idx_take) | ((~has_swing) & (~has_take) & (evidence[:, -1] >= 0.0))
        p_swing = float(np.mean(swing_won))

        commit_steps = np.minimum(idx_swing, idx_take)
        commit_steps = np.where(commit_steps <= n_steps, commit_steps, n_steps - 1)
        mean_time = float(np.mean(commit_steps) * self.dt)

        return p_swing, mean_time

    def fit_batter(
        self,
        p_strike_values: np.ndarray,
        swing_labels: np.ndarray,
        bat_speed: float = 70.0,
        swing_length: float = 7.2,
        t_plate: float = DEFAULT_T_PLATE,
    ) -> Dict[str, Any]:
        """
        Fits DDM parameters theta_i = (nu_i, b0_i, kappa_i) for a single batter.
        
        Uses an analytical logistic-diffusion approximation for ultra-fast, robust MLE:
            P(Swing | u) ~= 1 / (1 + exp(-2 * nu * u * B_effective / sigma^2))
        where B_effective incorporates threshold b0 and collapse kappa over the window.
        """
        t_swing = compute_swing_duration(bat_speed, swing_length)
        t_deadline = compute_motor_deadline(t_plate, t_swing)

        # Centered sensory utility u in [-0.5, 0.5]
        u = p_strike_values - 0.50
        y = swing_labels.astype(np.float64)

        def neg_log_lik(params):
            nu, b0, kappa = params
            # Effective boundary integrated across flight window
            b_eff = b0 * (1.0 - 0.5 * kappa)
            # Drift scale
            scaling = 2.0 * nu * b_eff / (self.sigma_sensory ** 2)
            logit_p = scaling * u
            # Add small baseline bias
            p = expit(logit_p)
            p = np.clip(p, 1e-6, 1.0 - 1e-6)
            nll = -np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
            # Regularization towards reasonable cognitive values
            reg = 0.01 * (nu - 5.0)**2 + 0.01 * (b0 - 1.5)**2 + 0.01 * (kappa - 0.5)**2
            return nll + reg

        # Parameter bounds: nu in [0.5, 20.0], b0 in [0.2, 5.0], kappa in [0.0, 1.0]
        init_guess = [5.0, 1.5, 0.5]
        bounds = [(0.5, 20.0), (0.2, 5.0), (0.0, 1.0)]

        res = minimize(neg_log_lik, init_guess, bounds=bounds, method='L-BFGS-B')
        nu_hat, b0_hat, kappa_hat = res.x

        return {
            "drift_rate_nu": float(nu_hat),
            "boundary_b0": float(b0_hat),
            "collapse_rate_kappa": float(kappa_hat),
            "t_swing_sec": float(t_swing),
            "t_deadline_sec": float(t_deadline),
            "converged": bool(res.success),
        }
