"""
Cognitive & Sequential Bayesian Arsenal Fusion (Pillar 1)
=========================================================

Batters do not perceive pitches in isolation. Prior to release (t = 0),
the batter holds a Dirichlet-Multinomial Bayesian repertoire prior conditioned on
pitcher repertoire, count state (b, s), and sequential pitch history H_{n-1}.

During flight, visual evidence updates this prior via sequential multi-hypothesis
filtering. Information-theoretic tunneling deception is quantified by the
Kullback-Leibler divergence surprise between recognition horizon (tau = 0.20)
and decision commit point (tau = 0.40):
    S_tunnel = D_KL( P_{0.40} || P_{0.20} )
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Standard pitch types in MLB Statcast
CANONICAL_PITCH_TYPES = ['FF', 'SI', 'FC', 'SL', 'ST', 'CH', 'CU', 'FS']
PITCH_TYPE_TO_IDX = {pt: i for i, pt in enumerate(CANONICAL_PITCH_TYPES)}


def bayesian_conjugate_update(
    mu_prior: Union[np.ndarray, torch.Tensor],
    sigma_prior: Union[np.ndarray, torch.Tensor],
    mu_vis: Union[np.ndarray, torch.Tensor],
    sigma_vis: Union[np.ndarray, torch.Tensor],
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """
    Computes conjugate 2D Gaussian precision-weighted update:
        Sigma_post = [(Sigma_prior)^-1 + (Sigma_vis)^-1]^-1
        mu_post = Sigma_post * [(Sigma_prior)^-1 * mu_prior + (Sigma_vis)^-1 * mu_vis]

    Guarantees det(Sigma_post) <= min(det(Sigma_prior), det(Sigma_vis)).
    Supports both 2x2 covariance matrices and diagonal (2,) variance vectors.
    """
    is_torch = isinstance(mu_prior, torch.Tensor)

    if is_torch:
        # Convert diagonal vectors to 2x2 if necessary
        if sigma_prior.ndim == 1:
            cov_p = torch.diag(sigma_prior)
        else:
            cov_p = sigma_prior

        if sigma_vis.ndim == 1:
            cov_v = torch.diag(sigma_vis)
        else:
            cov_v = sigma_vis

        inv_p = torch.linalg.inv(cov_p)
        inv_v = torch.linalg.inv(cov_v)
        inv_post = inv_p + inv_v
        cov_post = torch.linalg.inv(inv_post)

        # mu updates
        mu_p_vec = mu_prior.unsqueeze(-1) if mu_prior.ndim == 1 else mu_prior
        mu_v_vec = mu_vis.unsqueeze(-1) if mu_vis.ndim == 1 else mu_vis

        mu_post = cov_post @ (inv_p @ mu_p_vec + inv_v @ mu_v_vec)
        if mu_prior.ndim == 1:
            mu_post = mu_post.squeeze(-1)

        return mu_post, cov_post
    else:
        mu_p = np.asarray(mu_prior, dtype=np.float64)
        mu_v = np.asarray(mu_vis, dtype=np.float64)
        sig_p = np.asarray(sigma_prior, dtype=np.float64)
        sig_v = np.asarray(sigma_vis, dtype=np.float64)

        if sig_p.ndim == 1:
            cov_p = np.diag(sig_p)
        else:
            cov_p = sig_p

        if sig_v.ndim == 1:
            cov_v = np.diag(sig_v)
        else:
            cov_v = sig_v

        inv_p = np.linalg.inv(cov_p)
        inv_v = np.linalg.inv(cov_v)
        inv_post = inv_p + inv_v
        cov_post = np.linalg.inv(inv_post)

        mu_p_vec = mu_p[:, None] if mu_p.ndim == 1 else mu_p
        mu_v_vec = mu_v[:, None] if mu_v.ndim == 1 else mu_v

        mu_post = cov_post @ (inv_p @ mu_p_vec + inv_v @ mu_v_vec)
        if mu_p.ndim == 1:
            mu_post = mu_post.squeeze(-1)

        return mu_post, cov_post


def gaussian_kl_divergence(
    mu_late: Union[np.ndarray, torch.Tensor],
    sigma_late: Union[np.ndarray, torch.Tensor],
    mu_early: Union[np.ndarray, torch.Tensor],
    sigma_early: Union[np.ndarray, torch.Tensor],
) -> Union[float, torch.Tensor]:
    """
    Closed-form Kullback-Leibler divergence between two 2D Gaussian distributions:
        D_KL(P_late || P_early) = 0.5 * [ tr(Sigma_early^-1 * Sigma_late)
                                         + (mu_early - mu_late)^T * Sigma_early^-1 * (mu_early - mu_late)
                                         - 2 + ln(det(Sigma_early) / det(Sigma_late)) ]
    
    Measures information-theoretic tunneling surprise S_tunnel = D_KL(P_0.40 || P_0.20).
    """
    is_torch = isinstance(mu_late, torch.Tensor)

    if is_torch:
        cov_late = torch.diag(sigma_late) if sigma_late.ndim == 1 else sigma_late
        cov_early = torch.diag(sigma_early) if sigma_early.ndim == 1 else sigma_early

        inv_early = torch.linalg.inv(cov_early)
        diff = (mu_early - mu_late).unsqueeze(-1) if mu_early.ndim == 1 else (mu_early - mu_late)

        tr_term = torch.trace(inv_early @ cov_late)
        quad_term = (diff.transpose(-2, -1) @ inv_early @ diff).squeeze()
        
        det_early = torch.linalg.det(cov_early)
        det_late = torch.linalg.det(cov_late)
        log_det_ratio = torch.log(torch.clamp(det_early / torch.clamp(det_late, min=1e-12), min=1e-12))

        kl = 0.5 * (tr_term + quad_term - 2.0 + log_det_ratio)
        return torch.clamp(kl, min=0.0)
    else:
        mu_l = np.asarray(mu_late, dtype=np.float64)
        mu_e = np.asarray(mu_early, dtype=np.float64)
        sig_l = np.asarray(sigma_late, dtype=np.float64)
        sig_e = np.asarray(sigma_early, dtype=np.float64)

        cov_late = np.diag(sig_l) if sig_l.ndim == 1 else sig_l
        cov_early = np.diag(sig_e) if sig_e.ndim == 1 else sig_e

        inv_early = np.linalg.inv(cov_early)
        diff = (mu_e - mu_l)[:, None] if mu_e.ndim == 1 else (mu_e - mu_l)

        tr_term = np.trace(inv_early @ cov_late)
        quad_term = float((diff.T @ inv_early @ diff).squeeze())

        det_early = np.linalg.det(cov_early)
        det_late = np.linalg.det(cov_late)
        log_det_ratio = np.log(max(det_early / max(det_late, 1e-12), 1e-12))

        kl = 0.5 * (tr_term + quad_term - 2.0 + log_det_ratio)
        return float(max(0.0, kl))


def compute_tunnel_amplification_factor(
    r_n_40: np.ndarray,
    r_prev_40: np.ndarray,
    plate_n: np.ndarray,
    plate_prev: np.ndarray,
    min_dist_threshold: float = 0.05,
) -> float:
    """
    Computes Tunnel Amplification Factor (TAF) between consecutive pitches n and n-1:
        TAF_n = ||x_plate,n - x_plate,n-1||_2 / max(||r_n(t_0.40) - r_n-1(t_0.40)||_2, 0.05)
    
    High TAF implies pitch trajectories are nearly indistinguishable at commit horizon
    (tau = 0.40) but diverge dramatically at the plate.
    """
    tunnel_dist_40 = float(np.linalg.norm(r_n_40 - r_prev_40))
    plate_dist = float(np.linalg.norm(plate_n - plate_prev))
    denom = max(tunnel_dist_40, min_dist_threshold)
    return plate_dist / denom


class DirichletMultinomialPrior:
    """
    Dirichlet-Multinomial Bayesian Shrinkage Repertoire Prior at t = 0.
    Conditioned on pitcher p, count state (b, s), and sequential pitch history.
    """

    def __init__(self, alpha_0: float = 20.0, pitch_types: Optional[List[str]] = None):
        self.alpha_0 = alpha_0
        self.pitch_types = pitch_types or CANONICAL_PITCH_TYPES
        self.n_types = len(self.pitch_types)
        self.pt_to_idx = {pt: i for i, pt in enumerate(self.pitch_types)}
        
        # League count-level priors: (b, s) -> array of probabilities
        self.league_count_priors: Dict[Tuple[int, int], np.ndarray] = {}
        # Pitcher count-level observed frequency: (pitcher, b, s) -> array of counts
        self.pitcher_counts: Dict[Tuple[int, int, int], np.ndarray] = {}

    def fit_league(self, df_pitches: pd.DataFrame):
        """Fit empirical baseline frequencies across count states (b, s)."""
        counts = df_pitches.groupby(['balls', 'strikes', 'pitch_type']).size().unstack(fill_value=0)
        # Ensure all pitch types exist
        for pt in self.pitch_types:
            if pt not in counts.columns:
                counts[pt] = 0
        counts = counts[self.pitch_types]

        for (b, s), row in counts.iterrows():
            total = row.sum()
            if total > 0:
                self.league_count_priors[(int(b), int(s))] = (row / total).values.astype(np.float64)
            else:
                self.league_count_priors[(int(b), int(s))] = np.ones(self.n_types) / self.n_types

    def fit_pitchers(self, df_pitches: pd.DataFrame):
        """Fit pitcher-specific usage counts across count states."""
        p_counts = df_pitches.groupby(['pitcher', 'balls', 'strikes', 'pitch_type']).size().unstack(fill_value=0)
        for pt in self.pitch_types:
            if pt not in p_counts.columns:
                p_counts[pt] = 0
        p_counts = p_counts[self.pitch_types]

        for (pitcher, b, s), row in p_counts.iterrows():
            self.pitcher_counts[(int(pitcher), int(b), int(s))] = row.values.astype(np.float64)

    def get_prior(
        self,
        pitcher: int,
        balls: int,
        strikes: int,
        prev_pitch_type: Optional[str] = None,
    ) -> np.ndarray:
        """
        Computes posterior Dirichlet-Multinomial pitch type probabilities:
            alpha_k = alpha_0 * pi_league_k(b, s) + N_pitcher_k(b, s)
            pi_k^(0) = alpha_k / sum(alpha)
        """
        count_key = (balls, strikes)
        base_prior = self.league_count_priors.get(count_key, np.ones(self.n_types) / self.n_types)

        pitcher_obs = self.pitcher_counts.get((pitcher, balls, strikes), np.zeros(self.n_types))
        
        # Dirichlet shrinkage update
        alpha = self.alpha_0 * base_prior + pitcher_obs

        # Optional sequence conditioning
        if prev_pitch_type and prev_pitch_type in self.pt_to_idx:
            # Slight empirical autocorrelation prior (pitchers often repeat fastballs, avoid back-to-back offspeed)
            prev_idx = self.pt_to_idx[prev_pitch_type]
            alpha[prev_idx] += 1.0

        pi_0 = alpha / np.sum(alpha)
        return pi_0


class SequentialArsenalFilter(nn.Module):
    """
    Neural Sequential Multi-Hypothesis Arsenal Filter matching Pillar 1 specification.
    """

    def __init__(self, n_pitch_types: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.n_pitch_types = n_pitch_types
        self.hidden_dim = hidden_dim

        # Input context vector: count (4) + score/runner context (12) + prev_pitch_type (n_pitch_types)
        # Total = 12 + n_pitch_types + 4
        in_dim = 12 + n_pitch_types + 4
        self.prior_net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_pitch_types),
            nn.Softmax(dim=-1),
        )
        # Likelihood head outputs [mux, muz, sigmax, sigmaz, rho] for each pitch type
        self.likelihood_head = nn.Linear(hidden_dim, n_pitch_types * 5)

    def forward(self, h_t: torch.Tensor, context_vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_t: Hidden state tensor from perception LSTM (batch, hidden_dim)
            context_vector: Context vector (batch, 12 + n_pitch_types + 4)
            
        Returns:
            pi_0: Repertoire prior weights (batch, n_pitch_types)
            params: Gaussian component parameters (batch, n_pitch_types, 5)
        """
        pi_0 = self.prior_net(context_vector)
        raw_params = self.likelihood_head(h_t)
        
        batch_size = h_t.shape[0]
        params = raw_params.view(batch_size, self.n_pitch_types, 5)

        # Activations for component parameters
        mu_x = params[..., 0]
        mu_z = params[..., 1]
        sigma_x = torch.exp(params[..., 2])
        sigma_z = torch.exp(params[..., 3])
        rho = torch.tanh(params[..., 4])

        structured_params = torch.stack([mu_x, mu_z, sigma_x, sigma_z, rho], dim=-1)
        return pi_0, structured_params
