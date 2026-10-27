"""
Physics-Informed Retinal Optics & Psychophysics (Pillar 4)
==========================================================

Biological batters do not perceive 3D Cartesian coordinates (x, y, z) or constant
accelerations (ax, ay, az). They receive optical projections on the spherical retina:
azimuth (θ), elevation (φ), subtended visual angle (ψ), looming rate (tau-dot),
and angular velocities, corrupted by Weber-Fechner sensory noise.

This module provides coordinate transformations from Cartesian 3D trajectories
to 9D Retinal Optic Flow tensors without leaking full-trajectory Magnus acceleration.
"""

import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

# Physical constants
BASEBALL_RADIUS_FT = 0.1208  # ~1.45 inches in feet (diameter = 2 * 0.1208 = 0.2416 ft)
DEFAULT_DT = 0.005  # Statcast trajectory temporal discretization (5 ms)

# Weber-Fechner sensory noise defaults
DEFAULT_SIGMA_0_ANG = 0.01    # Baseline angular velocity noise (rad/s)
DEFAULT_WEBER_ANG = 0.05      # Weber fraction for angular motion (5%)
DEFAULT_SIGMA_0_LOOM = 0.02   # Baseline looming rate noise (1/s)
DEFAULT_WEBER_LOOM = 0.08     # Weber fraction for looming expansion (8%)


def get_eye_position(
    stand: Union[str, torch.Tensor, np.ndarray],
    sz_top: Union[float, torch.Tensor, np.ndarray],
    eye_x_right: float = -1.75,
    eye_x_left: float = 1.75,
    eye_y: float = 1.417,
    eye_z_offset: float = 0.65,
) -> Tuple[Union[float, torch.Tensor, np.ndarray], float, Union[float, torch.Tensor, np.ndarray]]:
    """
    Computes 3D batter eye position (p_eye) in Statcast coordinate frame.
    
    Args:
        stand: Batter handedness ('R' or 'L', or tensor/array of 'R'/'L' or boolean is_right).
        sz_top: Top of batter's strike zone (ft).
        eye_x_right: Lateral eye position for right-handed batter (standing in LH box, x = -1.75).
        eye_x_left: Lateral eye position for left-handed batter (standing in RH box, x = +1.75).
        eye_y: Longitudinal eye position relative to back tip of home plate (ft, 1.417 ft).
        eye_z_offset: Vertical offset of eyes above top of strike zone (ft, default 0.65 ft).
        
    Returns:
        (eye_x, eye_y, eye_z)
    """
    if isinstance(stand, torch.Tensor):
        if stand.dtype == torch.bool:
            eye_x = torch.where(stand, eye_x_right, eye_x_left)
        elif stand.dtype in (torch.int32, torch.int64):
            eye_x = torch.where(stand == 1, eye_x_right, eye_x_left)
        else:
            # Assumed string or char tensor if supported, or converted
            eye_x = torch.where(stand == 'R', eye_x_right, eye_x_left)
        eye_z = sz_top + eye_z_offset
        return eye_x, eye_y, eye_z
    elif isinstance(stand, np.ndarray):
        if stand.dtype == bool:
            eye_x = np.where(stand, eye_x_right, eye_x_left)
        elif np.issubdtype(stand.dtype, np.integer):
            eye_x = np.where(stand == 1, eye_x_right, eye_x_left)
        else:
            eye_x = np.where(stand == 'R', eye_x_right, eye_x_left)
        eye_z = sz_top + eye_z_offset
        return eye_x, eye_y, eye_z
    else:
        # Scalar python
        is_r = (stand == 'R' or stand is True or stand == 1)
        eye_x = eye_x_right if is_r else eye_x_left
        eye_z = float(sz_top) + eye_z_offset
        return eye_x, eye_y, eye_z


def cartesian_to_retinal_tensor(
    traj_xyz_v: torch.Tensor,
    stand: Union[str, torch.Tensor],
    sz_top: Union[float, torch.Tensor],
    dt: float = DEFAULT_DT,
    sigma_0_ang: float = DEFAULT_SIGMA_0_ANG,
    w_weber: float = DEFAULT_WEBER_ANG,
    sigma_0_loom: float = DEFAULT_SIGMA_0_LOOM,
    w_loom: float = DEFAULT_WEBER_LOOM,
    use_analytical_rates: bool = True,
) -> torch.Tensor:
    """
    Transforms Cartesian trajectory [x, y, z, vx, vy, vz, ...] into 9D Retinal Optic Flow.
    
    The 9 output channels are:
    [θ, φ, ψ, θ_dot, φ_dot, ψ_dot/ψ, σ_θ_dot, σ_φ_dot, σ_loom]
    
    Args:
        traj_xyz_v: Tensor of shape (batch, seq_len, >=6) or (seq_len, >=6).
        stand: Batter stand ('R' or 'L' or Tensor).
        sz_top: Strike zone top (ft).
        dt: Time step between trajectory points (default 0.005s).
        use_analytical_rates: If True, uses instantaneous kinematic projection
                              for angular rates (prevents finite-difference border artifacts
                              and guarantees zero leakage of acceleration at t=0).
                              
    Returns:
        retinal_tensor: Tensor of shape (..., seq_len, 9) containing retinal optics.
    """
    has_batch = traj_xyz_v.ndim == 3
    if not has_batch:
        traj_xyz_v = traj_xyz_v.unsqueeze(0)
    
    batch_size, seq_len, _ = traj_xyz_v.shape
    device = traj_xyz_v.device
    dtype = traj_xyz_v.dtype

    # Eye position
    if not isinstance(stand, torch.Tensor):
        is_r = (stand == 'R' or stand is True or stand == 1)
        eye_x_val = -1.75 if is_r else 1.75
        eye_x = torch.full((batch_size, 1), eye_x_val, device=device, dtype=dtype)
    else:
        stand_t = stand.to(device)
        if stand_t.ndim == 0:
            stand_t = stand_t.expand(batch_size)
        if stand_t.dtype == torch.bool:
            eye_x = torch.where(stand_t, -1.75, 1.75).view(batch_size, 1).to(dtype)
        elif stand_t.dtype in (torch.int32, torch.int64):
            eye_x = torch.where(stand_t == 1, -1.75, 1.75).view(batch_size, 1).to(dtype)
        else:
            eye_x = torch.where(stand_t == 'R', -1.75, 1.75).view(batch_size, 1).to(dtype)

    eye_y = 1.417
    if not isinstance(sz_top, torch.Tensor):
        eye_z = torch.full((batch_size, 1), float(sz_top) + 0.65, device=device, dtype=dtype)
    else:
        sz_top_t = sz_top.to(device, dtype=dtype)
        if sz_top_t.ndim == 0:
            sz_top_t = sz_top_t.expand(batch_size)
        eye_z = (sz_top_t + 0.65).view(batch_size, 1)

    # Displacements
    x = traj_xyz_v[..., 0]
    y = traj_xyz_v[..., 1]
    z = traj_xyz_v[..., 2]
    vx = traj_xyz_v[..., 3]
    vy = traj_xyz_v[..., 4]
    vz = traj_xyz_v[..., 5]

    dx = x - eye_x
    dy = y - eye_y
    dz = z - eye_z

    d_xy_sq = dx**2 + dy**2
    d_xy = torch.sqrt(torch.clamp(d_xy_sq, min=1e-8))
    dist_sq = d_xy_sq + dz**2
    dist = torch.sqrt(torch.clamp(dist_sq, min=1e-8))

    # Spherical Angles
    theta = torch.atan2(dx, dy)
    phi = torch.atan2(dz, d_xy)
    psi = 2.0 * BASEBALL_RADIUS_FT / torch.clamp(dist, min=1.0)

    # Looming rate (tau-dot: psi_dot / psi = - (dr . v) / dist^2)
    dr_dot_v = dx * vx + dy * vy + dz * vz
    loom_rate = -dr_dot_v / torch.clamp(dist_sq, min=1e-6)

    # Angular velocities
    if use_analytical_rates:
        # Analytical projection of relative velocity into angular rates
        theta_dot = (vx * dy - vy * dx) / torch.clamp(d_xy_sq, min=1e-6)
        phi_dot = (vz * d_xy_sq - dz * (dx * vx + dy * vy)) / (torch.clamp(dist_sq, min=1e-6) * d_xy)
    else:
        # Finite-difference across sequence axis
        theta_dot = torch.gradient(theta, spacing=dt, dim=1)[0]
        phi_dot = torch.gradient(phi, spacing=dt, dim=1)[0]

    # Weber-Fechner sensory uncertainty
    sigma_theta_dot = torch.sqrt(sigma_0_ang**2 + (w_weber**2) * (theta_dot**2))
    sigma_phi_dot = torch.sqrt(sigma_0_ang**2 + (w_weber**2) * (phi_dot**2))
    sigma_loom = torch.sqrt(sigma_0_loom**2 + (w_loom**2) * (loom_rate**2))

    retinal_9d = torch.stack([
        theta,
        phi,
        psi,
        theta_dot,
        phi_dot,
        loom_rate,
        sigma_theta_dot,
        sigma_phi_dot,
        sigma_loom
    ], dim=-1)

    if not has_batch:
        retinal_9d = retinal_9d.squeeze(0)

    return retinal_9d


def cartesian_to_retinal_numpy(
    traj_xyz_v: np.ndarray,
    stand: Union[str, np.ndarray],
    sz_top: Union[float, np.ndarray],
    dt: float = DEFAULT_DT,
    sigma_0_ang: float = DEFAULT_SIGMA_0_ANG,
    w_weber: float = DEFAULT_WEBER_ANG,
    sigma_0_loom: float = DEFAULT_SIGMA_0_LOOM,
    w_loom: float = DEFAULT_WEBER_LOOM,
    use_analytical_rates: bool = True,
) -> np.ndarray:
    """
    NumPy implementation of Cartesian to 9D Retinal Optic Flow.
    """
    has_batch = traj_xyz_v.ndim == 3
    if not has_batch:
        traj_xyz_v = np.expand_dims(traj_xyz_v, axis=0)

    batch_size, seq_len, _ = traj_xyz_v.shape

    if isinstance(stand, (str, bool, int, np.bool_)):
        is_r = (stand == 'R' or stand is True or stand == 1)
        eye_x = np.full((batch_size, 1), -1.75 if is_r else 1.75, dtype=np.float32)
    else:
        stand_arr = np.asarray(stand)
        if stand_arr.ndim == 0:
            stand_arr = np.repeat(stand_arr, batch_size)
        if stand_arr.dtype == bool:
            eye_x = np.where(stand_arr, -1.75, 1.75)[:, None].astype(np.float32)
        else:
            eye_x = np.where(stand_arr == 'R', -1.75, 1.75)[:, None].astype(np.float32)

    eye_y = 1.417
    if np.isscalar(sz_top):
        eye_z = np.full((batch_size, 1), float(sz_top) + 0.65, dtype=np.float32)
    else:
        sz_top_arr = np.asarray(sz_top)
        if sz_top_arr.ndim == 0:
            sz_top_arr = np.repeat(sz_top_arr, batch_size)
        eye_z = (sz_top_arr + 0.65)[:, None].astype(np.float32)

    x = traj_xyz_v[..., 0]
    y = traj_xyz_v[..., 1]
    z = traj_xyz_v[..., 2]
    vx = traj_xyz_v[..., 3]
    vy = traj_xyz_v[..., 4]
    vz = traj_xyz_v[..., 5]

    dx = x - eye_x
    dy = y - eye_y
    dz = z - eye_z

    d_xy_sq = dx**2 + dy**2
    d_xy = np.sqrt(np.maximum(d_xy_sq, 1e-8))
    dist_sq = d_xy_sq + dz**2
    dist = np.sqrt(np.maximum(dist_sq, 1e-8))

    theta = np.arctan2(dx, dy)
    phi = np.arctan2(dz, d_xy)
    psi = 2.0 * BASEBALL_RADIUS_FT / np.maximum(dist, 1.0)

    dr_dot_v = dx * vx + dy * vy + dz * vz
    loom_rate = -dr_dot_v / np.maximum(dist_sq, 1e-6)

    if use_analytical_rates:
        theta_dot = (vx * dy - vy * dx) / np.maximum(d_xy_sq, 1e-6)
        phi_dot = (vz * d_xy_sq - dz * (dx * vx + dy * vy)) / (np.maximum(dist_sq, 1e-6) * d_xy)
    else:
        theta_dot = np.gradient(theta, dt, axis=1)
        phi_dot = np.gradient(phi, dt, axis=1)

    sigma_theta_dot = np.sqrt(sigma_0_ang**2 + (w_weber**2) * (theta_dot**2))
    sigma_phi_dot = np.sqrt(sigma_0_ang**2 + (w_weber**2) * (phi_dot**2))
    sigma_loom = np.sqrt(sigma_0_loom**2 + (w_loom**2) * (loom_rate**2))

    retinal_9d = np.stack([
        theta,
        phi,
        psi,
        theta_dot,
        phi_dot,
        loom_rate,
        sigma_theta_dot,
        sigma_phi_dot,
        sigma_loom
    ], axis=-1)

    if not has_batch:
        retinal_9d = np.squeeze(retinal_9d, axis=0)

    return retinal_9d


def inject_weber_noise(
    retinal_tensor: torch.Tensor,
    rng: Optional[torch.Generator] = None,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Injects zero-mean Gaussian observation noise with standard deviations
    given by the Weber-Fechner noise channels (indices 6, 7, 8).
    
    Returns a copy of retinal_tensor where (theta_dot, phi_dot, loom_rate)
    are corrupted by psychophysical sensory noise.
    """
    noisy = retinal_tensor.clone()
    # Channels: 3=theta_dot, 4=phi_dot, 5=loom_rate
    # Sigmas:   6=sigma_theta_dot, 7=sigma_phi_dot, 8=sigma_loom
    noise_theta = torch.randn(noisy[..., 3].shape, generator=rng, device=noisy.device, dtype=noisy.dtype)
    noise_phi = torch.randn(noisy[..., 4].shape, generator=rng, device=noisy.device, dtype=noisy.dtype)
    noise_loom = torch.randn(noisy[..., 5].shape, generator=rng, device=noisy.device, dtype=noisy.dtype)

    noisy[..., 3] = noisy[..., 3] + scale * noise_theta * noisy[..., 6]
    noisy[..., 4] = noisy[..., 4] + scale * noise_phi * noisy[..., 7]
    noisy[..., 5] = noisy[..., 5] + scale * noise_loom * noisy[..., 8]

    return noisy
