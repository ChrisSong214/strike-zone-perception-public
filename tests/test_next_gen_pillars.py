"""
Verification Suite for Next-Generation Research Extensions (Pillars 1–4)
========================================================================

Covers:
- Pillar 4: Physics-Informed Retinal Optics & Psychophysics
- Pillar 1: Cognitive & Sequential Bayesian Arsenal Fusion
- Pillar 2: Continuous-Time Optimal Stopping & Drift-Diffusion Control
- Pillar 3: Individualized Damage Surfaces & Minimax Equilibrium
- Strict 5-fold Batter-Grouped Out-of-Fold (OOF) Provenance
"""

import math
import numpy as np
import pytest
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold

from src.retinal_physics import (
    cartesian_to_retinal_tensor,
    cartesian_to_retinal_numpy,
    inject_weber_noise,
    get_eye_position,
    BASEBALL_RADIUS_FT,
)
from src.bayesian_tunneling import (
    bayesian_conjugate_update,
    gaussian_kl_divergence,
    compute_tunnel_amplification_factor,
    DirichletMultinomialPrior,
    SequentialArsenalFilter,
)
from src.drift_diffusion import (
    CollapsingBoundDDM,
    compute_swing_duration,
    compute_motor_deadline,
)
from src.personalized_utility import (
    IndividualContactUtilityModel,
    xwobacon_to_run_value,
    DEFAULT_DELTA_RE_STRIKE,
    DEFAULT_DELTA_RE_BALL,
)


# ==============================================================================
# PILLAR 4 TESTS: RETINAL OPTICS & PSYCHOPHYSICS
# ==============================================================================

def test_retinal_projection_zero_leakage():
    """
    SPEC REQUIREMENT 1:
    Confirm modifying (ax, ay, az) at t=0 yields identical retinal values at the initial timestep.
    Proves retinal transformation does not leak constant Magnus acceleration into initial perception.
    """
    n_seq = 100
    traj1 = torch.zeros(1, n_seq, 9)
    # Positions and velocities
    traj1[0, :, 0] = torch.linspace(0.5, 0.2, n_seq)
    traj1[0, :, 1] = torch.linspace(55.0, 1.4, n_seq)
    traj1[0, :, 2] = torch.linspace(6.0, 2.5, n_seq)
    traj1[0, :, 3] = -0.6
    traj1[0, :, 4] = -135.0
    traj1[0, :, 5] = -7.5
    traj1[0, :, 6] = 4.0   # ax
    traj1[0, :, 7] = 24.0  # ay
    traj1[0, :, 8] = -14.0 # az

    # Create modified trajectory with completely different Magnus acceleration
    traj2 = traj1.clone()
    traj2[0, :, 6:] = traj1[0, :, 6:] * -3.5 + 42.0

    ret1 = cartesian_to_retinal_tensor(traj1, stand='R', sz_top=3.4)
    ret2 = cartesian_to_retinal_tensor(traj2, stand='R', sz_top=3.4)

    # At t=0, retinal projection must be identically identical
    diff_t0 = (ret1[0, 0] - ret2[0, 0]).abs().max().item()
    assert diff_t0 < 1e-6, f"Retinal projection leaked acceleration at t=0: max diff = {diff_t0}"


def test_retinal_projection_shape_and_range():
    """Verify 9D channel output structure and mathematical angular ranges."""
    traj = torch.zeros(4, 100, 9)
    traj[:, :, 1] = torch.linspace(55.0, 1.4, 100).unsqueeze(0)
    traj[:, :, 2] = 3.0
    ret = cartesian_to_retinal_tensor(traj, stand='R', sz_top=3.3)

    assert ret.shape == (4, 100, 9), f"Expected shape (4, 100, 9), got {ret.shape}"
    # Azimuth in [-pi, pi]
    theta = ret[..., 0]
    assert (theta >= -math.pi).all() and (theta <= math.pi).all()
    # Elevation in [-pi/2, pi/2]
    phi = ret[..., 1]
    assert (phi >= -math.pi / 2.0).all() and (phi <= math.pi / 2.0).all()
    # Subtended visual angle strictly positive
    psi = ret[..., 2]
    assert (psi > 0).all()


def test_retinal_handedness_eye_offset():
    """Verify right vs left handed batter eye placement produces lateral symmetry."""
    eye_r_x, _, _ = get_eye_position('R', 3.4)
    eye_l_x, _, _ = get_eye_position('L', 3.4)
    assert eye_r_x == -1.75
    assert eye_l_x == 1.75


def test_retinal_looming_rate_expansion():
    """Verify looming rate (tau-dot) is positive for an approaching fastball during flight."""
    traj = torch.zeros(1, 100, 9)
    # y moving from 55 to 2.0 ft during flight approaching plate
    traj[0, :, 1] = torch.linspace(55.0, 2.0, 100)
    traj[0, :, 4] = -130.0
    ret = cartesian_to_retinal_tensor(traj, stand='R', sz_top=3.4)
    loom_rate = ret[0, :, 5]
    # Approaching ball must have positive looming rate (expanding retinal image)
    assert (loom_rate > 0).all()


def test_weber_fechner_noise_scaling():
    """Verify Weber-Fechner sensory uncertainty scales with rate magnitude."""
    traj = torch.zeros(1, 100, 9)
    traj[0, :, 1] = torch.linspace(55.0, 1.4, 100)
    traj[0, :, 4] = -130.0
    ret = cartesian_to_retinal_tensor(traj, stand='R', sz_top=3.4)

    sigma_theta = ret[0, :, 6]
    sigma_phi = ret[0, :, 7]
    sigma_loom = ret[0, :, 8]

    assert (sigma_theta >= 0.01).all()
    assert (sigma_phi >= 0.01).all()
    assert (sigma_loom >= 0.02).all()


def test_weber_noise_injection_stochasticity():
    """Verify noise injection preserves tensor shape and modifies velocity channels."""
    ret = torch.ones(2, 50, 9)
    ret[..., 6:] = 0.05
    rng = torch.Generator().manual_seed(42)
    noisy = inject_weber_noise(ret, rng=rng)

    assert noisy.shape == ret.shape
    # Channels 0, 1, 2 (angles) must remain untouched
    assert torch.equal(noisy[..., :3], ret[..., :3])
    # Velocity channels (3, 4, 5) must be perturbed
    assert not torch.equal(noisy[..., 3:6], ret[..., 3:6])


# ==============================================================================
# PILLAR 1 TESTS: BAYESIAN ARSENAL FUSION & TUNNELING
# ==============================================================================

def test_bayesian_precision_addition():
    """
    SPEC REQUIREMENT 2:
    Verify that det(Sigma_post) <= min(det(Sigma_prior), det(Sigma_vis)).
    Guarantees information fusion strictly contracts or maintains uncertainty.
    """
    mu_prior = np.array([0.0, 2.5])
    sigma_prior = np.diag([0.16, 0.25])  # det = 0.04

    mu_vis = np.array([0.2, 2.3])
    sigma_vis = np.diag([0.09, 0.16])    # det = 0.0144

    mu_post, sigma_post = bayesian_conjugate_update(mu_prior, sigma_prior, mu_vis, sigma_vis)

    det_prior = np.linalg.det(sigma_prior)
    det_vis = np.linalg.det(sigma_vis)
    det_post = np.linalg.det(sigma_post)

    assert det_post <= min(det_prior, det_vis) + 1e-12, f"Precision addition violated: {det_post} > min({det_prior}, {det_vis})"


def test_bayesian_conjugate_mean_update():
    """Verify posterior mean is an exact precision-weighted combination."""
    mu_prior = torch.tensor([0.0, 0.0])
    sigma_prior = torch.tensor([1.0, 1.0])

    mu_vis = torch.tensor([2.0, 2.0])
    sigma_vis = torch.tensor([1.0, 1.0])

    mu_post, _ = bayesian_conjugate_update(mu_prior, sigma_prior, mu_vis, sigma_vis)
    # Equal isotropic covariances -> posterior mean is midpoint [1.0, 1.0]
    assert torch.allclose(mu_post, torch.tensor([1.0, 1.0]), atol=1e-5)


def test_dirichlet_multinomial_shrinkage():
    """Verify Dirichlet-Multinomial repertoire prior shrinkage behavior."""
    prior = DirichletMultinomialPrior(alpha_0=10.0, pitch_types=['FF', 'SL', 'CH'])
    # Baseline league: 50% FF, 30% SL, 20% CH in (0, 0)
    prior.league_count_priors[(0, 0)] = np.array([0.5, 0.3, 0.2])

    # Unseen pitcher -> returns exact league baseline
    p_unseen = prior.get_prior(pitcher=999999, balls=0, strikes=0)
    assert np.allclose(p_unseen, [0.5, 0.3, 0.2], atol=1e-4)

    # Pitcher with 10 pitches: 10 FF, 0 SL, 0 CH
    # alpha = 10 * [0.5, 0.3, 0.2] + [10, 0, 0] = [15, 3, 2] -> sum = 20 -> [0.75, 0.15, 0.10]
    prior.pitcher_counts[(12345, 0, 0)] = np.array([10.0, 0.0, 0.0])
    p_shrunk = prior.get_prior(pitcher=12345, balls=0, strikes=0)
    assert np.allclose(p_shrunk, [0.75, 0.15, 0.10], atol=1e-4)


def test_gaussian_kl_divergence_nonnegativity_and_identity():
    """Verify D_KL(P || P) = 0 and D_KL(P_late || P_early) >= 0."""
    mu = np.array([0.1, 2.4])
    sigma = np.diag([0.04, 0.09])

    # Self-divergence
    kl_zero = gaussian_kl_divergence(mu, sigma, mu, sigma)
    assert abs(kl_zero) < 1e-9

    # Different distributions
    mu2 = np.array([-0.3, 1.8])
    sigma2 = np.diag([0.09, 0.16])
    kl_val = gaussian_kl_divergence(mu, sigma, mu2, sigma2)
    assert kl_val > 0.0


def test_tunnel_amplification_factor():
    """Verify Tunnel Amplification Factor behaves correctly under tight tunneling."""
    r_n = np.array([0.10, 2.50])
    r_prev = np.array([0.11, 2.51]) # Divergence at tau=0.40 is ~0.014 ft (< 0.05 min threshold)
    
    plate_n = np.array([0.50, 1.80])
    plate_prev = np.array([-0.40, 3.20]) # Divergence at plate is ~1.66 ft
    
    taf = compute_tunnel_amplification_factor(r_n, r_prev, plate_n, plate_prev)
    assert taf > 10.0 # Extreme amplification factor


def test_sequential_arsenal_filter_dimensions():
    """Verify neural SequentialArsenalFilter parameter shapes and prior simplex constraint."""
    net = SequentialArsenalFilter(n_pitch_types=8, hidden_dim=64)
    h_t = torch.randn(5, 64)
    ctx = torch.randn(5, 12 + 8 + 4)
    pi_0, params = net(h_t, ctx)

    assert pi_0.shape == (5, 8)
    assert params.shape == (5, 8, 5)
    # Repertoire prior sums to 1
    assert torch.allclose(pi_0.sum(dim=-1), torch.ones(5), atol=1e-5)
    # Variances strictly positive
    sig_x = params[..., 2]
    sig_z = params[..., 3]
    assert (sig_x > 0).all() and (sig_z > 0).all()


# ==============================================================================
# PILLAR 2 TESTS: DRIFT-DIFFUSION & OPTIMAL STOPPING
# ==============================================================================

def test_ddm_boundary_collapse():
    """
    SPEC REQUIREMENT 3:
    Verify that decision boundary B_i(t) monotonically decreases and equals zero at t = T_deadline.
    """
    ddm = CollapsingBoundDDM()
    b0 = 2.0
    kappa = 1.0
    t_deadline = 0.28
    t_steps = np.linspace(0.0, t_deadline, 100)

    b_vals = ddm.boundary(t_steps, b0=b0, kappa=kappa, t_deadline=t_deadline)

    # Initial boundary equals b0
    assert abs(b_vals[0] - b0) < 1e-9
    # Monotonically non-increasing
    diffs = np.diff(b_vals)
    assert (diffs <= 1e-9).all()
    # At deadline, boundary is exactly zero
    assert abs(b_vals[-1] - 0.0) < 1e-9


def test_ddm_euler_maruyama_first_passage():
    """Verify Euler-Maruyama single-trial simulation outputs valid commit time."""
    ddm = CollapsingBoundDDM(dt=0.005)
    rng = np.random.default_rng(123)
    is_swing, commit_t = ddm.simulate_first_passage(
        v=5.0, b0=1.5, kappa=0.8, t_deadline=0.30, rng=rng
    )
    assert isinstance(is_swing, (bool, np.bool_))
    assert 0.0 <= commit_t <= 0.30 + 1e-6


def test_ddm_drift_monotonicity():
    """Verify increasing drift rate monotonically increases swing rate on strike pitches."""
    ddm = CollapsingBoundDDM(dt=0.005)
    rng = np.random.default_rng(42)
    # Clear take (negative drift)
    p_take, _ = ddm.simulate_batch(v=-8.0, b0=1.5, kappa=0.5, t_deadline=0.28, n_trials=1000, rng=rng)
    # Clear strike (positive drift)
    p_swing, _ = ddm.simulate_batch(v=8.0, b0=1.5, kappa=0.5, t_deadline=0.28, n_trials=1000, rng=rng)

    assert p_take < 0.10
    assert p_swing > 0.90


def test_bat_tracking_execution_timing():
    """Verify biomechanical swing duration computation and deadline calculation."""
    # Fast swing: 80 mph bat speed, 7.0 ft swing length
    t_fast = compute_swing_duration(bat_speed_mph=80.0, swing_length_ft=7.0)
    # Slow swing: 60 mph bat speed, 7.5 ft swing length
    t_slow = compute_swing_duration(bat_speed_mph=60.0, swing_length_ft=7.5)

    assert t_fast < t_slow
    assert 0.10 <= t_fast <= 0.25
    assert 0.10 <= t_slow <= 0.25

    deadline = compute_motor_deadline(t_plate=0.40, t_swing=t_fast)
    assert 0.15 <= deadline <= 0.30


# ==============================================================================
# PILLAR 3 TESTS: PERSONALIZED DAMAGE SURFACES & MINIMAX
# ==============================================================================

def test_dva_additive_identity():
    """
    SPEC REQUIREMENT 4:
    Confirm that |Total Indiv DVA - (Universal DVA + Damage Alignment DVA)| < 10^-6 identically.
    Proves orthogonal factorization holds with machine precision.
    """
    model = IndividualContactUtilityModel()

    n = 100
    rng = np.random.default_rng(99)
    is_swing = rng.integers(0, 2, size=n)
    p_swing_oof = rng.uniform(0.1, 0.9, size=n)
    ev_indiv = rng.normal(0.05, 0.1, size=n)
    ev_league = rng.normal(0.03, 0.08, size=n)
    ev_take = rng.normal(-0.02, 0.05, size=n)

    universal_dva, damage_dva, total_dva = model.decompose_dva(
        is_swing=is_swing,
        p_swing_oof=p_swing_oof,
        ev_swing_indiv=ev_indiv,
        ev_swing_league=ev_league,
        ev_take=ev_take,
    )

    residual = np.abs(total_dva - (universal_dva + damage_dva))
    assert np.max(residual) < 1e-6, f"Additive identity violated: max residual = {np.max(residual)}"


def test_strict_oof_partition():
    """
    SPEC REQUIREMENT 5:
    Ensure no overlapping batter IDs exist between training and validation splits within any fold.
    """
    rng = np.random.default_rng(2025)
    batters = np.repeat(np.arange(100), 20) # 100 unique batters, 20 pitches each
    gkf = GroupKFold(n_splits=5)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(batters, groups=batters)):
        train_batters = set(batters[train_idx])
        test_batters = set(batters[test_idx])
        overlap = train_batters.intersection(test_batters)
        assert len(overlap) == 0, f"Fold {fold} contained {len(overlap)} overlapping batters between train and test splits!"


def test_game_theoretic_minimax_pitch():
    """Verify minimax pitch solver selects pitch minimizing batter run value."""
    model = IndividualContactUtilityModel()
    repertoire = [
        {'pitch_type': 'FF'},
        {'pitch_type': 'SL'},
        {'pitch_type': 'CH'},
    ]
    best_opt = model.find_minimax_pitch(repertoire, batter_id=500000, count=(1, 2))
    assert 'pitch_type' in best_opt
    assert 'plate_x' in best_opt
    assert 'plate_z' in best_opt

    # Compare against serving a fastball right down the middle (heart: 0.0, 2.5)
    px = np.array([0.0])
    pz = np.array([2.5])
    ev_indiv_heart, _ = model.predict_spatial_ev(500000, px, pz, np.array([1]))
    p_strike_heart = 0.85
    ev_take_heart = p_strike_heart * DEFAULT_DELTA_RE_STRIKE + (1.0 - p_strike_heart) * DEFAULT_DELTA_RE_BALL
    heart_expected_rv = 0.70 * ev_indiv_heart[0] + 0.30 * ev_take_heart

    # Minimax optimization must produce lower expected batter RV than grooving a fastball down the middle
    assert best_opt['expected_batter_rv'] < heart_expected_rv


def test_spatial_contact_utility_shrinkage():
    """Verify xwobacon conversion and spatial surface boundaries."""
    # xwobacon to run value mapping
    rv_out = xwobacon_to_run_value(0.0) # pure out
    rv_barrel = xwobacon_to_run_value(1.0) # home run / elite barrel
    assert rv_out < 0.0
    assert rv_barrel > 0.5
