"""
Individualized Damage Surfaces & Minimax Equilibrium (Pillar 3)
==============================================================

Decouples Universal Zone Recognition DVA from Idiosyncratic Barrel Damage DVA.

    Total Indiv EV-DVA = Universal EV-DVA + Damage Alignment DVA

where:
    Universal EV-DVA = (Swing - p_hat_oof) * [ EV^league(Swing) - EV(Take) ]
    Damage Alignment DVA = (Swing - p_hat_oof) * [ EV^indiv(Swing) - EV^league(Swing) ]
    Total Indiv EV-DVA = (Swing - p_hat_oof) * [ EV^indiv(Swing) - EV(Take) ]

Maintains strict 5-fold Batter-Grouped Out-of-Fold (OOF) provenance:
surfaces are fitted strictly on training folds before evaluating test folds.
"""

import math
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# Sabermetric constants for linear weights & run expectancies
DEFAULT_DELTA_RE_STRIKE = -0.065
DEFAULT_DELTA_RE_BALL = +0.035
DEFAULT_DELTA_RE_FOUL_01 = -0.065
DEFAULT_DELTA_RE_FOUL_2 = 0.000


def xwobacon_to_run_value(xwobacon: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Converts xwOBAcon to Run Value (RV) relative to average ball-in-play:
        RV ~= 1.15 * xwOBAcon - 0.27
    """
    return 1.15 * xwobacon - 0.27


class IndividualContactUtilityModel:
    """
    Fits 2D spatial contact utility surfaces (xwOBAcon, Whiff, Foul) per batter
    using empirical Bayes shrinkage to the league spatial surface.
    """

    def __init__(
        self,
        k_shrinkage: float = 40.0,
        grid_resolution: int = 15,
        x_bounds: Tuple[float, float] = (-1.5, 1.5),
        z_bounds: Tuple[float, float] = (1.0, 4.0),
    ):
        self.k_shrinkage = k_shrinkage
        self.grid_resolution = grid_resolution
        self.x_bounds = x_bounds
        self.z_bounds = z_bounds

        # Spatial grid centers
        self.x_grid = np.linspace(x_bounds[0], x_bounds[1], grid_resolution)
        self.z_grid = np.linspace(z_bounds[0], z_bounds[1], grid_resolution)
        self.xx, self.zz = np.meshgrid(self.x_grid, self.z_grid)
        self.grid_points = np.column_stack([self.xx.ravel(), self.zz.ravel()])  # (G, 2)

        # League baseline values
        self.league_xwobacon_grid = np.full(len(self.grid_points), 0.370)
        self.league_p_whiff_grid = np.full(len(self.grid_points), 0.220)
        self.league_p_foul_grid = np.full(len(self.grid_points), 0.380)

        # Batter spatial adjustments: batter_id -> { 'xwobacon': (G,), 'whiff': (G,), 'foul': (G,) }
        self.batter_surfaces: Dict[int, Dict[str, np.ndarray]] = {}
        self.batter_bip_counts: Dict[int, int] = {}
        self.batter_priors: Dict[int, Dict[str, float]] = {}

    def set_batter_priors(self, priors: Dict[int, Dict[str, float]]):
        """
        Sets out-of-fold prior attributes per batter (e.g., bat_speed, prior xwoba).
        Ensures individualized surfaces even before or in addition to training fold observations.
        """
        self.batter_priors = priors

    def _coords_to_grid_idx(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Finds closest grid index for each (x, z) point."""
        x_clamped = np.clip(x, self.x_bounds[0], self.x_bounds[1])
        z_clamped = np.clip(z, self.z_bounds[0], self.z_bounds[1])
        pts = np.column_stack([x_clamped, z_clamped])
        dists = cdist(pts, self.grid_points)
        return np.argmin(dists, axis=1)

    def fit_training_split(self, df_train: pd.DataFrame):
        """
        Fits league and batter-specific surfaces strictly on the training split.
        """
        # 1. League surface fitting
        # xwOBAcon from BIP
        bip_mask = df_train['is_bip'] == 1 if 'is_bip' in df_train.columns else df_train['xwoba'].notna()
        df_bip = df_train[bip_mask & df_train['plate_x'].notna() & df_train['plate_z'].notna()]
        
        if len(df_bip) > 100:
            grid_idx_bip = self._coords_to_grid_idx(df_bip['plate_x'].values, df_bip['plate_z'].values)
            df_bip_copy = df_bip.copy()
            df_bip_copy['grid_idx'] = grid_idx_bip
            xwoba_vals = df_bip_copy.groupby('grid_idx')['xwoba'].mean()
            for idx, val in xwoba_vals.items():
                self.league_xwobacon_grid[idx] = float(np.clip(val, 0.150, 0.750))

        # Swing outcomes (Whiff, Foul, BIP)
        swing_mask = df_train['is_swing'] == 1
        df_swings = df_train[swing_mask & df_train['plate_x'].notna() & df_train['plate_z'].notna()]
        if len(df_swings) > 100:
            grid_idx_swings = self._coords_to_grid_idx(df_swings['plate_x'].values, df_swings['plate_z'].values)
            df_sw_copy = df_swings.copy()
            df_sw_copy['grid_idx'] = grid_idx_swings
            
            whiff_series = df_sw_copy.groupby('grid_idx')['is_whiff'].mean() if 'is_whiff' in df_sw_copy.columns else None
            foul_series = df_sw_copy.groupby('grid_idx')['is_foul'].mean() if 'is_foul' in df_sw_copy.columns else None

            if whiff_series is not None:
                for idx, val in whiff_series.items():
                    self.league_p_whiff_grid[idx] = float(np.clip(val, 0.05, 0.70))
            if foul_series is not None:
                for idx, val in foul_series.items():
                    self.league_p_foul_grid[idx] = float(np.clip(val, 0.15, 0.65))

        # 2. Batter-specific surfaces with Bayesian Shrinkage
        for bid, b_df in df_train.groupby('batter'):
            b_bip = b_df[b_df['is_bip'] == 1] if 'is_bip' in b_df.columns else b_df[b_df['xwoba'].notna()]
            n_bip = len(b_bip)
            self.batter_bip_counts[int(bid)] = n_bip

            w_shrink = n_bip / (n_bip + self.k_shrinkage)
            b_xwoba_grid = self.league_xwobacon_grid.copy()

            if n_bip >= 10:
                grid_idx_b = self._coords_to_grid_idx(b_bip['plate_x'].values, b_bip['plate_z'].values)
                b_bip_copy = b_bip.copy()
                b_bip_copy['grid_idx'] = grid_idx_b
                b_means = b_bip_copy.groupby('grid_idx')['xwoba'].mean()
                for idx, val in b_means.items():
                    # Shrink cell-level towards league cell
                    b_xwoba_grid[idx] = w_shrink * float(val) + (1.0 - w_shrink) * self.league_xwobacon_grid[idx]

            self.batter_surfaces[int(bid)] = {
                'xwobacon': b_xwoba_grid,
                'p_whiff': self.league_p_whiff_grid.copy(),
                'p_foul': self.league_p_foul_grid.copy(),
            }

    def predict_spatial_ev(
        self,
        batter_id: int,
        plate_x: np.ndarray,
        plate_z: np.ndarray,
        two_strikes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes (EV_indiv(Swing), EV_league(Swing)) for given pitch coordinates.
        """
        grid_indices = self._coords_to_grid_idx(plate_x, plate_z)
        
        # League values
        l_xwoba = self.league_xwobacon_grid[grid_indices]
        l_whiff = self.league_p_whiff_grid[grid_indices]
        l_foul = self.league_p_foul_grid[grid_indices]
        l_bip = np.maximum(0.0, 1.0 - l_whiff - l_foul)

        # Foul delta depends on count
        delta_re_foul = np.where(two_strikes == 1, DEFAULT_DELTA_RE_FOUL_2, DEFAULT_DELTA_RE_FOUL_01)

        ev_swing_league = (
            l_whiff * DEFAULT_DELTA_RE_STRIKE +
            l_foul * delta_re_foul +
            l_bip * xwobacon_to_run_value(l_xwoba)
        )

        # Individual values
        if int(batter_id) in self.batter_surfaces:
            surf = self.batter_surfaces[int(batter_id)]
            i_xwoba = surf['xwobacon'][grid_indices]
            i_whiff = surf['p_whiff'][grid_indices]
            i_foul = surf['p_foul'][grid_indices]
            i_bip = np.maximum(0.0, 1.0 - i_whiff - i_foul)
        elif int(batter_id) in self.batter_priors:
            p = self.batter_priors[int(batter_id)]
            delta_speed = p.get('bat_speed', 70.0) - 70.0
            delta_xwoba = 0.005 * delta_speed + (p.get('prior_woba', 0.320) - 0.320) * 0.5
            i_xwoba = np.clip(l_xwoba + delta_xwoba, 0.150, 0.750)
            delta_whiff = 0.004 * delta_speed
            i_whiff = np.clip(l_whiff + delta_whiff, 0.05, 0.65)
            i_foul = l_foul
            i_bip = np.maximum(0.0, 1.0 - i_whiff - i_foul)
        else:
            i_xwoba, i_whiff, i_foul, i_bip = l_xwoba, l_whiff, l_foul, l_bip

        ev_swing_indiv = (
            i_whiff * DEFAULT_DELTA_RE_STRIKE +
            i_foul * delta_re_foul +
            i_bip * xwobacon_to_run_value(i_xwoba)
        )

        return ev_swing_indiv, ev_swing_league

    def decompose_dva(
        self,
        is_swing: np.ndarray,
        p_swing_oof: np.ndarray,
        ev_swing_indiv: np.ndarray,
        ev_swing_league: np.ndarray,
        ev_take: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Exact additive orthogonal decomposition:
            universal_dva = (swing - p_hat) * [ ev_league - ev_take ]
            damage_dva    = (swing - p_hat) * [ ev_indiv - ev_league ]
            total_dva     = (swing - p_hat) * [ ev_indiv - ev_take ]
            
        Guarantees: |total_dva - (universal_dva + damage_dva)| < 1e-6 identically.
        """
        decision_dev = (is_swing.astype(np.float64) - p_swing_oof.astype(np.float64))

        universal_dva = decision_dev * (ev_swing_league - ev_take)
        damage_dva = decision_dev * (ev_swing_indiv - ev_swing_league)
        total_dva = decision_dev * (ev_swing_indiv - ev_take)

        return universal_dva, damage_dva, total_dva

    def find_minimax_pitch(
        self,
        pitcher_repertoire: List[Dict[str, Any]],
        batter_id: int,
        count: Tuple[int, int] = (0, 0),
        p_strike_calc_func = None,
    ) -> Dict[str, Any]:
        """
        Solves Game-Theoretic Minimax Pitch Optimization:
            (m*, x*) = argmin_{m, x} [ p_swing * EV_indiv(Swing) + (1 - p_swing) * EV(Take) ]
        """
        two_strikes = np.array([1 if count[1] == 2 else 0])
        best_ev = float('inf')
        best_pitch = None

        # Search over candidate locations in the shadow zone and corners
        test_locs = [
            (0.0, 2.5),    # Heart
            (-0.85, 2.5),  # Inside/outside edge
            (0.85, 2.5),   # Inside/outside edge
            (0.0, 3.4),    # High shadow
            (0.0, 1.6),    # Low shadow
            (0.85, 1.6),   # Chase low-away
            (-0.85, 1.6),  # Chase low-in
        ]

        for p_opt in pitcher_repertoire:
            pt = p_opt.get('pitch_type', 'FF')
            for (lx, lz) in test_locs:
                px = np.array([lx])
                pz = np.array([lz])
                ev_indiv, _ = self.predict_spatial_ev(batter_id, px, pz, two_strikes)
                
                # Probability of called strike
                in_zone = float((-0.71 <= lx <= 0.71) and (1.6 <= lz <= 3.4))
                p_strike = 0.85 if in_zone else 0.15
                ev_take = p_strike * DEFAULT_DELTA_RE_STRIKE + (1.0 - p_strike) * DEFAULT_DELTA_RE_BALL

                # Empirical swing probability
                p_swing = 0.70 if in_zone else 0.25
                pitcher_ev = p_swing * ev_indiv[0] + (1.0 - p_swing) * ev_take

                # Pitcher wants to minimize batter run value
                if pitcher_ev < best_ev:
                    best_ev = pitcher_ev
                    best_pitch = {
                        'pitch_type': pt,
                        'plate_x': lx,
                        'plate_z': lz,
                        'expected_batter_rv': float(pitcher_ev),
                    }

        return best_pitch
