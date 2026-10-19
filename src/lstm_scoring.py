"""
LSTM-based Eye Rating Scoring Engine

Uses the trained LSTM+MDN model to compute batter eye ratings.

Key concept: The model outputs a 2D Gaussian belief at each timestep.
At the "decision point" (configurable, default = final timestep), we compute:
    P(Strike) = integral of the predicted Gaussian over the strike zone

A batter's "eye rating" is derived from how well their swing decisions align
with the model's predicted P(Strike) at the decision point.

Better eye = swings more at high P(Strike), takes more at low P(Strike)
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import PitchPerceptionLSTM
from src.physics import calculate_trajectory

# --- Constants ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'models')
FEATURES = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az']
SEQ_LEN = 100


class LSTMEyeScorer:
    """
    Scores batters using the trained LSTM+MDN streaming perception model.
    
    For each pitch:
        1. Generate trajectory sequence from Statcast data
        2. Run LSTM → get predicted Gaussian (μx, μz, σx, σz, ρ) at each timestep
        3. Compute P(Strike) at the decision point using the predicted Gaussian
        4. Score the batter on how well swings align with P(Strike)
    """
    
    def __init__(self, model_path=None, device='cpu', decision_frac=0.40):
        """
        Load trained model and normalization stats.
        
        Args:
            model_path: Path to best_model.pt checkpoint
            device: 'cpu', 'mps', or 'cuda'
            decision_frac: Fraction through the pitch's flight at which to
                evaluate the model.  0.0 = release, 1.0 = plate.
                Default 0.65 ≈ 140ms before plate on a 95-mph fastball,
                corresponding to the batter's swing/take decision point.
        """
        if model_path is None:
            model_path = os.path.join(MODEL_DIR, 'best_model.pt')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained model found at {model_path}. "
                f"Run scripts/train_model.py first."
            )
        
        if device == 'cuda' and not torch.cuda.is_available():
            print("Note: CUDA requested but torch.cuda.is_available() is False. Using CPU.", flush=True)
            device = 'cpu'
            
        self.model_path = model_path
        self.device = device
        self.decision_frac = decision_frac
        checkpoint = torch.load(model_path, weights_only=False, map_location=device)
        
        # Reconstruct model
        args = checkpoint.get('args', {})
        self.model = PitchPerceptionLSTM(
            input_size=len(FEATURES),
            hidden_size=args.get('hidden_size', 128),
            num_layers=args.get('num_layers', 2)
        ).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Normalization stats
        self.norm_mean = checkpoint['norm_mean'].to(device)
        self.norm_std = checkpoint['norm_std'].to(device)
        
        print(f"Loaded model from epoch {checkpoint.get('epoch', '?')}")
        print(f"  Val loss: {checkpoint.get('val_loss', '?')}")
        print(f"  Decision point: {decision_frac:.0%} through flight")
        if 'val_stats' in checkpoint:
            vs = checkpoint['val_stats']
            print(f"  R²: ({vs.get('r2_x', '?'):.4f}, {vs.get('r2_z', '?'):.4f})")
    
    def _pitch_to_sequence(self, row):
        """
        Convert a Statcast pitch row → normalized (1, SEQ_LEN, 9) tensor.
        
        Returns:
            seq: (1, SEQ_LEN, 9) tensor
            valid_len: (1,) tensor
            success: bool
        """
        traj = calculate_trajectory(row)
        
        if traj is None or len(traj) == 0:
            return None, None, False
        
        feat = traj[FEATURES].values.astype(np.float32)
        actual_len = len(feat)
        
        seq = np.zeros((SEQ_LEN, len(FEATURES)), dtype=np.float32)
        
        if actual_len > SEQ_LEN:
            indices = np.linspace(0, actual_len - 1, SEQ_LEN).round().astype(int)
            seq = feat[indices]
            valid_len = SEQ_LEN
        else:
            seq[-actual_len:] = feat
            valid_len = actual_len
        
        # Normalize
        seq_t = torch.from_numpy(seq).to(self.device)
        start = SEQ_LEN - valid_len
        if valid_len > 0:
            seq_t[start:] = (seq_t[start:] - self.norm_mean) / self.norm_std
        
        return seq_t.unsqueeze(0), torch.tensor([valid_len]), True
    
    @torch.no_grad()
    def predict_perception(self, row):
        """
        Get the model's full streaming perception for a single pitch.
        
        Returns:
            dict with keys:
                'params': (SEQ_LEN, 5) numpy array — [μx, μz, σx, σz, ρ] at each step
                'valid_len': int
                'final': dict with μx, μz, σx, σz, ρ at the final timestep
            or None if trajectory generation fails
        """
        seq, vlen, ok = self._pitch_to_sequence(row)
        if not ok:
            return None
        
        params = self.model(seq, vlen)  # (1, SEQ_LEN, 5)
        params_np = params[0].cpu().numpy()
        
        # Final timestep
        final_idx = vlen[0].item() - 1
        actual_idx = SEQ_LEN - vlen[0].item() + final_idx  # offset for padding
        fp = params_np[SEQ_LEN - 1]  # last position
        
        return {
            'params': params_np,
            'valid_len': vlen[0].item(),
            'final': {
                'mu_x': fp[0], 'mu_z': fp[1],
                'sigma_x': fp[2], 'sigma_z': fp[3],
                'rho': fp[4]
            }
        }
    
    @torch.no_grad()
    def batch_predict(self, df):
        """
        Run perception model on a batch of pitches.
        
        Evaluates at the decision point (self.decision_frac through flight),
        NOT at the final timestep.  This ensures the model's predictions
        reflect the batter's genuine uncertainty at the moment they must
        commit to swing or take.
        
        Args:
            df: DataFrame with Statcast columns needed by calculate_trajectory
            
        Returns:
            results: DataFrame with columns: mu_x, mu_z, sigma_x, sigma_z, 
                     rho, plus original index
        """
        sequences = []
        valid_lengths = []
        indices = []
        
        for idx, row in df.iterrows():
            seq, vlen, ok = self._pitch_to_sequence(row)
            if ok:
                sequences.append(seq)
                valid_lengths.append(vlen)
                indices.append(idx)
        
        if not sequences:
            return pd.DataFrame()
        
        # Stack into batches
        batch_seq = torch.cat(sequences, dim=0)       # (N, SEQ_LEN, 9)
        batch_vlen = torch.cat(valid_lengths, dim=0)   # (N,)
        
        # Run model — process in chunks to avoid OOM
        chunk_size = 1024
        all_decision = []
        
        for i in range(0, len(sequences), chunk_size):
            chunk_seq = batch_seq[i:i+chunk_size]
            chunk_vlen = batch_vlen[i:i+chunk_size]
            
            # Evaluate at decision point, not final timestep
            decision_params = self.model.predict_at_fraction(
                chunk_seq, chunk_vlen, fraction=self.decision_frac
            )  # (chunk, 5)
            all_decision.append(decision_params.cpu())
        
        decision_params = torch.cat(all_decision, dim=0).numpy()
        
        # Build results DataFrame
        results = pd.DataFrame({
            'mu_x': decision_params[:, 0],
            'mu_z': decision_params[:, 1],
            'sigma_x': decision_params[:, 2],
            'sigma_z': decision_params[:, 3],
            'rho': decision_params[:, 4],
        }, index=indices)
        
        return results
    
    @staticmethod
    def _bvn_cdf(h, k, r):
        """
        Vectorized standard bivariate normal cumulative distribution function.
        P(X <= h, Y <= k; rho=r) using Owen's T function.
        """
        h = np.atleast_1d(h).astype(np.float64)
        k = np.atleast_1d(k).astype(np.float64)
        if r is None:
            r = np.zeros_like(h)
        else:
            r = np.atleast_1d(r).astype(np.float64)
            
        h, k, r = np.broadcast_arrays(h, k, r)
        out = np.zeros_like(h, dtype=np.float64)
        
        # Zero correlation: product of standard normal CDFs
        mask_zero = np.abs(r) < 1e-7
        out[mask_zero] = norm.cdf(h[mask_zero]) * norm.cdf(k[mask_zero])
        
        mask = ~mask_zero
        if np.any(mask):
            from scipy.special import owens_t
            hh = h[mask]
            kk = k[mask]
            rr = np.clip(r[mask], -0.999999, 0.999999)
            sq = np.sqrt(1.0 - rr**2)
            
            eps = 1e-15
            hh_safe = np.where(hh == 0, eps, hh)
            kk_safe = np.where(kk == 0, eps, kk)
            
            ah = (kk_safe - rr * hh_safe) / (hh_safe * sq)
            ak = (hh_safe - rr * kk_safe) / (kk_safe * sq)
            
            term = 0.5 * norm.cdf(hh) + 0.5 * norm.cdf(kk) - owens_t(hh, ah) - owens_t(kk, ak)
            cond = (hh * kk < 0) | ((hh * kk == 0) & (hh + kk < 0))
            term = term - np.where(cond, 0.5, 0.0)
            out[mask] = np.clip(term, 0.0, 1.0)
            
        return out

    @staticmethod
    def compute_p_strike(mu_x, mu_z, sigma_x, sigma_z, sz_top, sz_bot, 
                         rho=None, sz_width=17/12):
        """
        Probability that the pitch is perceived as a strike given the 
        model's predicted bivariate Gaussian belief, incorporating correlation rho.
        
        Integrates the bivariate Gaussian over the strike zone rectangle:
            x in [-sz_width/2, sz_width/2], z in [sz_bot, sz_top]
        """
        sigma_x = np.maximum(sigma_x, 1e-6)
        sigma_z = np.maximum(sigma_z, 1e-6)
        
        x_min = -sz_width / 2
        x_max = sz_width / 2
        
        u1 = (x_min - mu_x) / sigma_x
        u2 = (x_max - mu_x) / sigma_x
        v1 = (sz_bot - mu_z) / sigma_z
        v2 = (sz_top - mu_z) / sigma_z
        
        if rho is None or np.all(np.abs(rho) < 1e-7):
            p_x = norm.cdf(u2) - norm.cdf(u1)
            p_z = norm.cdf(v2) - norm.cdf(v1)
            return p_x * p_z
        
        bvn = LSTMEyeScorer._bvn_cdf
        p_strike = bvn(u2, v2, rho) - bvn(u1, v2, rho) - bvn(u2, v1, rho) + bvn(u1, v1, rho)
        return np.clip(p_strike, 0.0, 1.0)
    
    def score_batter(self, df_batter):
        """
        Score a single batter's perception using LSTM predictions.
        
        The scoring uses log-likelihood of swing decisions given P(Strike):
            swing → log(P(Strike))      [good swings are at strikes]
            take  → log(1 - P(Strike))  [good takes are on balls]
        
        Higher (less negative) log-likelihood = better eye.
        
        Args:
            df_batter: DataFrame of a single batter's pitches
            
        Returns:
            dict with eye_score, mean_sigma, swing_accuracy, etc.
        """
        # Run model on all pitches
        predictions = self.batch_predict(df_batter)
        
        if len(predictions) == 0:
            return None
        
        # Merge predictions with original data
        merged = df_batter.loc[predictions.index].copy()
        merged['mu_x'] = predictions['mu_x']
        merged['mu_z'] = predictions['mu_z']
        merged['sigma_x'] = predictions['sigma_x']
        merged['sigma_z'] = predictions['sigma_z']
        merged['rho'] = predictions['rho']
        
        # Compute P(Strike) for each pitch
        merged['p_strike'] = self.compute_p_strike(
            merged['mu_x'].values, merged['mu_z'].values,
            merged['sigma_x'].values, merged['sigma_z'].values,
            merged['sz_top'].values, merged['sz_bot'].values,
            rho=merged['rho'].values
        )
        
        # Determine swings
        swing_events = {
            'hit_into_play', 'foul', 'swinging_strike', 'swinging_strike_blocked',
            'foul_tip', 'foul_bunt', 'missed_bunt', 'bunt_foul_tip'
        }
        is_swing = merged['description'].isin(swing_events).values
        
        p = np.clip(merged['p_strike'].values, 0.01, 0.99)
        
        # Log-likelihood
        ll = np.sum(is_swing * np.log(p) + (~is_swing) * np.log(1 - p))
        ll_per_pitch = ll / len(merged)
        
        return {
            'n_pitches': len(merged),
            'n_failed': len(df_batter) - len(predictions),
            'log_likelihood': ll,
            'll_per_pitch': ll_per_pitch,
            'mean_sigma_x': merged['sigma_x'].mean(),
            'mean_sigma_z': merged['sigma_z'].mean(),
            'swing_rate': is_swing.mean(),
            'mean_p_strike': merged['p_strike'].mean(),
        }
