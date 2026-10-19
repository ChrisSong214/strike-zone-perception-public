"""
LSTM with Bivariate Gaussian Probabilistic Output Head (Single-Component K=1 MDN)
for Pitch Perception — Streaming Version

The model processes a pitch trajectory and outputs a 2D Gaussian belief
about the plate-crossing location AT EVERY TIMESTEP, not just the final one.

This is the "Living Cloud" — early in the flight the model should have high σ
(large uncertainty), and as the ball approaches the plate, σ should shrink
(batter resolves the pitch).

Architecture:
    Input:  (batch, seq_len, 9)  — [x, y, z, vx, vy, vz, ax, ay, az] per timestep
    LSTM:   2-layer, hidden=128, dropout=0.2
    Head:   Linear(128 → 5) at each timestep → [μx, μz, σx, σz, ρ]
            parameterizing a correlated bivariate Gaussian distribution.

Loss: Negative log-likelihood of the actual (plate_x, plate_z) under
      the predicted Gaussian at EACH timestep, averaged over the sequence.
      Only valid (non-padded) timesteps contribute to the loss.
"""

import torch
import torch.nn as nn
import math

# Default decision commitment point across flight (40% flight, ~175ms after release)
DEFAULT_DECISION_FRAC = 0.40


class PitchPerceptionLSTM(nn.Module):
    """
    Streaming LSTM with Bivariate Gaussian Probabilistic Output Head
    (Single-Component K=1 Mixture Density Network) — outputs predictions at every timestep.
    """
    
    def __init__(self, input_size=9, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM backbone
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # MDN output head: 5 parameters for bivariate Gaussian
        # Applied at each timestep: [μx, μz, σx, σz, ρ]
        self.mdn_head = nn.Linear(hidden_size, 5)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier init for linear, bias σ to start with moderate uncertainty."""
        nn.init.xavier_uniform_(self.mdn_head.weight)
        with torch.no_grad():
            self.mdn_head.bias.fill_(0.0)
            # Start sigmas around exp(0) = 1.0 ft (deliberately large)
            self.mdn_head.bias[2] = 0.0
            self.mdn_head.bias[3] = 0.0
    
    def forward(self, x, lengths=None):
        """
        Args:
            x: (batch, seq_len, 9) trajectory sequences (left-padded with zeros)
            lengths: (batch,) valid sequence lengths (optional)
            
        Returns:
            params: (batch, seq_len, 5) — [μx, μz, σx, σz, ρ] at each timestep
        """
        # Note: sequences are left-padded (valid data ends at index seq_len - 1).
        # We process x directly through the LSTM without pack_padded_sequence,
        # as PyTorch packing strictly assumes right-padding and truncates the
        # critical final timesteps approaching the plate.
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden)
        
        # Apply MDN head at every timestep
        raw = self.mdn_head(lstm_out)  # (batch, seq_len, 5)
        
        # Apply activations
        mu_x = raw[:, :, 0]                    # unconstrained
        mu_z = raw[:, :, 1]                    # unconstrained
        sigma_x = torch.exp(raw[:, :, 2])      # > 0
        sigma_z = torch.exp(raw[:, :, 3])      # > 0
        rho = torch.tanh(raw[:, :, 4])         # (-1, 1)
        
        return torch.stack([mu_x, mu_z, sigma_x, sigma_z, rho], dim=2)
    
    def predict_final(self, x, lengths=None):
        """
        Convenience: returns only the final-timestep prediction per sample.
        For left-padded sequences, the final timestep (at plate) is always index -1.
        
        Returns:
            params: (batch, 5) — [μx, μz, σx, σz, ρ] at the last valid timestep
        """
        all_params = self.forward(x, lengths)  # (batch, seq_len, 5)
        return all_params[:, -1, :]
    
    def predict_at_fraction(self, x, lengths=None, fraction=DEFAULT_DECISION_FRAC):
        """
        Returns prediction at a specific fraction through each sequence's
        valid portion.  This is the "decision point" — the moment in the 
        pitch's flight when the batter must commit to swing or take.
        
        Args:
            x:        (batch, seq_len, 9) trajectory sequences
            lengths:  (batch,) valid sequence lengths
            fraction: 0.0 = first valid timestep (release)
                      1.0 = last valid timestep  (plate)
                      DEFAULT_DECISION_FRAC = 0.40 (40% through flight, ~175ms after release)
        
        Returns:
            params: (batch, 5) — [μx, μz, σx, σz, ρ] at the decision point
        """
        all_params = self.forward(x, lengths)  # (batch, seq_len, 5)
        
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        if lengths is not None:
            # Sequences are LEFT-padded: valid data occupies the last `length` positions
            # start_idx = first valid position, end_idx = seq_len - 1
            start_idx = (seq_len - lengths).long()
            offset = (fraction * (lengths.float() - 1.0)).round().long()
            decision_idx = (start_idx + offset).clamp(min=0, max=seq_len - 1)
            final_params = all_params[torch.arange(batch_size, device=x.device), decision_idx]
        else:
            idx = int(round(fraction * (seq_len - 1)))
            final_params = all_params[:, idx, :]
        
        return final_params


def streaming_mdn_loss(params, target_x, target_z, valid_lengths, seq_len):
    """
    NLL of the target plate location under the predicted Gaussian at EACH timestep.
    Only valid (non-padded) timesteps contribute. 
    
    This is the key: the same target (plate_x, plate_z) is used at every timestep.
    Early predictions should have large σ (high NLL tolerated), late predictions 
    should have small σ (low NLL required).
    
    Args:
        params: (batch, seq_len, 5) — [μx, μz, σx, σz, ρ] per timestep
        target_x: (batch,) — actual plate_x
        target_z: (batch,) — actual plate_z
        valid_lengths: (batch,) — number of valid timesteps per sample
        seq_len: int — total sequence length (for mask creation)
        
    Returns:
        loss: scalar — mean NLL across all valid timesteps
    """
    batch_size = params.size(0)
    
    mu_x = params[:, :, 0]
    mu_z = params[:, :, 1]
    sigma_x = params[:, :, 2]
    sigma_z = params[:, :, 3]
    rho = params[:, :, 4]
    
    # Clamp for stability
    sigma_x = torch.clamp(sigma_x, min=1e-4, max=10.0)
    sigma_z = torch.clamp(sigma_z, min=1e-4, max=10.0)
    rho = torch.clamp(rho, min=-0.99, max=0.99)
    
    # Broadcast targets to (batch, seq_len)
    tx = target_x.unsqueeze(1).expand(-1, seq_len)
    tz = target_z.unsqueeze(1).expand(-1, seq_len)
    
    # Standardized residuals
    dx = (tx - mu_x) / sigma_x
    dz = (tz - mu_z) / sigma_z
    
    # Bivariate Gaussian NLL at each timestep
    one_minus_rho2 = 1.0 - rho ** 2
    
    exponent = (dx ** 2 + dz ** 2 - 2 * rho * dx * dz) / (2 * one_minus_rho2)
    
    log_norm = (math.log(2 * math.pi) 
                + torch.log(sigma_x) 
                + torch.log(sigma_z) 
                + 0.5 * torch.log(one_minus_rho2))
    
    nll = log_norm + exponent  # (batch, seq_len)
    
    # Mask: only valid (non-padded) timesteps
    # Padding is at the START (left-padded), valid data is at the END
    timestep_idx = torch.arange(seq_len, device=params.device).unsqueeze(0)  # (1, seq_len)
    start_idx = (seq_len - valid_lengths).unsqueeze(1)  # (batch, 1)
    mask = (timestep_idx >= start_idx).float()  # (batch, seq_len)
    
    # Masked mean
    masked_nll = (nll * mask).sum() / mask.sum()
    
    return masked_nll
