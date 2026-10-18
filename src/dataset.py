"""
PyTorch Dataset for pitch trajectory sequences.

Loads pre-computed .pt files from data/training/ and provides
(sequence, labels) pairs for training the LSTM+MDN.
"""

import os
import torch
from torch.utils.data import Dataset


class PitchTrajectoryDataset(Dataset):
    """
    Loads trajectory sequences and labels from .pt files.
    
    Each sample is:
        sequence: (seq_len, 9) float32 — [x, y, z, vx, vy, vz, ax, ay, az]
        labels: dict with plate_x, plate_z, sz_top, sz_bot, is_swing, batter_id
        valid_length: int — timesteps before padding
    """
    
    def __init__(self, data_dir, years=None, normalize=True):
        """
        Args:
            data_dir: Path to data/training/ directory
            years: List of years to load (e.g., [2025]). If None, loads all.
            normalize: Whether to z-score normalize features
        """
        # Find files
        if years is None:
            files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt')])
        else:
            files = [f'trajectories_{y}.pt' for y in years]
        
        valid_files = [f for f in files if os.path.exists(os.path.join(data_dir, f))]
        if not valid_files:
            raise FileNotFoundError(f"No trajectory files found in {data_dir} for {years}")
            
        if len(valid_files) == 1:
            path = os.path.join(data_dir, valid_files[0])
            print(f"Loading {valid_files[0]}...")
            data = torch.load(path, weights_only=False)
            self.sequences = data['sequences']
            self.valid_lengths = data['valid_lengths']
            self.labels = data['labels']
        else:
            loaded_data = []
            for fname in valid_files:
                path = os.path.join(data_dir, fname)
                print(f"Loading {fname}...")
                loaded_data.append(torch.load(path, weights_only=False))
                
            total_samples = sum(d['sequences'].shape[0] for d in loaded_data)
            seq_len = loaded_data[0]['sequences'].shape[1]
            n_features = loaded_data[0]['sequences'].shape[2]
            label_keys = list(loaded_data[0]['labels'].keys())
            
            # Pre-allocate final tensors to avoid doubling peak memory
            self.sequences = torch.empty((total_samples, seq_len, n_features), dtype=loaded_data[0]['sequences'].dtype)
            self.valid_lengths = torch.empty(total_samples, dtype=loaded_data[0]['valid_lengths'].dtype)
            self.labels = {k: torch.empty(total_samples, dtype=loaded_data[0]['labels'][k].dtype) for k in label_keys}
            
            offset = 0
            while loaded_data:
                d = loaded_data.pop(0)  # free dict reference sequentially
                n = d['sequences'].shape[0]
                self.sequences[offset:offset+n] = d['sequences']
                self.valid_lengths[offset:offset+n] = d['valid_lengths']
                for k in label_keys:
                    self.labels[k][offset:offset+n] = d['labels'][k]
                del d
                offset += n
        
        self.n_samples = len(self.sequences)
        self.seq_len = self.sequences.shape[1]
        self.n_features = self.sequences.shape[2]
        
        # Normalization stats
        self.normalize = normalize
        self.feature_mean = None
        self.feature_std = None
        
        if normalize:
            self._compute_normalization_stats()
            
    def set_normalization_stats(self, mean, std):
        """Set normalization stats externally (e.g. from training fold only)."""
        self.feature_mean = mean
        self.feature_std = std
        self.normalize = True
    
    def _compute_normalization_stats(self, indices=None, chunk_size=25000):
        """
        Compute per-feature mean/std across valid timesteps in memory-bounded chunks.
        Avoids materializing large tensor copies or boolean masks of the entire dataset.
        """
        if indices is None:
            indices = torch.arange(self.n_samples)
        elif not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices, dtype=torch.long)
            
        n_features = self.n_features
        seq_len = self.seq_len
        
        sum_x = torch.zeros(n_features, dtype=torch.float64)
        sum_x2 = torch.zeros(n_features, dtype=torch.float64)
        total_valid_steps = 0
        
        step_indices = torch.arange(seq_len)
        
        # Iterate over indices in chunks to keep peak memory bounded (<150 MB)
        for start_idx in range(0, len(indices), chunk_size):
            chunk_idx = indices[start_idx:start_idx + chunk_size]
            chunk_seqs = self.sequences[chunk_idx]  # (B, seq_len, 9)
            chunk_lens = self.valid_lengths[chunk_idx]  # (B,)
            
            # Mask valid timesteps (padding is at the beginning)
            start_pos = (seq_len - chunk_lens).unsqueeze(1)  # (B, 1)
            valid_mask = step_indices.unsqueeze(0) >= start_pos  # (B, seq_len)
            
            valid_pts = chunk_seqs[valid_mask].to(torch.float64)
            if len(valid_pts) > 0:
                sum_x += valid_pts.sum(dim=0)
                sum_x2 += (valid_pts ** 2).sum(dim=0)
                total_valid_steps += len(valid_pts)
                
        if total_valid_steps == 0:
            self.feature_mean = torch.zeros(n_features, dtype=torch.float32)
            self.feature_std = torch.ones(n_features, dtype=torch.float32)
            return

        mean = sum_x / total_valid_steps
        var = (sum_x2 / total_valid_steps) - (mean ** 2)
        var = torch.clamp(var, min=1e-12)
        std = torch.sqrt(var)
        
        self.feature_mean = mean.to(torch.float32)
        self.feature_std = torch.clamp(std.to(torch.float32), min=1e-6)
        
        print(f"Normalization stats computed over {total_valid_steps:,} timesteps (memory-bounded streaming):")
        features = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az']
        for i, f in enumerate(features):
            print(f"  {f:>3}: mean={self.feature_mean[i]:8.3f}, std={self.feature_std[i]:8.3f}")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        seq = self.sequences[idx].clone()  # (seq_len, 9)
        valid_len = self.valid_lengths[idx]
        
        # Apply normalization only to non-padded region
        if self.normalize and self.feature_mean is not None:
            # Normalize only the valid (non-padded) timesteps
            start = self.seq_len - valid_len  # padding is at the start
            if valid_len > 0:
                seq[start:] = (seq[start:] - self.feature_mean) / self.feature_std
        
        labels = {k: self.labels[k][idx] for k in self.labels}
        
        return seq, labels, valid_len
