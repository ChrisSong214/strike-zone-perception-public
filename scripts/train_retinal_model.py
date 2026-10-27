"""
Training & Evaluation Pipeline for Retinal Perception LSTM (Pillar 4)
=====================================================================

Processes trajectories in biologically grounded 9D retinal space:
    [θ, φ, ψ, θ_dot, φ_dot, ψ_dot/ψ, σ_θ_dot, σ_φ_dot, σ_loom]
excluding all constant Cartesian acceleration features (ax, ay, az).

Evaluates perceptual uncertainty Σ_{0.40} and compares against Cartesian baseline.
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.model import PitchPerceptionLSTM, mdn_loss
from src.retinal_physics import cartesian_to_retinal_tensor, inject_weber_noise


def parse_args():
    parser = argparse.ArgumentParser(description="Train / Evaluate Retinal Perception LSTM")
    parser.add_argument("--year", type=int, default=2025, help="Season year")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--sample_size", type=int, default=20000, help="Subset size for quick training/eval")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda, mps)")
    parser.add_argument("--eval_only", action="store_true", help="Run evaluation without full training")
    return parser.parse_args()


def run_retinal_pipeline(year: int = 2025, sample_size: int = 20000, epochs: int = 2, device: str = "cpu", eval_only: bool = False):
    traj_path = os.path.join(_PROJECT_ROOT, "data", "training", f"trajectories_{year}.pt")
    if not os.path.exists(traj_path):
        print(f"[WARN] Trajectory file not found: {traj_path}. Generating synthetic verification dataset.")
        n_samples = 1000
        seq_len = 100
        cartesian = torch.randn(n_samples, seq_len, 9)
        # realistic positions: x near 0, y from 55 to 1.4, z from 6 to 2.5
        y_steps = torch.linspace(55.0, 1.4, seq_len)
        cartesian[:, :, 1] = y_steps.unsqueeze(0)
        plate_x = torch.randn(n_samples) * 0.5
        plate_z = torch.randn(n_samples) * 0.5 + 2.5
        sz_top = torch.full((n_samples,), 3.4)
        stand = torch.tensor(['R'] * (n_samples // 2) + ['L'] * (n_samples - n_samples // 2))
        valid_lengths = torch.full((n_samples,), 100, dtype=torch.int32)
    else:
        print(f"[INFO] Loading trajectories from {traj_path}...")
        raw = torch.load(traj_path, map_location="cpu", weights_only=False)
        total = raw["sequences"].shape[0]
        n_use = min(sample_size, total)
        cartesian = raw["sequences"][:n_use]
        plate_x = raw["labels"]["plate_x"][:n_use]
        plate_z = raw["labels"]["plate_z"][:n_use]
        sz_top = raw["labels"]["sz_top"][:n_use]
        valid_lengths = raw["valid_lengths"][:n_use]
        # Stand default or right
        stand = torch.tensor(['R'] * n_use)

    print(f"[INFO] Transforming {len(cartesian)} trajectories to 9D Retinal Coordinates...")
    retinal_9d = cartesian_to_retinal_tensor(cartesian, stand=stand, sz_top=sz_top)
    
    # Check dimensions
    assert retinal_9d.shape[-1] == 9, f"Expected 9 channels, got {retinal_9d.shape[-1]}"
    print(f"[INFO] Retinal tensor shape: {retinal_9d.shape}")

    # Initialize model
    model = PitchPerceptionLSTM(input_size=9, hidden_size=128, num_layers=2).to(device)
    
    targets = torch.stack([plate_x, plate_z], dim=-1).to(device)
    retinal_9d = retinal_9d.to(device)
    valid_lengths = valid_lengths.to(device)

    dataset = TensorDataset(retinal_9d, targets, valid_lengths)
    n_train = int(len(dataset) * 0.8)
    n_val = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if not eval_only:
        print(f"[INFO] Training Retinal LSTM for {epochs} epochs...")
        for ep in range(epochs):
            model.train()
            total_loss = 0.0
            n_batches = 0
            for bx, by, blen in train_loader:
                optimizer.zero_grad()
                pred = model(bx, blen)
                loss = mdn_loss(pred, by, blen)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            print(f"Epoch {ep+1}/{epochs}: Train NLL = {total_loss / max(1, n_batches):.4f}")

    # Evaluate at decision horizon tau = 0.40
    model.eval()
    val_loss = 0.0
    val_sigma_40_x = []
    val_sigma_40_z = []
    with torch.no_grad():
        for bx, by, blen in val_loader:
            pred = model(bx, blen)
            loss = mdn_loss(pred, by, blen)
            val_loss += loss.item() * len(bx)
            # Evaluate tau = 0.40 (step 40)
            tau_idx = 40
            val_sigma_40_x.append(pred[:, tau_idx, 2].cpu().numpy())
            val_sigma_40_z.append(pred[:, tau_idx, 3].cpu().numpy())

    mean_loss = val_loss / len(val_ds)
    sig_x = np.concatenate(val_sigma_40_x).mean()
    sig_z = np.concatenate(val_sigma_40_z).mean()
    print(f"[RESULT] Retinal Model Val NLL: {mean_loss:.4f}")
    print(f"[RESULT] Decision Point (tau=0.40) Mean Uncertainty: σ_x={sig_x:.3f} ft, σ_z={sig_z:.3f} ft")
    return {
        "val_nll": mean_loss,
        "mean_sigma_x_40": float(sig_x),
        "mean_sigma_z_40": float(sig_z),
    }


if __name__ == "__main__":
    args = parse_args()
    run_retinal_pipeline(
        year=args.year,
        sample_size=args.sample_size,
        epochs=args.epochs,
        device=args.device,
        eval_only=args.eval_only,
    )
