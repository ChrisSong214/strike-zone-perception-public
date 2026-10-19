"""
Train the LSTM+MDN Pitch Perception Model (Streaming Version)

The model predicts plate-crossing distributions at EVERY timestep.
Loss is NLL of the actual plate location under each timestep's prediction.
The model must learn to express uncertainty early and confidence late.

Usage:
    python3 scripts/train_model.py                    # Train on 2025
    python3 scripts/train_model.py --years 2020-2025  # Train on all years
    python3 scripts/train_model.py --epochs 30        # Custom epochs
    python3 scripts/train_model.py --device cpu        # Force CPU
"""

import sys
import os
import argparse
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.dataset import PitchTrajectoryDataset
from src.model import PitchPerceptionLSTM, streaming_mdn_loss

# --- Defaults ---
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'training')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'models')
DEVICE = (
    'mps' if torch.backends.mps.is_available()
    else 'cuda' if torch.cuda.is_available()
    else 'cpu'
)


def collate_fn(batch):
    """Custom collate to handle the label dicts."""
    seqs = torch.stack([b[0] for b in batch])
    valid_lens = torch.stack([b[2] for b in batch])
    
    labels = {}
    for key in batch[0][1]:
        labels[key] = torch.stack([b[1][key] for b in batch])
    
    return seqs, labels, valid_lens


def train_epoch(model, loader, optimizer, device, seq_len):
    """Train one epoch with streaming loss, return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for seqs, labels, lengths in loader:
        seqs = seqs.to(device)
        target_x = labels['plate_x'].to(device)
        target_z = labels['plate_z'].to(device)
        lengths = lengths.to(device)
        
        optimizer.zero_grad()
        params = model(seqs, lengths)  # (batch, seq_len, 5)
        loss = streaming_mdn_loss(params, target_x, target_z, lengths, seq_len)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  WARNING: NaN/Inf loss detected, skipping batch")
            continue
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_epoch(model, loader, device, seq_len):
    """Evaluate one epoch. Reports stats at FINAL timestep (decision time)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    # Collect final-timestep predictions
    all_mu_x, all_mu_z = [], []
    all_sigma_x, all_sigma_z = [], []
    all_target_x, all_target_z = [], []
    
    # Collect decision-point predictions (40% flight)
    all_mu_x_dec, all_mu_z_dec = [], []
    all_sigma_x_dec, all_sigma_z_dec = [], []
    
    # Collect early-timestep sigmas (first valid timestep) for comparison
    all_sigma_x_early, all_sigma_z_early = [], []
    
    for seqs, labels, lengths in loader:
        seqs = seqs.to(device)
        target_x = labels['plate_x'].to(device)
        target_z = labels['plate_z'].to(device)
        lengths = lengths.to(device)
        
        params = model(seqs, lengths)  # (batch, seq_len, 5)
        loss = streaming_mdn_loss(params, target_x, target_z, lengths, seq_len)
        
        total_loss += loss.item()
        n_batches += 1
        
        batch_size = seqs.size(0)
        
        # Final timestep predictions (using predict_final)
        final_params = model.predict_final(seqs, lengths)  # (batch, 5)
        all_mu_x.append(final_params[:, 0].cpu())
        all_mu_z.append(final_params[:, 1].cpu())
        all_sigma_x.append(final_params[:, 2].cpu())
        all_sigma_z.append(final_params[:, 3].cpu())
        all_target_x.append(target_x.cpu())
        all_target_z.append(target_z.cpu())
        
        # Decision point predictions at 40% flight
        dec_params = model.predict_at_fraction(seqs, lengths, fraction=0.40)
        all_mu_x_dec.append(dec_params[:, 0].cpu())
        all_mu_z_dec.append(dec_params[:, 1].cpu())
        all_sigma_x_dec.append(dec_params[:, 2].cpu())
        all_sigma_z_dec.append(dec_params[:, 3].cpu())
        
        # Early timestep sigmas (first valid timestep per sample)
        start_idx = (seq_len - lengths).clamp(min=0).long()
        early_params = params[torch.arange(batch_size, device=device), start_idx]
        all_sigma_x_early.append(early_params[:, 2].cpu())
        all_sigma_z_early.append(early_params[:, 3].cpu())
    
    # Aggregate
    mu_x = torch.cat(all_mu_x)
    mu_z = torch.cat(all_mu_z)
    sigma_x = torch.cat(all_sigma_x)
    sigma_z = torch.cat(all_sigma_z)
    target_x = torch.cat(all_target_x)
    target_z = torch.cat(all_target_z)
    sigma_x_early = torch.cat(all_sigma_x_early)
    sigma_z_early = torch.cat(all_sigma_z_early)
    
    mu_x_dec = torch.cat(all_mu_x_dec)
    mu_z_dec = torch.cat(all_mu_z_dec)
    sigma_x_dec = torch.cat(all_sigma_x_dec)
    sigma_z_dec = torch.cat(all_sigma_z_dec)
    
    # R² for final-timestep predictions
    ss_res_x = ((mu_x - target_x) ** 2).sum()
    ss_tot_x = ((target_x - target_x.mean()) ** 2).sum()
    r2_x = 1 - ss_res_x / ss_tot_x
    
    ss_res_z = ((mu_z - target_z) ** 2).sum()
    ss_tot_z = ((target_z - target_z.mean()) ** 2).sum()
    r2_z = 1 - ss_res_z / ss_tot_z

    # R² at 40% decision point
    r2_x_dec = 1 - (((mu_x_dec - target_x) ** 2).sum() / ss_tot_x)
    r2_z_dec = 1 - (((mu_z_dec - target_z) ** 2).sum() / ss_tot_z)
    
    stats = {
        'r2_x': r2_x.item(),
        'r2_z': r2_z.item(),
        'mae_x': (mu_x - target_x).abs().mean().item(),
        'mae_z': (mu_z - target_z).abs().mean().item(),
        # Decision point (40% flight) metrics
        'r2_x_dec': r2_x_dec.item(),
        'r2_z_dec': r2_z_dec.item(),
        'mae_x_dec': (mu_x_dec - target_x).abs().mean().item(),
        'mae_z_dec': (mu_z_dec - target_z).abs().mean().item(),
        'sigma_x_dec': sigma_x_dec.mean().item(),
        'sigma_z_dec': sigma_z_dec.mean().item(),
        # Final timestep uncertainty
        'sigma_x_final': sigma_x.mean().item(),
        'sigma_z_final': sigma_z.mean().item(),
        # Early timestep uncertainty
        'sigma_x_early': sigma_x_early.mean().item(),
        'sigma_z_early': sigma_z_early.mean().item(),
    }
    
    return total_loss / max(n_batches, 1), stats


def main():
    parser = argparse.ArgumentParser(description='Train LSTM+MDN Pitch Perception Model')
    parser.add_argument('--years', type=str, default='2025',
                        help='Years to train on, e.g. "2025" or "2020-2025"')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val-split', type=float, default=0.15,
                        help='Fraction of data for validation')
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--device', type=str, default=None,
                        help='Force device (cpu, mps, cuda). Default: auto-detect.')
    args = parser.parse_args()
    
    # Device selection
    if args.device:
        device = args.device
    else:
        device = DEVICE
    
    # Parse years
    if '-' in args.years:
        start, end = args.years.split('-')
        years = list(range(int(start), int(end) + 1))
    else:
        years = [int(y) for y in args.years.split(',')]
    
    print(f"{'='*60}")
    print(f"  LSTM+MDN Streaming Perception Training")
    print(f"{'='*60}")
    print(f"  Device: {device}")
    print(f"  Years: {years}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Hidden size: {args.hidden_size}")
    print(f"  Num layers: {args.num_layers}")
    print(f"  Val split: {args.val_split}")
    print()
    
    # Load data without premature full-dataset normalization
    print("Loading dataset...")
    dataset = PitchTrajectoryDataset(DATA_DIR, years=years, normalize=False)
    print(f"Total samples: {len(dataset):,}")
    seq_len = dataset.seq_len
    
    # Train / Val split by indices to prevent data leakage
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Compute normalization statistics on training subset ONLY
    dataset._compute_normalization_stats(indices=train_indices)
    dataset.normalize = True
    
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    print(f"Train: {train_size:,} | Val: {val_size:,}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_fn,
        num_workers=0, pin_memory=(device != 'cpu')
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn,
        num_workers=0, pin_memory=(device != 'cpu')
    )
    
    # Model
    model = PitchPerceptionLSTM(
        input_size=dataset.n_features,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Optimizer + Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )
    
    # Training loop
    os.makedirs(MODEL_DIR, exist_ok=True)
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(MODEL_DIR, 'best_model.pt')
    
    print(f"\n{'='*60}")
    print(f"  Training (streaming loss over all timesteps)...")
    print(f"{'='*60}\n")
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, device, seq_len)
        val_loss, val_stats = eval_epoch(model, val_loader, device, seq_len)
        
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0
        
        # Check for improvement
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_stats': val_stats,
                'norm_mean': dataset.feature_mean,
                'norm_std': dataset.feature_std,
                'args': vars(args),
            }, best_model_path)
            marker = ' ★ SAVED'
        else:
            patience_counter += 1
            marker = ''
        
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"R²(x): {val_stats['r2_x']:.4f} | R²(z): {val_stats['r2_z']:.4f} | "
              f"σ_early: ({val_stats['sigma_x_early']:.2f},{val_stats['sigma_z_early']:.2f}) | "
              f"σ_final: ({val_stats['sigma_x_final']:.3f},{val_stats['sigma_z_final']:.3f}) | "
              f"lr: {lr:.2e} | {elapsed:.1f}s{marker}")
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break
    
    # Final evaluation
    print(f"\n{'='*60}")
    print(f"  Training Complete")
    print(f"{'='*60}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Model saved to: {best_model_path}")
    
    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    final_loss, fs = eval_epoch(model, val_loader, device, seq_len)
    
    print(f"\n  Final Validation Stats (at plate):")
    print(f"    R²(x):         {fs['r2_x']:.4f}")
    print(f"    R²(z):         {fs['r2_z']:.4f}")
    print(f"    MAE(x):        {fs['mae_x']:.4f} ft ({fs['mae_x']*12:.2f} in)")
    print(f"    MAE(z):        {fs['mae_z']:.4f} ft ({fs['mae_z']*12:.2f} in)")
    print(f"    σ at release:  ({fs['sigma_x_early']:.3f}, {fs['sigma_z_early']:.3f}) ft")
    print(f"    σ at plate:    ({fs['sigma_x_final']:.3f}, {fs['sigma_z_final']:.3f}) ft")
    print(f"    σ ratio:       x={fs['sigma_x_early']/max(fs['sigma_x_final'],1e-6):.1f}x, "
          f"z={fs['sigma_z_early']/max(fs['sigma_z_final'],1e-6):.1f}x")


if __name__ == "__main__":
    main()
