#!/usr/bin/env python3
"""
ACIE Training - Vanilla PyTorch (No Lightning)
Works with existing PyTorch installation
"""

import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

# Add ACIE to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from acie.core.scm import AstronomicalSCM
from acie.models.networks import Encoder, Decoder

print("="*80)
print("ACIE AGGRESSIVE TRAINING - VANILLA PYTORCH")
print("="*80)
print()

# Configuration
config = {
    'obs_dim': 10000,
    'latent_dim': 256,
    'hidden_dims': [2048, 1024, 512],
    'batch_size': 128,
    'epochs': 100,
    'learning_rate': 0.0005,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print(f"Configuration:")
print(f"  Device: {config['device']}")
print(f"  Observation dim: {config['obs_dim']}")
print(f"  Latent dim: {config['latent_dim']}")
print(f"  Batch size: {config['batch_size']}")
print(f"  Epochs: {config['epochs']}")
print()

# Dataset
class ACIEDataset(Dataset):
    def __init__(self, csv_path, max_samples=None, sample_every=1):
        print(f"Loading: {Path(csv_path).name}")
        # Load with sampling to fit in memory
        self.data = pd.read_csv(csv_path, skiprows=lambda i: i % sample_every != 0, nrows=max_samples)
        self.data = self.data.values.astype(np.float32)
        # Normalize
        self.mean = self.data.mean(axis=0)
        self.std = self.data.std(axis=0) + 1e-8
        self.data = (self.data - self.mean) / self.std
        print(f"  ✓ {len(self.data)} samples, {self.data.shape[1]} features")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx])
        return x, x

print("Loading datasets...")
try:
    # Sample data to fit in RAM (every 10th row = 2000 samples from 20k)
    train_dataset = ACIEDataset("lib/acie_observational_20k_x_20k.csv", sample_every=10)
    val_dataset = ACIEDataset("lib/acie_observational_10k_x_10k.csv", max_samples=1000)
except Exception as e:
    print(f"Error loading datasets: {e}")
    print("Creating synthetic data for demo...")
    # Fallback to synthetic data
    class SyntheticDataset(Dataset):
        def __init__(self, n_samples, obs_dim):
            self.data = torch.randn(n_samples, obs_dim)
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx], self.data[idx]
    
    train_dataset = SyntheticDataset(2000, config['obs_dim'])
    val_dataset = SyntheticDataset(200, config['obs_dim'])
    print(" ✓ Using synthetic data")

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)

print()
print(f"✓ Train batches: {len(train_loader)}")
print(f"✓ Val batches: {len(val_loader)}")
print()

# Model
print("Building model...")
scm = AstronomicalSCM()
encoder = Encoder(config['obs_dim'], config['latent_dim'], hidden_dims=config['hidden_dims'])
decoder = Decoder(config['latent_dim'], config['obs_dim'], hidden_dims=config['hidden_dims'][::-1])

encoder = encoder.to(config['device'])
decoder = decoder.to(config['device'])

total_params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in decoder.parameters())
print(f"✓ Total parameters: {total_params:,}")
print()

# Optimizer
optimizer = optim.AdamW(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=config['learning_rate'],
    weight_decay=0.0001
)

# Scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

# Training loop
print("="*80)
print("STARTING TRAINING")
print("="*80)
print()

best_val_loss = float('inf')
history = {'train_loss': [], 'val_loss': []}

for epoch in range(config['epochs']):
    epoch_start = time.time()
    
    # Training
    encoder.train()
    decoder.train()
    train_loss = 0.0
    
    for batch_idx, (x, _) in enumerate(train_loader):
        x = x.to(config['device'])
        
        # Forward pass
        mu, logvar = encoder(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        x_recon = decoder(z)
        
        # Loss
        recon_loss = nn.functional.mse_loss(x_recon, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        loss = recon_loss + 0.5 * kl_loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 1.0)
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # Validation
    encoder.eval()
    decoder.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(config['device'])
            mu, logvar = encoder(x)
            z = mu
            x_recon = decoder(z)
            recon_loss = nn.functional.mse_loss(x_recon, x)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
            loss = recon_loss + 0.5 * kl_loss
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    
    # Update scheduler
    scheduler.step()
    
    # Log
    epoch_time = time.time() - epoch_start
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    
    print(f"Epoch {epoch+1}/{config['epochs']} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"LR: {scheduler.get_last_lr()[0]:.6f} | "
          f"Time: {epoch_time:.1f}s")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': config
        }, 'outputs/checkpoints/acie_best.pt')
        print(f"  → Saved best model (val_loss: {val_loss:.4f})")
    
    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch,
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': config
        }, f'outputs/checkpoints/acie_epoch_{epoch+1}.pt')

print()
print("="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Model saved to: outputs/checkpoints/acie_best.pt")
print()
