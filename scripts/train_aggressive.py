#!/usr/bin/env python3
"""
ACIE Aggressive Training Script
Hyperparameter-optimized training with all available data
"""

import sys
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
import pandas as pd
import numpy as np

# Add ACIE to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from acie.core.scm import StructuralCausalModel, AstronomicalSCM
from acie.models.networks import Encoder, Decoder
from acie.training.train import ACIELightningModule
from torch.utils.data import Dataset, DataLoader

print("="*80)
print("ACIE AGGRESSIVE TRAINING - HYPERPARAMETER OPTIMIZED")
print("="*80)
print()

# Load configuration
config_path = Path(__file__).parent.parent / "config" / "aggressive_config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

print(f"✓ Loaded config from: {config_path}")
print(f"  - OBS DIM: {config['model']['obs_dim']}")
print(f"  - LATENT DIM: {config['model']['latent_dim']}")
print(f"  - BATCH SIZE: {config['training']['batch_size']}")
print(f"  - EPOCHS: {config['training']['epochs']}")
print()

# Custom dataset for large CSV files
class LargeCSVDataset(Dataset):
    """Memory-efficient dataset for large CSV files"""
    
    def __init__(self, csv_path, max_samples=None):
        self.csv_path = csv_path
        print(f"Loading dataset: {Path(csv_path).name}...")
        
        # Read CSV in chunks
        self.data = pd.read_csv(csv_path, nrows=max_samples)
        self.data = self.data.values.astype(np.float32)
        
        print(f"  ✓ Loaded {len(self.data)} samples, {self.data.shape[1]} features")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx])
        return x, x  # Return (input, target) for reconstruction

# Load datasets
print("Loading datasets...")
train_dataset = LargeCSVDataset(
    config['data']['observational_train'],
    max_samples=20000  # Use all 20k samples
)

val_dataset = LargeCSVDataset(
    config['data']['observational_val'],
    max_samples=10000  # Use all 10k samples
)

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=config['training']['batch_size'],
    shuffle=True,
    num_workers=config['hardware']['num_workers'],
    pin_memory=config['hardware']['pin_memory'],
    persistent_workers=config['hardware']['persistent_workers']
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config['training']['batch_size'],
    shuffle=False,
    num_workers=config['hardware']['num_workers'],
    pin_memory=config['hardware']['pin_memory'],
    persistent_workers=config['hardware']['persistent_workers']
)

print()
print("✓ Dataloaders ready")
print(f"  - Train batches: {len(train_loader)}")
print(f"  - Val batches: {len(val_loader)}")
print()

# Create SCM
print("Building model...")
scm = AstronomicalSCM()
print(f"  ✓ SCM with {len(scm.graph.nodes())} nodes")

# Create encoder/decoder
obs_dim = config['model']['obs_dim']
latent_dim = config['model']['latent_dim']
hidden_dims = config['model']['hidden_dims']

encoder = Encoder(obs_dim, latent_dim, hidden_dims=hidden_dims)
decoder = Decoder(latent_dim, obs_dim, hidden_dims=hidden_dims[::-1])

print(f"  ✓ Encoder: {obs_dim} → {latent_dim}")
print(f"  ✓ Decoder: {latent_dim} → {obs_dim}")
print(f"  ✓ Hidden dims: {hidden_dims}")
print()

# Create Lightning module
model = ACIELightningModule(
    scm=scm,
    encoder=encoder,
    decoder=decoder,
    learning_rate=config['training']['learning_rate'],
    config=config
)

print("✓ Lightning module created")
print()

# Callbacks
callbacks = []

# Model checkpoint
checkpoint_callback = ModelCheckpoint(
    dirpath="outputs/checkpoints",
    filename="acie-{epoch:02d}-{val_loss:.4f}",
    monitor="val/total_loss",
    mode="min",
    save_top_k=config['checkpointing']['save_top_k'],
    save_last=True,
    verbose=True
)
callbacks.append(checkpoint_callback)

# Early stopping
early_stop_callback = EarlyStopping(
    monitor="val/total_loss",
    patience=config['early_stopping']['patience'],
    mode="min",
    verbose=True
)
callbacks.append(early_stop_callback)

# Learning rate monitor
lr_monitor = LearningRateMonitor(logging_interval='step')
callbacks.append(lr_monitor)

print("✓ Callbacks configured")
print()

# Logger
logger = TensorBoardLogger(
    save_dir=config['logging']['log_dir'],
    name="aggressive_training"
)

print("✓ TensorBoard logger ready")
print()

# Trainer
trainer = pl.Trainer(
    max_epochs=config['training']['epochs'],
    accelerator=config['hardware']['accelerator'],
    devices=config['hardware']['devices'],
    callbacks=callbacks,
    logger=logger,
    log_every_n_steps=config['logging']['log_every_n_steps'],
    precision="16-mixed" if config['training']['use_amp'] else 32,
    accumulate_grad_batches=config['training']['accumulate_grad_batches'],
    gradient_clip_val=config['training']['clip_grad_norm'],
    deterministic=False,  # Faster training
    benchmark=True  # cuDNN auto-tuner
)

print("="*80)
print("STARTING AGGRESSIVE TRAINING")
print("="*80)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print()
print("Training configuration:")
print(f"  - Epochs: {config['training']['epochs']}")
print(f"  - Batch size: {config['training']['batch_size']} (effective: {config['training']['batch_size'] * config['training']['accumulate_grad_batches']})")
print(f"  - Learning rate: {config['training']['learning_rate']}")
print(f"  - Mixed precision: {config['training']['use_amp']}")
print(f"  - Device: {trainer.device_ids if hasattr(trainer, 'device_ids') else 'CPU'}")
print()
print("="*80)
print()

# Train!
try:
    trainer.fit(model, train_loader, val_loader)
    
    print()
    print("="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best val loss: {checkpoint_callback.best_model_score:.4f}")
    print()
    print("Model saved to: outputs/checkpoints/")
    print("Logs available at: outputs/logs/")
    print()
    print("To view training progress:")
    print("  tensorboard --logdir outputs/logs")
    print()
    
except KeyboardInterrupt:
    print("\n\nTraining interrupted by user!")
    print(f"Latest checkpoint: {checkpoint_callback.last_model_path}")
    
except Exception as e:
    print(f"\n\nTraining failed with error: {e}")
    import traceback
    traceback.print_exc()
