#!/usr/bin/env python3
"""
Quick start training script for ACIE
"""

from pathlib import Path
from acie.training.train import train_acie

if __name__ == "__main__":
    # Configure training
    data_dir = Path("lib")
    output_dir = Path("outputs/quickstart")
    
    print("="*60)
    print("ACIE Quick Start Training")
    print("="*60)
    print(f"\nData directory: {data_dir}")
    print(f"Output directory: {output_dir}\n")
    
    # Train on 10k dataset with counterfactuals
    model, trainer = train_acie(
        data_dir=data_dir,
        output_dir=output_dir,
        obs_dim=6000,
        latent_dim=2000,
        batch_size=128,
        max_epochs=50,
        learning_rate=1e-4,
        dataset_size="10k",
        use_counterfactual=True,
        gpus=1,
        num_workers=4,
        fast_dev_run=False,
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nModel saved to: {output_dir}/acie_final.ckpt")
    print(f"Logs saved to: {output_dir}/logs")
    print("\nTo perform inference:")
    print(f"  python -m acie.cli infer --checkpoint {output_dir}/acie_final.ckpt \\")
    print(f"    --observation-file <your_data.csv> \\")
    print(f"    --intervention 'mass=1.5' \\")
    print(f"    --output-dir results/")
