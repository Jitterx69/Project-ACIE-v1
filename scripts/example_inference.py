#!/usr/bin/env python3
"""
Example: Counterfactual query on trained ACIE model
"""

import torch
import pandas as pd
from pathlib import Path
from acie.training.train import ACIELightningModule

def run_example():
    # Load trained model
    checkpoint_path = "outputs/quickstart/acie_final.ckpt"
    
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train a model first using scripts/train_quickstart.py")
        return
    
    print("Loading ACIE model...")
    model = ACIELightningModule.load_from_checkpoint(checkpoint_path)
    engine = model.get_acie_engine()
    engine.eval()
    
    # Load some observational data
    data_path = "lib/acie_observational_10k_x_10k.csv"
    print(f"Loading observations from {data_path}...")
    
    # Load a single observation
    df = pd.read_csv(data_path, nrows=10)
    obs = torch.tensor(df.values[:5], dtype=torch.float32)  # Take 5 samples
    
    print(f"Observation shape: {obs.shape}")
    
    # Define interventions to test
    interventions = [
        {"mass": 1.5},
        {"mass": 2.0},
        {"metallicity": 0.02},
        {"environment": 1.5},
    ]
    
    results = []
    
    for intervention in interventions:
        print(f"\nIntervention: {intervention}")
        
        # Perform counterfactual query
        result = engine.counterfactual_query(obs, intervention)
        
        # Compute effect size
        effect = (result["counterfactual_obs"] - result["factual_obs"]).abs().mean()
        print(f"  Mean effect size: {effect.item():.6f}")
        
        results.append({
            "intervention": str(intervention),
            "effect_size": effect.item(),
        })
    
    # Print summary
    print("\n" + "="*60)
    print("Counterfactual Inference Results")
    print("="*60)
    
    for r in results:
        print(f"{r['intervention']:30s} Effect: {r['effect_size']:.6f}")
    
    print("\nExample complete!")

if __name__ == "__main__":
    run_example()
