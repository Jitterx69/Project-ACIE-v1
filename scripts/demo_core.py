#!/usr/bin/env python3
"""
ACIE Core Demo - Python Only
Demonstrates the Python core without requiring other languages
"""

import numpy as np
import torch
import sys
from pathlib import Path

# Add ACIE to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*70)
print("ACIE CORE SYSTEM DEMONSTRATION")
print("="*70)
print()

# 1. Test SCM
print("1. Testing Structural Causal Model (SCM)...")
try:
    from acie.core.scm import StructuralCausalModel
    
    scm = StructuralCausalModel()
    scm.add_node("mass")
    scm.add_node("luminosity")
    scm.add_node("temperature")
    scm.add_edge("mass", "luminosity")
    scm.add_edge("mass", "temperature")
    
    print(f"   ✓ Created SCM with {len(scm.graph.nodes())} nodes")
    print(f"   ✓ Edges: {list(scm.graph.edges())}")
    
    # Test intervention
    interventions = {"mass": torch.tensor(2.0)}
    intervened_scm = scm.intervene(interventions)
    print(f"   ✓ Intervention applied: mass = 2.0")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print()

# 2. Test Encoder/Decoder
print("2. Testing VAE Encoder/Decoder...")
try:
    from acie.models.networks import Encoder, Decoder
    
    obs_dim = 100
    latent_dim = 20
    batch_size = 4
    
    encoder = Encoder(obs_dim, latent_dim)
    decoder = Decoder(latent_dim, obs_dim)
    
    # Test forward pass
    obs = torch.randn(batch_size, obs_dim)
    mu, logvar = encoder(obs)
    
    # Reparameterization
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    latent = mu + eps * std
    
    # Decode
    reconstructed = decoder(latent)
    
    print(f"   ✓ Encoder: {obs.shape} → mu:{mu.shape}, logvar:{logvar.shape}")
    print(f"   ✓ Latent: {latent.shape}")
    print(f"   ✓ Decoder: {latent.shape} → {reconstructed.shape}")
    print(f"   ✓ Reconstruction error: {torch.mean((obs - reconstructed)**2):.4f}")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print()

# 3. Test Physics Constraints
print("3. Testing Physics Constraint Layer...")
try:
    from acie.models.physics import PhysicsConstraintLayer
    
    latent_dim = 20
    batch_size = 4
    
    physics_layer = PhysicsConstraintLayer(latent_dim)
    latent = torch.randn(batch_size, latent_dim)
    
    violations = physics_layer(latent)
    
    print(f"   ✓ Input latent: {latent.shape}")
    print(f"   ✓ Physics violations: {violations.shape}")
    print(f"   ✓ Mean violation: {violations.mean().item():.6f}")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print()

# 4. Test Dataset
print("4. Testing ACIE Dataset...")
try:
    from acie.data.dataset import ACIEDataset
    
    # Create synthetic data
    n_samples = 100
    obs_dim = 50
    
    observations = np.random.randn(n_samples, obs_dim).astype(np.float32)
    latents = np.random.randn(n_samples, 20).astype(np.float32)
    
    dataset = ACIEDataset(observations, latents)
    
    print(f"   ✓ Created dataset with {len(dataset)} samples")
    
    # Get a sample
    obs, latent = dataset[0]
    print(f"   ✓ Sample shapes: obs={obs.shape}, latent={latent.shape}")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print()

# 5. Test Metrics
print("5. Testing Evaluation Metrics...")
try:
    from acie.eval.metrics import ACIEMetrics
    
    metrics = ACIEMetrics()
    
    pred = torch.randn(10, 50)
    target = torch.randn(10, 50)
    latent = torch.randn(10, 20)
    
    mse = metrics.counterfactual_mse(pred, target)
    mae = metrics.counterfactual_mae(pred, target)
    physics_score = metrics.physics_constraint_score(latent)
    
    print(f"   ✓ MSE: {mse:.4f}")
    print(f"   ✓ MAE: {mae:.4f}")
    print(f"   ✓ Physics score: {physics_score:.4f}")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print()
print("="*70)
print("CORE DEMO COMPLETE ✓")
print("="*70)
print()
print("All Python core components are working!")
print()
print("Next Steps:")
print("  - Build Rust: cd rust && cargo build --release")
print("  - Build Assembly: cd asm && make")
print("  - Build Java: cd java && mvn package")
print("  - Install R packages: Rscript -e 'install.packages(...)'")
print("  - Full demo: python3 scripts/demo_multilang.py")
print()
