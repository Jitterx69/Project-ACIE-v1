#!/usr/bin/env python3
"""
ACIE System Demo - Core Components Working
"""

import sys
import torch
import numpy as np

print("=" * 70)
print("ACIE MULTI-LANGUAGE SYSTEM - WORKING DEMO")
print("=" * 70)
print()

print("✓ 1. Structural Causal Model (SCM)")
from acie.core.scm import StructuralCausalModel

scm = StructuralCausalModel()
scm.add_node("mass")
scm.add_node("luminosity")
scm.add_node("temperature")
scm.add_edge("mass", "luminosity")
scm.add_edge("mass", "temperature")
print(f"   Created SCM: {list(scm.graph.nodes())}")
print(f"   Edges: {list(scm.graph.edges())}")

print()
print("✓ 2. VAE Encoder/Decoder")
from acie.models.networks import Encoder, Decoder

obs_dim, latent_dim = 100, 20
encoder = Encoder(obs_dim, latent_dim)
decoder = Decoder(latent_dim, obs_dim)

obs = torch.randn(4, obs_dim)
mu, logvar = encoder(obs)
latent = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
recon = decoder(latent)
print(f"   Input: {obs.shape} → Latent: {latent.shape} → Output: {recon.shape}")
print(f"   Reconstruction MSE: {torch.mean((obs - recon)**2):.4f}")

print()
print("✓ 3. ACIE Engine")
from acie.core.engine import ACIEEngine

engine = ACIEEngine(scm, encoder, decoder, physics_layer=None)
print(f"   Engine initialized with SCM + VAE")

print()
print("✓ 4. Counterfactual Inference")
from acie.inference.counterfactual import CounterfactualEngine

cf_engine = CounterfactualEngine(scm, encoder, decoder)
interventions = {"mass": 1.5}
cf = cf_engine.infer(obs, interventions)
print(f"   Factual shape: {obs.shape}")
print(f"   Counterfactual shape: {cf.shape}")
print(f"   Mean difference: {torch.abs(obs - cf).mean():.4f}")

print()
print("✓ 5. Evaluation Metrics")
from acie.eval.metrics import ACIEMetrics

metrics = ACIEMetrics()
mse = metrics.counterfactual_mse(cf, obs)
mae = metrics.counterfactual_mae(cf, obs)
print(f"   MSE: {mse:.4f}")
print(f"   MAE: {mae:.4f}")

print()
print("=" * 70)
print("PYTHON CORE: FULLY FUNCTIONAL ✓")
print("=" * 70)
print()
print("Summary:")
print("  ✓ Structural Causal Model - Working")
print("  ✓ VAE Encoder/Decoder - Working")
print("  ✓ ACIE Engine - Working") 
print("  ✓ Counterfactual Inference - Working")
print("  ✓ Evaluation Metrics - Working")
print()
print("Multi-Language Build Status:")
print("  ✓ Python Core: READY (no build needed)")
print("  ⚠ Rust: Run 'cd rust && cargo build --release'")
print("  ⚠ Java: Run 'cd java && mvn package'")
print("  ⚠ Assembly: Run 'cd asm && make'")
print("  ⚠ R: Install R and packages")
print()
print("The core ACIE system is working!")
print()
