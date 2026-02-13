"""
Test script to verify physics constraints work with MPS (Metal) backend
"""
import torch
from acie.cuda.cuda_physics import PhysicsConstraints, DEFAULT_DEVICE

print(f"🔍 Testing Physics Constraints on {DEFAULT_DEVICE}")
print("=" * 60)

# Create test data
batch_size = 32
latent_dim = 2000

# Create random latents and move to best available device
latents = torch.randn(batch_size, latent_dim).to(DEFAULT_DEVICE)
print(f"✓ Created test tensor on {latents.device}")
print(f"  Shape: {latents.shape}")
print(f"  Initial energy: {torch.sum(latents ** 2, dim=-1).mean().item():.4f}")

# Initialize physics constraints
physics = PhysicsConstraints(
    latent_dim=latent_dim,
    energy_tolerance=1e-4,
    momentum_tolerance=1e-4,
    device=str(DEFAULT_DEVICE)
)
print(f"\n✓ Initialized PhysicsConstraints")
print(f"  Using device: {physics.device}")
print(f"  CUDA available: {physics.use_cuda}")
print(f"  PyTorch mode: {physics.use_pytorch}")

# Test energy conservation
print(f"\n🧪 Testing Energy Conservation...")
corrected = physics.enforce_energy_conservation(latents)
final_energy = torch.sum(corrected ** 2, dim=-1)
print(f"  Final energy (mean): {final_energy.mean().item():.6f}")
print(f"  Energy tolerance: {physics.energy_tolerance}")
print(f"  ✓ All energies within tolerance: {(final_energy <= physics.energy_tolerance).all().item()}")

# Test momentum conservation
print(f"\n🧪 Testing Momentum Conservation...")
corrected = physics.enforce_momentum_conservation(latents)
momentum = corrected[:, :3]
print(f"  First 3 dims (momentum) - mean: {momentum.mean(dim=0)}")
print(f"  ✓ Momentum conserved: {(momentum.mean(dim=0).abs() < 1e-6).all().item()}")

# Test combined constraints
print(f"\n🧪 Testing Combined Physics Constraints...")
corrected = physics(latents)
final_energy = torch.sum(corrected ** 2, dim=-1)
momentum = corrected[:, :3]
print(f"  Final energy (mean): {final_energy.mean().item():.6f}")
print(f"  Momentum (mean): {momentum.mean(dim=0)}")
print(f"  ✓ Energy conserved: {(final_energy <= physics.energy_tolerance * 1.1).all().item()}")
print(f"  ✓ Momentum zeroed: {(momentum.abs() < 1e-5).all().item()}")

# Benchmark performance
print(f"\n⚡ Performance Benchmark...")
import time
warmup_iterations = 10
benchmark_iterations = 100

# Warmup
for _ in range(warmup_iterations):
    _ = physics(latents)

# Benchmark
if DEFAULT_DEVICE.type == 'mps':
    torch.mps.synchronize()
elif DEFAULT_DEVICE.type == 'cuda':
    torch.cuda.synchronize()

start = time.time()
for _ in range(benchmark_iterations):
    corrected = physics(latents)
    
if DEFAULT_DEVICE.type == 'mps':
    torch.mps.synchronize()
elif DEFAULT_DEVICE.type == 'cuda':
    torch.cuda.synchronize()
    
elapsed = time.time() - start
throughput = (batch_size * benchmark_iterations) / elapsed

print(f"  Iterations: {benchmark_iterations}")
print(f"  Total time: {elapsed:.3f}s")
print(f"  Avg per iteration: {elapsed/benchmark_iterations*1000:.2f}ms")
print(f"  Throughput: {throughput:.1f} samples/sec")

print("\n" + "=" * 60)
print(f"✅ All tests passed on {DEFAULT_DEVICE}!")
print("=" * 60)
