"""
Python bindings for CUDA physics constraint kernels
Provides PyTorch integration for custom CUDA operations
"""

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
from pathlib import Path

# Get the directory containing this file
CUDA_DIR = Path(__file__).parent

# Load CUDA extension
try:
    physics_cuda = load(
        name='physics_cuda',
        sources=[
            str(CUDA_DIR / 'physics_constraints.cu'),
        ],
        extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_70'],
        verbose=True
    )
    CUDA_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not load CUDA kernels: {e}")
    print("Falling back to PyTorch implementations")
    CUDA_AVAILABLE = False


class PhysicsConstraints(nn.Module):
    """
    PyTorch module wrapping CUDA physics constraint kernels
    """
    
    def __init__(
        self,
        latent_dim: int = 2000,
        energy_tolerance: float = 1e-4,
        momentum_tolerance: float = 1e-4,
        use_cuda: bool = True
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.energy_tolerance = energy_tolerance
        self.momentum_tolerance = momentum_tolerance
        self.use_cuda = use_cuda and CUDA_AVAILABLE and torch.cuda.is_available()
    
    def enforce_energy_conservation(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Enforce energy conservation on latent states
        
        Args:
            latents: Tensor of shape (batch_size, latent_dim)
            
        Returns:
            Corrected latents with energy conservation enforced
        """
        if self.use_cuda and latents.is_cuda:
            # Use CUDA kernel
            corrected = torch.empty_like(latents)
            physics_cuda.enforce_energy_conservation(
                latents.contiguous(),
                corrected,
                latents.size(0),
                latents.size(1),
                self.energy_tolerance
            )
            return corrected
        else:
            # PyTorch fallback
            energy = torch.sum(latents ** 2, dim=-1, keepdim=True)
            scale = torch.where(
                energy > self.energy_tolerance,
                torch.sqrt(self.energy_tolerance / energy),
                torch.ones_like(energy)
            )
            return latents * scale
    
    def enforce_momentum_conservation(
        self,
        latents: torch.Tensor,
        momentum_start_idx: int = 0,
        momentum_dim: int = 3
    ) -> torch.Tensor:
        """
        Enforce momentum conservation on latent states
        
        Args:
            latents: Tensor of shape (batch_size, latent_dim)
            momentum_start_idx: Starting index of momentum components
            momentum_dim: Number of momentum dimensions
            
        Returns:
            Corrected latents with momentum conservation enforced
        """
        if self.use_cuda and latents.is_cuda:
            # Use CUDA kernel
            corrected = torch.empty_like(latents)
            physics_cuda.enforce_momentum_conservation(
                latents.contiguous(),
                corrected,
                latents.size(0),
                latents.size(1),
                momentum_start_idx,
                momentum_dim
            )
            return corrected
        else:
            # PyTorch fallback
            corrected = latents.clone()
            momentum_slice = latents[:, momentum_start_idx:momentum_start_idx + momentum_dim]
            mean_momentum = torch.mean(momentum_slice, dim=0, keepdim=True)
            corrected[:, momentum_start_idx:momentum_start_idx + momentum_dim] -= mean_momentum
            return corrected
    
    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Apply all physics constraints
        
        Args:
            latents: Tensor of shape (batch_size, latent_dim)
            
        Returns:
            Corrected latents with all constraints enforced
        """
        if self.use_cuda and latents.is_cuda:
            # Use combined CUDA kernel
            corrected = torch.empty_like(latents)
            physics_cuda.enforce_physics_constraints(
                latents.contiguous(),
                corrected,
                latents.size(0),
                latents.size(1),
                self.energy_tolerance,
                self.momentum_tolerance
            )
            return corrected
        else:
            # Sequential PyTorch fallback
            latents = self.enforce_energy_conservation(latents)
            latents = self.enforce_momentum_conservation(latents)
            return latents


class CUDAMatMul(nn.Module):
    """
    Custom CUDA matrix multiplication with optimizations
    """
    
    def __init__(self, use_cuda: bool = True):
        super().__init__()
        self.use_cuda = use_cuda and CUDA_AVAILABLE and torch.cuda.is_available()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Perform matrix multiplication C = A @ B
        
        Args:
            A: Tensor of shape (M, K)
            B: Tensor of shape (K, N)
            
        Returns:
            C: Tensor of shape (M, N)
        """
        if self.use_cuda and A.is_cuda and B.is_cuda:
            M, K = A.shape
            K2, N = B.shape
            assert K == K2, "Matrix dimensions must match"
            
            C = torch.empty(M, N, device=A.device, dtype=A.dtype)
            physics_cuda.cuda_matmul(
                A.contiguous(),
                B.contiguous(),
                C,
                M, N, K
            )
            return C
        else:
            return torch.matmul(A, B)


# Convenience functions
def enforce_energy_conservation(latents: torch.Tensor, tolerance: float = 1e-4) -> torch.Tensor:
    """Apply energy conservation constraint"""
    constraint = PhysicsConstraints(
        latent_dim=latents.size(-1),
        energy_tolerance=tolerance
    )
    return constraint.enforce_energy_conservation(latents)


def enforce_momentum_conservation(latents: torch.Tensor, tolerance: float = 1e-4) -> torch.Tensor:
    """Apply momentum conservation constraint"""
    constraint = PhysicsConstraints(
        latent_dim=latents.size(-1),
        momentum_tolerance=tolerance
    )
    return constraint.enforce_momentum_conservation(latents)


def enforce_physics_constraints(latents: torch.Tensor) -> torch.Tensor:
    """Apply all physics constraints"""
    constraint = PhysicsConstraints(latent_dim=latents.size(-1))
    return constraint(latents)
