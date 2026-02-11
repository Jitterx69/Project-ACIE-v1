"""
Python bindings for CUDA physics constraint kernels
Provides PyTorch integration for custom CUDA operations
"""

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import sys
from pathlib import Path
import warnings

# Get the directory containing this file
CUDA_DIR = Path(__file__).parent

# Detect best available device
def get_best_device():
    """
    Detect the best available compute device.
    Priority: CUDA > MPS (Apple Silicon/Metal) > CPU
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS is available on macOS 12.3+ with PyTorch 1.12+
        return torch.device('mps')
    else:
        return torch.device('cpu')

# Global device
DEFAULT_DEVICE = get_best_device()

# Try to load CUDA extension
CUDA_AVAILABLE = False
physics_cuda = None

try:
    # First check if CUDA toolkit is available
    import subprocess
    nvcc_check = subprocess.run(['which', 'nvcc'], capture_output=True, text=True)
    has_nvcc = nvcc_check.returncode == 0
    
    if not has_nvcc:
        device_info = "CPU"
        if DEFAULT_DEVICE.type == 'mps':
            device_info = "Metal (Apple GPU)"
        print(f"ℹ️  CUDA Toolkit not detected - using {device_info} for physics constraints")
        print("   (This is normal on Apple Silicon Macs)")
    
    # Try to load the pre-built shared library
    lib_path = CUDA_DIR / 'physics_cuda.so'
    if lib_path.exists():
        # Load using ctypes (simpler than JIT compilation)
        import ctypes
        physics_cuda = ctypes.CDLL(str(lib_path))
        CUDA_AVAILABLE = True
        if has_nvcc:
            print("✓ Loaded physics kernels (CUDA version)")
        else:
            print("✓ Loaded physics kernels (CPU version)")
    else:
        print(f"⚠️  Physics kernel library not found at {lib_path}")
        print("   Run 'make' in the cuda/ directory to build it")
        print("   Falling back to pure PyTorch implementations")
        
except Exception as e:
    print(f"ℹ️  Could not load physics kernels: {e}")
    print("   Using pure PyTorch fallback implementations")
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
        use_cuda: bool = True,
        device: str = None
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.energy_tolerance = energy_tolerance
        self.momentum_tolerance = momentum_tolerance
        
        # Determine device to use
        if device:
            self.device = torch.device(device)
        else:
            self.device = DEFAULT_DEVICE
        
        # Only use C library for actual CUDA
        self.use_cuda = use_cuda and CUDA_AVAILABLE and torch.cuda.is_available()
        
        # Use PyTorch on MPS or CPU (better than C library)
        self.use_pytorch = self.device.type in ['mps', 'cpu']
    
    def enforce_energy_conservation(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Enforce energy conservation on latent states
        
        Args:
            latents: Tensor of shape (batch_size, latent_dim)
            
        Returns:
            Corrected latents with energy conservation enforced
        """
        # Move to appropriate device if needed
        if latents.device != self.device:
            latents = latents.to(self.device)
        
        if self.use_cuda and latents.is_cuda:
            # Use CUDA kernel (NVIDIA GPUs only)
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
            # PyTorch implementation (works on MPS, CPU, and as CUDA fallback)
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
        # Move to appropriate device if needed
        if latents.device != self.device:
            latents = latents.to(self.device)
        
        if self.use_cuda and latents.is_cuda:
            # Use CUDA kernel (NVIDIA GPUs only)
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
            # PyTorch implementation (works on MPS, CPU, and as CUDA fallback)
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
        # Move to appropriate device if needed
        if latents.device != self.device:
            latents = latents.to(self.device)
        
        if self.use_cuda and latents.is_cuda:
            # Use combined CUDA kernel (NVIDIA GPUs only)
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
            # PyTorch implementation (works on MPS, CPU, and as CUDA fallback)
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
