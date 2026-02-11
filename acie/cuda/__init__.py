"""ACIE CUDA module"""

try:
    from .cuda_physics import (
        PhysicsConstraints,
        CUDAMatMul,
        enforce_energy_conservation,
        enforce_momentum_conservation,
        enforce_physics_constraints
    )
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("CUDA physics extension not available in acie.cuda")
