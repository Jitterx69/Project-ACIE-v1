"""
Physics-Constrained Layers

Implements differentiable physics constraints:
- Conservation laws (energy, momentum, etc.)
- Stability constraints
- Observational boundaries
- Soft and hard constraint enforcement
"""

import torch
import torch.nn as nn
from typing import Optional, List


class PhysicsConstraintLayer(nn.Module):
    """
    Base class for physics constraint layers.
    
    Constraints are enforced as soft penalties during training
    and can be used for hard projection during inference.
    """
    
    def __init__(self, penalty_weight: float = 1.0):
        super().__init__()
        self.penalty_weight = penalty_weight
    
    def compute_violation(self, *args, **kwargs) -> torch.Tensor:
        """Compute constraint violation. Override in subclass."""
        raise NotImplementedError
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Return penalty for violation."""
        violation = self.compute_violation(*args, **kwargs)
        return self.penalty_weight * violation


class ConservationLayer(nn.Module):
    """
    Enforce conservation laws on latent physical variables.
    
    For astronomy, relevant conservation laws include:
    - Total mass conservation
    - Energy conservation
    - Angular momentum conservation
    """
    
    def __init__(
        self,
        conserved_indices: List[int],
        penalty_weight: float = 1.0,
    ):
        super().__init__()
        self.conserved_indices = conserved_indices
        self.penalty_weight = penalty_weight
    
    def forward(
        self,
        latent_before: torch.Tensor,
        latent_after: torch.Tensor,
    ) -> torch.Tensor:
        """
        Penalize changes in conserved quantities.
        
        Args:
            latent_before: Latent state before intervention
            latent_after: Latent state after intervention/propagation
            
        Returns:
            Conservation violation penalty
        """
        # Extract conserved quantities
        conserved_before = latent_before[:, self.conserved_indices]
        conserved_after = latent_after[:, self.conserved_indices]
        
        # Compute violation (should be zero for perfect conservation)
        violation = torch.mean((conserved_before - conserved_after) ** 2)
        
        return self.penalty_weight * violation


class StabilityLayer(nn.Module):
    """
    Enforce dynamical stability constraints.
    
    For astronomical systems:
    - Orbital stability (e.g., Hill stability)
    - Virial equilibrium
    - Hydrostatic equilibrium
    """
    
    def __init__(
        self,
        stability_type: str = "virial",
        penalty_weight: float = 1.0,
    ):
        super().__init__()
        self.stability_type = stability_type
        self.penalty_weight = penalty_weight
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Compute stability violation.
        
        Args:
            latent: Latent physical state
            
        Returns:
            Stability violation penalty
        """
        if self.stability_type == "virial":
            # Virial theorem: 2K + U = 0 (for bound systems)
            # Approximate with latent variables
            kinetic = latent[:, :100].pow(2).sum(dim=1)
            potential = latent[:, 100:200].sum(dim=1)
            
            virial_violation = (2 * kinetic + potential).pow(2).mean()
            return self.penalty_weight * virial_violation
        
        elif self.stability_type == "bounds":
            # Physical bounds: certain quantities must be positive
            # e.g., mass, temperature, luminosity
            positive_violation = torch.relu(-latent[:, :500]).pow(2).mean()
            return self.penalty_weight * positive_violation
        
        else:
            return torch.tensor(0.0, device=latent.device)


class ObservationalBoundaryLayer(nn.Module):
    """
    Enforce observational boundaries and selection effects.
    
    Observations must lie within physically plausible ranges:
    - Flux limits (positive, within detector range)
    - Magnitude limits
    - Spectral line ratios
    """
    
    def __init__(
        self,
        obs_min: Optional[torch.Tensor] = None,
        obs_max: Optional[torch.Tensor] = None,
        penalty_weight: float = 1.0,
    ):
        super().__init__()
        self.penalty_weight = penalty_weight
        
        if obs_min is not None:
            self.register_buffer("obs_min", obs_min)
        else:
            self.obs_min = None
            
        if obs_max is not None:
            self.register_buffer("obs_max", obs_max)
        else:
            self.obs_max = None
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Penalize observations outside physical bounds.
        
        Args:
            observations: Generated observations
            
        Returns:
            Boundary violation penalty
        """
        violation = torch.tensor(0.0, device=observations.device)
        
        if self.obs_min is not None:
            min_violation = torch.relu(self.obs_min - observations).pow(2).mean()
            violation = violation + min_violation
        
        if self.obs_max is not None:
            max_violation = torch.relu(observations - self.obs_max).pow(2).mean()
            violation = violation + max_violation
        
        return self.penalty_weight * violation


class PhysicsConstraintValidator:
    """
    Validates physics constraints for inference.
    
    Used to detect and report constraint violations in generated
    counterfactuals.
    """
    
    def __init__(
        self,
        conservation_layer: Optional[ConservationLayer] = None,
        stability_layer: Optional[StabilityLayer] = None,
        boundary_layer: Optional[ObservationalBoundaryLayer] = None,
    ):
        self.conservation_layer = conservation_layer
        self.stability_layer = stability_layer
        self.boundary_layer = boundary_layer
    
    def __call__(
        self,
        latent: torch.Tensor,
        observations: torch.Tensor,
        latent_before: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Check all physics constraints.
        
        Returns:
            Boolean tensor indicating violations per sample
        """
        violations = torch.zeros(latent.shape[0], dtype=torch.bool, device=latent.device)
        
        if self.conservation_layer is not None and latent_before is not None:
            conservation_penalty = self.conservation_layer(latent_before, latent)
            violations |= (conservation_penalty > 0.01)
        
        if self.stability_layer is not None:
            stability_penalty = self.stability_layer(latent)
            violations |= (stability_penalty > 0.01)
        
        if self.boundary_layer is not None:
            boundary_penalty = self.boundary_layer(observations)
            violations |= (boundary_penalty > 0.01)
        
        return violations


class DifferentiablePhysics(nn.Module):
    """
    Differentiable physics layer that enforces physics through
    learned corrections.
    
    Instead of hard constraints, learns to adjust predictions to
    satisfy physics using gradient descent.
    """
    
    def __init__(
        self,
        latent_dim: int,
        num_constraints: int = 10,
    ):
        super().__init__()
        
        # Learn constraint functions
        self.constraint_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
            for _ in range(num_constraints)
        ])
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Compute physics constraint violations.
        
        Args:
            latent: Latent physical state
            
        Returns:
            Total constraint violation
        """
        violations = []
        for constraint_net in self.constraint_nets:
            violation = constraint_net(latent).pow(2)
            violations.append(violation)
        
        total_violation = torch.cat(violations, dim=1).mean()
        return total_violation
