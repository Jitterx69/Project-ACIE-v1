import torch
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from acie.logging.structured_logger import logger

@dataclass
class GlobalPhysicsState:
    total_mass: float = 0.0
    total_energy: float = 0.0
    virial_kinetic: float = 0.0
    virial_potential: float = 0.0
    tile_count: int = 0
    
    # Store min/max bounds observed across all tiles
    global_min: float = float('inf')
    global_max: float = float('-inf')

class GlobalPhysicsAggregator:
    """
    Aggregates physical quantities across streaming tiles to enforce 
    global laws (Conservation of Mass, Virial Equilibrium).
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = GlobalPhysicsState()
        self.density_map_preview = [] # Coarse-grained map if needed

    def update(self, tile_result: torch.Tensor, tile_box: tuple):
        """
        Update global state with result from a single tile.
        
        Args:
            tile_result: Decrypted tensor [OutputDim] or [Channels, H, W]
            tile_box: (x, y, w, h) coordinates of the tile
        """
        # For this example, we assume specific indices represent physical quantities.
        # This must align with the OutputDim of the generation model.
        # Let's assume:
        # Index 0: Mass/Density
        # Index 1: Velocity Magnitude
        # Index 2: Temperature
        
        if tile_result.numel() < 3:
            # Not enough channels for physics check
            return

        with torch.no_grad():
            # 1. Total Mass Integration (Sum of density)
            mass = tile_result[0].item() if tile_result.dim() == 1 else tile_result[0].sum().item()
            self.state.total_mass += max(0, mass) # Enforce physical positivity locally for sum
            
            # 2. Global Max/Min
            curr_max = tile_result.max().item()
            curr_min = tile_result.min().item()
            if curr_max > self.state.global_max: self.state.global_max = curr_max
            if curr_min < self.state.global_min: self.state.global_min = curr_min
            
            # 3. Virial Kinetic Energy Contribution (Approximation)
            # K ~ 0.5 * Mass * Velocity^2
            velocity = tile_result[1].item() if tile_result.dim() == 1 else tile_result[1].mean().item()
            kinetic = 0.5 * mass * (velocity ** 2)
            self.state.virial_kinetic += kinetic

            self.state.tile_count += 1
            
            # Log significant updates (e.g. every 10 tiles)
            if self.state.tile_count % 10 == 0:
                logger.debug(f"Physics Aggregator: Processed {self.state.tile_count} tiles. Total Mass: {self.state.total_mass:.2e}")

    def validate_global_constraints(self) -> Dict[str, float]:
        """
        Check constraints that apply to the whole image after processing is complete.
        
        Returns:
            Dictionary of violation metrics (0.0 means satisfied).
        """
        violations = {}
        
        # 1. Mass Positivity (Global)
        if self.state.total_mass <= 1e-6:
             violations['zero_mass_error'] = 1.0
        
        # 2. Virial Equilibrium Check (Global)
        # In a stable system, 2K + U = 0. 
        # Since we don't have full U (requires N^2 pairs), we estimation bounds.
        # This is a placeholder for the real astrophysical logic.
        
        # 3. Statistical Anomalies
        # If max is too high (artifact)
        if self.state.global_max > 1e6:
            violations['unphysical_flux_spike'] = self.state.global_max
            
        logger.info(f"Global Physics Validation Complete. Mass: {self.state.total_mass:.2e}, Max: {self.state.global_max:.2f}")
        return violations
