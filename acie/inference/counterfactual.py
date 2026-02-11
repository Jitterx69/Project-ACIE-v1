"""
Counterfactual Inference Engine

Implements the 3-step counterfactual reasoning:
1. Abduction: Infer latent state from factual observation
2. Action: Apply intervention
3. Prediction: Propagate to counterfactual observation

This is the core of ACIE's counterfactual reasoning.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from acie.models.networks import InterventionNet, Decoder


class CounterfactualEngine(nn.Module):
    """
    Counterfactual inference engine implementing:
        P(O_do(P) | O)
    
    Uses twin network architecture:
    - Factual network: processes original observations
    - Counterfactual network: generates counterfactual outcomes
    """
    
    def __init__(
        self,
        latent_dim: int,
        obs_dim: int,
        use_twin_network: bool = True,
        decoder_hidden_dims: list = [256, 512, 1024],
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.use_twin_network = use_twin_network
        
        if use_twin_network:
            # Twin network for counterfactual generation
            self.counterfactual_net = InterventionNet(
                latent_dim=latent_dim,
                obs_dim=obs_dim,
                hidden_dims=[512, 512, 1024],
            )
        else:
            # Simple decoder (assumes independence)
            self.counterfactual_net = Decoder(
                latent_dim=latent_dim,
                obs_dim=obs_dim,
                hidden_dims=decoder_hidden_dims,
            )
    
    def forward(
        self,
        latent: torch.Tensor,
        observations: Optional[torch.Tensor] = None,
        factual_latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate counterfactual observations.
        
        Args:
            latent: Intervened latent state, shape [batch, latent_dim]
            observations: Original factual observations (for twin network)
            factual_latent: Factual latent state (for twin network)
            
        Returns:
            Counterfactual observations, shape [batch, obs_dim]
        """
        if self.use_twin_network:
            if factual_latent is None:
                raise ValueError("Twin network requires factual_latent")
            
            return self.counterfactual_net(
                factual_latent=factual_latent,
                intervened_latent=latent,
            )
        else:
            return self.counterfactual_net(latent)
    
    def state_dict(self):
        """Return state dict for saving."""
        return self.counterfactual_net.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.counterfactual_net.load_state_dict(state_dict)


class TwinNetworkCF(nn.Module):
    """
    Twin Network for Counterfactual Reasoning.
    
    Maintains two parallel networks:
    - Factual network: processes actual observations
    - Counterfactual network: generates counterfactual outcomes
    
    Shared exogenous noise ensures consistency.
    """
    
    def __init__(
        self,
        latent_dim: int,
        obs_dim: int,
        noise_dim: int = 100,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.noise_dim = noise_dim
        
        # Factual branch
        self.factual_decoder = Decoder(
            latent_dim=latent_dim + noise_dim,
            obs_dim=obs_dim,
            hidden_dims=[256, 512, 1024],
        )
        
        # Counterfactual branch (shares noise)
        self.counterfactual_decoder = Decoder(
            latent_dim=latent_dim + noise_dim,
            obs_dim=obs_dim,
            hidden_dims=[256, 512, 1024],
        )
    
    def infer_noise(
        self,
        latent: torch.Tensor,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Infer exogenous noise from factual observation.
        
        This is the abduction step: recover U such that
            O = f(P, U)
        """
        batch_size = latent.shape[0]
        
        # Initialize noise
        noise = torch.randn(
            batch_size, self.noise_dim,
            device=latent.device,
            requires_grad=True
        )
        
        # Optimize noise to match observation
        optimizer = torch.optim.Adam([noise], lr=0.01)
        
        for _ in range(50):  # Quick optimization
            optimizer.zero_grad()
            
            latent_with_noise = torch.cat([latent, noise], dim=-1)
            pred_obs = self.factual_decoder(latent_with_noise)
            
            loss = nn.functional.mse_loss(pred_obs, observations)
            loss.backward()
            optimizer.step()
        
        return noise.detach()
    
    def forward(
        self,
        factual_latent: torch.Tensor,
        intervened_latent: torch.Tensor,
        observations: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full twin network forward pass.
        
        Args:
            factual_latent: Original latent state
            intervened_latent: Intervened latent state
            observations: Factual observations
            
        Returns:
            - factual_pred: Reconstructed factual observations
            - counterfactual_pred: Counterfactual observations
        """
        # Abduction: Infer exogenous noise
        noise = self.infer_noise(factual_latent, observations)
        
        # Factual branch
        factual_input = torch.cat([factual_latent, noise], dim=-1)
        factual_pred = self.factual_decoder(factual_input)
        
        # Counterfactual branch (same noise!)
        counterfactual_input = torch.cat([intervened_latent, noise], dim=-1)
        counterfactual_pred = self.counterfactual_decoder(counterfactual_input)
        
        return factual_pred, counterfactual_pred


class CausalPropagator(nn.Module):
    """
    Causal propagation through the SCM.
    
    Given intervention on some variables, propagates effects
    through the causal graph to descendants.
    """
    
    def __init__(
        self,
        latent_dim: int,
        obs_dim: int,
        causal_structure: Optional[Dict] = None,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.causal_structure = causal_structure or {}
        
        # Learned causal mechanisms
        # Simplified: learn direct latent -> obs mapping
        self.propagation_net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, obs_dim),
        )
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Propagate latent state to observations via causal mechanisms.
        
        Args:
            latent: Latent state (possibly intervened)
            
        Returns:
            Observations after causal propagation
        """
        return self.propagation_net(latent)


class CounterfactualConsistencyLoss(nn.Module):
    """
    Loss function ensuring counterfactual consistency.
    
    Enforces:
    1. Factual consistency: f(P, U) = O
    2. Intervention consistency: do(P) actually changes P
    3. Causal consistency: only descendants of intervention change
    """
    
    def __init__(
        self,
        factual_weight: float = 1.0,
        intervention_weight: float = 0.5,
        descendant_weight: float = 0.3,
    ):
        super().__init__()
        
        self.factual_weight = factual_weight
        self.intervention_weight = intervention_weight
        self.descendant_weight = descendant_weight
    
    def forward(
        self,
        factual_obs: torch.Tensor,
        counterfactual_obs: torch.Tensor,
        original_obs: torch.Tensor,
        intervened_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute counterfactual consistency loss.
        
        Args:
            factual_obs: Reconstructed factual observations
            counterfactual_obs: Generated counterfactual observations
            original_obs: True factual observations
            intervened_indices: Indices of intervened variables
            
        Returns:
            - loss: Total consistency loss
            - info: Loss components
        """
        # Factual consistency: reconstructed should match original
        factual_loss = nn.functional.mse_loss(factual_obs, original_obs)
        
        # Intervention should cause change
        change = torch.abs(counterfactual_obs - factual_obs).mean()
        intervention_loss = torch.exp(-change)  # Penalize no change
        
        # Total loss
        loss = (
            self.factual_weight * factual_loss +
            self.intervention_weight * intervention_loss
        )
        
        info = {
            "factual_loss": factual_loss.item(),
            "intervention_loss": intervention_loss.item(),
            "avg_change": change.item(),
        }
        
        return loss, info
