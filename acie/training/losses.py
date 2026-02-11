"""
Loss Functions for ACIE Training

Implements:
1. Reconstruction loss (ELBO)
2. Counterfactual consistency loss
3. Physics constraint penalties
4. Causal identifiability regularization
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class ACIELoss(nn.Module):
    """
    Combined loss function for ACIE training.
    
    Total Loss = α*ELBO + β*CF_Loss + γ*Physics + δ*Identifiability
    
    Where:
    - ELBO: Reconstruction + KL divergence
    - CF_Loss: Counterfactual consistency
    - Physics: Constraint violations
    - Identifiability: Mutual information I(P; O_do(P))
    """
    
    def __init__(
        self,
        elbo_weight: float = 1.0,
        cf_weight: float = 0.5,
        physics_weight: float = 0.1,
        identifiability_weight: float = 0.01,
        kl_beta: float = 1.0,
    ):
        super().__init__()
        
        self.elbo_weight = elbo_weight
        self.cf_weight = cf_weight
        self.physics_weight = physics_weight
        self.identifiability_weight = identifiability_weight
        self.kl_beta = kl_beta
    
    def compute_elbo(
        self,
        obs: torch.Tensor,
        reconstruction: torch.Tensor,
        mean: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute ELBO loss."""
        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(reconstruction, obs, reduction="mean")
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)
        )
        
        # Total ELBO
        elbo = recon_loss + self.kl_beta * kl_loss
        
        return elbo, {
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
            "elbo": elbo.item(),
        }
    
    def compute_counterfactual_loss(
        self,
        factual_obs: torch.Tensor,
        counterfactual_obs_pred: torch.Tensor,
        counterfactual_obs_true: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute counterfactual prediction loss."""
        cf_loss = nn.functional.mse_loss(
            counterfactual_obs_pred,
            counterfactual_obs_true,
            reduction="mean"
        )
        
        # Measure intervention effect magnitude
        effect_magnitude = torch.abs(
            counterfactual_obs_pred - factual_obs
        ).mean()
        
        return cf_loss, {
            "cf_loss": cf_loss.item(),
            "effect_magnitude": effect_magnitude.item(),
        }
    
    def compute_physics_penalty(
        self,
        latent: torch.Tensor,
        physics_layer: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute physics constraint violation penalty."""
        if physics_layer is None:
            return torch.tensor(0.0, device=latent.device), {"physics_penalty": 0.0}
        
        penalty = physics_layer(latent)
        
        return penalty, {"physics_penalty": penalty.item()}
    
    def compute_identifiability_loss(
        self,
        latent_samples: torch.Tensor,
        obs_samples: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Estimate mutual information I(P; O) for identifiability.
        
        Uses MINE (Mutual Information Neural Estimation) approximation.
        Higher MI = better identifiability.
        
        We want to MAXIMIZE this, so return negative MI as loss.
        """
        # Simplified: use correlation as proxy
        # In practice, would use MINE or other MI estimators
        correlation = torch.corrcoef(
            torch.cat([latent_samples.mean(0, keepdim=True),
                      obs_samples.mean(0, keepdim=True)], dim=0)
        )[0, 1]
        
        # Negative because we want to maximize
        mi_loss = -torch.abs(correlation)
        
        return mi_loss, {
            "identifiability": -mi_loss.item(),
            "mi_proxy": correlation.item(),
        }
    
    def forward(
        self,
        obs: torch.Tensor,
        reconstruction: torch.Tensor,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        counterfactual_obs_pred: Optional[torch.Tensor] = None,
        counterfactual_obs_true: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
        physics_layer: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute total ACIE loss.
        
        Returns:
            - total_loss: Weighted sum of all losses
            - info: Dict with individual loss values
        """
        info = {}
        
        # ELBO loss (always computed)
        elbo, elbo_info = self.compute_elbo(obs, reconstruction, mean, logvar)
        info.update(elbo_info)
        total_loss = self.elbo_weight * elbo
        
        # Counterfactual loss (if data available)
        if counterfactual_obs_pred is not None and counterfactual_obs_true is not None:
            cf_loss, cf_info = self.compute_counterfactual_loss(
                obs, counterfactual_obs_pred, counterfactual_obs_true
            )
            info.update(cf_info)
            total_loss = total_loss + self.cf_weight * cf_loss
        
        # Physics penalty (if layer provided)
        if latent is not None:
            physics_loss, physics_info = self.compute_physics_penalty(
                latent, physics_layer
            )
            info.update(physics_info)
            total_loss = total_loss + self.physics_weight * physics_loss
        
        # Identifiability (if latent available)
        if latent is not None:
            id_loss, id_info = self.compute_identifiability_loss(latent, obs)
            info.update(id_info)
            total_loss = total_loss + self.identifiability_weight * id_loss
        
        info["total_loss"] = total_loss.item()
        
        return total_loss, info


class ContrastiveCausalLoss(nn.Module):
    """
    Contrastive loss for causal representation learning.
    
    Encourages:
    - Similar latents for similar observables
    - Different latents under interventions
    """
    
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        latent: torch.Tensor,
        latent_intervened: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss between factual and intervened latents.
        
        Args:
            latent: Factual latent representations
            latent_intervened: Intervened latent representations
            
        Returns:
            Contrastive loss
        """
        batch_size = latent.shape[0]
        
        # Compute similarity matrix
        similarity = torch.matmul(latent, latent_intervened.T) / self.temperature
        
        # Positive pairs: same sample before/after intervention
        positives = similarity.diag()
        
        # Contrastive loss: maximize distance between factual and intervened
        # (they should be different after intervention)
        loss = -torch.log(
            torch.exp(-positives) /
            (torch.exp(-similarity).sum(dim=1) + 1e-8)
        ).mean()
        
        return loss
