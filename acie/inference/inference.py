"""
Latent State Inference System

Implements P(P|O) - inferring latent physical states from observations.

Uses variational inference with physics-regularized posteriors:
    q_θ(P|O) ∝ p(O|P) * p_physics(P)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import torch.distributions as dist

from acie.models.networks import Encoder, DeepEncoder, Decoder


class LatentInferenceModel(nn.Module):
    """
    Latent state inference via variational autoencoder.
    
    Learns to infer posterior P(P|O) where:
    - P: Latent physical variables (mass, metallicity, age, etc.)
    - O: Observable variables (photometry, spectra, etc.)
    
    Architecture:
    - Encoder: q_θ(P|O) - variational posterior
    - Decoder: p_θ(O|P) - generative model
    - Prior: p(P) - physics-informed prior
    """
    
    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        encoder_type: str = "deep",
        encoder_hidden_dims: list = [1024, 512, 256],
        decoder_hidden_dims: list = [256, 512, 1024],
        prior_std: float = 1.0,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.prior_std = prior_std
        
        # Encoder: q_θ(P|O)
        if encoder_type == "deep":
            self.encoder = DeepEncoder(
                obs_dim=obs_dim,
                latent_dim=latent_dim,
                num_blocks=4,
                hidden_dim=512,
            )
        else:
            self.encoder = Encoder(
                obs_dim=obs_dim,
                latent_dim=latent_dim,
                hidden_dims=encoder_hidden_dims,
            )
        
        # Decoder: p_θ(O|P)
        self.decoder = Decoder(
            latent_dim=latent_dim,
            obs_dim=obs_dim,
            hidden_dims=decoder_hidden_dims,
        )
        
        # Prior: p(P) - standard normal for now
        self.register_buffer(
            "prior_loc",
            torch.zeros(latent_dim)
        )
        self.register_buffer(
            "prior_scale",
            torch.ones(latent_dim) * prior_std
        )
    
    def encode(self, obs: torch.Tensor) -> dist.Normal:
        """
        Encode observations to latent posterior.
        
        Args:
            obs: Observations, shape [batch, obs_dim]
            
        Returns:
            Posterior distribution q_θ(P|O)
        """
        return self.encoder.encode(obs)
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent state to observations.
        
        Args:
            latent: Latent state, shape [batch, latent_dim]
            
        Returns:
            Reconstructed observations, shape [batch, obs_dim]
        """
        return self.decoder(latent)
    
    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling.
        
        z = μ + σ * ε, where ε ~ N(0, 1)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(
        self,
        obs: torch.Tensor,
        return_latent: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode -> sample -> decode.
        
        Args:
            obs: Observations, shape [batch, obs_dim]
            return_latent: Whether to return latent samples
            
        Returns:
            - reconstruction: Reconstructed observations
            - mean: Posterior mean
            - logvar: Posterior log-variance
            - (optional) latent: Sampled latent state
        """
        # Encode
        mean, logvar = self.encoder(obs)
        
        # Sample latent
        latent = self.reparameterize(mean, logvar)
        
        # Decode
        reconstruction = self.decoder(latent)
        
        if return_latent:
            return reconstruction, mean, logvar, latent
        
        return reconstruction, mean, logvar
    
    def prior(self) -> dist.Normal:
        """Return prior distribution p(P)."""
        return dist.Normal(self.prior_loc, self.prior_scale)
    
    def compute_elbo(
        self,
        obs: torch.Tensor,
        beta: float = 1.0,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute Evidence Lower Bound (ELBO).
        
        ELBO = E_q[log p(O|P)] - β * KL(q(P|O) || p(P))
        
        Args:
            obs: Observations
            beta: KL weighting (β-VAE)
            
        Returns:
            - loss: Negative ELBO (to minimize)
            - info: Dict with loss components
        """
        # Forward pass
        reconstruction, mean, logvar = self.forward(obs)
        
        # Reconstruction loss: -log p(O|P)
        recon_loss = nn.functional.mse_loss(
            reconstruction, obs, reduction="mean"
        )
        
        # KL divergence: KL(q(P|O) || p(P))
        # For diagonal Gaussian: 0.5 * sum(1 + log(σ²) - μ² - σ²)
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)
        )
        
        # Total loss
        loss = recon_loss + beta * kl_loss
        
        info = {
            "loss": loss.item(),
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
        }
        
        return loss, info
    
    def sample_prior(self, num_samples: int, device: str = "cpu") -> torch.Tensor:
        """Sample from prior p(P)."""
        prior_dist = self.prior()
        return prior_dist.sample((num_samples,)).to(device)
    
    def reconstruct(self, obs: torch.Tensor) -> torch.Tensor:
        """Reconstruct observations (deterministic using mean)."""
        with torch.no_grad():
            mean, _ = self.encoder(obs)
            reconstruction = self.decoder(mean)
        return reconstruction


class PhysicsInformedInference(nn.Module):
    """
    Physics-informed latent inference.
    
    Extends basic VAE with physics constraints in the latent space.
    """
    
    def __init__(
        self,
        base_model: LatentInferenceModel,
        physics_constraint_weight: float = 0.1,
    ):
        super().__init__()
        
        self.base_model = base_model
        self.physics_constraint_weight = physics_constraint_weight
        
        # Learnable physics constraints
        from acie.models.physics_layers import DifferentiablePhysics
        self.physics_layer = DifferentiablePhysics(
            latent_dim=base_model.latent_dim,
            num_constraints=10,
        )
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with physics constraints."""
        return self.base_model(obs)
    
    def encode(self, obs: torch.Tensor):
        """Encode with physics-regularized posterior."""
        return self.base_model.encode(obs)
    
    def decode(self, latent: torch.Tensor):
        """Decode latent to observations."""
        return self.base_model.decode(latent)
    
    def compute_loss(
        self,
        obs: torch.Tensor,
        beta: float = 1.0,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute loss with physics constraints.
        
        Total loss = ELBO + λ * Physics_Penalty
        """
        # Base ELBO loss
        elbo_loss, info = self.base_model.compute_elbo(obs, beta)
        
        # Sample latent for physics check
        mean, logvar = self.base_model.encoder(obs)
        latent = self.base_model.reparameterize(mean, logvar)
        
        # Physics constraint penalty
        physics_penalty = self.physics_layer(latent)
        
        # Total loss
        total_loss = elbo_loss + self.physics_constraint_weight * physics_penalty
        
        info["physics_penalty"] = physics_penalty.item()
        info["total_loss"] = total_loss.item()
        
        return total_loss, info
