"""
Neural Network Architectures for ACIE

Implements:
- Encoder: O → P (observation to latent)
- Decoder: P → O (latent to observation)  
- Causal Mechanism Networks: f_i(Pa(X_i), U_i)
- Intervention-aware architectures
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class MLP(nn.Module):
    """Multi-layer perceptron with flexible architecture."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list = [512, 512],
        activation: str = "relu",
        dropout: float = 0.1,
        batch_norm: bool = True,
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self._get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def _get_activation(self, name: str):
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.2),
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
        }
        return activations.get(name, nn.ReLU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Encoder(nn.Module):
    """
    Encoder network: q_θ(P|O)
    
    Maps observations to latent physical state posterior.
    Uses variational inference with diagonal Gaussian posterior.
    """
    
    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        hidden_dims: list = [1024, 512, 256],
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        
        # Shared encoder backbone
        self.backbone = MLP(
            input_dim=obs_dim,
            output_dim=hidden_dims[-1],
            hidden_dims=hidden_dims[:-1],
            dropout=dropout,
        )
        
        # Mean and log-variance heads
        self.mean_head = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar_head = nn.Linear(hidden_dims[-1], latent_dim)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode observations to latent posterior parameters.
        
        Args:
            obs: Observations, shape [batch, obs_dim]
            
        Returns:
            - mean: Posterior mean, shape [batch, latent_dim]
            - logvar: Posterior log-variance, shape [batch, latent_dim]
        """
        h = self.backbone(obs)
        mean = self.mean_head(h)
        logvar = self.logvar_head(h)
        
        return mean, logvar
    
    def encode(self, obs: torch.Tensor) -> torch.distributions.Normal:
        """Return posterior distribution."""
        mean, logvar = self.forward(obs)
        std = torch.exp(0.5 * logvar)
        return torch.distributions.Normal(mean, std)


class Decoder(nn.Module):
    """
    Decoder network: p_θ(O|P)
    
    Maps latent physical state to observations.
    This is the learned generative model.
    """
    
    def __init__(
        self,
        latent_dim: int,
        obs_dim: int,
        hidden_dims: list = [256, 512, 1024],
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        
        self.network = MLP(
            input_dim=latent_dim,
            output_dim=obs_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent state to observations.
        
        Args:
            latent: Latent physical state, shape [batch, latent_dim]
            
        Returns:
            obs: Reconstructed observations, shape [batch, obs_dim]
        """
        return self.network(latent)


class CausalMechanismNet(nn.Module):
    """
    Learns individual causal mechanism: X_i = f_i(Pa(X_i), U_i)
    
    Used within the SCM for causal propagation.
    """
    
    def __init__(
        self,
        parent_dim: int,
        noise_dim: int,
        output_dim: int,
        hidden_dims: list = [128, 128],
    ):
        super().__init__()
        
        self.network = MLP(
            input_dim=parent_dim + noise_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            dropout=0.0,  # No dropout for causal mechanisms
        )
    
    def forward(
        self,
        parents: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply causal mechanism.
        
        Args:
            parents: Parent variable values
            noise: Exogenous noise
            
        Returns:
            Output variable value
        """
        inputs = torch.cat([parents, noise], dim=-1)
        return self.network(inputs)


class InterventionNet(nn.Module):
    """
    Intervention-aware network for counterfactual generation.
    
    Takes both factual and intervened latent states to generate
    counterfactual observations.
    """
    
    def __init__(
        self,
        latent_dim: int,
        obs_dim: int,
        hidden_dims: list = [512, 512, 1024],
    ):
        super().__init__()
        
        # Process factual latent
        self.factual_encoder = MLP(
            input_dim=latent_dim,
            output_dim=hidden_dims[0],
            hidden_dims=[],
            dropout=0.0,
        )
        
        # Process intervened latent
        self.intervened_encoder = MLP(
            input_dim=latent_dim,
            output_dim=hidden_dims[0],
            hidden_dims=[],
            dropout=0.0,
        )
        
        # Combine and decode
        self.decoder = MLP(
            input_dim=hidden_dims[0] * 2,
            output_dim=obs_dim,
            hidden_dims=hidden_dims[1:],
            dropout=0.1,
        )
    
    def forward(
        self,
        factual_latent: torch.Tensor,
        intervened_latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate counterfactual observations.
        
        Args:
            factual_latent: Original latent state
            intervened_latent: Intervened latent state
            
        Returns:
            Counterfactual observations
        """
        h_factual = self.factual_encoder(factual_latent)
        h_intervened = self.intervened_encoder(intervened_latent)
        
        combined = torch.cat([h_factual, h_intervened], dim=-1)
        return self.decoder(combined)


class ResidualBlock(nn.Module):
    """Residual block for deeper networks."""
    
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.layers(x))


class DeepEncoder(nn.Module):
    """
    Deeper encoder with residual connections for complex mappings.
    """
    
    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        num_blocks: int = 4,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Initial projection
        self.input_proj = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout)
            for _ in range(num_blocks)
        ])
        
        # Output heads
        self.mean_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(obs)
        
        for block in self.blocks:
            h = block(h)
        
        mean = self.mean_head(h)
        logvar = self.logvar_head(h)
        
        return mean, logvar
    
    def encode(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean, logvar = self.forward(obs)
        std = torch.exp(0.5 * logvar)
        return torch.distributions.Normal(mean, std)
