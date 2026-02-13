"""
Logistic Regression / Propensity Score Models for ACIE.

In Causal Inference, "Propensity Score" is the probability of treatment assignment
conditioned on observed covariates: e(x) = P(T=1 | X=x).

This is traditionally modeled using Logistic Regression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LogisticPropensityModel(nn.Module):
    """
    Standard Logistic Regression model.
    Linear layer followed by Sigmoid activation.
    
    P(y=1|x) = σ(Mx + b)
    """
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        
        # Single linear layer (weights M, bias b)
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features (e.g. latent variables or observations)
               Shape: [batch_size, input_dim]
               
        Returns:
            Probability of positive class (value between 0 and 1)
            Shape: [batch_size, 1]
        """
        # Linear transformation z = Mx + b
        logits = self.linear(x)
        
        # Sigmoid activation σ(z) = 1 / (1 + exp(-z))
        probs = torch.sigmoid(logits)
        
        return probs

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Predict binary class labels.
        """
        probs = self.forward(x)
        return (probs >= threshold).float()

class DeepPropensityModel(nn.Module):
    """
    Multi-layer Perceptron for non-linear propensity estimation.
    For comparison with Logistic Regression.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
