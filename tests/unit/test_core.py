"""
Unit tests for ACIE core components
"""

import pytest
import torch
import torch.nn as nn
from acie.core.scm import StructuralCausalModel, AstronomicalSCM
from acie.models.networks import Encoder, Decoder
from acie.inference.inference import LatentInferenceModel
from acie.inference.counterfactual import CounterfactualEngine


def test_scm_creation():
    """Test SCM creation and basic operations."""
    scm = StructuralCausalModel()
    
    # Add nodes
    scm.add_node("X", is_latent=True)
    scm.add_node("Y", is_latent=False)
    scm.add_node("Z", is_latent=False)
    
    # Add edges
    scm.add_edge("X", "Y")
    scm.add_edge("Y", "Z")
    
    assert len(scm.nodes) == 3
    assert len(scm.graph.edges()) == 2
    
    # Test topological order
    topo_order = scm.topological_order()
    assert topo_order.index("X") < topo_order.index("Y")
    assert topo_order.index("Y") < topo_order.index("Z")


def test_intervention():
    """Test intervention mechanism."""
    scm = StructuralCausalModel()
    scm.add_node("X")
    scm.add_node("Y")
    scm.add_edge("X", "Y")
    
    # Apply intervention
    intervened_scm = scm.intervene({"X": torch.tensor([1.0])})
    
    # Check that intervention is stored
    assert hasattr(intervened_scm, 'interventions')
    assert "X" in intervened_scm.interventions


def test_encoder_decoder():
    """Test encoder and decoder."""
    obs_dim = 100
    latent_dim = 50
    batch_size = 32
    
    encoder = Encoder(obs_dim, latent_dim)
    decoder = Decoder(latent_dim, obs_dim)
    
    # Forward pass
    obs = torch.randn(batch_size, obs_dim)
    mean, logvar = encoder(obs)
    
    assert mean.shape == (batch_size, latent_dim)
    assert logvar.shape == (batch_size, latent_dim)
    
    # Decode
    recon = decoder(mean)
    assert recon.shape == (batch_size, obs_dim)


def test_latent_inference():
    """Test latent inference model."""
    obs_dim = 100
    latent_dim = 50
    batch_size = 32
    
    model = LatentInferenceModel(obs_dim, latent_dim)
    
    obs = torch.randn(batch_size, obs_dim)
    recon, mean, logvar = model(obs)
    
    assert recon.shape == (batch_size, obs_dim)
    assert mean.shape == (batch_size, latent_dim)
    assert logvar.shape == (batch_size, latent_dim)


def test_counterfactual_engine():
    """Test counterfactual engine."""
    latent_dim = 50
    obs_dim = 100
    batch_size = 32
    
    engine = CounterfactualEngine(latent_dim, obs_dim)
    
    latent = torch.randn(batch_size, latent_dim)
    factual_latent = torch.randn(batch_size, latent_dim)
    
    cf_obs = engine(latent, factual_latent=factual_latent)
    
    assert cf_obs.shape == (batch_size, obs_dim)


def test_astronomical_scm():
    """Test astronomical-specific SCM."""
    scm = AstronomicalSCM(latent_dim=100, observable_dim=200, noise_dim=50)
    
    assert len(scm.nodes) == 350  # 100 + 200 + 50
    
    # Add physics structure
    scm.add_physics_structure(edge_probability=0.01)
    
    # Check that some edges were added
    assert len(scm.graph.edges()) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
