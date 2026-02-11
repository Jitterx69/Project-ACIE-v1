"""
Integration tests for ACIE training pipeline
"""

import pytest
import torch
from pathlib import Path
from acie.training.train import ACIELightningModule
from acie.data.dataset import ACIEDataset


def test_dataset_loading():
    """Test dataset loading from CSV."""
    data_path = Path("lib/acie_observational_10k_x_10k.csv")
    
    if not data_path.exists():
        pytest.skip("Dataset not available")
    
    dataset = ACIEDataset(
        data_path,
        latent_dim=2000,
        obs_dim=6000,
        noise_dim=2000,
        max_rows=100,  # Load only 100 rows for test
    )
    
    assert len(dataset) == 100
    
    sample = dataset[0]
    assert "latent" in sample
    assert "obs" in sample
    assert sample["obs"].shape == (6000,)


def test_model_forward_pass():
    """Test model forward pass."""
    model = ACIELightningModule(
        obs_dim=100,
        latent_dim=50,
        use_physics_constraints=False,
    )
    
    batch = {
        "obs": torch.randn(16, 100),
    }
    
    loss = model.training_step(batch, 0)
    
    assert loss is not None
    assert loss.requires_grad


def test_counterfactual_training_step():
    """Test training step with counterfactual data."""
    model = ACIELightningModule(
        obs_dim=100,
        latent_dim=50,
        cf_weight=0.5,
    )
    
    batch = {
        "factual_obs": torch.randn(16, 100),
        "counterfactual_obs": torch.randn(16, 100),
        "factual_latent": torch.randn(16, 50),
        "counterfactual_latent": torch.randn(16, 50),
    }
    
    loss = model.training_step(batch, 0)
    
    assert loss is not None
    assert loss.requires_grad


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
