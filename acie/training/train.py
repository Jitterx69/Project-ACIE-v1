"""
Main Training Pipeline for ACIE

Uses PyTorch Lightning for:
- Distributed training
- Checkpointing
- Logging
- Mixed precision
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
from typing import Optional, Dict

from acie.inference.inference import LatentInferenceModel, PhysicsInformedInference
from acie.inference.counterfactual import CounterfactualEngine
from acie.training.losses import ACIELoss
from acie.models.physics_layers import DifferentiablePhysics


class ACIELightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for ACIE training.
    
    Handles:
    - Observational data (VAE training)
    - Counterfactual pairs (CF training)
    - Physics constraints
    - Multi-objective optimization
    """
    
    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        learning_rate: float = 1e-4,
        elbo_weight: float = 1.0,
        cf_weight: float = 0.5,
        physics_weight: float = 0.1,
        kl_beta: float = 1.0,
        use_physics_constraints: bool = True,
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        
        # Build models
        base_inference = LatentInferenceModel(
            obs_dim=obs_dim,
            latent_dim=latent_dim,
            encoder_type="deep",
        )
        
        if use_physics_constraints:
            self.inference_model = PhysicsInformedInference(
                base_model=base_inference,
                physics_constraint_weight=physics_weight,
            )
        else:
            self.inference_model = base_inference
        
        self.counterfactual_engine = CounterfactualEngine(
            latent_dim=latent_dim,
            obs_dim=obs_dim,
            use_twin_network=True,
        )
        
        # Loss function
        self.loss_fn = ACIELoss(
            elbo_weight=elbo_weight,
            cf_weight=cf_weight,
            physics_weight=physics_weight,
            kl_beta=kl_beta,
        )
        
        # Physics layer (optional)
        if use_physics_constraints:
            # Prefer CUDA implementation if available
            try:
                from acie.models.physics_layers import CUDAPhysicsConstraintLayer
                self.physics_layer = CUDAPhysicsConstraintLayer(
                    latent_dim=latent_dim,
                    energy_tolerance=1e-4,
                    momentum_tolerance=1e-4,
                    penalty_weight=1.0
                )
            except ImportError:
                # Fallback to differentiable physics
                self.physics_layer = DifferentiablePhysics(
                    latent_dim=latent_dim,
                    num_constraints=10,
                )
        else:
            self.physics_layer = None
    
    def forward(self, obs: torch.Tensor):
        """Forward pass through inference model."""
        return self.inference_model(obs)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step - handles both observational and counterfactual data."""
        
        # Check if counterfactual data is available
        if "factual_obs" in batch:
            return self._train_step_counterfactual(batch)
        else:
            return self._train_step_observational(batch)
    
    def _train_step_observational(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Training on observational data only (VAE)."""
        obs = batch["obs"]
        
        # Forward pass
        reconstruction, mean, logvar, latent = self.inference_model(
            obs, return_latent=True
        )
        
        # Compute loss
        loss, info = self.loss_fn(
            obs=obs,
            reconstruction=reconstruction,
            mean=mean,
            logvar=logvar,
            latent=latent,
            physics_layer=self.physics_layer,
        )
        
        # Log metrics
        self.log("train/loss", loss, prog_bar=True)
        for key, value in info.items():
            self.log(f"train/{key}", value)
        
        return loss
    
    def _train_step_counterfactual(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Training on paired counterfactual data."""
        factual_obs = batch["factual_obs"]
        counterfactual_obs = batch["counterfactual_obs"]
        factual_latent = batch["factual_latent"]
        counterfactual_latent = batch["counterfactual_latent"]
        
        # Infer factual latent
        recon, mean, logvar, inferred_latent = self.inference_model(
            factual_obs, return_latent=True
        )
        
        # Generate counterfactual prediction
        cf_pred = self.counterfactual_engine(
            latent=counterfactual_latent,
            observations=factual_obs,
            factual_latent=inferred_latent,
        )
        
        # Compute loss
        loss, info = self.loss_fn(
            obs=factual_obs,
            reconstruction=recon,
            mean=mean,
            logvar=logvar,
            counterfactual_obs_pred=cf_pred,
            counterfactual_obs_true=counterfactual_obs,
            latent=inferred_latent,
            physics_layer=self.physics_layer,
        )
        
        # Log metrics
        self.log("train/loss", loss, prog_bar=True)
        for key, value in info.items():
            self.log(f"train/{key}", value)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        if "factual_obs" in batch:
            obs = batch["factual_obs"]
        else:
            obs = batch["obs"]
        
        # Forward pass
        reconstruction, mean, logvar = self.inference_model(obs)
        
        # Compute ELBO only for validation
        elbo, info = self.loss_fn.compute_elbo(obs, reconstruction, mean, logvar)
        
        # Log metrics
        self.log("val/elbo", elbo, prog_bar=True)
        self.log("val/recon_loss", info["recon_loss"])
        self.log("val/kl_loss", info["kl_loss"])
        
        return elbo
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            verbose=True,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/elbo",
            },
        }
    
    def get_acie_engine(self):
        """Export trained ACIE engine for inference."""
        from acie.core.acie_core import ACIEEngine
        from acie.core.scm import AstronomicalSCM
        
        scm = AstronomicalSCM(
            latent_dim=self.latent_dim,
            observable_dim=self.obs_dim,
        )
        
        engine = ACIEEngine(
            scm=scm,
            inference_model=self.inference_model.base_model if hasattr(self.inference_model, 'base_model') else self.inference_model,
            counterfactual_engine=self.counterfactual_engine,
            device=self.device,
        )
        
        return engine


def train_acie(
    data_dir: Path,
    output_dir: Path,
    obs_dim: int = 6000,
    latent_dim: int = 2000,
    batch_size: int = 128,
    max_epochs: int = 100,
    learning_rate: float = 1e-4,
    dataset_size: str = "10k",
    use_counterfactual: bool = True,
    gpus: int = 1,
    num_workers: int = 4,
    fast_dev_run: bool = False,
):
    """
    Main training function for ACIE.
    
    Args:
        data_dir: Directory containing CSV datasets
        output_dir: Directory for checkpoints and logs
        obs_dim: Observable dimension
        latent_dim: Latent dimension
        batch_size: Training batch size
        max_epochs: Maximum training epochs
        learning_rate: Learning rate
        dataset_size: "10k" or "20k"
        use_counterfactual: Whether to use counterfactual pairs
        gpus: Number of GPUs
        num_workers: DataLoader workers
        fast_dev_run: Quick test run
    """
    # Create output directories
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Load data
    from acie.data.dataset import create_dataloaders
    
    dataloaders = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        dataset_size=dataset_size,
        max_rows=1000 if fast_dev_run else None,
    )
    
    # Select appropriate dataloader
    if use_counterfactual and "counterfactual" in dataloaders:
        train_loader = dataloaders["counterfactual"]
        val_loader = dataloaders.get("counterfactual")
    else:
        train_loader = dataloaders.get("observational")
        val_loader = dataloaders.get("observational")
    
    if train_loader is None:
        raise ValueError("No training data found!")
    
    # Create model
    model = ACIELightningModule(
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        learning_rate=learning_rate,
        elbo_weight=1.0,
        cf_weight=0.5 if use_counterfactual else 0.0,
        physics_weight=0.1,
        use_physics_constraints=True,
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="acie-{epoch:02d}-{val/elbo:.4f}",
            monitor="val/elbo",
            mode="min",
            save_top_k=3,
        ),
        EarlyStopping(
            monitor="val/elbo",
            patience=10,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=output_dir / "logs",
        name="acie_training",
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if gpus > 0 else "cpu",
        devices=gpus if gpus > 0 else 1,
        callbacks=callbacks,
        logger=logger,
        fast_dev_run=fast_dev_run,
        gradient_clip_val=1.0,
        precision="16-mixed" if gpus > 0 else 32,
    )
    
    # Train
    print("Starting ACIE training...")
    trainer.fit(model, train_loader, val_loader)
    
    # Save final model
    final_path = output_dir / "acie_final.ckpt"
    trainer.save_checkpoint(final_path)
    print(f"Training complete! Model saved to {final_path}")
    
    return model, trainer
