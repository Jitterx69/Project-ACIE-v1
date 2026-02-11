"""
Evaluation Metrics for ACIE

Metrics for assessing:
1. Counterfactual prediction accuracy
2. Physics violation rate
3. Intervention consistency
4. Latent reconstruction quality
5. Causal identifiability
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import numpy as np


class ACIEMetrics:
    """
    Comprehensive metrics for ACIE evaluation.
    """
    
    @staticmethod
    def counterfactual_mse(
        predicted: torch.Tensor,
        true: torch.Tensor,
    ) -> float:
        """Mean squared error for counterfactual predictions."""
        return nn.functional.mse_loss(predicted, true).item()
    
    @staticmethod
    def counterfactual_mae(
        predicted: torch.Tensor,
        true: torch.Tensor,
    ) -> float:
        """Mean absolute error for counterfactual predictions."""
        return nn.functional.l1_loss(predicted, true).item()
    
    @staticmethod
    def intervention_effect_size(
        factual: torch.Tensor,
        counterfactual: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Measure intervention effect magnitude.
        
        Returns:
            - mean_effect: Average change magnitude
            - max_effect: Maximum change
            - affected_fraction: Fraction of variables significantly affected
        """
        diff = torch.abs(counterfactual - factual)
        
        return {
            "mean_effect": diff.mean().item(),
            "max_effect": diff.max().item(),
            "std_effect": diff.std().item(),
            "affected_fraction": (diff > 0.1).float().mean().item(),
        }
    
    @staticmethod
    def latent_reconstruction_quality(
        original: torch.Tensor,
        reconstructed: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Assess latent reconstruction quality.
        
        Returns:
            - mse: Mean squared error
            - correlation: Average correlation
            - r2_score: R² score
        """
        mse = nn.functional.mse_loss(reconstructed, original).item()
        
        # Compute correlation per dimension
        original_centered = original - original.mean(dim=0)
        recon_centered = reconstructed - reconstructed.mean(dim=0)
        
        correlation = (
            (original_centered * recon_centered).sum(dim=0) /
            (original_centered.pow(2).sum(dim=0).sqrt() *
             recon_centered.pow(2).sum(dim=0).sqrt() + 1e-8)
        ).mean().item()
        
        # R² score
        ss_res = (original - reconstructed).pow(2).sum()
        ss_tot = (original - original.mean(dim=0)).pow(2).sum()
        r2 = (1 - ss_res / (ss_tot + 1e-8)).item()
        
        return {
            "mse": mse,
            "correlation": correlation,
            "r2_score": r2,
        }
    
    @staticmethod
    def physics_violation_rate(
        latent: torch.Tensor,
        observations: torch.Tensor,
        physics_validator: Optional[nn.Module] = None,
        threshold: float = 0.01,
    ) -> Dict[str, float]:
        """
        Compute physics constraint violation rate.
        
        Returns:
            - violation_rate: Fraction of samples violating constraints
            - avg_violation: Average violation magnitude
        """
        if physics_validator is None:
            return {"violation_rate": 0.0, "avg_violation": 0.0}
        
        violations = physics_validator(latent, observations, latent)
        
        violation_rate = violations.float().mean().item()
        
        return {
            "violation_rate": violation_rate,
            "num_violations": violations.sum().item(),
        }
    
    @staticmethod
    def intervention_consistency(
        latent_before: torch.Tensor,
        latent_after: torch.Tensor,
        intervention_indices: list,
    ) -> Dict[str, float]:
        """
        Check if intervention actually changed targeted variables.
        
        Returns:
            - target_change: Change in intervened variables
            - non_target_change: Change in non-intervened variables
            - consistency_score: Ratio (should be >> 1)
        """
        # Extract intervened and non-intervened variables
        all_indices = set(range(latent_before.shape[1]))
        non_intervention_indices = list(all_indices - set(intervention_indices))
        
        target_change = torch.abs(
            latent_after[:, intervention_indices] -
            latent_before[:, intervention_indices]
        ).mean().item()
        
        non_target_change = torch.abs(
            latent_after[:, non_intervention_indices] -
            latent_before[:, non_intervention_indices]
        ).mean().item()
        
        consistency_score = target_change / (non_target_change + 1e-8)
        
        return {
            "target_change": target_change,
            "non_target_change": non_target_change,
            "consistency_score": consistency_score,
        }
    
    @staticmethod
    def causal_identifiability_score(
        latent_samples: torch.Tensor,
        obs_samples: torch.Tensor,
    ) -> float:
        """
        Estimate causal identifiability via mutual information proxy.
        
        Higher scores indicate better identifiability of causal structure.
        """
        # Use average absolute correlation as proxy for MI
        latent_mean = latent_samples.mean(dim=0)
        obs_mean = obs_samples.mean(dim=0)
        
        latent_centered = latent_samples - latent_mean
        obs_centered = obs_samples - obs_mean
        
        # Compute cross-correlation matrix
        cross_corr = torch.matmul(latent_centered.T, obs_centered) / len(latent_samples)
        
        # Average absolute correlation
        identifiability = torch.abs(cross_corr).mean().item()
        
        return identifiability


class CounterfactualEvaluator:
    """
    Comprehensive evaluator for counterfactual inference.
    """
    
    def __init__(self, metrics_computer: Optional[ACIEMetrics] = None):
        self.metrics = metrics_computer or ACIEMetrics()
        self.results = {}
    
    def evaluate(
        self,
        model,
        dataloader,
        device: str = "cuda",
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation on a dataset.
        
        Args:
            model: Trained ACIE model
            dataloader: DataLoader with test data
            device: Device for computation
            
        Returns:
            Dict of all metrics
        """
        model.eval()
        
        all_factual_obs = []
        all_cf_obs_true = []
        all_cf_obs_pred = []
        all_latent = []
        
        with torch.no_grad():
            for batch in dataloader:
                factual_obs = batch["factual_obs"].to(device)
                cf_obs_true = batch["counterfactual_obs"].to(device)
                
                # Infer latent
                latent_dist = model.inference_model.encode(factual_obs)
                latent = latent_dist.mean
                
                # Generate counterfactual
                cf_obs_pred = model.counterfactual_engine(
                    latent=latent,
                    observations=factual_obs,
                    factual_latent=latent,
                )
                
                all_factual_obs.append(factual_obs.cpu())
                all_cf_obs_true.append(cf_obs_true.cpu())
                all_cf_obs_pred.append(cf_obs_pred.cpu())
                all_latent.append(latent.cpu())
        
        # Concatenate all batches
        factual_obs = torch.cat(all_factual_obs, dim=0)
        cf_obs_true = torch.cat(all_cf_obs_true, dim=0)
        cf_obs_pred = torch.cat(all_cf_obs_pred, dim=0)
        latent = torch.cat(all_latent, dim=0)
        
        # Compute all metrics
        results = {}
        
        # Counterfactual accuracy
        results["cf_mse"] = self.metrics.counterfactual_mse(cf_obs_pred, cf_obs_true)
        results["cf_mae"] = self.metrics.counterfactual_mae(cf_obs_pred, cf_obs_true)
        
        # Intervention effect
        effect_metrics = self.metrics.intervention_effect_size(factual_obs, cf_obs_pred)
        results.update({f"effect_{k}": v for k, v in effect_metrics.items()})
        
        # Identifiability
        results["identifiability"] = self.metrics.causal_identifiability_score(
            latent, factual_obs
        )
        
        self.results = results
        return results
    
    def print_results(self):
        """Print evaluation results in a formatted way."""
        if not self.results:
            print("No results to display. Run evaluate() first.")
            return
        
        print("\n" + "="*60)
        print("ACIE Evaluation Results")
        print("="*60)
        
        print("\nCounterfactual Prediction:")
        print(f"  MSE: {self.results.get('cf_mse', 0):.6f}")
        print(f"  MAE: {self.results.get('cf_mae', 0):.6f}")
        
        print("\nIntervention Effects:")
        print(f"  Mean Effect: {self.results.get('effect_mean_effect', 0):.6f}")
        print(f"  Max Effect: {self.results.get('effect_max_effect', 0):.6f}")
        print(f"  Affected Fraction: {self.results.get('effect_affected_fraction', 0):.2%}")
        
        print("\nCausal Identifiability:")
        print(f"  Score: {self.results.get('identifiability', 0):.6f}")
        
        print("="*60 + "\n")
