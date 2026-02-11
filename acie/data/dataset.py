"""
Data Loading Pipeline for ACIE

Handles large CSV datasets with efficient chunk loading.
Supports:
- Observational data
- Counterfactual data (paired with observations)
- Intervention data
- Environment/instrument shift data
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
import warnings


class ACIEDataset(Dataset):
    """
    Dataset for astronomical observations with optional counterfactuals.
    
    Data structure expected (from generation scripts):
    - Columns 0-1999 (or 0-3999): Latent variables P
    - Columns 2000-7999 (or 4000-14999): Observable variables O
    - Remaining columns: Noise variables N
    """
    
    def __init__(
        self,
        data_path: Path,
        latent_dim: int = 2000,
        obs_dim: int = 6000,
        noise_dim: int = 2000,
        normalize: bool = True,
        max_rows: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.data_path = Path(data_path)
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.noise_dim = noise_dim
        self.normalize = normalize
        self.dtype = dtype
        
        # Load data
        print(f"Loading data from {self.data_path}...")
        self.data = self._load_data(max_rows)
        
        # Extract components
        self.latent = self.data[:, :latent_dim]
        self.obs = self.data[:, latent_dim:latent_dim + obs_dim]
        self.noise = self.data[:, latent_dim + obs_dim:]
        
        # Normalize if requested
        if normalize:
            self.obs_mean = self.obs.mean(dim=0)
            self.obs_std = self.obs.std(dim=0) + 1e-8
            self.obs = (self.obs - self.obs_mean) / self.obs_std
        else:
            self.obs_mean = None
            self.obs_std = None
        
        print(f"Loaded {len(self)} samples")
    
    def _load_data(self, max_rows: Optional[int]) -> torch.Tensor:
        """Load CSV data efficiently."""
        try:
            # Try loading all at once for small files
            if max_rows:
                df = pd.read_csv(self.data_path, nrows=max_rows)
            else:
                df = pd.read_csv(self.data_path)
            
            data = torch.tensor(df.values, dtype=self.dtype)
            return data
            
        except MemoryError:
            warnings.warn("File too large, loading in chunks...")
            return self._load_chunked(max_rows)
    
    def _load_chunked(self, max_rows: Optional[int], chunk_size: int = 10000) -> torch.Tensor:
        """Load large CSV in chunks."""
        chunks = []
        total_rows = 0
        
        for chunk in pd.read_csv(self.data_path, chunksize=chunk_size):
            chunks.append(torch.tensor(chunk.values, dtype=self.dtype))
            total_rows += len(chunk)
            
            if max_rows and total_rows >= max_rows:
                break
        
        return torch.cat(chunks, dim=0)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "latent": self.latent[idx],
            "obs": self.obs[idx],
            "noise": self.noise[idx],
        }


class PairedCounterfactualDataset(Dataset):
    """
    Dataset with paired (factual, counterfactual) observations.
    
    Used for training counterfactual inference.
    """
    
    def __init__(
        self,
        factual_path: Path,
        counterfactual_path: Path,
        latent_dim: int = 2000,
        obs_dim: int = 6000,
        noise_dim: int = 2000,
        normalize: bool = True,
        max_rows: Optional[int] = None,
    ):
        print("Loading paired counterfactual dataset...")
        
        self.factual_dataset = ACIEDataset(
            factual_path,
            latent_dim=latent_dim,
            obs_dim=obs_dim,
            noise_dim=noise_dim,
            normalize=normalize,
            max_rows=max_rows,
        )
        
        self.counterfactual_dataset = ACIEDataset(
            counterfactual_path,
            latent_dim=latent_dim,
            obs_dim=obs_dim,
            noise_dim=noise_dim,
            normalize=False,  # Use same normalization as factual
            max_rows=max_rows,
        )
        
        # Apply same normalization to counterfactual
        if normalize and self.factual_dataset.obs_mean is not None:
            self.counterfactual_dataset.obs = (
                (self.counterfactual_dataset.obs - self.factual_dataset.obs_mean) /
                self.factual_dataset.obs_std
            )
        
        assert len(self.factual_dataset) == len(self.counterfactual_dataset), \
            "Factual and counterfactual datasets must have same length"
        
        print(f"Loaded {len(self)} paired samples")
    
    def __len__(self) -> int:
        return len(self.factual_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        factual = self.factual_dataset[idx]
        counterfactual = self.counterfactual_dataset[idx]
        
        return {
            "factual_latent": factual["latent"],
            "factual_obs": factual["obs"],
            "counterfactual_latent": counterfactual["latent"],
            "counterfactual_obs": counterfactual["obs"],
        }


class MultiDatasetLoader:
    """
    Manages multiple datasets for comprehensive ACIE training.
    
    Handles:
    - Observational data
    - Counterfactual pairs
    - Intervention data
    - Distribution shift data (environment, instrument)
    """
    
    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 32,
        num_workers: int = 4,
        max_rows_per_dataset: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_rows = max_rows_per_dataset
        
        self.datasets = {}
        self.loaders = {}
    
    def load_observational(self, dataset_size: str = "10k"):
        """Load observational dataset."""
        path = self.data_dir / f"acie_observational_{dataset_size}_x_{dataset_size}.csv"
        
        if path.exists():
            self.datasets["observational"] = ACIEDataset(
                path,
                latent_dim=2000 if dataset_size == "10k" else 4000,
                obs_dim=6000 if dataset_size == "10k" else 11000,
                noise_dim=2000 if dataset_size == "10k" else 5000,
                max_rows=self.max_rows,
            )
            
            self.loaders["observational"] = DataLoader(
                self.datasets["observational"],
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )
            print(f"Loaded observational dataset: {len(self.datasets['observational'])} samples")
        else:
            print(f"Warning: {path} not found")
    
    def load_counterfactual(self, dataset_size: str = "10k"):
        """Load paired counterfactual dataset."""
        factual_path = self.data_dir / f"acie_observational_{dataset_size}_x_{dataset_size}.csv"
        cf_path = self.data_dir / f"acie_counterfactual_{dataset_size}_x_{dataset_size}.csv"
        
        if factual_path.exists() and cf_path.exists():
            self.datasets["counterfactual"] = PairedCounterfactualDataset(
                factual_path,
                cf_path,
                latent_dim=2000 if dataset_size == "10k" else 4000,
                obs_dim=6000 if dataset_size == "10k" else 11000,
                noise_dim=2000 if dataset_size == "10k" else 5000,
                max_rows=self.max_rows,
            )
            
            self.loaders["counterfactual"] = DataLoader(
                self.datasets["counterfactual"],
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )
            print(f"Loaded counterfactual dataset: {len(self.datasets['counterfactual'])} pairs")
        else:
            print(f"Warning: Counterfactual data not found")
    
    def load_all(self, dataset_size: str = "10k"):
        """Load all available datasets."""
        self.load_observational(dataset_size)
        self.load_counterfactual(dataset_size)
        
        # Load intervention datasets if available
        for intervention_type in ["hard_intervention", "environment_shift", "instrument_shift"]:
            path = self.data_dir / f"acie_{intervention_type}_20k_x_20k.csv"
            if path.exists():
                self.datasets[intervention_type] = ACIEDataset(
                    path,
                    latent_dim=4000,
                    obs_dim=11000,
                    noise_dim=5000,
                    max_rows=self.max_rows,
                )
                self.loaders[intervention_type] = DataLoader(
                    self.datasets[intervention_type],
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                )
                print(f"Loaded {intervention_type} dataset: {len(self.datasets[intervention_type])} samples")


def create_dataloaders(
    data_dir: Path,
    batch_size: int = 32,
    num_workers: int = 4,
    dataset_size: str = "10k",
    max_rows: Optional[int] = None,
) -> Dict[str, DataLoader]:
    """
    Convenience function to create all dataloaders.
    
    Args:
        data_dir: Directory containing CSV files
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        dataset_size: "10k" or "20k"
        max_rows: Maximum rows to load per dataset (for development)
        
    Returns:
        Dict of dataloaders
    """
    loader = MultiDatasetLoader(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        max_rows_per_dataset=max_rows,
    )
    
    loader.load_all(dataset_size=dataset_size)
    
    return loader.loaders
