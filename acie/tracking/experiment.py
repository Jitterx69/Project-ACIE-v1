"""
Experiment tracking integration using MLflow
"""

import mlflow
import os
import torch
from typing import Dict, Any, Optional, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Wrapper for MLflow experiment tracking.
    
    Usage:
        tracker = ExperimentTracker("ACIE_Experiments")
        with tracker.start_run("run_name"):
            tracker.log_params({"lr": 0.001})
            tracker.log_metrics({"loss": 0.5})
            tracker.log_model(model, "model")
    """
    
    def __init__(self, experiment_name: str = "ACIE_Default"):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
        """
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
        mlflow.set_tracking_uri(self.tracking_uri)
        
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                self.experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"Created new experiment: {experiment_name} (ID: {self.experiment_id})")
            else:
                self.experiment_id = experiment.experiment_id
                logger.debug(f"Using existing experiment: {experiment_name} (ID: {self.experiment_id})")
                
            mlflow.set_experiment(experiment_name)
            
        except Exception as e:
            logger.warning(f"Failed to setup MLflow experiment: {e}")
            self.experiment_id = None
            
    def start_run(self, run_name: Optional[str] = None, nested: bool = False):
        """
        Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run
            nested: Whether this is a nested run
            
        Returns:
            ActiveRun object (use as context manager)
        """
        return mlflow.start_run(run_name=run_name, nested=nested)
        
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters (hyperparameters, config)"""
        try:
            mlflow.log_params(params)
        except Exception as e:
            logger.warning(f"Failed to log params: {e}")
            
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics (loss, accuracy, etc.)"""
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")
            
    def log_model(
        self, 
        model: torch.nn.Module, 
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None
    ) -> None:
        """
        Log PyTorch model.
        
        Args:
            model: PyTorch model
            artifact_path: Path within the artifact directory
            registered_model_name: If provided, register model in registry
        """
        try:
            mlflow.pytorch.log_model(
                model, 
                artifact_path,
                registered_model_name=registered_model_name
            )
            logger.info(f"Logged model to {artifact_path}")
        except Exception as e:
            logger.error(f"Failed to log model: {e}")
            
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log a local file or directory as an artifact"""
        try:
            mlflow.log_artifact(local_path, artifact_path)
        except Exception as e:
            logger.warning(f"Failed to log artifact {local_path}: {e}")
            
    def auto_log(self) -> None:
        """Enable PyTorch autologging"""
        try:
            mlflow.pytorch.autolog()
        except Exception as e:
            logger.warning(f"Failed to enable autologging: {e}")
