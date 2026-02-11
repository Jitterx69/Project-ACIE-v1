"""
Model Registry management for ACIE
"""

import mlflow
from mlflow.tracking import MlflowClient
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Manager for MLflow Model Registry.
    Handles model registration, versioning, and stage transitions.
    """
    
    def __init__(self):
        self.client = MlflowClient()
        
    def register_model(self, run_id: str, model_name: str, artifact_path: str = "model") -> str:
        """
        Register a model from a specific run.
        
        Args:
            run_id: MLflow run ID
            model_name: Name for the registered model
            artifact_path: Path to model artifact in the run
            
        Returns:
            Registered model version
        """
        model_uri = f"runs:/{run_id}/{artifact_path}"
        try:
            result = mlflow.register_model(model_uri, model_name)
            logger.info(f"Registered model {model_name} version {result.version}")
            return result.version
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise
            
    def promote_model(self, model_name: str, version: str, stage: str) -> None:
        """
        Promote a model version to a specific stage.
        
        Args:
            model_name: Registered model name
            version: Version number string
            stage: Target stage ("Staging", "Production", "Archived", "None")
        """
        valid_stages = {"Staging", "Production", "Archived", "None"}
        if stage not in valid_stages:
            raise ValueError(f"Invalid stage: {stage}. Must be one of {valid_stages}")
            
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            logger.info(f"Transitioned {model_name} v{version} to {stage}")
        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            raise
            
    def get_latest_version(self, model_name: str, stage: str = "None") -> Optional[str]:
        """
        Get latest version of model in specific stage.
        
        Args:
            model_name: Registered model name
            stage: Stage filter
            
        Returns:
            Latest version string or None
        """
        try:
            versions = self.client.get_latest_versions(model_name, stages=[stage])
            if versions:
                return versions[0].version
            return None
        except Exception:
            return None
            
    def load_model(self, model_name: str, version: Optional[str] = None, stage: Optional[str] = None):
        """
        Load a model from registry.
        
        Args:
            model_name: Registered model name
            version: Specific version (optional)
            stage: Stage alias (optional, e.g. "Production")
            
        Returns:
            Loaded PyTorch model
        """
        if version:
            model_uri = f"models:/{model_name}/{version}"
        elif stage:
            model_uri = f"models:/{model_name}/{stage}"
        else:
            # Default to latest version
            model_uri = f"models:/{model_name}/latest"
            
        logger.info(f"Loading model from {model_uri}")
        return mlflow.pytorch.load_model(model_uri)
        
    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models"""
        models = []
        for rm in self.client.search_registered_models():
            models.append({
                "name": rm.name,
                "latest_versions": [
                    {"version": v.version, "stage": v.current_stage} 
                    for v in rm.latest_versions
                ]
            })
        return models
