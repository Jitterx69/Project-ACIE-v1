"""
Cache key generation utilities for ACIE
"""

import hashlib
import json
from typing import Dict, Any, List


def make_inference_key(observation: List[float], intervention: Dict[str, float]) -> str:
    """
    Generate a unique cache key for inference results.
    
    Args:
        observation: Observation vector
        intervention: Intervention dictionary
        
    Returns:
        Unique cache key string
    """
    # Create deterministic representation
    data = {
        "observation": observation,
        "intervention": sorted(intervention.items())  # Sort for consistent hashing
    }
    
    # Hash to create key
    data_str = json.dumps(data, sort_keys=True)
    hash_val = hashlib.md5(data_str.encode()).hexdigest()
    
    return f"acie:inference:{hash_val}"


def make_model_key(model_version: str = "latest") -> str:
    """
    Generate cache key for model metadata.
    
    Args:
        model_version: Model version identifier
        
    Returns:
        Model cache key
    """
    return f"acie:model:{model_version}"


def make_batch_key(batch_id: str) -> str:
    """
    Generate cache key for batch inference results.
    
    Args:
        batch_id: Unique batch identifier
        
    Returns:
        Batch cache key
    """
    return f"acie:batch:{batch_id}"
