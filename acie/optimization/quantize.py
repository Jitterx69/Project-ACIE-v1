"""
Quantization utilities for ACIE models
"""

import torch
import torch.quantization
from typing import Set, Type
import logging

logger = logging.getLogger(__name__)


def quantize_dynamic(
    model: torch.nn.Module,
    qconfig_spec: Set[Type[torch.nn.Module]] = None,
    dtype: torch.dtype = torch.qint8
) -> torch.nn.Module:
    """
    Apply dynamic quantization to the model.
    Converts weights to INT8 or other quantized types.
    
    Args:
        model: PyTorch model to quantize
        qconfig_spec: Set of module types to quantize (default: Linear, LSTM)
        dtype: Target data type (default: qint8)
        
    Returns:
        Quantized model
    """
    if qconfig_spec is None:
        qconfig_spec = {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU}
        
    logger.info(f"Quantizing model with spec: {qconfig_spec}")
    
    # Ensure model is in eval mode
    model.eval()
    
    try:
        quantized_model = torch.quantization.quantize_dynamic(
            model, 
            qconfig_spec=qconfig_spec, 
            dtype=dtype
        )
        logger.info("Dynamic quantization completed successfully")
        return quantized_model
        
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        raise
