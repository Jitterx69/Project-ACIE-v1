"""
Pruning utilities for ACIE models
"""

import torch
import torch.nn.utils.prune as prune
from typing import Optional, List, Type
import logging

logger = logging.getLogger(__name__)


def prune_model(
    model: torch.nn.Module,
    amount: float = 0.3,
    method: str = 'l1_unstructured',
    target_modules: Optional[List[Type[torch.nn.Module]]] = None,
    make_permanent: bool = True
) -> torch.nn.Module:
    """
    Apply pruning to the model to remove redundant weights.
    
    Args:
        model: PyTorch model to prune
        amount: Fraction of weights to prune (0.0 to 1.0)
        method: Pruning method ('l1_unstructured', 'random_unstructured')
        target_modules: List of module types to prune (default: Linear)
        make_permanent: Whether to make pruning permanent (remove masks)
        
    Returns:
        Pruned model
    """
    if target_modules is None:
        target_modules = [torch.nn.Linear]
        
    logger.info(f"Pruning model with amount={amount}, method={method}")
    
    count = 0
    for name, module in model.named_modules():
        if any(isinstance(module, t) for t in target_modules):
            if method == 'l1_unstructured':
                prune.l1_unstructured(module, name='weight', amount=amount)
            elif method == 'random_unstructured':
                prune.random_unstructured(module, name='weight', amount=amount)
            # Add other methods as needed
            
            if make_permanent:
                prune.remove(module, 'weight')
            
            count += 1
            
    logger.info(f"Pruned {count} modules")
    return model


def get_sparsity(model: torch.nn.Module) -> float:
    """
    Calculate global sparsity of the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Sparsity percentage (0.0 to 100.0)
    """
    total_zeros = 0
    total_elements = 0
    
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            if hasattr(module, 'weight'):
                total_zeros += torch.sum(module.weight == 0).item()
                total_elements += module.weight.nelement()
                
    if total_elements == 0:
        return 0.0
        
    return 100.0 * total_zeros / total_elements
