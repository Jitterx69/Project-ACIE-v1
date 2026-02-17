
import torch
import numpy as np
import logging

try:
    from asm import vector_entropy
    SimpleNamespace = None # Placeholder
    ASM_AVAILABLE = True
except ImportError:
    ASM_AVAILABLE = False

logger = logging.getLogger(__name__)

class EntropyMonitor:
    """
    Monitors the entropy of distributions to detect mode collapse.
    Uses AVX-512 accelerated entropy calculation if available.
    """
    
    def __init__(self, threshold: float = 1e-4):
        self.threshold = threshold
        self.history = []
        
    def calculate_entropy(self, probs: torch.Tensor) -> float:
        """
        Calculate Shannon entropy of a probability distribution.
        Args:
            probs: Probability tensor (N_samples, Dim) or (Dim,). Sum must be 1.
        Returns:
            Average entropy per sample.
        """
        if not isinstance(probs, torch.Tensor):
            probs = torch.tensor(probs)
            
        # Ensure probs:
        # If batch, we sum over last dim.
        if probs.ndim > 1:
            # Flatten to 1D array for vector processing?
            # Our kernel computes -p*ln(p) elementwise.
            # So we can just flatten, compute elementwise, reshape, sum last dim.
            
            flat_probs = probs.detach().cpu().numpy().flatten().astype(np.float32)
            
            if ASM_AVAILABLE:
                terms = vector_entropy(flat_probs)
            else:
                # Fallback in numpy
                flat_probs = np.maximum(flat_probs, 1e-9)
                terms = -flat_probs * np.log(flat_probs)
                
            # Reshape back to (N, D)
            terms = terms.reshape(probs.shape)
            
            # Sum over last dim (entropy per sample)
            sample_entropies = terms.sum(axis=-1)
            return float(sample_entropies.mean())
            
        else:
            # Single distribution
            flat_probs = probs.detach().cpu().numpy().astype(np.float32)
            if ASM_AVAILABLE:
                terms = vector_entropy(flat_probs)
            else:
                flat_probs = np.maximum(flat_probs, 1e-9)
                terms = -flat_probs * np.log(flat_probs)
            return float(terms.sum())

    def check_collapse(self, probs: torch.Tensor) -> bool:
        """Returns True if entropy is below threshold (Mode Collapse)."""
        ent = self.calculate_entropy(probs)
        self.history.append(ent)
        if ent < self.threshold:
            logger.warning(f"Mode Collapse Detected! Entropy: {ent:.4f} < {self.threshold}")
            return True
        return False
