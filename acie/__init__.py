"""
Astronomical Counterfactual Inference Engine (ACIE)

A physics-constrained, causal inference system for astronomical observations.
"""

__version__ = "0.1.0"

from acie.core.acie_core import ACIEEngine
from acie.core.scm import StructuralCausalModel

# Secure Infrastructure
from acie.cipher_embeddings.acie_h import ACIEHomomorphicCipher
from acie.cipher_embeddings.tensor import CipherTensor
from acie.models.secure_layers import SecureLinear
from acie.data.secure_dataset import SecureACIEDataset


__all__ = ["ACIEEngine", "StructuralCausalModel", "ACIEHomomorphicCipher", "CipherTensor", "SecureLinear", "SecureACIEDataset"]
