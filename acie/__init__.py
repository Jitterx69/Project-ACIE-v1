"""
Astronomical Counterfactual Inference Engine (ACIE)

A physics-constrained, causal inference system for astronomical observations.
"""

__version__ = "0.1.0"

from acie.core.acie_core import ACIEEngine
from acie.core.scm import StructuralCausalModel

__all__ = ["ACIEEngine", "StructuralCausalModel"]
