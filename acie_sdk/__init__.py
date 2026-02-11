"""
ACIE SDK - Python client library for ACIE Inference API

This package provides a simple interface for interacting with ACIE services.
"""

__version__ = "1.0.0"
__author__ = "ACIE Team"

from acie_sdk.client import ACIEClient
from acie_sdk.exceptions import (
    ACIEError,
    ACIEConnectionError,
    ACIEAuthenticationError,
    ACIENotFoundError,
    ACIEServerError,
)

__all__ = [
    "ACIEClient",
    "ACIEError",
    "ACIEConnectionError",
    "ACIEAuthenticationError",
    "ACIENotFoundError",
    "ACIEServerError",
]
