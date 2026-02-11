"""
Python interface to Assembly kernels via ctypes
Simpler alternative to compiled extension module
"""

import ctypes
import numpy as np
from pathlib import Path

# Load the shared library
lib_path = Path(__file__).parent / "libacie_asm.dylib"

if lib_path.exists():
    _lib = ctypes.CDLL(str(lib_path))
    
    # Define function signatures
    _lib.fast_matmul_wrapper.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # A
        ctypes.POINTER(ctypes.c_float),  # B
        ctypes.POINTER(ctypes.c_float),  # C
        ctypes.c_int64,  # M
        ctypes.c_int64,  # N
        ctypes.c_int64,  # K
    ]
    _lib.fast_matmul_wrapper.restype = None
    
    _lib.fast_relu_wrapper.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # input
        ctypes.POINTER(ctypes.c_float),  # output
        ctypes.c_int64,  # length
    ]
    _lib.fast_relu_wrapper.restype = None
    
    _lib.fast_sigmoid_wrapper.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # input
        ctypes.POINTER(ctypes.c_float),  # output
        ctypes.c_int64,  # length
    ]
    _lib.fast_sigmoid_wrapper.restype = None
    
    AVAILABLE = True
else:
    AVAILABLE = False
    print(f"Warning: Assembly library not found at {lib_path}")


def fast_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Fast matrix multiplication using Assembly"""
    if not AVAILABLE:
        return np.dot(A, B)
    
    assert A.dtype == np.float32 and B.dtype == np.float32
    assert A.shape[1] == B.shape[0]
    
    M, N = A.shape
    K = B.shape[1]
    C = np.zeros((M, K), dtype=np.float32)
    
    _lib.fast_matmul_wrapper(
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        M, N, K
    )
    
    return C


def fast_relu(x: np.ndarray) -> np.ndarray:
    """Fast ReLU using Assembly"""
    if not AVAILABLE:
        return np.maximum(0, x)
    
    assert x.dtype == np.float32
    
    output = np.zeros_like(x)
    
    _lib.fast_relu_wrapper(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        x.size
    )
    
    return output


def fast_sigmoid(x: np.ndarray) -> np.ndarray:
    """Fast sigmoid using Assembly"""
    if not AVAILABLE:
        return 1 / (1 + np.exp(-x))
    
    assert x.dtype == np.float32
    
    output = np.zeros_like(x)
    
    _lib.fast_sigmoid_wrapper(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        x.size
    )
    
    return output
