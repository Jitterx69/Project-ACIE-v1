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
    try:
        _lib = ctypes.CDLL(str(lib_path))
        AVAILABLE = True
    except OSError as e:
        AVAILABLE = False
        print(f"Warning: Could not load assembly library at {lib_path}: {e}")
        _lib = None

    if AVAILABLE:
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

        _lib.fast_minkowski_wrapper.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # input (N x 4)
            ctypes.POINTER(ctypes.c_float),  # output (N)
            ctypes.c_int64,  # num_points
        ]
        _lib.fast_minkowski_wrapper.restype = None

        _lib.vector_mul_u64_wrapper.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_int64,
        ]
        _lib.vector_mul_u64_wrapper.restype = None

        _lib.vector_entropy_wrapper.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # p
            ctypes.POINTER(ctypes.c_float),  # out
            ctypes.c_int64,  # N
        ]
        _lib.vector_entropy_wrapper.restype = None
        
        _lib.montgomery_mul_wrapper.argtypes = [
            ctypes.POINTER(ctypes.c_uint64), # A
            ctypes.POINTER(ctypes.c_uint64), # B
            ctypes.POINTER(ctypes.c_uint64), # N (scalar ptr)
            ctypes.POINTER(ctypes.c_uint64), # Out
            ctypes.c_uint64,                 # k0
            ctypes.c_int64,                  # count
        ]
        _lib.montgomery_mul_wrapper.restype = None
    
    # AVAILABLE = True set above
else:
    AVAILABLE = False
    print(f"Warning: Assembly library not found at {lib_path}")


def fast_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Fast matrix multiplication using Assembly"""
    if not AVAILABLE:
        return np.dot(A, B)
    
    assert A.dtype == np.float32 and B.dtype == np.float32
    assert A.shape[1] == B.shape[0]
    
    M, A_cols = A.shape
    K = B.shape[1]
    
    # We need contiguous arrays for ctypes
    if not A.flags['C_CONTIGUOUS']: A = np.ascontiguousarray(A)
    if not B.flags['C_CONTIGUOUS']: B = np.ascontiguousarray(B)
    
    C = np.zeros((M, K), dtype=np.float32)
    
    _lib.fast_matmul_wrapper(
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int64(M), 
        ctypes.c_int64(A_cols), 
        ctypes.c_int64(K)
    )
    
    return C


def fast_relu(x: np.ndarray) -> np.ndarray:
    """Fast ReLU using Assembly"""
    if not AVAILABLE:
        return np.maximum(0, x)
    
    assert x.dtype == np.float32
    if not x.flags['C_CONTIGUOUS']: x = np.ascontiguousarray(x)
    
    output = np.zeros_like(x)
    
    _lib.fast_relu_wrapper(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int64(x.size)
    )
    
    return output


def fast_sigmoid(x: np.ndarray) -> np.ndarray:
    """Fast sigmoid using Assembly"""
    if not AVAILABLE:
        return 1 / (1 + np.exp(-x))
    
    assert x.dtype == np.float32
    if not x.flags['C_CONTIGUOUS']: x = np.ascontiguousarray(x)
    
    output = np.zeros_like(x)
    
    _lib.fast_sigmoid_wrapper(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int64(x.size)
    )
    
    return output


def fast_minkowski(points: np.ndarray) -> np.ndarray:
    """
    Fast Minkowski Metric Calculation using AVX-512 Assembly.
    Input: points (N, 4) array of [t, x, y, z]
    Output: ds2 (N,) array of -t^2 + x^2 + y^2 + z^2
    """
    if not AVAILABLE:
        # Numpy fallback
        t = points[:, 0]
        space = points[:, 1:]
        return -t**2 + np.sum(space**2, axis=1)

    assert points.dtype == np.float32
    assert points.ndim == 2 and points.shape[1] == 4
    
    if not points.flags['C_CONTIGUOUS']:
        points = np.ascontiguousarray(points)

    num_points = points.shape[0]
    output = np.zeros(num_points, dtype=np.float32)

    _lib.fast_minkowski_wrapper(
        points.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int64(num_points)
    )
    
    return output


def vector_multiply_u64(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    SIMD Vector Multiplication for 64-bit integers (AVX-512).
    C[i] = A[i] * B[i] (modulo 2^64)
    """
    if not AVAILABLE:
        return np.multiply(A, B)
    
    assert A.dtype == np.uint64 and B.dtype == np.uint64
    assert A.shape == B.shape
    
    if not A.flags['C_CONTIGUOUS']: A = np.ascontiguousarray(A)
    if not B.flags['C_CONTIGUOUS']: B = np.ascontiguousarray(B)
    
    C = np.zeros_like(A)
    
    _lib.vector_mul_u64_wrapper(
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        ctypes.c_int64(A.size)
    )
    
    return C


def vector_entropy(p: np.ndarray) -> np.ndarray:
    """
    Fast Shannon Entropy term calculation:
    out[i] = -p[i] * ln(p[i])
    Uses AVX-512 fast log approximation.
    """
    if not AVAILABLE:
        # Numpy fallback
        # Handle zero logs gracefully
        out = np.zeros_like(p)
        mask = p > 0
        out[mask] = -p[mask] * np.log(p[mask])
        return out
    
    assert p.dtype == np.float32
    if not p.flags['C_CONTIGUOUS']: p = np.ascontiguousarray(p)
    
    out = np.zeros_like(p)
    
    _lib.vector_entropy_wrapper(
        p.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int64(p.size)
    )
    
    return out


def montgomery_mul(A: np.ndarray, B: np.ndarray, N: int, k0: int) -> np.ndarray:
    """
    Vectorized Montgomery Multiplication (AVX-512)
    C[i] = A[i] * B[i] * R^-1 mod N
    Assumes A, B are arrays of 64-bit integers.
    N and k0 are scalars.
    result = A*B*R^-1 mod N.
    """
    assert isinstance(N, int) and isinstance(k0, int)
    
    if not AVAILABLE:
        # Software fallback (using Python's BigInt support)
        # R = 2^64. Montgomery reduction: t = (T + m*N) / R
        # Effective operation is A * B * R^-1 mod N
        try:
            R_inv = pow(1 << 64, -1, N)
            return (A.astype(object) * B.astype(object) * R_inv % N).astype(np.uint64)
        except ValueError:
            # If N is even, inverse doesn't exist
            print("Warning: N must be odd for Montgomery Mul")
            return (A * B) % N
        
    assert A.dtype == np.uint64 and B.dtype == np.uint64
    if not A.flags['C_CONTIGUOUS']: A = np.ascontiguousarray(A)
    if not B.flags['C_CONTIGUOUS']: B = np.ascontiguousarray(B)
    
    C = np.zeros_like(A)
    c_N = ctypes.c_uint64(N)
    
    _lib.montgomery_mul_wrapper(
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        ctypes.byref(c_N),
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        ctypes.c_uint64(k0),
        ctypes.c_int64(A.size)
    )
    
    return C
