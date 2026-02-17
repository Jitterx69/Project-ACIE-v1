
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from asm.asm_python import montgomery_mul, AVAILABLE

def test_montgomery():
    print("Testing Montgomery Multiplication...")
    
    if not AVAILABLE:
        print("AVX-512 Library not loaded. Skipping low-level verification.")
        # But we can test the fallback logic
        print("Testing Python fallback logic...")
    else:
        print("AVX-512 Library loaded. Running assembly tests.")

    # Parameters
    N = 101  # Small prime for testing
    k0 = pow(2**64, -1, N) # This works if we assume 64-bit word size. 
                       # Wait, k0 formula depends on R.
                       # R = 2^64.
                       # k0 = -N^-1 mod R.
    # Actually, Python's pow(a, -1, b) works.
    # k0 = -pow(N, -1, 2**64) mod 2**64
    try:
        n_inv = pow(N, -1, 2**64)
        k0 = (1 << 64) - n_inv
    except ValueError:
        print("N is not invertible mod 2^64 (must be odd).")
        return

    # Random inputs
    count = 16
    A = np.random.randint(0, N, size=count, dtype=np.uint64)
    B = np.random.randint(0, N, size=count, dtype=np.uint64)
    
    # Expected Result: A * B * R^-1 mod N
    R_inv = pow(1 << 64, -1, N)
    expected = (A.astype(object) * B.astype(object) * R_inv) % N
    expected = expected.astype(np.uint64)
    
    # Actual Result
    result = montgomery_mul(A, B, N, int(k0))
    
    # Compare
    if np.allclose(result, expected):
        print("SUCCESS: Montgomery Mul matches expected values.")
    else:
        print("FAILURE: Mismatch detected.")
        print("Expected:", expected)
        print("Actual:  ", result)
        # Debug first element
        print(f"Index 0: A={A[0]}, B={B[0]}, N={N}")
        a0, b0 = int(A[0]), int(B[0])
        exp0 = (a0 * b0 * R_inv) % N
        res0 = result[0]
        print(f"  Exp: {exp0}")
        print(f"  Got: {res0}")

if __name__ == "__main__":
    test_montgomery()
