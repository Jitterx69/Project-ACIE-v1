# ACIE Assembly Kernels (x86-64 AVX-512)

This directory contains hand-optimized assembly kernels for critical mathematical operations in the ACIE pipeline, specifically targeting high-dimensional physics simulations and cryptographic primitives.

## Features

### 1. Relativistic Physics (New!)
*   **`_minkowski_metric_avx512`**: Simultaneously computes spacetime intervals ($ds^2 = -t^2 + x^2 + y^2 + z^2$) for **4 points per clock cycle** using 512-bit ZMM registers. This replaces the standard scalar loop in `physics_layers.py`, providing a theoretical **4x-16x speedup** for large batch constraints.

### 2. Cryptography Acceleration (New!)
*   **`_vector_mul_u64_avx512`**: Vectorized 64-bit integer multiplication. Computes 8 multiplications per cycle. Foundational for accelerating large-integer arithmetic in Paillier encryption.

### 3. Generative Monitoring (New!)
*   **`_vector_entropy_term_avx512`**: Computes Shannon entropy terms $-p \cdot \ln(p)$ using a fast AVX-512 log approximation ($\approx 0.693(E+M-1)$). Essential for real-time mode collapse detection in the VAE latent space.

### 4. Matrix Operations
*   **`_fast_matrix_multiply_asm`**: Cache-oblivious matrix multiplication kernel.
*   **`_fast_relu_asm`**: Vectorized Rectified Linear Unit activation.
*   **`_fast_sigmoid_asm`**: Fast approximation of sigmoid $\sigma(x) \approx 0.5 \cdot \frac{x}{1+|x|} + 0.5$.

## Usage

These kernels are exposed to Python via `asm/asm_python.py` using `ctypes`.

```python
import numpy as np
from acie.asm.asm_python import fast_minkowski, vector_multiply_u64, vector_entropy

# Physics: Compute ds^2 interval
points = np.random.randn(10000, 4).astype(np.float32)
intervals = fast_minkowski(points)

# Crypto: Multiply large integer arrays
A = np.random.randint(0, 100, 1000, dtype=np.uint64)
B = np.random.randint(0, 100, 1000, dtype=np.uint64)
C = vector_multiply_u64(A, B)

# Entropy: Detect mode collapse
probs = np.random.uniform(0, 1, 1000).astype(np.float32)
probs /= probs.sum()
entropy_terms = vector_entropy(probs)
total_entropy = entropy_terms.sum()
```

### 4. Vectorized Montgomery Multiplication (AVX-512)
**Signature**: `void _montgomery_mul_avx512(uint64_t *A, uint64_t *B, uint64_t *N, uint64_t *Out, uint64_t k0, int64_t count)`
**Description**: Computes batched modular multiplication $C[i] = A[i] \times B[i] \times R^{-1} \pmod N$ for arrays of 64-bit integers.
- **Optimization**: Uses `vpmuludq` to emulate 128-bit products and parallelize modular reduction across 8 lanes.
- **Usage**: Critical for accelerating standard BigInt cryptography (e.g., Paillier) by vectorizing the core modular multiplication loop.

## Compilation

Requires `nasm` and a C compiler (GCC/Clang).

```bash
cd asm
make
```

This produces `libacie_asm.dylib` (macOS) which is automatically loaded by the Python wrapper. If not found, the Python code gracefully falls back to NumPy implementation.
