# ACIE-H: Analytical Cipher for Intelligent Embeddings (Homomorphic)
**Technical White Paper & System Specification**

**Version:** 1.0.0
**Date:** February 13, 2026
**Author:** ACIE Engineering Team

---

## 1. Abstract
The **ACIE-H (Analytical Cipher for Intelligent Embeddings - Homomorphic)** is a proprietary cryptographic system designed to enable privacy-preserving computation on sensitive astronomical and physical datasets. Unlike traditional encryption schemes (AES/RC2) which require data to be decrypted before processing, ACIE-H allows mathematical operations—specifically addition and scalar multiplication—to be performed directly on the ciphertext. This ensures that the underlying data remains encrypted throughout the entire computational pipeline ("Data in Use"), exposing only the final result to authorized entities with the private key.

## 2. Problem Statement
Standard encryption methods (Symmetric/Asymmetric) protect data at rest (storage) and in transit (network). However, they fail to protect data during computation (in use). To compute a weighted average of galaxy fluxes or combine multi-modal sensor inputs, traditional systems must decrypt the data in memory, exposing it to potential side-channel attacks, memory dumps, or compromised runtime environments.

**ACIE-H solves this by enabling the following workflow:**
1.  **Encrypt Inputs**: $E(x_1), E(x_2), \dots$
2.  **Compute blindly**: $E(y) = f(E(x_1), E(x_2), \dots)$
3.  **Decrypt Output**: $y = D(E(y))$
The intermediate value $x_i$ is never exposed in plaintext form during step 2.

---

## 3. Mathematical Mechanism

### 3.1. Foundation
ACIE-H is a probabilistic public-key cryptosystem based on the **Paillier Cryptosystem**, leveraging the **Composite Residuosity Class Problem** (CRCP). The security of the system relies on the computational intractability of computing $n$-th residue classes modulo $n^2$, which is mathematically related to the integer factorization problem.

### 3.2. Key Generation Algorithm
The system generates a public/private key pair as follows:

1.  **Prime Selection**: generating two large prime numbers $p$ and $q$ of equal bit-length $k$ (e.g., $k=1024$ bits).
    *   *Constraint*: $\gcd(pq, (p-1)(q-1)) = 1$.
2.  **Modulus Computation**:
    $$n = p \cdot q$$
    $$n^2 = n \cdot n$$ (The ciphertext space)
3.  **Generator Selection ($g$)**:
    ACIE-H enforces a specific generator optimization for performance:
    $$g = n + 1$$
    *Rationale*: This simplifies modular exponentiation because $(1+n)^x \equiv 1 + nx \pmod{n^2}$, reducing encryption complexity.
4.  **Lambda ($\lambda$) Computation** (Private):
    $$\lambda = \text{lcm}(p-1, q-1)$$
5.  **Mu ($\mu$) Computation** (Private):
    $$\mu = (L(g^\lambda \bmod n^2))^{-1} \bmod n$$
    *Where $L(x) = \frac{x-1}{n}$ is the discrete logarithmic function for this subgroup.*

### 3.3. Encryption Primitive ($E$)
To encrypt a plaintext message $m$ where $0 \le m < n$:

1.  **Random Blinding**: Select a random integer $r$ such that $0 < r < n$ and $\gcd(r, n) = 1$.
2.  **Ciphertext Computation**:
    $$c = g^m \cdot r^n \bmod n^2$$
    *Using ACIE-H optimization ($g=n+1$):*
    $$c = (1 + m \cdot n) \cdot r^n \bmod n^2$$

**Properties:**
*   **Probabilistic**: Encrypting the same $m$ twice yields different $c$ values due to random $r$.
*   **Expansion**: The ciphertext size is $2 \cdot |n|$ (2048 bits for a 1024-bit key).

### 3.4. Decryption Primitive ($D$)
To decrypt a ciphertext $c < n^2$:

1.  **Remove Randomness**:
    $$u = c^\lambda \bmod n^2$$
2.  **Apply L-Function**:
    $$L(u) = \frac{u-1}{n}$$
3.  **Recover Plaintext**:
    $$m = L(u) \cdot \mu \bmod n$$

---

## 4. Homomorphic Operations & Policies
ACIE-H supports specific "Calculative Purposes".

### 4.1. Additive Homomorphism ($E(a) \oplus E(b)$)
**Policy**: The sum of two encrypted values is the product of their ciphertexts modulo $n^2$.
$$D(E(m_1) \cdot E(m_2) \bmod n^2) = m_1 + m_2 \pmod n$$
*Use Case*: Aggregating total flux from multiple sensors.

### 4.2. Scalar Multiplication ($E(a) \otimes k$)
**Policy**: An encrypted value raised to a plaintext scalar $k$ results in the encrypted product.
$$D(E(m)^k \bmod n^2) = m \cdot k \pmod n$$
*Use Case*: Applying weighted importance to features (e.g., $0.3 \times \text{Feature}_A$).

### 4.3. Scalar Addition ($E(a) \oplus k$)
**Policy**: Adding a constant $k$ to an encrypted value involves multiplying by $g^k$.
$$D(E(m) \cdot g^k \bmod n^2) = m + k \pmod n$$
*Use Case*: Adding bias terms in a linear regression model.

---

## 5. System Constraints & Limitations

1.  **Data Types**: ACIE-H fundamentally operates on **Non-Negative Integers**.
    *   *Constraint*: Floating point numbers must be scaled (quantized) to integers (e.g., multiply by $10^6$) before encryption.
    *   *Constraint*: Negative numbers require a special encoding representation (e.g., using $n - x$ to represent $-x$) or managing a sign bit externally.
2.  **Ciphertext Expansion**: The ciphertext is twice the size of the modulus $n$. This impacts storage bandwidth.
    *   *Policy*: Use ACIE-H only for critical metadata/features, not for bulk raw data (like full-resolution video streams) unless necessary.
3.  **Multiplication Limitation**: ACIE-H is **Partially Homomorphic**.
    *   *Constraint*: It supports $Enc \times Plain$, but **NOT** $Enc \times Enc$. To multiply two encrypted values, users must decrypt one (interactive protocol) or switch to a Fully Homomorphic Encryption (FHE) scheme (much slower).

## 6. Implementation Architecture

### 6.1. `CryptoEngine` Interface
The `CryptoEngine` serves as the high-level API.
*   **Integrity Layer**: Uses **GOST** (OpenSSL) for hashing/fingerprinting.
*   **Storage Layer**: Uses **RC2** (OpenSSL) for bulk file encryption.
*   **Compute Layer**: Uses **ACIE-H** (Python/GMP) for calculative metadata.

### 6.2. Key Management Policy
*   **Generation**: Keys must be generated on a secure, air-gapped machine or HSM.
*   **Storage**: Private keys ($\lambda, \mu$) must never be stored in the cloud compute environment. They reside only on the client/analyst side.
*   **Rotation**: Keys should be rotated annually or upon employee churn.

## 7. Operational Workflow

1.  **Data Owner**:
    *   Quantizes data (Float -> Int).
    *   Encrypts with Public Key $(n, g)$.
    *   Uploads Ciphertexts to Cloud/Cluster.
2.  **Compute Node (Cloud)**:
    *   Receives Ciphertexts.
    *   Performs Aggregation/Weighted Sums using Homomorphic properties.
    *   Has **NO ACCESS** to Private Key. Cannot see inputs or intermediate results.
    *   Returns Result Ciphertext.
3.  **Data Owner**:
    *   Downloads Result Ciphertext.
    *   Decrypts with Private Key $(\lambda, \mu)$.
    *   De-quantizes (Int -> Float).

---

**End of Specification**
ACIE Engineering Team
