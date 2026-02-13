# ACIE-H: Mathematical Theory & Mechanism
**Analytical Cipher for Intelligent Embeddings (Homomorphic)**

## 1. Introduction
ACIE-H is a **Partial Homomorphic Encryption (PHE)** scheme derived from the **Paillier Cryptosystem**. Its primary feature is the ability to perform arithmetic operations (specifically addition and scalar multiplication) on *ciphertext* without ever decrypting it. This ensures that the data remains secure during calculation (e.g., aggregating galaxy flux values).

---

## 2. Key Generation
The security relies on the hardness of computing $n$-th residue classes, closely related to the integer factorization problem.

1.  **Select Primes**: Choose two large prime numbers $p$ and $q$ of equivalent bit-length (e.g., 512 bits each).
2.  **Compute Modulus**: 
    $$n = p \cdot q$$
    $$n^2 = n \cdot n$$
3.  **Compute Lambda** (Private Key component):
    $$\lambda = \text{lcm}(p-1, q-1)$$
4.  **Select Generator**:
    We choose a generator $g$ such that its order is a multiple of $n$. For efficiency in ACIE-H, we strictly use:
    $$g = n + 1$$
5.  **Compute Mu** (Private Key component):
    $$\mu = (L(g^\lambda \bmod n^2))^{-1} \bmod n$$
    *Where $L(x) = \frac{x-1}{n}$*

**Keys:**
*   **Public Key**: $(n, g)$
*   **Private Key**: $(\lambda, \mu)$

---

## 3. Encryption
To encrypt a plaintext message $m$ (where $0 \le m < n$):

1.  Select a random integer $r$ where $0 < r < n$ and $\gcd(r, n) = 1$.
2.  Compute Ciphertext $c$:
    $$c = g^m \cdot r^n \bmod n^2$$

**Optimization in ACIE-H:**
Since $g = n+1$, we can use the binomial expansion property $(1+n)^m = 1 + mn \pmod{n^2}$ to speed up encryption:
$$c = (1 + m \cdot n) \cdot r^n \bmod n^2$$

---

## 4. Decryption
To decrypt ciphertext $c$:

1.  Compute the raw plaintext structure using the private key $\lambda$:
    $$u = c^\lambda \bmod n^2$$
2.  Apply the $L$ function:
    $$L(u) = \frac{u-1}{n}$$
3.  Recover message $m$:
    $$m = L(u) \cdot \mu \bmod n$$

---

## 5. Homomorphic Properties
This is the core "Calculative Purpose" of ACIE-H.

### A. Modular Addition (Sum of Encrypted Values)
**Goal**: Compute $E(m_1 + m_2)$ given $E(m_1)$ and $E(m_2)$.

**Operation**: Multiply the ciphertexts modulo $n^2$.
$$E(m_1) \cdot E(m_2) = (g^{m_1} r_1^n) \cdot (g^{m_2} r_2^n) \bmod n^2$$
$$= g^{m_1+m_2} \cdot (r_1 r_2)^n \bmod n^2$$
$$= E(m_1 + m_2)$$

*Result*: The product of two ciphertexts decrypts to the **sum** of their plaintexts.

### B. Scalar Multiplication (Multiplying Encrypted Value by Constant)
**Goal**: Compute $E(m \cdot k)$ given $E(m)$ and a known constant $k$.

**Operation**: Raise the ciphertext to the power of $k$ modulo $n^2$.
$$E(m)^k = (g^m r^n)^k \bmod n^2$$
$$= g^{mk} \cdot (r^k)^n \bmod n^2$$
$$= E(m \cdot k)$$

*Result*: Raising a ciphertext to a scalar power decrypts to the **product** of the plaintext and that scalar.

### C. Scalar Addition (Adding Constant to Encrypted Value)
**Goal**: Compute $E(m + k)$.

**Operation**: Multiply ciphertext by $g^k$.
$$E(m) \cdot g^k = (g^m r^n) \cdot g^k \bmod n^2$$
$$= g^{m+k} \cdot r^n \bmod n^2$$
$$= E(m + k)$$

---

## 6. Mathematical Security Proof
ACIE-H provides **semantic security** against chosen-plaintext attacks (IND-CPA). Because of the random factor $r^n$, encrypting the same message $m$ twice results in different ciphertexts, preventing attackers from identifying data patterns.
