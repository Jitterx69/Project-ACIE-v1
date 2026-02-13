"""
ACIE-H: Analytical Cipher for Intelligent Embeddings (Homomorphic)

This module implements a Partial Homomorphic Encryption (PHE) scheme based on the 
Paillier cryptosystem. It enables arithmetic operations on encrypted data:

1. Enc(m1) + Enc(m2) -> Enc(m1 + m2)
2. Enc(m1) * k       -> Enc(m1 * k)
3. Enc(m1) + k       -> Enc(m1 + k)

Designed for "Calculative Purpose" on ciphered metadata as requested.
"""

import random
import math
from typing import Tuple, Union

class ACIEHomomorphicCipher:
    """
    ACIE-H Cipher Implementation.
    Supports secure addition and scalar multiplication on encrypted integers.
    """
    
    def __init__(self, key_size: int = 1024):
        self.key_size = key_size
        self.n = None
        self.g = None
        self.lmbda = None
        self.mu = None
        self._generate_keys()

    def _generate_primes(self, bits: int) -> int:
        """Generate a large prime number (simplified for demo)."""
        # In production, use standard crypto lib for prime gen.
        # This is a simplified probabilistic implementation.
        while True:
            p = random.getrandbits(bits)
            if p % 2 == 0: continue
            if self._is_prime(p):
                return p

    def _is_prime(self, n: int, k: int = 5) -> bool:
        """Miller-Rabin primality test."""
        if n == 2 or n == 3: return True
        if n < 2 or n % 2 == 0: return False

        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        for _ in range(k):
            a = random.randrange(2, n - 1)
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    def _lcm(self, a, b):
        return abs(a * b) // math.gcd(a, b)

    def _generate_keys(self):
        """Generate Public (n, g) and Private (lambda, mu) keys."""
        # 1. Generate two large primes p and q
        p = self._generate_primes(self.key_size // 2)
        q = self._generate_primes(self.key_size // 2)
        
        # 2. Compute n = p*q and lambda = lcm(p-1, q-1)
        self.n = p * q
        self.nsquare = self.n * self.n
        self.lmbda = self._lcm(p - 1, q - 1)
        
        # 3. Choose g = n + 1 (simple variant)
        self.g = self.n + 1
        
        # 4. Compute mu = (L(g^lambda mod n^2))^-1 mod n
        # For g = n+1, mu = lmbda^-1 mod n
        # Modular inverse
        try:
            self.mu = pow(self.lmbda, -1, self.n)
        except ValueError:
            # If lmbda is not coprime to n (very rare), retry
            self._generate_keys()

    def encrypt(self, m: int) -> int:
        """
        Encrypt plaintext integer m.
        c = g^m * r^n mod n^2
        """
        if self.n is None: raise ValueError("Keys not generated")
        
        # Random r in Z*_n
        while True:
            r = random.randrange(1, self.n)
            if math.gcd(r, self.n) == 1:
                break
                
        # c = (g^m mod n^2) * (r^n mod n^2) mod n^2
        # optimized: g^m = (1+n)^m = (1 + m*n) mod n^2
        c1 = (1 + m * self.n) % self.nsquare
        c2 = pow(r, self.n, self.nsquare)
        
        return (c1 * c2) % self.nsquare

    def decrypt(self, c: int) -> int:
        """
        Decrypt ciphertext c.
        m = L(c^lambda mod n^2) * mu mod n
        where L(x) = (x-1)/n
        
        Handles negative numbers: if m > n/2, returns m - n.
        """
        if self.lmbda is None: raise ValueError("Private key missing")
        
        u = pow(c, self.lmbda, self.nsquare)
        l_u = (u - 1) // self.n
        m = (l_u * self.mu) % self.n
        
        # Handle signed integers
        if m > self.n // 2:
            return m - self.n
        return m

    # --- Homomorphic Operations ---

    def add(self, c1: int, c2: int) -> int:
        """
        Add two encrypted values: E(m1) + E(m2)
        Result: E(m1 + m2) = c1 * c2 mod n^2
        """
        return (c1 * c2) % self.nsquare

    def add_scalar(self, c: int, k: int) -> int:
        """
        Add ciphertext and plaintext: E(m) + k
        Result: E(m + k) = c * g^k mod n^2
        """
        # g^k = (1 + k*n) mod n^2
        g_k = (1 + k * self.n) % self.nsquare
        return (c * g_k) % self.nsquare

    def multiply_scalar(self, c: int, k: int) -> int:
        """
        Multiply ciphertext by plaintext scalar: E(m) * k
        Result: E(m * k) = c^k mod n^2
        """
        return pow(c, k, self.nsquare)
