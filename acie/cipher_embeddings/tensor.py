"""
CipherTensor: A PyTorch-like wrapper for ACIE-H Encrypted Data.

Allows treating encrypted integers as a tensor for high-level operations.
Supports:
- Addition: Enc + Enc, Enc + Plain
- Multiplication: Enc * Plain (Scalar)
- MatMul: Enc_Vector @ Plain_Matrix
"""

import torch
from typing import List, Union
from .acie_h import ACIEHomomorphicCipher

class CipherTensor:
    def __init__(self, data: List[int], cipher: ACIEHomomorphicCipher):
        """
        data: List of ciphertexts (integers)
        cipher: The ACIE-H instance used for operations
        """
        self.data = data
        self.cipher = cipher
        self.shape = (len(data),)

    def decrypt(self) -> torch.Tensor:
        """Decrypts all values and returns a PyTorch FloatTensor."""
        decrypted = [self.cipher.decrypt(c) for c in self.data]
        return torch.tensor(decrypted, dtype=torch.float32)

    def __repr__(self):
        return f"CipherTensor(shape={self.shape}, cipher=ACIE-H, key_size={self.cipher.key_size})"

    def __add__(self, other):
        """
        Supports:
        - CipherTensor + CipherTensor (Element-wise)
        - CipherTensor + int (Scalar add)
        - CipherTensor + List[int] (Element-wise scalar add)
        """
        new_data = []
        
        if isinstance(other, CipherTensor):
            if len(self.data) != len(other.data):
                raise ValueError("Shape mismatch for addition")
            for c1, c2 in zip(self.data, other.data):
                new_data.append(self.cipher.add(c1, c2))
                
        elif isinstance(other, int):
            for c in self.data:
                new_data.append(self.cipher.add_scalar(c, other))
                
        elif isinstance(other, (list, torch.Tensor)):
            if len(self.data) != len(other):
                raise ValueError("Shape mismatch for addition")
            for c, val in zip(self.data, other):
                new_data.append(self.cipher.add_scalar(c, int(val)))
        else:
            raise TypeError(f"Unsupported type for addition: {type(other)}")
            
        return CipherTensor(new_data, self.cipher)

    def __mul__(self, other):
        """
        Supports:
        - CipherTensor * int (Scalar multiplication)
        - CipherTensor * List[int] (Element-wise scalar multiplication)
        """
        new_data = []
        
        if isinstance(other, int):
            for c in self.data:
                new_data.append(self.cipher.multiply_scalar(c, other))
                
        elif isinstance(other, (list, torch.Tensor)):
            if len(self.data) != len(other):
                raise ValueError("Shape mismatch for multiplication")
            for c, val in zip(self.data, other):
                new_data.append(self.cipher.multiply_scalar(c, int(val)))
        else:
            raise TypeError(f"Unsupported type for multiplication: {type(other)}")
            
        return CipherTensor(new_data, self.cipher)

    def matmul(self, matrix: Union[List[List[int]], torch.Tensor]) -> 'CipherTensor':
        """
        Perform Vector-Matrix Multiplication: y = x @ W
        x: Encrypted Vector [D_in]
        W: Plaintext Matrix [D_in, D_out]
        y: Encrypted Vector [D_out]
        
        Rule: y[j] = Sum(x[i] * W[i][j])
        """
        # Convert torch tensor to list if needed
        if isinstance(matrix, torch.Tensor):
            matrix = matrix.tolist()
            
        # Check dimensions
        d_in = len(self.data)
        if len(matrix) != d_in:
            raise ValueError(f"Shape mismatch: Vector {d_in} vs Matrix {len(matrix)}x?")
            
        d_out = len(matrix[0])
        new_data = []
        
        # For each output neuron
        for j in range(d_out):
            # Calculate dot product: Sum(x[i] * W[i][j])
            # Initialize with Enc(0)
            accum = self.cipher.encrypt(0)
            
            for i in range(d_in):
                weight = int(matrix[i][j])
                # Enc(x_i) * weight
                weighted_val = self.cipher.multiply_scalar(self.data[i], weight)
                # Accumulate
                accum = self.cipher.add(accum, weighted_val)
                
            new_data.append(accum)
            
        return CipherTensor(new_data, self.cipher)
