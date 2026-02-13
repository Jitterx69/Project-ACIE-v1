"""
Secure Neural Network Layers (ACIE-H).

Implements layers that can operate on Encrypted Data.
"""

import torch
import torch.nn as nn
from acie.cipher_embeddings.tensor import CipherTensor

class SecureLinear(nn.Module):
    """
    A Linear Layer that accepts Encrypted Input and produces Encrypted Output.
    
    Operation: y = x @ W^T + b
    - x: CipherTensor (Encrypted)
    - W: Plaintext Weights
    - b: Plaintext Bias
    - y: CipherTensor (Encrypted)
    
    Compatible with standard PyTorch weights.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard PyTorch parameters (Plaintext)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Initialize
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: CipherTensor) -> CipherTensor:
        """
        Forward pass with Encrypted Input.
        Note: Weights must be quantized to integers for ACIE-H.
        """
        # 1. Quantize Weights & Bias (Float -> Int)
        # Scale factor (e.g. 1000) to preserve precision
        SCALE = 1000
        w_int = (self.weight.t() * SCALE).long() # [In, Out]
        b_int = (self.bias * SCALE * SCALE).long() # Bias needs double scale if Input was scaled? 
        # Actually standard dot product: (x*S) * (w*S) = y * S^2. So bias needs S^2.
        
        # 2. Matrix Multiplication: x @ W.t
        # CipherTensor.matmul expects [In, Out] matrix. 
        # PyTorch stores weights as [Out, In], so we transpose.
        y = x.matmul(w_int)
        
        # 3. Add Bias
        y = y + b_int.tolist()
        
        return y

import math
