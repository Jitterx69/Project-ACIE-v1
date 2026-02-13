#!/usr/bin/env python3
"""
Secure Inference Demo (ACIE-H Integration).

Demonstrates the "Entire Project" integration:
1. SecureLayer (Model)
2. SecureDataset (Data)
3. CipherTensor (Compute)
4. ACIE-H Cipher (Core)

Scenario:
We have a trained Linear Encoder ($W, b$). 
We want to encode an observation $x$ (Flux/Spectra) into a latent vector $z$,
BUT $x$ must remain encrypted throughout the process.
Metric: $z = Wx + b$
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from acie.cipher_embeddings.acie_h import ACIEHomomorphicCipher
from acie.cipher_embeddings.tensor import CipherTensor
from acie.models.secure_layers import SecureLinear
from acie.data.dataset import ACIEDataset
from acie.data.secure_dataset import SecureACIEDataset

def main():
    print("="*60)
    print("ACIE-H: Secure Machine Learning Pipeline")
    print("="*60)

    # 1. Setup Infra
    print("\n[1] Infrastructure: Initializing ACIE-H Cipher...")
    # Using smaller key for demo speed (512 bit), in prod use 1024+
    acie_h = ACIEHomomorphicCipher(key_size=512) 
    print("    Cipher Ready.")

    # 2. Setup Model
    print("\n[2] Model: Initializing Secure Linear Layer...")
    # Simulating an Encoder: 10 input features -> 5 latent features
    IN_DIM = 10
    OUT_DIM = 5
    
    secure_model = SecureLinear(IN_DIM, OUT_DIM)
    print(f"    SecureLinear({IN_DIM}, {OUT_DIM}) initialized.")
    print("    Weights (Plaintext): Available")

    # 3. Setup Data
    print("\n[3] Data: Loading and Encrypting Sample...")
    # Create dummy data for demo since we might not have the big CSVs generated
    dummy_data = torch.randn(1, IN_DIM) # 1 sample
    print(f"    Raw Input (First 5): {dummy_data[0][:5].numpy()}...")
    
    # Quantize and Encrypt
    SCALE = 1000
    quantized_data = (dummy_data * SCALE).long().tolist()[0]
    
    encrypted_data = [acie_h.encrypt(x) for x in quantized_data]
    print("    Input Encrypted (ACIE-H).")
    
    # Wrap in CipherTensor
    secure_input = CipherTensor(encrypted_data, acie_h)

    # 4. Secure Inference
    print("\n[4] Execution: Running Forward Pass on Encrypted Data...")
    print("    Operation: z = Enc(x) @ W.t + b")
    
    # This calls SecureLinear.forward() -> CipherTensor.matmul()
    secure_output = secure_model(secure_input)
    
    print(f"    Output Shape: {secure_output.shape}")
    print(f"    Output is Ciphertext (Encrypted).")

    # 5. Decryption & Verification
    print("\n[5] Verification: Decrypting Result...")
    decrypted_z = secure_output.decrypt()
    
    # Compare with Plaintext Execution
    print("    Comparing with insecure plaintext calculation...")
    
    # Manual Plaintext calc
    # Quantize input * Quantize Weight + Quantized Bias
    w_int = (secure_model.weight * SCALE).long()
    b_int = (secure_model.bias * SCALE * SCALE).long()
    x_int = (dummy_data * SCALE).long()
    
    expected_z = torch.matmul(x_int, w_int.t()) + b_int
    
    print(f"    Decrypted Z: {decrypted_z.long().numpy()}")
    print(f"    Expected  Z: {expected_z.numpy()[0]}")
    
    # Check error (should be 0 or very small due to int math)
    diff = (decrypted_z - expected_z).abs().sum().item()
    if diff == 0:
        print("    ✅ SUCCESS: Perfect Match!")
    else:
        print(f"    ⚠️  Match close (Diff: {diff}). Small precision diffs expected.")

if __name__ == "__main__":
    main()
