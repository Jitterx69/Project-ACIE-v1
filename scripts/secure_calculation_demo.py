#!/usr/bin/env python3
"""
Secure Calculation Demo (ACIE-H Homomorphic Encryption).

Demonstrates performing calculations on data while it remains encrypted.
Scenario: Calculating the weighted average flux of a galaxy observation.

Workflow:
1. Input: [Flux1, Flux2, Flux3]
2. Encrypt: [Enc(F1), Enc(F2), Enc(F3)]
3. Calculate: Sum(Enc(Fi) * Weight_i)
   * performed WITHOUT decrypting *
4. Decrypt Result: Get the final weighted sum.
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from cipher_embeddings.acie_h import ACIEHomomorphicCipher, CipherTensor

def main():
    print("="*60)
    print("ACIE-H: Secure Calculation Demo (Homomorphic)")
    print("="*60)

    # 1. Setup Cipher
    print("\n[1] Initializing ACIE-H Cipher System...")
    print("    Generating 1024-bit keys (this may take a moment)...")
    acie_h = ACIEHomomorphicCipher(key_size=1024)
    print("    Keys Generated.")
    print(f"    Public Key (n): {str(acie_h.n)[:20]}...")

    # 2. Input Data (Simulated Galaxy Flux in 3 bands)
    inputs = [150, 200, 350]
    weights = [2, 3, 1] # Weighted importance
    
    print(f"\n[2] Input Data: {inputs}")
    print(f"    Weights:    {weights}")
    print(f"    Expected Weighted Sum: {sum(i*w for i,w in zip(inputs, weights))}")

    # 3. Encryption
    print("\n[3] Encrypting Inputs...")
    enc_inputs = []
    for x in inputs:
        # Encrypt the integer
        ciphertext = acie_h.encrypt(x)
        # Wrap it in CipherTensor for easy math
        enc_tensor = CipherTensor(ciphertext, acie_h)
        enc_inputs.append(enc_tensor)
        print(f"    Enc({x}) = {str(ciphertext)[:30]}...")

    # 4. Calculation in Ciphered Format
    # Formula: Result = Sum(Enc(Input_i) * Weight_i)
    # This involves: Scalar Multiplication and Encrypted Addition
    print("\n[4] Performing Calculation in Ciphered Format...")
    print("    Operation: Σ (Enc(Input) * Weight)")
    
    # Initialize with Enc(0)
    enc_result = CipherTensor(acie_h.encrypt(0), acie_h)
    
    for enc_val, w in zip(enc_inputs, weights):
        # Homomorphic Multiply by Scalar: Enc(val * w)
        weighted_val = enc_val * w 
        
        # Homomorphic Add: Enc(sum + val*w)
        enc_result = enc_result + weighted_val
        
    print(f"    Result Ciphertext: {str(enc_result.value)[:50]}...")

    # 5. Decryption
    print("\n[5] Decrypting Result...")
    decrypted_sum = enc_result.decrypt()
    
    print(f"    Decrypted Value: {decrypted_sum}")
    
    # Validation
    expected = sum(i*w for i,w in zip(inputs, weights))
    if decrypted_sum == expected:
        print(f"    ✅ SUCCESS: Analyzed in ciphered format correctly!")
        print(f"       {inputs} -> Encrypted -> Calculated -> {decrypted_sum}")
    else:
        print(f"    ❌ FAILURE: Expected {expected}, got {decrypted_sum}")

if __name__ == "__main__":
    main()
