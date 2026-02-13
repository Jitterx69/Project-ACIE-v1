#!/usr/bin/env python3
"""
Secure Image Classification Pipeline.

Implements the user-requested workflow:
1. Input Image
2. "Strings" extraction & GOST Hash (for Integrity/ID)
3. CNN Classification (using pixel data)
4. RC2 Encryption (for storage)
5. Decryption & Verification

Dependencies: standard python libs, pytorch, openssl command line tool
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from acie.models.cnn import SecureImageCNN
from acie.cipher_embeddings.engine import CryptoEngine

def secure_pipeline_demo():
    print("="*60)
    print("SECURE IMAGE CLASSIFICATION PIPELINE")
    print("="*60)
    
    # Initialize Crypto Engine
    crypto = CryptoEngine(password="mobile_secure_acie_key")
    
    # 1. Generate Dummy Image (if not exists)
    img_path = "galaxy_sample.png"
    print(f"\n[1] Input: Generating dummy astronomical image '{img_path}'...")
    
    # Create a random noise image simulating a galaxy
    data = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    img = Image.fromarray(data)
    img.save(img_path)
    
    # 2. Strings & GOST Hash
    print("\n[2] Integrity: Generating GOST Hash (via CryptoEngine)...")
    
    # Uses engine to run: strings <file> | openssl dgst -md_gost94
    img_hash = crypto.gost_hash_file(img_path)
    print(f"    GOST Hash ID: {img_hash}")
    
    # 3. CNN Classification
    print("\n[3] Classification: Running CNN on image pixels...")
    
    # Prepare image for PyTorch
    img_tensor = torch.tensor(data).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0) # [1, 3, 64, 64]
    
    # Init model
    model = SecureImageCNN(num_classes=2) 
    model.eval()
    
    with torch.no_grad():
        probs = model(img_tensor)
        predicted_class = torch.argmax(probs).item()
        confidence = probs[0][predicted_class].item()
        
    classes = ["Spiral Galaxy", "Elliptical Galaxy"]
    print(f"    Prediction: {classes[predicted_class]}")
    print(f"    Confidence: {confidence:.4f}")
    
    # 4. RC2 Encryption
    print("\n[4] Security: Encrypting image with RC2 (via CryptoEngine)...")
    enc_path = f"{img_path}.rc2"
    
    try:
        crypto.encrypt_file_rc2(img_path, enc_path)
        print(f"    Encrypted file saved to: {enc_path}")
    except Exception as e:
        print(f"    Encryption failed: {e}")
        return
    
    # 5. Verification (Decrypt & Re-hash)
    print("\n[5] Verification: Decrypting and verifying integrity...")
    dec_path = "decrypted_galaxy.png"
    
    try:
        crypto.decrypt_file_rc2(enc_path, dec_path)
        print(f"    Decrypted to: {dec_path}")
        
        # Re-hash
        verify_hash = crypto.gost_hash_file(dec_path)
        
        print(f"    Original Hash: {img_hash}")
        print(f"    Verified Hash: {verify_hash}")
        
        if img_hash == verify_hash:
            print("    ✅ SUCCESS: Integrity Verified!")
        else:
            print("    ❌ FAILURE: Hash mismatch!")

    except Exception as e:
        print(f"    Verification failed: {e}")

    # Cleanup
    if os.path.exists(img_path): os.remove(img_path)
    if os.path.exists(enc_path): os.remove(enc_path)
    if os.path.exists(dec_path): os.remove(dec_path)
    print("\nPipeline Demo Complete.")

if __name__ == "__main__":
    secure_pipeline_demo()
