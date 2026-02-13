#!/usr/bin/env python3
"""
Project-Wide Encryption/Decryption Tool.

Uses the `acie/cipher_embeddings` engine to:
1. Recursive encrypt a folder (e.g. models/ or data/)
2. Recursive decrypt a folder

This addresses the user request to "expand crypto for the entire ML project".
"""

import sys
import argparse
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from acie.cipher_embeddings.engine import CryptoEngine

def main():
    parser = argparse.ArgumentParser(description="ACIE Project Encryption Tool")
    parser.add_argument("action", choices=["encrypt", "decrypt"], help="Action to perform")
    parser.add_argument("target_dir", help="Directory to process")
    parser.add_argument("--password", default="acie_project_secret", help="Encryption password")
    
    args = parser.parse_args()
    
    engine = CryptoEngine(password=args.password)
    target = Path(args.target_dir)
    
    if not target.exists():
        print(f"Error: Target directory '{target}' does not exist.")
        sys.exit(1)
        
    if args.action == "encrypt":
        print(f"🔒 Encrypting files in: {target}")
        count = engine.encrypt_directory(str(target))
        print(f"Done. Encrypted {count} files.")
        
    elif args.action == "decrypt":
        print(f"🔓 Decrypting files in: {target}")
        count = engine.decrypt_directory(str(target))
        print(f"Done. Decrypted {count} files.")

if __name__ == "__main__":
    main()
