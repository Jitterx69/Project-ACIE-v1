"""
Core Cryptography Engine for ACIE.

Handles:
- GOST Hashing (Integrity)
- RC2 Encryption/Decryption (Confidentiality)

Uses OpenSSL subprocess calls for implementation due to library limitations.
"""

import subprocess
import os
import sys
from pathlib import Path
from typing import Optional

class CryptoEngine:
    """
    Wrapper for OpenSSL cryptographic operations.
    """
    
    def __init__(self, password: str = "default_acie_secure_key"):
        self.password = password
        self._check_openssl()

    def _check_openssl(self):
        """Verify OpenSSL is available."""
        try:
            subprocess.run(
                "openssl version", 
                shell=True, 
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError:
            print("Warning: OpenSSL not found. Crypto operations will fail.")

    def _run_command(self, cmd: str) -> str:
        """Run shell command and return output."""
        try:
            result = subprocess.run(
                cmd, 
                shell=True, 
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Crypto Command Failed: {cmd}\nError: {e.stderr}")

    def gost_hash_file(self, file_path: str) -> str:
        """
        Generate GOST hash for a file.
        Using: strings <file> | openssl dgst -md_gost94
        Fallback: SHA256 if GOST unavailable.
        """
        # Check for GOST support (once per instance or lazily?)
        # For robustness, we check availability 
        try:
            self._run_command("openssl dgst -md_gost94 < /dev/null")
            algo = "md_gost94"
        except RuntimeError:
            algo = "sha256"
            
        # Use 'strings' to get printable content as requested, or just hash file directly
        # The user specifically requested 'strings' interaction
        cmd = f"strings '{file_path}' | openssl dgst -{algo}"
        output = self._run_command(cmd)
        
        # Parse output format "algo(stdin)= hash"
        if "= " in output:
            return output.split("= ")[1]
        return output

    def encrypt_file_rc2(self, input_path: str, output_path: str = None) -> str:
        """
        Encrypt file using RC2-CBC.
        """
        if output_path is None:
            output_path = f"{input_path}.rc2"
            
        cmd = (
            f"openssl enc -rc2 -e -in '{input_path}' -out '{output_path}' "
            f"-k '{self.password}' -pbkdf2"
        )
        self._run_command(cmd)
        return output_path

    def decrypt_file_rc2(self, input_path: str, output_path: str = None) -> str:
        """
        Decrypt file using RC2-CBC.
        """
        if output_path is None:
            # Try to remove .rc2 extension
            if input_path.endswith(".rc2"):
                output_path = input_path[:-4]
            else:
                output_path = f"{input_path}.decrypted"
                
        cmd = (
            f"openssl enc -rc2 -d -in '{input_path}' -out '{output_path}' "
            f"-k '{self.password}' -pbkdf2"
        )
        self._run_command(cmd)
        return output_path

    def encrypt_directory(self, dir_path: str):
        """
        Recursively encrypt all files in a directory.
        Skips already encrypted (.rc2) files.
        """
        root = Path(dir_path)
        if not root.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
            
        print(f"Encrypting directory: {dir_path}")
        count = 0
        for path in root.rglob("*"):
            if path.is_file() and not path.name.endswith(".rc2"):
                try:
                    self.encrypt_file_rc2(str(path))
                    # Optionally remove original? 
                    # os.remove(path) 
                    # Keeping original for safety in this demo
                    count += 1
                except Exception as e:
                    print(f"Failed to encrypt {path}: {e}")
        return count

    def decrypt_directory(self, dir_path: str):
        """
        Recursively decrypt all .rc2 files in a directory.
        """
        root = Path(dir_path)
        if not root.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
            
        print(f"Decrypting directory: {dir_path}")
        count = 0
        for path in root.rglob("*.rc2"):
            if path.is_file():
                try:
                    self.decrypt_file_rc2(str(path))
                    count += 1
                except Exception as e:
                    print(f"Failed to decrypt {path}: {e}")
        return count
