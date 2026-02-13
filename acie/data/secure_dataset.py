"""
Secure Data Loading for ACIE-H.

wraps standard datasets to provide Encrypted Inputs on-the-fly.
"""

import torch
from torch.utils.data import Dataset
from acie.data.dataset import ACIEDataset
from acie.cipher_embeddings.acie_h import ACIEHomomorphicCipher
from acie.cipher_embeddings.tensor import CipherTensor

class SecureACIEDataset(Dataset):
    """
    Wraps an ACIEDataset and encrypts the 'obs' (observation) vector.
    Used for Secure Inference where data must be encrypted before entering the model.
    """
    
    def __init__(self, base_dataset: ACIEDataset, cipher: ACIEHomomorphicCipher):
        self.base_dataset = base_dataset
        self.cipher = cipher
        print("SecureACIEDataset: Initialized. Note: Accessing items will be slow due to encryption.")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # 1. Get raw data
        sample = self.base_dataset[idx]
        raw_obs = sample["obs"] # Shape [Obs_Dim] (Float Tensor)
        
        # 2. Quantize (Float -> Int) for ACIE-H
        # Scale by 1000 to keep 3 decimal places
        SCALE = 1000
        quantized_obs = (raw_obs * SCALE).long().tolist()
        
        # 3. Encrypt
        # This returns a list of huge integers (ciphertexts)
        # We wrap it in a lightweight struct/list to pass to Collate
        enc_obs = [self.cipher.encrypt(x) for x in quantized_obs]
        
        return {
            "indices": idx,
            "enc_obs": enc_obs, # List[int], ready for CipherTensor
            "raw_target": sample["latent"] # Kept plain for validation comparison
        }

def secure_collate_fn(batch, cipher):
    """
    Custom collate to handle List[int] ciphertexts.
    Returns CipherTensor properly.
    """
    # Batch is list of dicts
    enc_obs_batch = [item["enc_obs"] for item in batch]
    raw_targets = torch.stack([item["raw_target"] for item in batch])
    
    # We can't stack Ciphertexts into a Tensor (they are too big for int64).
    # We return a list of CipherTensors, one per sample. 
    # Or a single CipherTensor if we implement batch logic (complex).
    # For this demo, we return a List of CipherTensor objects.
    
    secure_batch = [CipherTensor(data, cipher) for data in enc_obs_batch]
    
    return secure_batch, raw_targets
