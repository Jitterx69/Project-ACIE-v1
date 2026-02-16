import os
import torch
from PIL import Image
import numpy as np
from acie.rag.pipeline import HEImageRAGPipeline
from acie.rag.config import RAGConfig

def test_rag_pipeline():
    # 1. Setup Dummy Image
    image_path = "test_image.png"
    # Create a 28x28 grayscale image with some pattern
    img_data = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
    img = Image.fromarray(img_data, mode='L')
    img.save(image_path)
    
    try:
        # 2. Initialize Pipeline with Config
        # Use small key size for speed in testing
        config = RAGConfig(key_size=128)
        pipeline = HEImageRAGPipeline(config=config)
        
        # 3. Add Context (Simulate "Retrieval Knowledge")
        # For a 784 -> 10 linear layer, weights are [10, 784]
        # We simulate retrieval by adding weights for a specific "model version" or "concept"
        context_key = "model_v1"
        # Weights must be quantized (long) for SecureLinear
        # Values should be small enough to avoid overflow during test with small key
        context_weights = torch.randint(-10, 10, (10, 784)).long()
        pipeline.add_context(context_key, context_weights)
        
        # 4. Run Pipeline
        metadata = {"query_key": context_key}
        result = pipeline.run(image_path, metadata)
        
        # 5. Verify Result
        print("Decrypted Result:", result)
        assert result.shape == (10,), f"Expected shape (10,), got {result.shape}"
        assert not torch.isnan(result).any(), "Result contains NaNs"
        
        print("Test Passed!")
        
    finally:
        # Cleanup
        if os.path.exists(image_path):
            os.remove(image_path)

if __name__ == "__main__":
    test_rag_pipeline()
