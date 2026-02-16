#!/usr/bin/env python3
"""
ACIE - Homomorphic Encryption RAG Pipeline Demo

This script demonstrates the usage of the HE RAG Pipeline.
It generates a sample image if one doesn't exist, runs the pipeline,
and prints the decryptedresults.

Usage:
    python examples/rag_demo.py
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from acie.rag.pipeline import HEImageRAGPipeline
from acie.rag.config import RAGConfig
from acie.logging.structured_logger import logger

def main():
    print("🚀 Starting ACIE-H RAG Pipeline Demo")
    
    # 1. Setup Sample Image
    image_path = "sample_galaxy.png"
    if not os.path.exists(image_path):
        print(f"Creating sample image: {image_path}")
        # Create a 28x28 grayscale image with random noise
        img_data = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
        img = Image.fromarray(img_data, mode='L')
        img.save(image_path)
    else:
        print(f"Using existing image: {image_path}")

    try:
        # 2. Configure Pipeline
        print("Initializing Pipeline...")
        # Use a smaller key for faster demo execution
        config = RAGConfig(
            key_size=512,  # 512-bit key for demo speed
            image_width=28,
            image_height=28,
            log_level="INFO"
        )
        pipeline = HEImageRAGPipeline(config=config)
        
        # 3. Add Knowledge Context
        # Simulating retrieving model weights or prototype vectors
        context_key = "galaxy_classifier_v1"
        print(f"Adding context for key: {context_key}")
        
        # Random weights for a 10-class classifier
        # Dimensions: [Output(10), Input(784)]
        context_weights = torch.randint(-5, 5, (10, 784)).long()
        pipeline.add_context(context_key, context_weights)
        
        # 4. Run Pipeline
        print("Running RAG Pipeline (Encrypt -> Retrieve -> Process -> Decrypt)...")
        metadata = {"query_key": context_key}
        
        result = pipeline.run(image_path, metadata)
        
        print("\n✅ Pipeline Execution Successful!")
        print("Decrypted Classification Logits:")
        print(result)
        
        # Simple analysis
        predicted_class = torch.argmax(result).item()
        print(f"Predicted Class: {predicted_class}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        # Print full traceback for debugging if needed
        import traceback
        traceback.print_exc()
        
    finally:
        print("\nDemo Complete.")
        # Note: We keep the sample image so the user can inspect it

if __name__ == "__main__":
    main()
