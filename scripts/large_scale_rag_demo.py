#!/usr/bin/env python3
"""
Large Scale RAG Demo using Tiled Streaming.

This script demonstrates how to process large-scale images (e.g., 5GB+) using
tiled streaming ingestion to prevent memory exhaustion.
"""

import sys
import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from acie.rag.pipeline import HEImageRAGPipeline
from acie.rag.config import RAGConfig
from acie.logging.structured_logger import logger

def create_large_dummy_image(path: str, width: int, height: int):
    """Creates a large dummy image (black/white noise) for testing."""
    logger.info(f"Creating dummy large image ({width}x{height}) at {path}...")
    # Use random noise
    # To save memory during creation, we could write chunks, but for this demo 
    # we'll use a moderate size that fits in RAM but exercises the tiler.
    # 4096 x 4096 = 16 tiles of 1024x1024
    
    # Create empty image
    img = Image.new('L', (width, height))
    
    # Fill with noise tile by tile (simulation)
    # Actually, let's just create a gradient or simple pattern
    pixels = np.random.randint(0, 255, (height, width), dtype=np.uint8)
    img = Image.fromarray(pixels)
    img.save(path)
    logger.info(f"Image created: {os.path.getsize(path) / (1024*1024):.2f} MB")

def run_large_scale_demo():
    print("="*60)
    print("LARGE SCALE SECURE RAG PIPELINE DEMO")
    print("="*60)
    
    # 1. Configuration
    # We set tile_size to 512 for this demo to see more tiles, 
    # but in production for 5GB images use 1024 or higher.
    config = RAGConfig.default()
    config.tile_size = 512 
    config.image_width = 512 # Set tile dimensions for model compatibility
    config.image_height = 512
    config.input_dim = 512 * 512
    
    pipeline = HEImageRAGPipeline(config)
    
    # 2. Setup Data
    img_path = "large_input_sample.png"
    # Create a 2048x2048 image -> 16 tiles of 512x512
    create_large_dummy_image(img_path, 2048, 2048)
    
    # Add dummy context
    context_key = "galaxy_cluster_v1"
    # Dummy context tensor (weights) - must match input_dim + bias or similar depending on SecureGenerationModel
    # Assuming SecureGenerationModel uses a linear layer Wx + b structure where context provides weights.
    # For this demo, we'll just register a placeholder.
    pipeline.add_context(context_key, torch.randn(10, config.input_dim)) 

    # 3. Run Streaming Pipeline
    print(f"\n[Action] Streaming processing for {img_path}...")
    
    metadata = {"query_key": context_key}
    
    try:
        # Get generator
        result_stream = pipeline.run_stream(img_path, metadata)
        
        tile_count = 0
        for i, result_tile in enumerate(result_stream):
            tile_count += 1
            # Result should be a tensor (e.g. classification logits or embedding)
            print(f"  > Processed Tile {i+1}: Result Shape {result_tile.shape}")
            
        print(f"\n[Success] Completed processing {tile_count} tiles.")
        
    except Exception as e:
        print(f"\n[Error] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

    # Cleanup
    if os.path.exists(img_path):
        os.remove(img_path)
        print("Cleanup complete.")

if __name__ == "__main__":
    run_large_scale_demo()
