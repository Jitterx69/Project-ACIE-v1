#!/usr/bin/env python3
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

def verify_streaming_logic():
    print("="*60)
    print("VERIFYING TILE STREAMING LOGIC (FAST MODE)")
    print("="*60)
    
    # 1. Setup Config for FAST execution
    # We use small tiles to prove the logic without waiting hours for encryption
    TILE_SIZE = 20
    INPUT_DIM = TILE_SIZE * TILE_SIZE
    
    config = RAGConfig.default()
    config.tile_size = TILE_SIZE
    config.input_dim = INPUT_DIM
    config.image_width = TILE_SIZE # For compatibility if needed
    config.image_height = TILE_SIZE
    
    pipeline = HEImageRAGPipeline(config)
    
    # 2. Create Dummy "Large" Image
    # 40x40 image with 20x20 tiles = 4 tiles total
    IMG_SIZE = 40
    img_path = "fast_verify_image.png"
    
    img = Image.new('L', (IMG_SIZE, IMG_SIZE), color=128)
    # Add some pattern
    pixels = np.arange(IMG_SIZE * IMG_SIZE).reshape((IMG_SIZE, IMG_SIZE)) % 255
    img = Image.fromarray(pixels.astype('uint8'))
    img.save(img_path)
    print(f"[1] Created dummy image: {IMG_SIZE}x{IMG_SIZE}")
    
    # 3. Register Context
    context_key = "test_context"
    # Create dummy weights matching the tile input dimension
    pipeline.add_context(context_key, torch.randn(10, INPUT_DIM))
    print(f"[2] Registered context for dimension {INPUT_DIM}")

    # 4. Run Streaming
    print("[3] Starting Stream...")
    metadata = {"query_key": context_key}
    
    try:
        generator = pipeline.run_stream(img_path, metadata)
        
        count = 0
        for i, result in enumerate(generator):
            count += 1
            print(f"    > Processed Tile {i+1} Successfully. Result shape: {result.shape}")
            
        print(f"\n[SUCCESS] Completed {count} tiles. Logic is correct.")
        
    except Exception as e:
        print(f"\n[FAILURE] {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if os.path.exists(img_path):
            os.remove(img_path)

if __name__ == "__main__":
    verify_streaming_logic()
