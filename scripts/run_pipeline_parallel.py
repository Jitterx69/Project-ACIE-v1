#!/usr/bin/env python3
import sys
import os
import torch
import torch.multiprocessing as mp
from pathlib import Path
from PIL import Image

# Add project root to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from acie.rag.pipeline import HEImageRAGPipeline
from acie.rag.config import RAGConfig
from acie.logging.structured_logger import logger

# Global pipeline instance for workers (prevent pickling issues)
pipeline = None

def worker_init(config_dict, context_key, context_tensor):
    """Initialize global pipeline in each worker process."""
    global pipeline
    # Reconstruct config from dict
    config = RAGConfig(**config_dict)
    
    # Initialize fresh pipeline instance per process 
    # (Avoids sharing connection/state issues)
    pipeline = HEImageRAGPipeline(config)
    pipeline.add_context(context_key, context_tensor)
    logger.info(f"Worker initialized in PID {os.getpid()}")

def process_tile_task(tile_args):
    """
    Worker function to process a single tile.
    Args:
        tile_args: tuple(image_path, box, context_key)
    """
    global pipeline
    image_path, box, context_key = tile_args
    try:
        # Load just the specific tile region
        # This keeps memory low per worker
        with Image.open(image_path) as img:
            tile = img.crop(box)
            
        # Transform & Encrypt
        # Pipeline ingestion returns (enc_tensor, box) tuple now
        encrypted_tile = pipeline.ingestion._encrypt_tensor(
            pipeline.ingestion.tile_transform(tile).view(-1)
        )
        
        # Retrieve Context (already in memory from init)
        metadata = {"query_key": context_key}
        context_weights = pipeline.retrieval.retrieve(metadata)
        
        # Secure Generation
        encrypted_result = pipeline.generation(encrypted_tile, context_weights)
        
        # Decrypt
        result = pipeline.ingestion.decrypt_result(encrypted_result)
        
        return (box, result) # Return result and location
        
    except Exception as e:
        logger.error(f"Tile {box} failed: {e}")
        return None

def main():
    # Use 'spawn' for CUDA/PyTorch compatibility
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    # 1. Setup Configuration
    config = RAGConfig.default()
    # tile_size = 1024 for real use (or config default)
    
    # 2. Get Image Path
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "large_input_sample.png"
        if not os.path.exists(image_path):
            print("Generating demo image...")
            from scripts.large_scale_rag_demo import create_large_dummy_image
            create_large_dummy_image(image_path, 2048, 2048)

    print(f"Starting PARALLEL pipeline for {image_path}...")

    # 3. Prepare Shared Context
    context_key = "galaxy_cluster_v1"
    # Create random context once in main process
    context_tensor = torch.randn(10, config.input_dim)
    
    # Convert config to dict for passing to workers
    config_dict = config.__dict__

    # 4. Generate Task List (Do not load image yet)
    tasks = []
    with Image.open(image_path) as img:
        width, height = img.size
        print(f"Image Size: {width}x{height}")
        
        for i in range(0, width, config.tile_size):
            for j in range(0, height, config.tile_size):
                box = (i, j, min(i + config.tile_size, width), min(j + config.tile_size, height))
                tasks.append((image_path, box, context_key))
                
    print(f"Generated {len(tasks)} tile tasks.")
    
    # 5. Launch Pool
    num_workers = min(os.cpu_count(), 8) # Cap at 8 workers or CPU count
    print(f"Launching {num_workers} workers...")
    
    try:
        with mp.Pool(processes=num_workers, initializer=worker_init, initargs=(config_dict, context_key, context_tensor)) as pool:
            # Use imap_unordered for streaming results as they finish
            for i, result in enumerate(pool.imap_unordered(process_tile_task, tasks)):
                if result:
                    box, tensor = result
                    print(f"[{i+1}/{len(tasks)}] Tile {box} finished. Shape: {tensor.shape}")
                else:
                    print(f"[{i+1}/{len(tasks)}] Tile failed.")
    except RuntimeError as e:
        logger.warning(f"Parallel execution failed ({e}). Falling back to serial processing...")
        # Fallback to serial
        # Initialize pipeline for serial
        if pipeline is None:
            worker_init(config_dict, context_key, context_tensor)
            
        for i, task in enumerate(tasks):
            result = process_tile_task(task)
            if result:
                box, tensor = result
                print(f"[{i+1}/{len(tasks)}] (Serial) Tile {box} finished. Shape: {tensor.shape}")
            else:
                print(f"[{i+1}/{len(tasks)}] (Serial) Tile failed.")

    print("Pipeline complete.")

if __name__ == "__main__":
    main()
