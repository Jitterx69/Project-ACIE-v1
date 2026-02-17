#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Add project root to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from acie.rag.pipeline import HEImageRAGPipeline
from acie.logging.structured_logger import logger

def main():
    # Initialize pipeline
    # Initialize pipeline
    pipeline = HEImageRAGPipeline()
    
    # Parse command line argument or use default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"Using user-provided image: {image_path}")
    else:
        # Fallback to demo image
        image_path = "large_input_sample.png"
        if not os.path.exists(image_path):
            print(f"File {image_path} not found. Running demo generator first...")
            # Create a dummy file if needed for the test to work
            from scripts.large_scale_rag_demo import create_large_dummy_image
            create_large_dummy_image(image_path, 2048, 2048)

    print(f"Starting streaming pipeline for {image_path}...")
    
    # Metadata for retrieval context
    context_key = "galaxy_cluster_v1"
    metadata = {"query_key": context_key}

    # Register dummy context for the demo run
    # In production, this would be loaded from a database or file
    import torch
    # Context dimension must match input_dim (1024*1024)
    # Using a small random tensor to simulate weights for the 1048576 dim
    # For a real Linear layer simulation, we need [out_dim, in_dim] or similiar
    # but the SecureGenerationModel likely expects a specific shape or handles it.
    # Let's check config.input_dim
    input_dim = pipeline.config.input_dim
    pipeline.add_context(context_key, torch.randn(10, input_dim))

    # This will now work for 5GB+ files without crashing RAM
    try:
        result_generator = pipeline.run_stream(image_path, metadata=metadata)

        for i, tile_result in enumerate(result_generator):
            # Process each tile result here
            # For example, save to disk or aggregate statistics
            print(f"Processed tile {i+1}, result shape: {tile_result.shape}")
            
        print("Pipeline completed successfully.")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
