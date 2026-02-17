#!/usr/bin/env python3
"""
Pipeline Orchestrator: Splits images into tiles and pushes jobs to Redis queue.
"""
import sys
import os
import json
import time
import uuid
import argparse
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from acie.rag.queue import TileJobQueue
from acie.logging.structured_logger import logger
from acie.rag.config import RAGConfig

def main():
    parser = argparse.ArgumentParser(description="Distributed RAG Pipeline Orchestrator")
    parser.add_argument("image_path", help="Path to large input image")
    parser.add_argument("--tile-size", type=int, default=1024, help="Tile size (pixels)")
    parser.add_argument("--context-key", default="galaxy_cluster_v1", help="Context key for retrieval")
    
    args = parser.parse_args()
    
    # 1. Image Check
    if not os.path.exists(args.image_path):
        logger.error(f"Image {args.image_path} not found.")
        sys.exit(1)
        
    config = RAGConfig.default()
    config.tile_size = args.tile_size
    
    # 2. Queue Setup
    # The queue manager handles fallback to file-based queue if Redis is unavailable
    queue_manager = TileJobQueue()
        
    # 3. Job Generation (Split Image)
    job_group_id = str(uuid.uuid4())
    logger.info(f"Starting distribution for Image: {args.image_path} (Group ID: {job_group_id})")
    
    jobs = []
    with Image.open(args.image_path) as img:
        width, height = img.size
        logger.info(f"Image Dimensions: {width}x{height}")
        
        for i in range(0, width, config.tile_size):
            for j in range(0, height, config.tile_size):
                box = (i, j, width, height) # Just bounds
                # We save only the box coordinates to the job payload,
                # worker loads the image path (assuming shared filesystem or URL)
                
                # Push Job
                job_id = queue_manager.push_job(
                    image_path=args.image_path,
                    tile_box=box,
                    context_key=args.context_key,
                    job_group_id=job_group_id
                )
                jobs.append(job_id)
                
    logger.info(f"Pushed {len(jobs)} jobs to queue. Monitoring progress...")
    
    # 4. Monitor Progress with TQDM
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False
        
    completed_jobs = 0
    start_time = time.time()
    
    # Prepare result storage
    results_file = f"pipeline_results_{job_group_id}.json"
    all_results = {}
    
    if has_tqdm:
        pbar = tqdm(total=len(jobs), unit="tile")
        
    while completed_jobs < len(jobs):
        time.sleep(2)
        
        # Check status of jobs
        current_completed = 0
        current_results = {}
        
        # In a real system, we'd query by group_id efficiently
        # Here we poll individuallly (okay for <1000 tiles in demo)
        for job_id in jobs:
            job_data = queue_manager.get_job_data(job_id) # Need consistent way to get data
            if job_data and job_data.get("status") == "completed":
                current_completed += 1
                current_results[job_id] = job_data.get("result")
                
        delta = current_completed - completed_jobs
        if delta > 0:
            if has_tqdm:
                pbar.update(delta)
            else:
                logger.info(f"Progress: {current_completed}/{len(jobs)} tiles processed.")
            
            # Save intermediate results
            all_results.update(current_results)
            with open(results_file, 'w') as f:
                json.dump({
                    "group_id": job_group_id,
                    "total_tiles": len(jobs),
                    "completed": current_completed,
                    "results": all_results
                }, f, indent=2)
                
            completed_jobs = current_completed
            
    if has_tqdm:
        pbar.close()
        
    total_time = time.time() - start_time
    logger.info(f"Distributed processing complete in {total_time:.2f}s. Results saved to {results_file}")

if __name__ == "__main__":
    main()
