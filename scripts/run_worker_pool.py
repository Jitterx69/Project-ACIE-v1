#!/usr/bin/env python3
"""
Worker Script for ACIE Distributed RAG Pipeline.
"""
import sys
import os
import torch
import torch.multiprocessing as mp
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from acie.rag.queue import TileJobQueue
from acie.rag.pipeline import HEImageRAGPipeline
from acie.rag.config import RAGConfig
from acie.logging.structured_logger import logger

def worker_main(worker_id: int):
    logger.info(f"Worker {worker_id} started (PID: {os.getpid()})")
    
    # Connect to Redis
    queue_manager = TileJobQueue()
    
    # Initialize Pipeline (Model weights loaded once per worker)
    # TODO: Load large models here so they persist
    config = RAGConfig.default()
    pipeline = HEImageRAGPipeline(config)
    
    # Fetch Context (for demo, context tensor is static or retrieved)
    context_key = "galaxy_cluster_v1"
    context_tensor = torch.randn(10, config.input_dim)
    pipeline.add_context(context_key, context_tensor)
    
    while True:
        try:
            # Poll for job (blocking)
            job = queue_manager.pop_job(timeout=5)
            
            if not job:
                # Idle loop or check for shutdown signal
                continue
                
            logger.info(f"Worker {worker_id} pulled job: {job['job_id']} (Tile: {job['tile_box']})")
            
            # --- PROCESS TILE logic ---
            image_path = job['image_path']
            x, y, w, h = job['tile_box']
            context_key = job['context_key']
            
            with Image.open(image_path) as img:
                tile = img.crop((x, y, x + w, y + h))
                
            # Encrypt
            encrypted_tile = pipeline.ingestion._encrypt_tensor(
                pipeline.ingestion.tile_transform(tile).view(-1)
            )
            
            # Retrieve / Generate
            metadata = {"query_key": context_key}
            context_weights = pipeline.retrieval.retrieve(metadata)
            encrypted_result = pipeline.generation(encrypted_tile, context_weights)
            
            # Decrypt
            result = pipeline.ingestion.decrypt_result(encrypted_result)
            
            # Save/Publish Result
            # In distributed system, we would upload 'result' to S3 or a DB,
            # and publish the reference. Here we just return metadata.
            result_summary = {
                "shape": list(result.shape),
                "mean": result.mean().item(),
                "std": result.std().item()
            }
            
            queue_manager.complete_job(job['job_id'], result_summary)
            logger.info(f"Worker {worker_id} completed job {job['job_id']}")
            
        except Exception as e:
            logger.error(f"Worker {worker_id} error: {e}")
            # Consider removing or re-queueing failed job
            
if __name__ == "__main__":
    # Launch multiple processes to act as workers
    num_workers = min(os.cpu_count(), 4)
    logger.info(f"Launching {num_workers} worker processes...")
    
    processes = []
    for i in range(num_workers):
        p = mp.Process(target=worker_main, args=(i,))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
