import json
import os
import uuid
import time
import glob
import shutil
from typing import Dict, Any, Optional
from acie.logging.structured_logger import logger

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

class FileJobQueue:
    """
    A file-system based queue for distributed jobs.
    Useful when Redis is not available.
    Structure:
    - .queue/pending/
    - .queue/processing/
    - .queue/completed/
    """
    def __init__(self, base_dir: str = ".acie_queue"):
        self.base_dir = base_dir
        self.pending_dir = os.path.join(base_dir, "pending")
        self.processing_dir = os.path.join(base_dir, "processing")
        self.completed_dir = os.path.join(base_dir, "completed")
        
        for d in [self.pending_dir, self.processing_dir, self.completed_dir]:
            os.makedirs(d, exist_ok=True)

    def push_job(self, image_path: str, tile_box: tuple, context_key: str, job_group_id: str) -> str:
        job_id = str(uuid.uuid4())
        job_payload = {
            "job_id": job_id,
            "group_id": job_group_id,
            "image_path": image_path,
            "tile_box": tile_box,
            "context_key": context_key,
            "timestamp": time.time(),
            "status": "pending"
        }
        
        # Write to pending
        job_file = os.path.join(self.pending_dir, f"{job_id}.json")
        with open(job_file, 'w') as f:
            json.dump(job_payload, f)
            
        return job_id

    def pop_job(self, timeout: int = 5) -> Optional[Dict[str, Any]]:
        # Simple polling
        start_time = time.time()
        while time.time() - start_time < timeout:
            pending_files = glob.glob(os.path.join(self.pending_dir, "*.json"))
            if not pending_files:
                time.sleep(0.1)
                continue
                
            # Try to acquire a job
            # We pick the first one and try to atomic rename it to processing
            target_file = pending_files[0]
            filename = os.path.basename(target_file)
            processing_file = os.path.join(self.processing_dir, filename)
            
            try:
                os.rename(target_file, processing_file)
                # Parse and return
                with open(processing_file, 'r') as f:
                    return json.load(f)
            except OSError:
                # Race condition: someone else took it
                continue
                
        return None

    def complete_job(self, job_id: str, result_metadata: Dict[str, Any]):
        filename = f"{job_id}.json"
        processing_file = os.path.join(self.processing_dir, filename)
        completed_file = os.path.join(self.completed_dir, filename)
        
        if os.path.exists(processing_file):
            # Update payload with result
            with open(processing_file, 'r') as f:
                payload = json.load(f)
            
            payload["status"] = "completed"
            payload["result"] = result_metadata
            payload["completed_at"] = time.time()
            
            # Write to completed
            with open(completed_file, 'w') as f:
                json.dump(payload, f)
                
            # Remove from processing
            os.remove(processing_file)

    def get_job_data(self, job_id: str) -> Optional[Dict[str, Any]]:
        # Check files
        filename = f"{job_id}.json"
        for d in [self.completed_dir, self.processing_dir, self.pending_dir]:
            filepath = os.path.join(d, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    return json.load(f)
        return None


class RedisJobQueue:
    """
    Manages a Redis-backed queue for distributing tile processing jobs.
    """
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis module not installed.")
        self.redis = redis.from_url(redis_url)
        self.job_queue_key = "acie:jobs:tiles"
        self.result_channel = "acie:results"
        self.job_status_hash = "acie:job_status"

    def push_job(self, image_path: str, tile_box: tuple, context_key: str, job_group_id: str) -> str:
        job_id = str(uuid.uuid4())
        job_payload = {
            "job_id": job_id,
            "group_id": job_group_id,
            "image_path": image_path,
            "tile_box": tile_box,
            "context_key": context_key,
            "timestamp": time.time(),
            "status": "pending"
        }
        self.redis.hset(self.job_status_hash, job_id, json.dumps(job_payload))
        self.redis.lpush(self.job_queue_key, job_id)
        return job_id

    def pop_job(self, timeout: int = 5) -> Optional[Dict[str, Any]]:
        result = self.redis.blpop(self.job_queue_key, timeout=timeout)
        if not result:
            return None
        job_id = result[1].decode('utf-8')
        job_data_str = self.redis.hget(self.job_status_hash, job_id)
        if not job_data_str:
            return None
        return json.loads(job_data_str)

    def complete_job(self, job_id: str, result_metadata: Dict[str, Any]):
        self.redis.hset(self.job_status_hash, job_id, json.dumps({
            "status": "completed",
            "result": result_metadata,
            "completed_at": time.time()
        }))
        self.redis.publish(self.result_channel, job_id)
        
    def get_job_status(self, job_id: str) -> str:
        job_data_str = self.redis.hget(self.job_status_hash, job_id)
        if not job_data_str: return "unknown"
        return json.loads(job_data_str).get("status", "unknown")

    def get_job_data(self, job_id: str) -> Optional[Dict[str, Any]]:
        job_data_str = self.redis.hget(self.job_status_hash, job_id)
        if not job_data_str: return None
        return json.loads(job_data_str)

# Factory function
def TileJobQueue(use_redis: bool = True):
    if use_redis and REDIS_AVAILABLE:
        try:
            return RedisJobQueue()
        except Exception:
            logger.warning("Failed to connect to Redis. Falling back to FileJobQueue.")
            return FileJobQueue()
    else:
        if use_redis:
            logger.warning("Redis not available. Falling back to FileJobQueue.")
        return FileJobQueue()
