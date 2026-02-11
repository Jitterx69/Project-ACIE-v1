"""
Redis cache wrapper for ACIE
Provides caching for inference results and model outputs
"""

import redis
import pickle
import json
from typing import Any, Optional, Union
import hashlib
from datetime import timedelta
import os

from acie.logging import logger


class RedisCache:
    """Redis cache manager for ACIE"""
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
        db: int = 0,
        password: Optional[str] = None,
        default_ttl: int = 3600
    ):
        """
        Initialize Redis cache
        
        Args:
            host: Redis host (default: from env or localhost)
            port: Redis port (default: from env or 6379)
            db: Redis database number
            password: Redis password
            default_ttl: Default time-to-live in seconds
        """
        self.host = host or os.getenv("REDIS_HOST", "localhost")
        self.port = port or int(os.getenv("REDIS_PORT", 6379))
        self.db = db
        self.password = password or os.getenv("REDIS_PASSWORD")
        self.default_ttl = default_ttl
        
        try:
            self.redis = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=False  # We'll handle encoding ourselves
            )
            # Test connection
            self.redis.ping()
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis = None
    
    def is_available(self) -> bool:
        """Check if Redis is available"""
        return self.redis is not None
    
    def _generate_key(self, prefix: str, **kwargs) -> str:
        """Generate cache key from parameters"""
        # Create a deterministic key from kwargs
        key_data = json.dumps(kwargs, sort_keys=True)
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        return f"{prefix}:{key_hash}"
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        if not self.is_available():
            return None
        
        try:
            cached = self.redis.get(key)
            if cached:
                logger.debug(f"Cache hit for key: {key}")
                return pickle.loads(cached)
            else:
                logger.debug(f"Cache miss for key: {key}")
                return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        overwrite: bool = True
    ) -> bool:
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (default: self.default_ttl)
            overwrite: Whether to overwrite existing key
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            return False
        
        try:
            ttl = ttl or self.default_ttl
            serialized = pickle.dumps(value)
            
            if overwrite:
                self.redis.setex(key, ttl, serialized)
            else:
                # Only set if key doesn't exist
                self.redis.set(key, serialized, ex=ttl, nx=True)
            
            logger.debug(f"Cached value for key: {key} (TTL: {ttl}s)")
            return True
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self.is_available():
            return False
        
        try:
            self.redis.delete(key)
            logger.debug(f"Deleted cache key: {key}")
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching pattern
        
        Args:
            pattern: Redis key pattern (e.g., "inference:*")
            
        Returns:
            Number of keys deleted
        """
        if not self.is_available():
            return 0
        
        try:
            keys = self.redis.keys(pattern)
            if keys:
                count = self.redis.delete(*keys)
                logger.info(f"Cleared {count} keys matching pattern: {pattern}")
                return count
            return 0
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return 0
    
    def cache_inference_result(
        self,
        observation: Any,
        intervention: dict,
        model_version: str,
        result: Any,
        ttl: int = 3600
    ) -> bool:
        """
        Cache inference result
        
        Args:
            observation: Input observation
            intervention: Intervention parameters
            model_version: Model version
            result: Inference result
            ttl: Time-to-live in seconds
            
        Returns:
            True if cached successfully
        """
        key = self._generate_key(
            "inference",
            observation_hash=hashlib.md5(str(observation).encode()).hexdigest(),
            intervention=intervention,
            model_version=model_version
        )
        return self.set(key, result, ttl=ttl)
    
    def get_cached_inference(
        self,
        observation: Any,
        intervention: dict,
        model_version: str
    ) -> Optional[Any]:
        """
        Get cached inference result
        
        Args:
            observation: Input observation
            intervention: Intervention parameters
            model_version: Model version
            
        Returns:
            Cached result or None
        """
        key = self._generate_key(
            "inference",
            observation_hash=hashlib.md5(str(observation).encode()).hexdigest(),
            intervention=intervention,
            model_version=model_version
        )
        return self.get(key)
    
    def get_stats(self) -> dict:
        """Get Redis stats"""
        if not self.is_available():
            return {"available": False}
        
        try:
            info = self.redis.info()
            return {
                "available": True,
                "used_memory_mb": info.get("used_memory", 0) / 1024 / 1024,
                "connected_clients": info.get("connected_clients", 0),
                "total_keys": self.redis.dbsize(),
                "uptime_seconds": info.get("uptime_in_seconds", 0)
            }
        except Exception as e:
            logger.error(f"Failed to get Redis stats: {e}")
            return {"available": False, "error": str(e)}


# Global cache instance
_cache_instance: Optional[RedisCache] = None


def get_cache() -> RedisCache:
    """Get global cache instance"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = RedisCache()
    return _cache_instance
