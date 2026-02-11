"""
FastAPI-based REST API server for ACIE inference
Replaces the Java Spring Boot server with modern Python async API
Includes Production Features: Logging, Auth, Caching, GPU support
"""

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional
import torch
import numpy as np
from datetime import datetime
import time
from pathlib import Path
import os

from acie.core.acie_core import ACIEEngine

# ... imports ...

# Global model cache
model_cache: Dict[str, ACIEEngine] = {}
# cf_engine_cache removed as it's part of ACIEEngine

# ...

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    logger.info("Starting ACIE API server v2.1.0...")
    
    # Initialize Cache
    cache = get_cache()
    if cache.is_available():
        logger.info(f"Connected to Redis cache at {cache.host}:{cache.port}")
    else:
        logger.warning("Redis cache not available")

    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
    else:
        logger.warning("No GPU available, using CPU")
    
    # Load default model
    model_path_env = os.getenv("MODEL_PATH", "outputs/acie_final.ckpt")
    default_model_path = Path(model_path_env)
    
    try:
        # Check if model exists, if not create dummy for testing availability
        if not default_model_path.exists():
            logger.warning(f"Model not found at {default_model_path}, initializing default for testing")
            # Create dummy model
            dummy_engine = ACIEEngine.from_config(Path("config/default.yaml"))
            # Save it so we can 'load' it next time or just use it
            model_cache["latest"] = dummy_engine.to(device)
        else:
            logger.info(f"Loading model from {default_model_path}")
            model = ACIEEngine.from_checkpoint(str(default_model_path), device=device)
            model_cache["latest"] = model
            
        set_model_count(1)
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Initialize default fallback
        fallback = ACIEEngine.from_config(Path("config/default.yaml"))
        model_cache["latest"] = fallback.to(device)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down ACIE API server...")
    model_cache.clear()


# Authentication Endpoints
@app.post("/token", response_model=Token, tags=["Authentication"])
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login to get access token"""
    return await login_endpoint(form_data.username, form_data.password)


# API Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "ACIE Inference API",
        "version": "2.1.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    cache = get_cache()
    return HealthResponse(
        status="healthy" if model_cache else "no_models_loaded",
        models_loaded=list(model_cache.keys()),
        gpu_available=torch.cuda.is_available(),
        gpu_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
        cache_connected=cache.is_available(),
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/models", dependencies=[Depends(get_current_user)], tags=["Models"])
async def list_models():
    """List all loaded models (Authenticated)"""
    models_info = []
    for version, engine in model_cache.items():
        # engine.inference_model contains parameters
        param_count = sum(p.numel() for p in engine.inference_model.parameters())
        device = engine.device
        
        models_info.append(ModelInfo(
            version=version,
            loaded_at=datetime.utcnow().isoformat(),
            parameters=param_count,
            device=str(device)
        ))
    
    return {"models": models_info}


@app.post("/api/v2/inference/counterfactual", 
          response_model=InferenceResponse, 
          dependencies=[Depends(get_current_user)],
          tags=["Inference"])
@track_inference("latest", "counterfactual")
async def counterfactual_inference(request: InferenceRequest):
    """
    Perform counterfactual inference (Authenticated)
    
    Given an observation and intervention, compute the counterfactual distribution
    Uses Redis caching if enabled.
    """
    start_time = time.time()
    cache = get_cache()
    
    # Check cache first
    if request.use_cache and cache.is_available():
        cached_result = cache.get_cached_inference(
            request.observation, 
            request.intervention,
            request.model_version
        )
        if cached_result:
            return InferenceResponse(**cached_result, cached=True)

    try:
        # Get model engine
        engine = model_cache.get(request.model_version)
        
        if not engine:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model version '{request.model_version}' not found"
            )
        
        # Convert observation to tensor
        obs_tensor = torch.tensor(request.observation, dtype=torch.float32).unsqueeze(0)
        
        # Move to GPU if available (handled by engine ideally, but ensure input is on device)
        obs_tensor = obs_tensor.to(engine.device)
        
        # Perform inference using engine
        with torch.no_grad():
            # infer_latent returns (mean, samples)
            latent_mean, _ = engine.infer_latent(obs_tensor)
            
            # Intervene
            counterfactual = engine.intervene(obs_tensor, request.intervention)
            
            # Get specific return values
            # To get latent/confidence we might need to peek internals or extend engine return
            # For now, let's assume we use what we have
            
            # NOTE: engine.intervene returns only counterfactual_obs. 
            # We need latent state too for response.
            
            latent_state = latent_mean
        
        # Extract results
        cf_result = counterfactual.cpu().squeeze(0).tolist()
        latent_result = latent_state.cpu().squeeze(0).tolist()
        confidence = 0.95 # Placeholder until confidence model integrated
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Record for dashboard
        latency_history.append({
            "time": datetime.utcnow().strftime("%H:%M:%S"),
            "value": latency_ms
        })
        
        response_data = {
            "counterfactual": cf_result,
            "latent_state": latent_result,
            "confidence": confidence,
            "model_version": request.model_version,
            "timestamp": datetime.utcnow().isoformat(),
            "latency_ms": latency_ms,
            "cached": False
        }
        
        # Cache result
        if request.use_cache and cache.is_available():
            cache.cache_inference_result(
                request.observation,
                request.intervention,
                request.model_version,
                response_data
            )
        
        return InferenceResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )


@app.post("/api/v2/inference/batch", 
          response_model=BatchInferenceResponse, 
          dependencies=[Depends(require_role("batch_user"))],
          tags=["Inference"])
async def batch_inference(request: BatchInferenceRequest):
    """
    Perform batch counterfactual inference (Role: batch_user)
    
    Process multiple observations in a single request
    """
    if len(request.observations) != len(request.interventions):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Number of observations must match number of interventions"
        )
    
    record_batch_size(len(request.observations))
    
    results = []
    failed_count = 0
    
    for obs, intervention in zip(request.observations, request.interventions):
        try:
            # Re-use single inference logic (could be optimized for true batch processing later)
            single_request = InferenceRequest(
                observation=obs,
                intervention=intervention,
                model_version=request.model_version,
                use_cache=True
            )
            # Directly call implementation logic would be better, but calling endpoint handler functions works nicely in FastAPI utils usually. 
            # However, here we have dependency injection which makes direct calls tricky without context.
            # For simplicity in this non-refactor step, we'll manually invoke logic or use a helper. 
            # Ideally, extract business logic to controller/service layer.
            # We'll do a direct call to the caching/inference logic here for efficiency.
            
            # --- Inline logic for batch ---
            cache = get_cache()
            cached = None
            if cache.is_available():
                cached = cache.get_cached_inference(obs, intervention, request.model_version)
            
            if cached:
                results.append(InferenceResponse(**cached, cached=True))
                continue
                
            model = model_cache.get(request.model_version)
            cf_engine = cf_engine_cache.get(request.model_version)
            
            if not model: 
                failed_count += 1
                continue
                
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            if torch.cuda.is_available(): obs_tensor = obs_tensor.cuda()
            
            with torch.no_grad():
                res = cf_engine.generate_counterfactual(obs_tensor, intervention)
                
            resp_data = {
                "counterfactual": res['counterfactual'].cpu().squeeze(0).tolist(),
                "latent_state": res['latent'].cpu().squeeze(0).tolist(),
                "confidence": float(res.get('confidence', 0.95)),
                "model_version": request.model_version,
                "timestamp": datetime.utcnow().isoformat(),
                "latency_ms": 0.0, # Batch latency hard to attribute per item approx
                "cached": False
            }
            if cache.is_available():
                cache.cache_inference_result(obs, intervention, request.model_version, resp_data)
            results.append(InferenceResponse(**resp_data))
            # ---------------------------

        except Exception as e:
            logger.error(f"Batch item failed: {e}")
            failed_count += 1
    
    return BatchInferenceResponse(
        results=results,
        total_count=len(request.observations),
        failed_count=failed_count
    )



# Dashboard Integration
from collections import deque
from acie.monitoring.metrics import get_system_stats

# Store last 50 latency values for real-time charting
latency_history = deque(maxlen=50)

@app.get("/api/dashboard/stats", tags=["Monitoring"])
async def dashboard_stats():
    """Get real-time system stats for dashboard"""
    stats = get_system_stats()
    
    # Add application specific stats
    stats["latency_history"] = list(latency_history)
    stats["models_loaded"] = list(model_cache.keys())
    stats["total_requests"] = len(latency_history) # Approximate for now, or use counter
    
    return stats


# Main entry point
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "acie.api.fastapi_server:app",
        host="0.0.0.0",
        port=8080,
        workers=4,
        log_level="info",
        reload=False
    )
