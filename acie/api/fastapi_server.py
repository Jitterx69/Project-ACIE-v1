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

from acie.core.acie_core import ACIECore
from acie.inference.counterfactual import CounterfactualEngine

# Production Features imports
from acie.logging import logger, RequestLoggingMiddleware
from acie.security import (
    Token, login_endpoint, get_current_user, 
    User, require_role
)
from acie.cache import get_cache
from acie.monitoring import (
    track_inference, metrics_endpoint, 
    set_model_count, record_batch_size
)

# Initialize FastAPI app
app = FastAPI(
    title="ACIE Inference API",
    description="Astronomical Counterfactual Inference Engine - Production API",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add Middleware
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache
model_cache: Dict[str, ACIECore] = {}
cf_engine_cache: Dict[str, CounterfactualEngine] = {}


# Pydantic models
class InferenceRequest(BaseModel):
    """Request schema for counterfactual inference"""
    observation: List[float] = Field(..., description="Observable values (6000-dim for 10k, 14000-dim for 20k)")
    intervention: Dict[str, float] = Field(..., description="Intervention parameters (e.g., {'mass': 1.5})")
    model_version: str = Field(default="latest", description="Model version to use")
    use_cache: bool = Field(default=True, description="Whether to use cached results if available")
    
    @validator('observation')
    def validate_observation_dim(cls, v):
        if len(v) not in [6000, 14000]:
            raise ValueError(f"Observation must be 6000 or 14000 dimensional, got {len(v)}")
        return v


class InferenceResponse(BaseModel):
    """Response schema for counterfactual inference"""
    counterfactual: List[float] = Field(..., description="Counterfactual observable values")
    latent_state: List[float] = Field(..., description="Inferred latent physical state")
    confidence: float = Field(..., description="Confidence score for the inference")
    model_version: str = Field(..., description="Model version used")
    timestamp: str = Field(..., description="Inference timestamp")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    cached: bool = Field(default=False, description="Whether result was served from cache")


class BatchInferenceRequest(BaseModel):
    """Request schema for batch inference"""
    observations: List[List[float]] = Field(..., description="Batch of observations")
    interventions: List[Dict[str, float]] = Field(..., description="Batch of interventions")
    model_version: str = Field(default="latest", description="Model version to use")


class BatchInferenceResponse(BaseModel):
    """Response schema for batch inference"""
    results: List[InferenceResponse]
    total_count: int
    failed_count: int


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    models_loaded: List[str]
    gpu_available: bool
    gpu_count: int
    cache_connected: bool
    timestamp: str


class ModelInfo(BaseModel):
    """Model metadata"""
    version: str
    loaded_at: str
    parameters: int
    device: str


# System Events
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
    if torch.cuda.is_available():
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
    else:
        logger.warning("No GPU available, using CPU")
    
    # Load default model
    model_path_env = os.getenv("MODEL_PATH", "outputs/acie_final.ckpt")
    default_model_path = Path(model_path_env)
    
    if default_model_path.exists():
        try:
            logger.info(f"Loading model from {default_model_path}")
            model = ACIECore.load_from_checkpoint(str(default_model_path))
            
            # Move to GPU if available
            if torch.cuda.is_available():
                model = model.cuda()
            
            model.eval()  # Set to evaluation mode
            
            model_cache["latest"] = model
            cf_engine_cache["latest"] = CounterfactualEngine(model)
            set_model_count(1)
            
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    else:
        logger.warning(f"Model not found at {default_model_path}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down ACIE API server...")
    model_cache.clear()
    cf_engine_cache.clear()


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
    for version, model in model_cache.items():
        param_count = sum(p.numel() for p in model.parameters())
        device = next(model.parameters()).device
        
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
        # Get model
        model = model_cache.get(request.model_version)
        cf_engine = cf_engine_cache.get(request.model_version)
        
        if not model or not cf_engine:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model version '{request.model_version}' not found"
            )
        
        # Convert observation to tensor
        obs_tensor = torch.tensor(request.observation, dtype=torch.float32).unsqueeze(0)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            obs_tensor = obs_tensor.cuda()
        
        # Perform inference
        with torch.no_grad():
            result = cf_engine.generate_counterfactual(obs_tensor, request.intervention)
        
        # Extract results
        counterfactual = result['counterfactual'].cpu().squeeze(0).tolist()
        latent_state = result['latent'].cpu().squeeze(0).tolist()
        confidence = float(result.get('confidence', 0.95))
        latency_ms = (time.time() - start_time) * 1000
        
        response_data = {
            "counterfactual": counterfactual,
            "latent_state": latent_state,
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


# Metrics Endpoint
app.add_route("/metrics", metrics_endpoint)


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
