"""
FastAPI-based REST API server for ACIE inference
Replaces the Java Spring Boot server with modern Python async API
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional
import torch
import numpy as np
from datetime import datetime
import logging
import time
from pathlib import Path

from acie.core.acie_core import ACIECore
from acie.inference.counterfactual import CounterfactualEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ACIE Inference API",
    description="Astronomical Counterfactual Inference Engine - Production API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
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
    timestamp: str


class ModelInfo(BaseModel):
    """Model metadata"""
    version: str
    loaded_at: str
    parameters: int
    device: str


# Startup and shutdown events
@app.on_event("startup")
async def load_models():
    """Load models on startup"""
    logger.info("Starting ACIE API server...")
    
    # Check GPU availability
    if torch.cuda.is_available():
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
    else:
        logger.warning("No GPU available, using CPU")
    
    # Load default model
    default_model_path = Path("outputs/acie_final.ckpt")
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


# API Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "ACIE Inference API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model_cache else "no_models_loaded",
        models_loaded=list(model_cache.keys()),
        gpu_available=torch.cuda.is_available(),
        gpu_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/models", tags=["Models"])
async def list_models():
    """List all loaded models"""
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


@app.post("/api/v2/inference/counterfactual", response_model=InferenceResponse, tags=["Inference"])
async def counterfactual_inference(request: InferenceRequest):
    """
    Perform counterfactual inference
    
    Given an observation and intervention, compute the counterfactual distribution
    """
    start_time = time.time()
    
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
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Inference completed in {latency_ms:.2f}ms")
        
        return InferenceResponse(
            counterfactual=counterfactual,
            latent_state=latent_state,
            confidence=confidence,
            model_version=request.model_version,
            timestamp=datetime.utcnow().isoformat(),
            latency_ms=latency_ms
        )
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )


@app.post("/api/v2/inference/batch", response_model=BatchInferenceResponse, tags=["Inference"])
async def batch_inference(request: BatchInferenceRequest):
    """
    Perform batch counterfactual inference
    
    Process multiple observations in a single request
    """
    if len(request.observations) != len(request.interventions):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Number of observations must match number of interventions"
        )
    
    results = []
    failed_count = 0
    
    for obs, intervention in zip(request.observations, request.interventions):
        try:
            single_request = InferenceRequest(
                observation=obs,
                intervention=intervention,
                model_version=request.model_version
            )
            result = await counterfactual_inference(single_request)
            results.append(result)
        except Exception as e:
            logger.error(f"Batch item failed: {e}")
            failed_count += 1
    
    return BatchInferenceResponse(
        results=results,
        total_count=len(request.observations),
        failed_count=failed_count
    )


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """
    Prometheus-compatible metrics endpoint
    """
    # TODO: Integrate with prometheus_client
    gpu_memory = 0
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**2  # MB
    
    return {
        "models_loaded": len(model_cache),
        "gpu_memory_mb": gpu_memory,
        "gpu_available": torch.cuda.is_available(),
    }


# Main entry point
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "acie.api.fastapi_server:app",
        host="0.0.0.0",
        port=8080,
        workers=4,
        log_level="info",
        reload=False  # Set to True for development
    )
