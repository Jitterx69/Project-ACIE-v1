"""
Multi-Language Integration Layer

Provides unified Python interface to:
- Rust high-performance kernels
- Java distributed inference server
- R statistical analysis
- Assembly optimized operations
"""

import subprocess
import requests
import json
from typing import Dict, Any, Optional
import numpy as np

class RustBackend:
    """Interface to Rust high-performance components"""
    
    def __init__(self):
        try:
            import acie_core
            self.rust_module = acie_core
            self.available = True
        except ImportError:
            print("Warning: Rust module not available")
            self.available = False
    
    def fast_matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fast matrix multiplication using Rust"""
        if not self.available:
            return np.dot(a, b)
        return self.rust_module.fast_matmul(a, b)
    
    def evaluate_physics(self, latent: np.ndarray, constraint_type: str) -> np.ndarray:
        """Evaluate physics constraints"""
        if not self.available:
            return np.zeros(len(latent))
        return self.rust_module.evaluate_physics_constraints(latent, constraint_type)


class JavaInferenceClient:
    """Client for Java distributed inference server"""
    
    def __init__(self, host: str = "localhost", port: int = 8080):
        self.base_url = f"http://{host}:{port}/api/v1"
        self._check_connection()
    
    def _check_connection(self):
        """Check if Java server is running"""
        try:
            response = requests.get(f"{self.base_url}/inference/health", timeout=2)
            self.available = response.status_code == 200
        except:
            self.available = False
            print("Warning: Java inference server not available")
    
    def infer_counterfactual(
        self,
        observations: np.ndarray,
        interventions: Dict[str, float],
        request_id: Optional[str] = None
    ) -> np.ndarray:
        """Submit counterfactual inference request to Java server"""
        if not self.available:
            raise RuntimeError("Java server not available")
        
        payload = {
            "requestId": request_id or "py-client-001",
            "observations": observations.tolist(),
            "interventions": interventions
        }
        
        response = requests.post(
            f"{self.base_url}/inference/counterfactual",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return np.array(result["counterfactuals"])
        else:
            raise RuntimeError(f"Inference failed: {response.text}")
    
    def batch_inference(
        self,
        requests_list: list
    ) -> list:
        """Submit batch inference request"""
        if not self.available:
            raise RuntimeError("Java server not available")
        
        response = requests.post(
            f"{self.base_url}/inference/counterfactual/batch",
            json=requests_list,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise RuntimeError(f"Batch inference failed: {response.text}")


class RAnalysis:
    """Interface to R statistical analysis"""
    
    def __init__(self):
        try:
            import rpy2.robjects as ro
            from rpy2.robjects import pandas2ri
            pandas2ri.activate()
            self.r = ro.r
            self.r.source("r/acie_analysis.R")
            self.available = True
        except ImportError:
            print("Warning: R integration not available (rpy2 not installed)")
            self.available = False
    
    def discover_causal_structure(self, data: np.ndarray, alpha: float = 0.05):
        """Discover causal structure using PC algorithm"""
        if not self.available:
            raise RuntimeError("R not available")
        
        import pandas as pd
        df = pd.DataFrame(data)
        
        result = self.r['discover_causal_structure'](df, alpha)
        return result
    
    def evaluate_counterfactuals(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray
    ) -> Dict[str, Any]:
        """Compute evaluation metrics in R"""
        if not self.available:
            raise RuntimeError("R not available")
        
        result = self.r['evaluate_counterfactuals'](predictions, ground_truth)
        
        return {
            "mse": float(result.rx2("mse")[0]),
            "mae": float(result.rx2("mae")[0]),
            "r2_mean": float(result.rx2("r2_mean")[0]),
        }


class AssemblyKernels:
    """Interface to Assembly optimized kernels"""
    
    def __init__(self):
        try:
            # Try to import ctypes interface
            import sys
            from pathlib import Path
            asm_path = Path(__file__).parent.parent.parent / "asm"
            if str(asm_path) not in sys.path:
                sys.path.insert(0, str(asm_path))
            
            from asm_python import fast_relu, AVAILABLE
            self.fast_relu_fn = fast_relu
            self.available = AVAILABLE
        except ImportError:
            print("Warning: Assembly kernels not available")
            self.available = False
    
    def fast_relu(self, input: np.ndarray) -> np.ndarray:
        """Ultra-fast ReLU using AVX2"""
        if not self.available:
            return np.maximum(0, input)
        
        # Ensure float32
        if input.dtype != np.float32:
            input = input.astype(np.float32)
        
        return self.fast_relu_fn(input)


class ACIEMultiLanguage:
    """
    Unified multi-language ACIE interface
    
    Orchestrates:
    - Python: ML models, training
    - Rust: High-performance kernels
    - Java: Distributed inference
    - R: Statistical analysis
    - Assembly: Ultra-optimized ops
    """
    
    def __init__(self):
        self.rust = RustBackend()
        self.java = JavaInferenceClient()
        self.r = RAnalysis()
        self.asm = AssemblyKernels()
        
        print("ACIE Multi-Language System Initialized")
        print(f"  Rust: {'✓' if self.rust.available else '✗'}")
        print(f"  Java: {'✓' if self.java.available else '✗'}")
        print(f"  R: {'✓' if self.r.available else '✗'}")
        print(f"  Assembly: {'✓' if self.asm.available else '✗'}")
    
    def infer(
        self,
        observations: np.ndarray,
        interventions: Dict[str, float],
        backend: str = "auto"
    ) -> np.ndarray:
        """
        Perform counterfactual inference using best available backend
        
        Args:
            observations: Input observations
            interventions: Intervention specification
            backend: "auto", "python", "java", or "rust"
        """
        if backend == "java" and self.java.available:
            return self.java.infer_counterfactual(observations, interventions)
        
        # Fall back to Python implementation
        from acie.training.train import ACIELightningModule
        model = ACIELightningModule.load_from_checkpoint("outputs/acie_final.ckpt")
        engine = model.get_acie_engine()
        
        import torch
        obs_tensor = torch.tensor(observations, dtype=torch.float32)
        result = engine.intervene(obs_tensor, interventions)
        
        return result.cpu().numpy()
    
    def evaluate(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        backend: str = "auto"
    ) -> Dict[str, Any]:
        """Evaluate predictions using best available backend"""
        if backend == "r" and self.r.available:
            return self.r.evaluate_counterfactuals(predictions, ground_truth)
        
        # Fall back to Python
        from acie.eval.metrics import ACIEMetrics
        metrics = ACIEMetrics()
        
        import torch
        pred_tensor = torch.tensor(predictions)
        gt_tensor = torch.tensor(ground_truth)
        
        return {
            "mse": metrics.counterfactual_mse(pred_tensor, gt_tensor),
            "mae": metrics.counterfactual_mae(pred_tensor, gt_tensor),
        }
