#!/usr/bin/env python3
"""
Multi-Language ACIE Demo
Demonstrates integration across Python, Rust, Java, R, and Assembly
"""

import numpy as np
from acie.integration.multi_language import ACIEMultiLanguage

def main():
    print("="*60)
    print("ACIE Multi-Language Integration Demo")
    print("="*60)
    
    # Initialize multi-language system
    acie = ACIEMultiLanguage()
    print()
    
    # Generate sample data
    print("Generating sample data...")
    observations = np.random.randn(10, 100).astype(np.float32)
    interventions = {"mass": 1.5, "metallicity": 0.02}
    print(f"Observations shape: {observations.shape}")
    print(f"Interventions: {interventions}\n")
    
    # Test Rust backend
    if acie.rust.available:
        print("Testing Rust high-performance kernels...")
        a = np.random.rand(100, 100).astype(np.float32)
        b = np.random.rand(100, 100).astype(np.float32)
        result = acie.rust.fast_matmul(a, b)
        print(f"✓ Rust matmul result shape: {result.shape}\n")
    
    # Test Assembly kernels
    if acie.asm.available:
        print("Testing Assembly AVX2 kernels...")
        input_data = np.random.randn(1000).astype(np.float32)
        output = acie.asm.fast_relu(input_data)
        print(f"✓ Assembly ReLU result shape: {output.shape}\n")
    
    # Test Java inference server
    if acie.java.available:
        print("Testing Java distributed inference...")
        try:
            result = acie.java.infer_counterfactual(
                observations,
                interventions,
                request_id="demo-001"
            )
            print(f"✓ Java inference result shape: {result.shape}\n")
        except Exception as e:
            print(f"Java inference test skipped: {e}\n")
    
    # Test R statistical analysis
    if acie.r.available:
        print("Testing R statistical analysis...")
        try:
            # Generate fake counterfactuals for evalution
            counterfactuals = observations + np.random.randn(*observations.shape) * 0.1
            
            metrics = acie.r.evaluate_counterfactuals(
                counterfactuals.astype(np.float64),
                observations.astype(np.float64)
            )
            print(f"✓ R evaluation metrics:")
            print(f"   MSE: {metrics['mse']:.4f}")
            print(f"   MAE: {metrics['mae']:.4f}")
            print(f"   R²: {metrics['r2_mean']:.4f}\n")
        except Exception as e:
            print(f"R analysis test skipped: {e}\n")
    
    # Unified inference
    print("Testing unified multi-backend inference...")
    try:
        result = acie.infer(observations, interventions, backend="auto")
        print(f"✓ Unified inference result shape: {result.shape}")
        print(f"   Mean difference: {np.abs(result - observations).mean():.4f}\n")
    except Exception as e:
        print(f"Inference test: {e}\n")
    
    print("="*60)
    print("Demo Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
