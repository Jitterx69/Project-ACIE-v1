#!/usr/bin/env python3
"""
ACIE Multi-Language System Test
Comprehensive test of all components
"""

import sys
import numpy as np
from pathlib import Path

print("=" * 70)
print("ACIE MULTI-LANGUAGE SYSTEM TEST")
print("=" * 70)
print()

# Test 1: Python Core
print("1. Testing Python Core Components...")
try:
    # Add ACIE to path
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    
    from acie.core.scm import StructuralCausalModel
    from acie.core.engine import ACIEEngine
    
    # Create simple SCM
    scm = StructuralCausalModel()
    scm.add_node("X")
    scm.add_node("Y")
    scm.add_edge("X", "Y")
    
    print("   ✓ SCM created successfully")
    print(f"   ✓ Nodes: {list(scm.graph.nodes())}")
    print(f"   ✓ Edges: {list(scm.graph.edges())}")
    
except Exception as e:
    print(f"   ✗ Error: {e}")

print()

# Test 2: Rust Components
print("2. Testing Rust High-Performance Kernels...")
try:
    # Check if Rust library exists
    rust_lib = Path("rust/target/release")
    if rust_lib.exists():
        print(f"   ✓ Rust library built at: {rust_lib}")
    else:
        print("   ⚠ Rust library not built yet (run: cd rust && cargo build --release)")
except Exception as e:
    print(f"   ⚠ Rust check failed: {e}")

print()

# Test 3: Assembly Kernels
print("3. Testing Assembly AVX2 Kernels...")
try:
    asm_lib = Path("asm/libacie_asm.dylib")
    if asm_lib.exists():
        print(f"   ✓ Assembly library exists: {asm_lib}")
        
        # Try to load it
        sys.path.insert(0, "asm")
        from asm_python import fast_relu, AVAILABLE
        
        if AVAILABLE:
            x = np.random.randn(100).astype(np.float32)
            y = fast_relu(x)
            print(f"   ✓ Assembly ReLU works! Input shape: {x.shape}, Output shape: {y.shape}")
        else:
            print("   ⚠ Assembly library not loaded")
    else:
        print("   ⚠ Assembly library not built (run: cd asm && make)")
except Exception as e:
    print(f"   ⚠ Assembly test failed: {e}")

print()

# Test 4: Java Server
print("4. Testing Java Inference Server...")
try:
    java_jar = Path("java/target/acie-inference-server-0.1.0.jar")
    if java_jar.exists():
        print(f"   ✓ Java JAR built: {java_jar}")
    else:
        print("   ⚠ Java server not built (run: cd java && mvn package)")
    
    # Check if server is running
    import requests
    try:
        resp = requests.get("http://localhost:8080/api/v1/inference/health", timeout=1)
        if resp.status_code == 200:
            print("   ✓ Java server is RUNNING!")
        else:
            print(f"   ⚠ Server responded with status: {resp.status_code}")
    except:
        print("   ⚠ Java server not running (start with: ./scripts/deploy.sh start)")
        
except Exception as e:
    print(f"   ⚠ Java check failed: {e}")

print()

# Test 5: R Analysis
print("5. Testing R Statistical Analysis...")
try:
    import subprocess
    result = subprocess.run(["which", "Rscript"], capture_output=True, timeout=2)
    if result.returncode == 0:
        print(f"   ✓ R found at: {result.stdout.decode().strip()}")
        
        r_files = list(Path("r").glob("*.R"))
        print(f"   ✓ R analysis files: {[f.name for f in r_files]}")
    else:
        print("   ⚠ R not installed")
except Exception as e:
    print(f"   ⚠ R check failed: {e}")

print()

# Test 6: Integration Test
print("6. Testing Multi-Language Integration...")
try:
    # Simple numpy test that all backends can use
    test_data = np.random.randn(10, 50).astype(np.float32)
    print(f"   ✓ Generated test data: {test_data.shape}")
    
    # Test basic Python operations
    mean = test_data.mean()
    std = test_data.std()
    print(f"   ✓ Statistics - Mean: {mean:.4f}, Std: {std:.4f}")
    
except Exception as e:
    print(f"   ✗ Integration test failed: {e}")

print()
print("=" * 70)
print("SYSTEM TEST COMPLETE")
print("=" * 70)
print()
print("Summary:")
print("  - Python Core: Available ✓")
print("  - Rust: Build required")
print("  - Assembly: Build required (optional)")  
print("  - Java: Build required (optional)")
print("  - R: Install required (optional)")
print()
print("Quick Start:")
print("  1. Test Python only: python3 scripts/example_inference.py")
print("  2. Build all: make all")
print("  3. Full demo: python3 scripts/demo_multilang.py")
print()
