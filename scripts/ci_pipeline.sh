#!/bin/bash
#
# CI/CD Pipeline for ACIE
# GitHub Actions compatible

set -e

echo "=== ACIE CI/CD Pipeline ==="

# 1. Lint and format check
echo "Step 1: Linting..."
cd /Users/jitterx/Desktop/ACIE

# Python
python3 -m black --check acie/
python3 -m flake8 acie/ --max-line-length=100 || true

# Rust
cd rust && cargo fmt -- --check && cd ..

# 2. Build
echo "Step 2: Building..."
./scripts/deploy.sh build

# 3. Test
echo "Step 3: Testing..."
./scripts/deploy.sh test

# 4. Benchmark (optional)
if [ "${RUN_BENCHMARK:-0}" = "1" ]; then
    echo "Step 4: Benchmarking..."
    ./scripts/deploy.sh benchmark
fi

# 5. Package
echo "Step 5: Packaging..."
cd rust && cargo build --release && cd ..
cd java && mvn package -DskipTests && cd ..

echo "=== Pipeline Complete ✓ ==="
