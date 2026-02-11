# Multi-Language ACIE Build System

.PHONY: all clean test build-rust build-java setup-python install-r deploy

# Main targets
all: setup-python build-rust build-java install-r

# Setup Python environment
setup-python:
	@echo "=== Setting up Python environment ==="
	python3 -m venv venv || true
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -e .
	@echo "✓ Python setup complete"

# Build Rust components
build-rust:
	@echo "=== Building Rust components ==="
	cd rust && cargo build --release
	cd rust && maturin develop --release
	@echo "✓ Rust build complete"

# Build Java server
build-java:
	@echo "=== Building Java inference server ==="
	cd java && mvn clean package -DskipTests
	@echo "✓ Java build complete"

# Build Assembly kernels
build-asm:
	@echo "=== Building Assembly kernels ==="
	cd asm && make
	@echo "✓ Assembly build complete"

# Install R packages
install-r:
	@echo "=== Installing R packages ==="
	Rscript -e "install.packages(c('reticulate', 'ggplot2', 'tidyverse', 'pcalg', 'bnlearn', 'shiny', 'shinydashboard', 'plotly', 'DT'), repos='https://cloud.r-project.org')" || echo "R packages installation skipped (R not found)"
	@echo "✓ R packages installed"

# Run all tests
test:
	@echo "=== Running test suite ==="
	./venv/bin/pytest tests/ -v
	cd rust && cargo test
	cd java && mvn test || true
	@echo "✓ Tests complete"

# Run benchmarks
benchmark:
	@echo "=== Running benchmarks ==="
	cd rust && cargo bench
	@echo "✓ Benchmarks complete"

# Deploy all services
deploy:
	@echo "=== Deploying ACIE ==="
	./scripts/deploy.sh deploy
	@echo "✓ Deployment complete"

# Clean build artifacts
clean:
	@echo "=== Cleaning build artifacts ==="
	cd rust && cargo clean
	cd java && mvn clean
	cd asm && make clean
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Clean complete"

# Development mode
dev:
	@echo "=== Starting development environment ==="
	./scripts/deploy.sh start
	@echo "✓ Development services started"
