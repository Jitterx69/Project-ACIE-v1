#!/bin/bash
#
# ACIE Deployment Script
# Automates building, testing, and deploying ACIE components

set -e  # Exit on error
set -u  # Exit on undefined variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

#  Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration
ACIE_ROOT="/Users/jitterx/Desktop/ACIE"
PYTHON_ENV="${ACIE_ROOT}/venv"
BUILD_DIR="${ACIE_ROOT}/build"
OUTPUT_DIR="${ACIE_ROOT}/outputs"

# Build Rust components
build_rust() {
    log_info "Building Rust components..."
    
    cd "${ACIE_ROOT}/rust"
    
    # Check if Cargo is installed
    if ! command -v cargo &> /dev/null; then
        log_error "Cargo not found. Please install Rust."
        exit 1
    fi
    
    # Build release
    cargo build --release
    
    # Run tests
    cargo test
    
    # Build Python bindings
    maturin develop --release
    
    log_info "Rust build complete ✓"
}

# Build Java server
build_java() {
    log_info "Building Java inference server..."
    
    cd "${ACIE_ROOT}/java"
    
    # Check for Maven
    if ! command -v mvn &> /dev/null; then
        log_error "Maven not found. Please install Maven."
        exit 1
    fi
    
    # Clean and build
    mvn clean package -DskipTests
    
    log_info "Java build complete ✓"
}

# Setup Python environment
setup_python() {
    log_info "Setting up Python environment..."
    
    cd "${ACIE_ROOT}"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "${PYTHON_ENV}" ]; then
        python3 -m venv "${PYTHON_ENV}"
    fi
    
    # Activate and install dependencies
    source "${PYTHON_ENV}/bin/activate"
    pip install --upgrade pip
    pip install -e .
    
    log_info "Python setup complete ✓"
}

# Run tests
run_tests() {
    log_info "Running test suite..."
    
    # Python tests
    log_info "Running Python tests..."
    cd "${ACIE_ROOT}"
    source "${PYTHON_ENV}/bin/activate"
    pytest tests/ -v
    
    # Rust tests
    log_info "Running Rust tests..."
    cd "${ACIE_ROOT}/rust"
    cargo test
    
    # Java tests
    log_info "Running Java tests..."
    cd "${ACIE_ROOT}/java"
    mvn test
    
    log_info "All tests passed ✓"
}

# Start services
start_services() {
    log_info "Starting ACIE services..."
    
    # Start Java server
    log_info "Starting Java inference server..."
    cd "${ACIE_ROOT}/java"
    java -jar target/acie-inference-server-0.1.0.jar &
    JAVA_PID=$!
    echo $JAVA_PID > /tmp/acie_java.pid
    
    # Start R Shiny dashboard
    if command -v Rscript &> /dev/null; then
        log_info "Starting R Shiny dashboard..."
        cd "${ACIE_ROOT}/r"
        Rscript -e "shiny::runApp('shiny_dashboard.R', port=8501)" &
        R_PID=$!
        echo $R_PID > /tmp/acie_shiny.pid
    else
        log_warn "R not found, skipping Shiny dashboard"
    fi
    
    log_info "Services started ✓"
    log_info "Java Server PID: ${JAVA_PID}"
    log_info "Shiny Dashboard PID: ${R_PID:-N/A}"
}

# Stop services
stop_services() {
    log_info "Stopping ACIE services..."
    
    if [ -f /tmp/acie_java.pid ]; then
        kill $(cat /tmp/acie_java.pid) 2>/dev/null || true
        rm /tmp/acie_java.pid
    fi
    
    if [ -f /tmp/acie_shiny.pid ]; then
        kill $(cat /tmp/acie_shiny.pid) 2>/dev/null || true
        rm /tmp/acie_shiny.pid
    fi
    
    log_info "Services stopped ✓"
}

# Performance benchmark
benchmark() {
    log_info "Running performance benchmarks..."
    
    cd "${ACIE_ROOT}/rust"
    cargo bench
    
    log_info "Benchmarks complete ✓"
}

# Main command dispatcher
case "${1:-help}" in
    build)
        log_info "Building all components..."
        setup_python
        build_rust
        build_java
        log_info "Build complete ✓"
        ;;
    test)
        run_tests
        ;;
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        stop_services
        sleep 2
        start_services
        ;;
    benchmark)
        benchmark
        ;;
    deploy)
        log_info "Deploying ACIE..."
        setup_python
        build_rust
        build_java
        run_tests
        start_services
        log_info "Deployment complete ✓"
        ;;
    *)
        echo "ACIE Deployment Script"
        echo ""
        echo "Usage: $0 {build|test|start|stop|restart|benchmark|deploy}"
        echo ""
        echo "Commands:"
        echo "  build      - Build all components"
        echo "  test       - Run test suite"
        echo "  start      - Start services"
        echo "  stop       - Stop services"
        echo "  restart    - Restart services"
        echo "  benchmark  - Run performance benchmarks"
        echo "  deploy     - Full deployment (build, test, start)"
        exit 1
        ;;
esac
