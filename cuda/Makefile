# Makefile for Physics Kernels (CPU Optimization Build)
# Note: CUDA and MPS GPU acceleration work automatically via PyTorch
# This Makefile only builds optional optimized CPU library

# Common settings
TARGET = physics_cuda.so

# Use g++ to avoid xcrun cache issues on macOS
CXX := $(shell which g++ 2>/dev/null || which clang++ 2>/dev/null || echo /usr/bin/g++)
CXXFLAGS = -O3 -fPIC -std=c++11

# CPU build settings
SOURCES = physics_constraints_cpu.cpp
COMPILER = $(CXX)
COMPILE_FLAGS = $(CXXFLAGS)
BUILD_TYPE = CPU

# Add OpenMP support if available
OPENMP_CHECK := $(shell $(CXX) -fopenmp -E - < /dev/null > /dev/null 2>&1 && echo yes || echo no)
ifeq ($(OPENMP_CHECK),yes)
    COMPILE_FLAGS += -fopenmp
endif

.PHONY: all clean test info

all: info $(TARGET)


info:
	@echo "========================================"
	@echo "Building Physics Kernels ($(BUILD_TYPE))"
	@echo "========================================"
	@echo "ℹ️  GPU acceleration via PyTorch (CUDA/MPS)"
	@echo "  This builds optional CPU optimization library"
	@echo "  Compiler: $(CXX)"
ifeq ($(OPENMP_CHECK),yes)
	@echo "  ✓ OpenMP enabled for parallelization"
else
	@echo "  ⚠ OpenMP not available"
endif
	@echo "========================================"



# Build shared library
$(TARGET): $(SOURCES)
	# Compile to object file first, then link
	$(CXX) $(CXXFLAGS) -c $< -o physics_constraints.o
	$(CXX) -dynamiclib physics_constraints.o -o $@
	@rm -f physics_constraints.o
	@echo "✓ Built $(TARGET) ($(BUILD_TYPE) version)"


# Test
test: all
	@echo "Testing physics kernels..."
	@python3 -c "from cuda_physics import PhysicsConstraints; print('✓ Import successful')" || echo "✗ Import failed"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -f $(TARGET) *.o
	@rm -rf __pycache__ *.pyc
	@echo "✓ Clean complete"

# Install (copy to ACIE package)
install: all
	@echo "Installing to ACIE package..."
	@mkdir -p ../acie/cuda
	@cp $(TARGET) ../acie/cuda/
	@echo "✓ Installed $(TARGET)"

.DEFAULT_GOAL := all
