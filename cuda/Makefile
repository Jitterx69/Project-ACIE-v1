# Makefile for CUDA Physics Kernels

# CUDA compiler
NVCC = nvcc

# CUDA flags
CUDA_FLAGS = -O3 --use_fast_math -Xcompiler -fPIC
CUDA_ARCH = -arch=sm_70  # Change based on your GPU (sm_70=V100, sm_80=A100, sm_86=RTX 3090)

# Python include path
PYTHON_INCLUDE = $(shell python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")
TORCH_INCLUDE = $(shell python3 -c "import torch; print(torch.utils.cpp_extension.include_paths()[0])")

# Source files
CU_SOURCES = physics_constraints.cu
CU_OBJECTS = $(CU_SOURCES:.cu=.o)

# Target
TARGET = physics_cuda.so

.PHONY: all clean test

all: $(TARGET)

# Compile CUDA source to object file
%.o: %.cu
	$(NVCC) $(CUDA_FLAGS) $(CUDA_ARCH) -c $< -o $@ \
		-I$(PYTHON_INCLUDE) \
		-I$(TORCH_INCLUDE)

# Link into shared library
$(TARGET): $(CU_OBJECTS)
	$(NVCC) -shared $(CUDA_FLAGS) $(CUDA_ARCH) $^ -o $@

# Test compilation
test: all
	python3 -c "import torch; from cuda_physics import PhysicsConstraints; \
		print('CUDA kernels loaded successfully'); \
		print(f'CUDA available: {torch.cuda.is_available()}')"

# Clean build artifacts
clean:
	rm -f $(CU_OBJECTS) $(TARGET)
	rm -rf __pycache__
	rm -rf *.pyc

# Install (copy to ACIE package)
install: all
	cp $(TARGET) ../acie/cuda/

.DEFAULT_GOAL := all
