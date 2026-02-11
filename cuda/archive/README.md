# CUDA Archive

This directory contains archived CUDA kernels that have been replaced with cross-platform alternatives.

## Archived Files

### physics_constraints.cu.backup
- **Original CUDA implementation** of physics constraint kernels
- **Replaced by**: PyTorch MPS (Metal) and CPU implementations in `cuda_physics.py`
- **Date archived**: 2026-02-11
- **Reason**: CUDA not supported on macOS; migrated to cross-platform solution

## Why Archived?

The CUDA implementation was specific to NVIDIA GPUs and not compatible with:
- Apple Silicon Macs (M1/M2/M3)
- Intel Macs (CUDA deprecated on macOS)
- Systems without NVIDIA GPUs

## Current Implementation

The project now uses:
1. **PyTorch MPS** - Apple GPU acceleration via Metal Performance Shaders
2. **PyTorch CPU** - Optimized CPU fallback for all platforms
3. **CUDA** (when available) - Via PyTorch's CUDA backend on Linux/Windows

All functionality is preserved with automatic device detection.

## Restoring CUDA

If you need NVIDIA CUDA support on Linux/Windows:
1. The `.backup` file contains the original implementation
2. Restore it as `physics_constraints.cu`
3. Update the Makefile to build CUDA version
4. Rebuild with `make`

The current PyTorch implementation already supports CUDA GPUs through PyTorch's CUDA backend, so custom CUDA kernels are typically unnecessary.
