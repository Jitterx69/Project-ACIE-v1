/*
 * CUDA Physics Constraints Kernel
 * High-performance GPU kernel for enforcing physics constraints
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Error checking macro
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(error));                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// ============================================================================
// Energy Conservation Kernel
// ============================================================================

__global__ void enforce_energy_conservation_kernel(
    const float *__restrict__ latents, float *__restrict__ corrected_latents,
    const int n_samples, const int latent_dim, const float tolerance) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n_samples) {
    // Calculate total energy for this sample
    float total_energy = 0.0f;
    for (int i = 0; i < latent_dim; i++) {
      int offset = idx * latent_dim + i;
      float val = latents[offset];
      total_energy += val * val;
    }

    // Determine scale factor
    float scale_factor = 1.0f;
    if (total_energy > tolerance && total_energy > 1e-8f) {
      scale_factor = sqrtf(tolerance / total_energy);
    }

    // Apply correction
    for (int i = 0; i < latent_dim; i++) {
      int offset = idx * latent_dim + i;
      corrected_latents[offset] = latents[offset] * scale_factor;
    }
  }
}

// ============================================================================
// Momentum Conservation Kernel
// ============================================================================

__global__ void enforce_momentum_conservation_kernel(
    const float *__restrict__ latents, float *__restrict__ corrected_latents,
    const int n_samples, const int latent_dim, const int momentum_start_idx,
    const int momentum_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n_samples) {
    // Copy all latents first
    for (int i = 0; i < latent_dim; i++) {
      int offset = idx * latent_dim + i;
      corrected_latents[offset] = latents[offset];
    }

    // Enforce zero momentum by setting momentum components to zero
    for (int i = 0; i < momentum_dim; i++) {
      int dim_idx = momentum_start_idx + i;
      if (dim_idx < latent_dim) {
        int offset = idx * latent_dim + dim_idx;
        corrected_latents[offset] = 0.0f;
      }
    }
  }
}

// ============================================================================
// Combined Physics Constraints Kernel
// ============================================================================

__global__ void enforce_physics_constraints_kernel(
    const float *__restrict__ latents, float *__restrict__ corrected_latents,
    const int n_samples, const int latent_dim, const float energy_tolerance,
    const float momentum_tolerance) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n_samples) {
    // Step 1: Calculate total energy
    float total_energy = 0.0f;
    for (int i = 0; i < latent_dim; i++) {
      int offset = idx * latent_dim + i;
      float val = latents[offset];
      total_energy += val * val;
    }

    // Step 2: Calculate energy scale factor
    float energy_scale = 1.0f;
    if (total_energy > energy_tolerance && total_energy > 1e-8f) {
      energy_scale = sqrtf(energy_tolerance / total_energy);
    }

    // Step 3: Apply corrections
    for (int i = 0; i < latent_dim; i++) {
      int offset = idx * latent_dim + i;
      float corrected = latents[offset] * energy_scale;

      // Zero out first 3 dimensions (momentum conservation)
      if (i < 3) {
        corrected = 0.0f;
      }

      corrected_latents[offset] = corrected;
    }
  }
}

// ============================================================================
// Fast Matrix Multiplication Kernel
// ============================================================================

__global__ void matmul_kernel(const float *__restrict__ A,
                              const float *__restrict__ B,
                              float *__restrict__ C, const int M, const int N,
                              const int K) {
  const int TILE_SIZE = 32;
  __shared__ float As[32][32];
  __shared__ float Bs[32][32];

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  float sum = 0.0f;

  int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

  for (int tile = 0; tile < num_tiles; tile++) {
    // Load tile from A
    int a_col = tile * TILE_SIZE + threadIdx.x;
    if (row < M && a_col < K) {
      As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
    } else {
      As[threadIdx.y][threadIdx.x] = 0.0f;
    }

    // Load tile from B
    int b_row = tile * TILE_SIZE + threadIdx.y;
    if (b_row < K && col < N) {
      Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
    } else {
      Bs[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

// Compute partial sum
#pragma unroll
    for (int k = 0; k < TILE_SIZE; k++) {
      sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }

    __syncthreads();
  }

  // Write result
  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

// ============================================================================
// C++ Interface Functions
// ============================================================================

extern "C" {

void enforce_energy_conservation(const float *latents, float *corrected_latents,
                                 int n_samples, int latent_dim,
                                 float tolerance) {
  int threads = 256;
  int blocks = (n_samples + threads - 1) / threads;

  enforce_energy_conservation_kernel<<<blocks, threads>>>(
      latents, corrected_latents, n_samples, latent_dim, tolerance);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

void enforce_momentum_conservation(const float *latents,
                                   float *corrected_latents, int n_samples,
                                   int latent_dim, int momentum_start_idx,
                                   int momentum_dim) {
  int threads = 256;
  int blocks = (n_samples + threads - 1) / threads;

  enforce_momentum_conservation_kernel<<<blocks, threads>>>(
      latents, corrected_latents, n_samples, latent_dim, momentum_start_idx,
      momentum_dim);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

void enforce_physics_constraints(const float *latents, float *corrected_latents,
                                 int n_samples, int latent_dim,
                                 float energy_tolerance,
                                 float momentum_tolerance) {
  int threads = 256;
  int blocks = (n_samples + threads - 1) / threads;

  enforce_physics_constraints_kernel<<<blocks, threads>>>(
      latents, corrected_latents, n_samples, latent_dim, energy_tolerance,
      momentum_tolerance);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_matmul(const float *A, const float *B, float *C, int M, int N,
                 int K) {
  dim3 threads(32, 32);
  dim3 blocks((N + 31) / 32, (M + 31) / 32);

  matmul_kernel<<<blocks, threads>>>(A, B, C, M, N, K);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

} // extern "C"
