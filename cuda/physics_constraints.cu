"""
CUDA Physics Constraints Kernel
High-performance GPU kernel for enforcing physics constraints
"""

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Error checking macro
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                  \
             cudaGetErrorString(error));                                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// ============================================================================
// Energy Conservation Kernel
// ============================================================================

__global__ void enforce_energy_conservation_kernel(
    const float* __restrict__ latents,
    float* __restrict__ corrected_latents,
    const int n_samples,
    const int latent_dim,
    const float tolerance
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n_samples) {
    // Calculate total energy for this sample
    float total_energy = 0.0f;
    for (int i = 0; i < latent_dim; i++) {
      int offset = idx * latent_dim + i;
      float val = latents[offset];
      total_energy += val * val; // Simplified energy calculation
    }

    // Normalize if energy exceeds tolerance
    float scale_factor = 1.0f;
    if (total_energy > tolerance) {
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
    // Calculate total momentum
    float total_momentum[3] = {0.0f, 0.0f, 0.0f};

    for (int i = 0; i < momentum_dim && i < 3; i++) {
      int offset = idx * latent_dim + momentum_start_idx + i;
      total_momentum[i] = latents[offset];
    }

    // Momentum should sum to zero (conservation)
    // Subtract mean to enforce this
    float mean_momentum[3];
    for (int i = 0; i < 3; i++) {
      mean_momentum[i] = total_momentum[i];
    }

    // Copy latents and apply momentum correction
    for (int i = 0; i < latent_dim; i++) {
      int offset = idx * latent_dim + i;
      corrected_latents[offset] = latents[offset];

      // Correct momentum components
      if (i >= momentum_start_idx && i < momentum_start_idx + momentum_dim &&
          i - momentum_start_idx < 3) {
        corrected_latents[offset] -= mean_momentum[i - momentum_start_idx];
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
    // Shared memory for reduction
    __shared__ float shared_energy[256];
    __shared__ float shared_momentum[3][256];

    int tid = threadIdx.x;

    // Initialize shared memory
    shared_energy[tid] = 0.0f;
    for (int i = 0; i < 3; i++) {
      shared_momentum[i][tid] = 0.0f;
    }

    // Calculate energy and momentum for this thread
    float energy = 0.0f;
    float momentum[3] = {0.0f, 0.0f, 0.0f};

    for (int i = tid; i < latent_dim; i += blockDim.x) {
      int offset = idx * latent_dim + i;
      float val = latents[offset];

      // Energy contribution
      energy += val * val;

      // Momentum contribution (first 3 components)
      if (i < 3) {
        momentum[i] = val;
      }
    }

    shared_energy[tid] = energy;
    for (int i = 0; i < 3; i++) {
      shared_momentum[i][tid] = momentum[i];
    }

    __syncthreads();

    // Reduce energy
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (tid < s) {
        shared_energy[tid] += shared_energy[tid + s];
        for (int i = 0; i < 3; i++) {
          shared_momentum[i][tid] += shared_momentum[i][tid + s];
        }
      }
      __syncthreads();
    }

    // Calculate correction factors
    float energy_scale = 1.0f;
    if (tid == 0) {
      float total_energy = shared_energy[0];
      if (total_energy > energy_tolerance) {
        energy_scale = sqrtf(energy_tolerance / total_energy);
      }
    }

    __syncthreads();

    // Apply corrections
    for (int i = tid; i < latent_dim; i += blockDim.x) {
      int offset = idx * latent_dim + i;
      float corrected = latents[offset] * energy_scale;

      // Momentum correction
      if (i < 3) {
        corrected -= shared_momentum[i][0];
      }

      corrected_latents[offset] = corrected;
    }
  }
}

// ============================================================================
// Fast Matrix Multiplication Kernel (for tensor operations)
// ============================================================================

__global__ void matmul_kernel(const float *__restrict__ A,
                              const float *__restrict__ B,
                              float *__restrict__ C, const int M, const int N,
                              const int K) {
  __shared__ float As[32][32];
  __shared__ float Bs[32][32];

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  float sum = 0.0f;

  for (int tile = 0; tile < (K + 31) / 32; tile++) {
    // Load tiles into shared memory
    if (row < M && (tile * 32 + threadIdx.x) < K) {
      As[threadIdx.y][threadIdx.x] = A[row * K + tile * 32 + threadIdx.x];
    } else {
      As[threadIdx.y][threadIdx.x] = 0.0f;
    }

    if ((tile * 32 + threadIdx.y) < K && col < N) {
      Bs[threadIdx.y][threadIdx.x] = B[(tile * 32 + threadIdx.y) * N + col];
    } else {
      Bs[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

// Compute partial sum
#pragma unroll
    for (int k = 0; k < 32; k++) {
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
