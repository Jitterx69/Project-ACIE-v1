/*
 * CPU Fallback Implementation for Physics Constraints
 * Provides same interface as CUDA version for systems without GPU support
 */

#include <cmath>
#include <cstdlib>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================================
// Energy Conservation (CPU)
// ============================================================================

void enforce_energy_conservation_cpu(const float *latents,
                                     float *corrected_latents,
                                     const int n_samples, const int latent_dim,
                                     const float tolerance) {
#pragma omp parallel for if (n_samples > 100)
  for (int idx = 0; idx < n_samples; idx++) {
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
// Momentum Conservation (CPU)
// ============================================================================

void enforce_momentum_conservation_cpu(const float *latents,
                                       float *corrected_latents,
                                       const int n_samples,
                                       const int latent_dim,
                                       const int momentum_start_idx,
                                       const int momentum_dim) {
#pragma omp parallel for if (n_samples > 100)
  for (int idx = 0; idx < n_samples; idx++) {
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
// Combined Physics Constraints (CPU)
// ============================================================================

void enforce_physics_constraints_cpu(const float *latents,
                                     float *corrected_latents,
                                     const int n_samples, const int latent_dim,
                                     const float energy_tolerance,
                                     const float momentum_tolerance) {
#pragma omp parallel for if (n_samples > 100)
  for (int idx = 0; idx < n_samples; idx++) {
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
// Matrix Multiplication (CPU)
// ============================================================================

void matmul_cpu(const float *A, const float *B, float *C, const int M,
                const int N, const int K) {
  // Initialize C to zero
  memset(C, 0, M * N * sizeof(float));

// Standard matrix multiplication with basic optimization
#pragma omp parallel for collapse(2) if (M * N > 10000)
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < K; k++) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

// ============================================================================
// C Interface (matching CUDA version)
// ============================================================================

extern "C" {

void enforce_energy_conservation(const float *latents, float *corrected_latents,
                                 int n_samples, int latent_dim,
                                 float tolerance) {
  enforce_energy_conservation_cpu(latents, corrected_latents, n_samples,
                                  latent_dim, tolerance);
}

void enforce_momentum_conservation(const float *latents,
                                   float *corrected_latents, int n_samples,
                                   int latent_dim, int momentum_start_idx,
                                   int momentum_dim) {
  enforce_momentum_conservation_cpu(latents, corrected_latents, n_samples,
                                    latent_dim, momentum_start_idx,
                                    momentum_dim);
}

void enforce_physics_constraints(const float *latents, float *corrected_latents,
                                 int n_samples, int latent_dim,
                                 float energy_tolerance,
                                 float momentum_tolerance) {
  enforce_physics_constraints_cpu(latents, corrected_latents, n_samples,
                                  latent_dim, energy_tolerance,
                                  momentum_tolerance);
}

void cuda_matmul(const float *A, const float *B, float *C, int M, int N,
                 int K) {
  matmul_cpu(A, B, C, M, N, K);
}

} // extern "C"
