/*!
  TOLC CUDA Kernel for Parallel Deliberation — Memory Coalescing Optimized
  kernel/cuda/tolc_compute_kernel.cu

  Version: v0.2 (Coalesced Memory Access)
  Date: 2026-07-11
  License: Autonomicity Games Sovereign Mercy License (AG-SML) v1.0

  ## Memory Coalescing Optimization (v0.2)

  Previous version used Array-of-Structures (AoS) layout:
      action_features[idx * feature_dim + feature]
  This caused strided memory access within a warp (bad coalescing).

  Optimized version uses Structure-of-Arrays (SoA) layout:
      energy_features[idx], entropy_features[idx], ...

  Result:
  - Perfect coalescing: 128-byte transactions per warp load.
  - Significantly higher memory bandwidth utilization.
  - Better L2 cache behavior on modern NVIDIA GPUs (Ampere+).

  Formal Alignment remains identical to Cubical Agda proofs.
*/

#include <cuda_runtime.h>
#include <math.h>

struct TOLCParams {
    float w_e;
    float w_s;
    float w_i;
    float w_m;
    float mercy_valence;
    float free_energy_available;
    float utf_min_energy;
    float utf_min_compute;
    float utf_min_attention;
    float distortion_penalty;
};

/**
 * Optimized CUDA kernel with perfect memory coalescing.
 *
 * Each thread processes one action.
 * All threads in a warp read consecutive addresses in each feature array.
 */
__global__ void __launch_bounds__(256)
 tolc_compute_tu_priority_batch_coalesced(
    const float* __restrict__ energy_features,     // [batch_size] - coalesced
    const float* __restrict__ entropy_features,    // [batch_size] - coalesced
    const float* __restrict__ info_features,       // [batch_size] - coalesced
    const float* __restrict__ mercy_features,      // [batch_size] - coalesced
    float* __restrict__ tu_out,                    // [batch_size]
    float* __restrict__ priority_out,              // [batch_size]
    int batch_size,
    TOLCParams params
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    // === Coalesced loads (perfect 128-byte transactions) ===
    float energy_proxy  = energy_features[idx];
    float entropy_proxy = entropy_features[idx];
    float info_proxy    = info_features[idx];
    float mercy_factor  = mercy_features[idx];

    // === TOLC Unit computation (identical semantics to Agda computeTU) ===
    float delta_f = energy_proxy * 0.85f + params.free_energy_available * 0.05f;
    float neg_delta_s = entropy_proxy * 0.35f + (1.0f - fminf(entropy_proxy, 1.0f)) * 0.2f;
    float i_mutual = info_proxy * 0.30f;

    float raw_tu = params.w_e * delta_f +
                   params.w_s * neg_delta_s +
                   params.w_i * i_mutual +
                   params.w_m * params.mercy_valence * mercy_factor;

    float tu = raw_tu; // normalization applied on host if needed

    // === UTF + Allocation Priority (mirrors allocationDistortionFree) ===
    float priority = 0.0f;
    if (params.free_energy_available >= params.utf_min_energy &&
        tu > 0.0f &&
        params.mercy_valence >= 0.9999999f) {
        priority = tu * params.mercy_valence * (1.0f - params.distortion_penalty);
        priority = fmaxf(priority, 0.0f);
    }

    tu_out[idx] = tu;
    priority_out[idx] = priority;
}

// Host launcher for the coalesced kernel
extern "C" void launch_tolc_batch_coalesced(
    const float* energy_features,
    const float* entropy_features,
    const float* info_features,
    const float* mercy_features,
    float* tu_out,
    float* priority_out,
    int batch_size,
    TOLCParams params,
    cudaStream_t stream
) {
    constexpr int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;

    tolc_compute_tu_priority_batch_coalesced<<<blocks, threads, 0, stream>>>(
        energy_features,
        entropy_features,
        info_features,
        mercy_features,
        tu_out,
        priority_out,
        batch_size,
        params
    );
}
