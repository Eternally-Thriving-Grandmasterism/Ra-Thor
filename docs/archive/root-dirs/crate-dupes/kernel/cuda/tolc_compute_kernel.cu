/*!
  TOLC CUDA Kernel — Warp Divergence Analysis & Optimization
  kernel/cuda/tolc_compute_kernel.cu

  Version: v0.3 (Warp Divergence Analysis)
  Date: 2026-07-11
  License: Autonomicity Games Sovereign Mercy License (AG-SML) v1.0

  ## Warp Divergence Analysis

  Sources of divergence in v0.2:

  1. Boundary check: `if (idx >= batch_size) return;`
     - Minor divergence only at the tail of the last block.
     - Standard and unavoidable without grid-stride loops.

  2. Priority computation branch:
     ```
     if (params.free_energy_available >= params.utf_min_energy &&
         tu > 0.0f &&
         params.mercy_valence >= 0.9999999f)
     ```
     - `params.*` values are **uniform** across the warp (broadcast from constant memory).
     - Only `tu > 0.0f` is **data-dependent** per thread.
     - When a warp contains a mix of high-TU and low-TU actions, threads diverge.
     - Divergence cost: ~10-20 cycles per divergent branch on modern NVIDIA GPUs.

  Impact:
  - In healthy Lattice states (high mercy_valence + sufficient free energy), divergence is mostly driven by `tu > 0.0f`.
  - Mixed-quality candidate batches can cause significant intra-warp divergence.
  - Output writes (`tu_out[idx]`, `priority_out[idx]`) are always coalesced.

  ## Optimization Applied (v0.3)

  Converted the priority logic to **branchless arithmetic** using `fmaxf` and multiplication.
  This eliminates data-dependent control flow inside the warp while preserving exact semantics.
*/

#include <cuda_runtime.h>
#include <math.h>

struct TOLCParams { ... };

__global__ void __launch_bounds__(256)
 tolc_compute_tu_priority_batch_coalesced(
    const float* __restrict__ energy_features,
    const float* __restrict__ entropy_features,
    const float* __restrict__ info_features,
    const float* __restrict__ mercy_features,
    float* __restrict__ tu_out,
    float* __restrict__ priority_out,
    int batch_size,
    TOLCParams params
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    // Coalesced loads
    float energy_proxy  = energy_features[idx];
    float entropy_proxy = entropy_features[idx];
    float info_proxy    = info_features[idx];
    float mercy_factor  = mercy_features[idx];

    // TOLC Unit
    float delta_f = energy_proxy * 0.85f + params.free_energy_available * 0.05f;
    float neg_delta_s = entropy_proxy * 0.35f + (1.0f - fminf(entropy_proxy, 1.0f)) * 0.2f;
    float i_mutual = info_proxy * 0.30f;

    float tu = params.w_e * delta_f +
               params.w_s * neg_delta_s +
               params.w_i * i_mutual +
               params.w_m * params.mercy_valence * mercy_factor;

    // === Branchless priority computation (no warp divergence) ===
    // Original branched version:
    //   if (params.free_energy_available >= params.utf_min_energy &&
    //       tu > 0.0f && params.mercy_valence >= 0.9999999f)
    //       priority = tu * params.mercy_valence * (1.0f - params.distortion_penalty);
    //
    // Branchless equivalent (semantically identical):
    float mercy_gate = (params.mercy_valence >= 0.9999999f) ? 1.0f : 0.0f;
    float energy_gate = (params.free_energy_available >= params.utf_min_energy) ? 1.0f : 0.0f;
    float tu_gate = (tu > 0.0f) ? 1.0f : 0.0f;

    float priority = tu * params.mercy_valence * (1.0f - params.distortion_penalty)
                     * mercy_gate * energy_gate * tu_gate;

    priority = fmaxf(priority, 0.0f);

    tu_out[idx] = tu;
    priority_out[idx] = priority;
}

extern "C" void launch_tolc_batch_coalesced(...) { ... }
