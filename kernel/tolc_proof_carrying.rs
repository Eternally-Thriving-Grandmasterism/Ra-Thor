/*!
# Proof-Carrying TOLC Quantification Module (kernel/tolc_proof_carrying.rs)

**Version**: v0.5 (CUDA Kernel Path Added)  
**Date**: 2026-07-11  
**License**: Autonomicity Games Sovereign Mercy License (AG-SML) v1.0

## CUDA Integration
Real CUDA kernel now available:
- `kernel/cuda/tolc_compute_kernel.cu`
- Launched via `cuda_deliberation_batch` in gpu_compute_pipeline.rs

New functions:
- `conduct_deliberation_batch_cuda`
- `allocation_priority_queue_cuda`
*/

use crate::kernel::gpu_compute_pipeline::{cuda_deliberation_batch, cuda_priority_queue_batch};

// ... (previous code) ...

/// GPU Batch (Rayon parallel)
pub fn conduct_deliberation_batch_gpu(...) { gpu_deliberation_batch(...) }
pub fn allocation_priority_queue_gpu(...) { gpu_priority_queue_batch(...) }

/// **CUDA Batch** — real kernel from tolc_compute_kernel.cu
pub fn conduct_deliberation_batch_cuda(
    candidate_actions: &[String],
    current_state: &LatticeState,
    weights: &TUWeights,
    utf_thresholds: &UTFThresholds,
) -> Vec<(String, f64, f64)> {
    cuda_deliberation_batch(candidate_actions, current_state, weights, utf_thresholds)
}

pub fn allocation_priority_queue_cuda(
    candidate_actions: &[String],
    current_state: &LatticeState,
    weights: &TUWeights,
    utf_thresholds: &UTFThresholds,
) -> Vec<(String, f64, f64)> {
    cuda_priority_queue_batch(candidate_actions, current_state, weights, utf_thresholds)
}

// ... (rest of file) ...
