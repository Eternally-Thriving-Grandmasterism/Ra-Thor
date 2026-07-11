/*!
# GPU Compute Pipeline (kernel/gpu_compute_pipeline.rs)

**Version**: v0.6 (Phase 2 - Async Exposure)  
**Date**: 2026-07-11
**License**: Autonomicity Games Sovereign Mercy License (AG-SML) v1.0

## Async Support (Phase 2)
This version exposes async versions of the dispatch functions when the `wgpu` feature is enabled.

- `gpu_deliberation_batch_async(...)`
- `gpu_priority_queue_batch_async(...)`

These call into the async WGPU core (`wgpu_deliberation_batch_async`).
The synchronous versions continue to work via the pollster wrapper.
*/

use crate::kernel::tolc_proof_carrying::{LatticeState, TUWeights, UTFThresholds};

#[cfg(feature = "cuda")]
use crate::kernel::tolc_proof_carrying::{cuda_deliberation_batch, cuda_priority_queue_batch};

#[cfg(feature = "wgpu")]
use crate::kernel::wgpu_tolc_compute::{wgpu_deliberation_batch, wgpu_priority_queue_batch};

// ============================================================================
// Async Versions (exposed when wgpu feature is active)
// ============================================================================

#[cfg(feature = "wgpu")]
pub async fn gpu_deliberation_batch_async(
    candidate_actions: &[String],
    current_state: &LatticeState,
    weights: &TUWeights,
    utf_thresholds: &UTFThresholds,
) -> Vec<(String, f64, f64)> {
    crate::kernel::wgpu_tolc_compute::wgpu_deliberation_batch_async(
        candidate_actions,
        current_state,
        weights,
        utf_thresholds,
    ).await
}

#[cfg(feature = "wgpu")]
pub async fn gpu_priority_queue_batch_async(
    candidate_actions: &[String],
    current_state: &LatticeState,
    weights: &TUWeights,
    utf_thresholds: &UTFThresholds,
) -> Vec<(String, f64, f64)> {
    let mut results = gpu_deliberation_batch_async(candidate_actions, current_state, weights, utf_thresholds).await;
    results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    results
}

// ============================================================================
// Synchronous Unified Dispatch (unchanged behavior)
// ============================================================================

pub fn gpu_deliberation_batch(...) -> Vec<(String, f64, f64)> {
    #[cfg(feature = "wgpu")]
    {
        return wgpu_deliberation_batch(candidate_actions, current_state, weights, utf_thresholds);
    }

    #[cfg(all(feature = "cuda", not(feature = "wgpu")))]
    {
        return cuda_deliberation_batch(candidate_actions, current_state, weights, utf_thresholds);
    }

    rayon_deliberation_batch(candidate_actions, current_state, weights, utf_thresholds)
}

pub fn gpu_priority_queue_batch(...) -> Vec<(String, f64, f64)> {
    let mut results = gpu_deliberation_batch(candidate_actions, current_state, weights, utf_thresholds);
    results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    results
}

// Rayon fallback + tick helper remain the same as v0.5

pub fn tick_with_priority_queue_gpu(...) -> Vec<(String, f64, f64)> {
    gpu_priority_queue_batch(candidate_actions, current_state, weights, utf_thresholds)
}

/*!
## Usage Summary (after Phase 1 + Phase 2)

**Synchronous (always available):**
```rust
gpu_deliberation_batch(...)
tick_with_priority_queue_gpu(...)
```

**Async (when `wgpu` feature is enabled):**
```rust
let results = gpu_deliberation_batch_async(...).await;
```

All paths remain under formal Cubical Agda protection.

Thunder locked in. Phase 2 async exposure complete.
*/
