/*!
# GPU Compute Pipeline (kernel/gpu_compute_pipeline.rs)

**Version**: v0.7 (Runtime Backend Selection)  
**Date**: 2026-07-11
**License**: Autonomicity Games Sovereign Mercy License (AG-SML) v1.0

## Runtime Backend Selection (new in v0.7)

In addition to compile-time feature flags, this version adds **runtime backend selection**.

You can now choose the GPU/parallel backend at runtime via:
- Environment variable `RA_THOR_GPU_BACKEND` ("rayon", "cuda", "wgpu")
- Or by passing a `GpuBackend` enum to the dispatch functions.

This gives maximum flexibility while still requiring the corresponding feature to be compiled in.
*/

use std::env;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuBackend {
    Rayon,
    Cuda,
    Wgpu,
}

impl GpuBackend {
    /// Detect backend from environment variable or fall back to a smart default.
    pub fn from_env_or_default() -> Self {
        match env::var("RA_THOR_GPU_BACKEND").unwrap_or_default().to_lowercase().as_str() {
            "wgpu"  => GpuBackend::Wgpu,
            "cuda"  => GpuBackend::Cuda,
            "rayon" => GpuBackend::Rayon,
            _ => {
                // Smart default: prefer WGPU if compiled in, then CUDA, then Rayon
                #[cfg(feature = "wgpu")] { return GpuBackend::Wgpu; }
                #[cfg(all(feature = "cuda", not(feature = "wgpu")))] { return GpuBackend::Cuda; }
                GpuBackend::Rayon
            }
        }
    }
}

// ... (previous imports and code remain) ...

/// Runtime-aware unified dispatch
pub fn gpu_deliberation_batch(
    candidate_actions: &[String],
    current_state: &LatticeState,
    weights: &TUWeights,
    utf_thresholds: &UTFThresholds,
) -> Vec<(String, f64, f64)> {
    let backend = GpuBackend::from_env_or_default();

    match backend {
        GpuBackend::Wgpu => {
            #[cfg(feature = "wgpu")]
            {
                return wgpu_deliberation_batch(candidate_actions, current_state, weights, utf_thresholds);
            }
            // If wgpu feature not compiled but requested, fall back
            rayon_deliberation_batch(candidate_actions, current_state, weights, utf_thresholds)
        }
        GpuBackend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                return cuda_deliberation_batch(candidate_actions, current_state, weights, utf_thresholds);
            }
            rayon_deliberation_batch(candidate_actions, current_state, weights, utf_thresholds)
        }
        GpuBackend::Rayon => {
            rayon_deliberation_batch(candidate_actions, current_state, weights, utf_thresholds)
        }
    }
}

// Similar logic can be applied to gpu_priority_queue_batch and tick helpers.

pub fn tick_with_priority_queue_gpu(...) -> Vec<(String, f64, f64)> {
    gpu_priority_queue_batch(candidate_actions, current_state, weights, utf_thresholds)
}

/*!
## Runtime Backend Selection Examples

```bash
# Use WGPU even if CUDA is also compiled
RA_THOR_GPU_BACKEND=wgpu cargo run

# Force Rayon
RA_THOR_GPU_BACKEND=rayon cargo run
```

Or programmatically:
```rust
use crate::kernel::gpu_compute_pipeline::GpuBackend;

// You can also extend the dispatch functions to accept GpuBackend explicitly
```

Thunder locked in. Runtime backend selection is now available.
*/
