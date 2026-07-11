/*!
# Master Kernel — ONE Organism Central Orchestrator (kernel/master_kernel.rs)

**Version**: v0.6 (GpuConfig Exposure)  
**Date**: 2026-07-11
**License**: Autonomicity Games Sovereign Mercy License (AG-SML) v1.0

## GpuConfig Exposure
The validated GPU backend configuration is now directly accessible from MasterKernel.

This allows higher layers (Lattice Conductor, PATSAGi Councils, self-evolution logic)
to inspect or react to the currently selected GPU backend.
*/

use crate::kernel::gpu_compute_pipeline::{GpuConfig, GpuBackend};

pub struct MasterKernel {
    pub current_state: LatticeState,
    pub weights: TUWeights,
    pub utf_thresholds: UTFThresholds,
    pub tick_count: u64,

    /// Validated GPU backend configuration (loaded from config file / env / default)
    pub gpu_config: GpuConfig,
}

impl MasterKernel {
    pub fn new(initial_state: LatticeState) -> Self {
        Self {
            current_state: initial_state,
            weights: TUWeights::default(),
            utf_thresholds: UTFThresholds::default(),
            tick_count: 0,
            gpu_config: GpuConfig::load(),
        }
    }

    /// Returns the currently active GPU backend
    pub fn gpu_backend(&self) -> GpuBackend {
        self.gpu_config.backend
    }

    /// Convenience: Check if WGPU is the active backend
    pub fn is_wgpu_active(&self) -> bool {
        self.gpu_backend() == GpuBackend::Wgpu
    }

    // ... existing tick methods remain ...

    pub fn tick_with_priority_queue_gpu(&self, candidates: &[String]) -> Vec<(String, f64, f64)> {
        // Can now internally decide based on self.gpu_backend() if desired
        crate::kernel::gpu_compute_pipeline::tick_with_priority_queue_gpu(
            candidates,
            &self.current_state,
            &self.weights,
            &self.utf_thresholds,
        )
    }
}

/*!
## Usage Example

```rust
let mut kernel = MasterKernel::new(state);

println!("Active GPU backend: {:?}", kernel.gpu_backend());

if kernel.is_wgpu_active() {
    // Special handling for WGPU path
}

let results = kernel.tick_with_priority_queue_gpu(&candidates);
```

Thunder locked in. GpuConfig is now exposed at the ONE Organism orchestration layer.
*/
