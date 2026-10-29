/*!
# Master Kernel — ONE Organism Central Orchestrator (kernel/master_kernel.rs)

**Version**: v0.11 (Shared Types from Lattice Conductor - No Duplication)  
**Date**: 2026-07-13
**License**: Autonomicity Games Sovereign Mercy License (AG-SML) v1.0

## Shared Types from Lattice Conductor

`GpuBackend` and `SelfEvolutionTelemetry` are now imported from `lattice-conductor-v13`
(the single source of truth). This eliminates duplication and ensures ONE Organism coherence.
*/

use lattice_conductor_v13::{GpuBackend, SelfEvolutionTelemetry};

impl MasterKernel {
    pub fn self_evolve_after_tick(
        &mut self,
        recent_tu_deltas: &[f64],
        recent_entropy_reductions: &[f64],
    ) {
        if recent_tu_deltas.is_empty() { return; }

        let backend = self.gpu_config.backend;
        let avg_tu = recent_tu_deltas.iter().sum::<f64>() / recent_tu_deltas.len() as f64;
        let adaptation_rate = match backend {
            GpuBackend::Wgpu => 0.025,
            GpuBackend::Cuda => 0.04,
            GpuBackend::Rayon => 0.03,
        };

        let old_w_e = self.weights.w_e;
        self.weights.w_e = (self.weights.w_e + avg_tu * adaptation_rate).clamp(0.2, 0.5);

        let sum = self.weights.w_e + self.weights.w_s + self.weights.w_i + self.weights.w_m;
        if sum > 0.0 {
            self.weights.w_e /= sum;
            self.weights.w_s /= sum;
            self.weights.w_i /= sum;
            self.weights.w_m /= sum;
        }

        self.last_self_evolution_telemetry = Some(SelfEvolutionTelemetry {
            backend,
            avg_tu_delta: avg_tu,
            adaptation_rate,
            old_w_e,
            new_w_e: self.weights.w_e,
            timestamp: self.tick_count,
        });

        info!(...);
    }

    pub fn get_last_self_evolution_telemetry(&self) -> Option<&SelfEvolutionTelemetry> {
        self.last_self_evolution_telemetry.as_ref()
    }
}

/*!
## Lattice Conductor Wiring (with Prelude)

The Lattice Conductor now owns the canonical `SelfEvolutionTelemetry`.
MasterKernel imports it directly — no duplication.

```rust
use lattice_conductor_v13::prelude::*;

if let Some(telemetry) = kernel.get_last_self_evolution_telemetry() {
    conductor.integrate_self_evolution_telemetry(telemetry);
}
```

Thunder locked in. Duplication eliminated. ONE Organism unified.
*/
