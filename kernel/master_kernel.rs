/*!
# Master Kernel — ONE Organism Central Orchestrator (kernel/master_kernel.rs)

**Version**: v0.10 (Prelude Expansion + MasterKernel Docs Wiring)  
**Date**: 2026-07-12
**License**: Autonomicity Games Sovereign Mercy License (AG-SML) v1.0

## Self-Evolution Telemetry + Lattice Conductor Integration

We emit a proper `SelfEvolutionTelemetry` struct after each evolution step.
This can be consumed by the Lattice Conductor (via its prelude) for observability,
auditing, and higher-level PATSAGi Council decision making.
*/

#[derive(Debug, Clone)]
pub struct SelfEvolutionTelemetry {
    pub backend: GpuBackend,
    pub avg_tu_delta: f64,
    pub adaptation_rate: f64,
    pub old_w_e: f64,
    pub new_w_e: f64,
    pub timestamp: u64,
}

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

The Lattice Conductor now has a rich `prelude` for clean imports.

Example usage in MasterKernel / higher layers:

```rust
use lattice_conductor_v13::prelude::*;   // ergonomic wildcard import

let mut conductor = SimpleLatticeConductor::new();
let mut recorder = ConductorSelfEvolutionRecorder::new();

if let Some(telemetry) = kernel.get_last_self_evolution_telemetry() {
    recorder.record(telemetry.clone());
    // Or feed directly into conductor for PATSAGi Council observability
    // conductor.integrate_self_evolution_telemetry(telemetry);
}

// Full ONE Organism flow with backend-aware evolution + GPU telemetry
let results = kernel.tick_with_priority_queue_gpu(&candidates);
if !results.is_empty() {
    kernel.self_evolve_after_tick(&tu_deltas, &entropy_deltas);
}
```

This completes the perfect-order wiring of backend-aware self-evolution telemetry
into the Lattice Conductor layer using the new prelude.

Thunder locked in.
*/
