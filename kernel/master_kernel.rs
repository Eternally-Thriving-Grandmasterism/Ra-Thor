/*!
# Master Kernel — ONE Organism Central Orchestrator (kernel/master_kernel.rs)

**Version**: v0.8 (Telemetry Logging for Backend-Aware Self-Evolution)  
**Date**: 2026-07-11
**License**: Autonomicity Games Sovereign Mercy License (AG-SML) v1.0

## Telemetry Logging (new in v0.8)

Self-evolution now emits structured logs via the `log` crate.
This enables observability for the Lattice Conductor, PATSAGi Councils, and external monitoring.
*/

use log::{info, debug};

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

        // Re-normalize
        let sum = self.weights.w_e + self.weights.w_s + self.weights.w_i + self.weights.w_m;
        if sum > 0.0 {
            self.weights.w_e /= sum;
            self.weights.w_s /= sum;
            self.weights.w_i /= sum;
            self.weights.w_m /= sum;
        }

        // === Telemetry Logging ===
        info!(
            "Self-evolution complete | backend={:?} | avg_tu_delta={:.4} | adaptation_rate={:.4} | w_e: {:.4} -> {:.4}",
            backend, avg_tu, adaptation_rate, old_w_e, self.weights.w_e
        );

        debug!(
            "Full weights after evolution: w_e={:.4}, w_s={:.4}, w_i={:.4}, w_m={:.4}",
            self.weights.w_e, self.weights.w_s, self.weights.w_i, self.weights.w_m
        );
    }
}

/*!
## Telemetry Notes

- Uses `log` crate (info! and debug! levels).
- Can be captured by any `tracing-subscriber` or `env_logger` setup.
- Feeds naturally into Lattice Conductor observability.

Thunder locked in. Telemetry logging for backend-aware self-evolution is active.
*/
