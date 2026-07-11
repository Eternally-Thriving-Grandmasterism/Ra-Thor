/*!
# Master Kernel — ONE Organism Central Orchestrator (kernel/master_kernel.rs)

**Version**: v0.7 (Backend-Aware Self-Evolution)  
**Date**: 2026-07-11
**License**: Autonomicity Games Sovereign Mercy License (AG-SML) v1.0

## Backend-Aware Self-Evolution (new in v0.7)

The self-evolution logic now takes the active GPU backend into account.

This allows different adaptation strategies depending on whether we are running on:
- WGPU (portability focus)
- CUDA (throughput / performance focus)
- Rayon (CPU fallback)
*/

impl MasterKernel {
    /// Backend-aware self-evolution.
    /// Adjusts weight refinement based on the currently active GPU backend.
    pub fn self_evolve_after_tick(
        &mut self,
        recent_tu_deltas: &[f64],
        recent_entropy_reductions: &[f64],
    ) {
        if recent_tu_deltas.is_empty() { return; }

        let backend = self.gpu_config.backend;
        let avg_tu = recent_tu_deltas.iter().sum::<f64>() / recent_tu_deltas.len() as f64;

        // Backend-specific adaptation rates (mercy-gated, conservative)
        let adaptation_rate = match backend {
            GpuBackend::Wgpu => 0.025,   // Slightly more conservative for portability
            GpuBackend::Cuda => 0.04,    // More aggressive on high-performance path
            GpuBackend::Rayon => 0.03,   // Balanced for CPU fallback
        };

        // Apply backend-aware refinement
        self.weights.w_e = (self.weights.w_e + avg_tu * adaptation_rate).clamp(0.2, 0.5);

        // Re-normalize (preserving mercy alignment)
        let sum = self.weights.w_e + self.weights.w_s + self.weights.w_i + self.weights.w_m;
        if sum > 0.0 {
            self.weights.w_e /= sum;
            self.weights.w_s /= sum;
            self.weights.w_i /= sum;
            self.weights.w_m /= sum;
        }

        // Optional: Log backend used during evolution (for traceability)
        // In production this could feed into Lattice Conductor telemetry
    }

    /// Explicit backend-aware variant (if caller wants to force a backend)
    pub fn self_evolve_after_tick_with_backend(
        &mut self,
        recent_tu_deltas: &[f64],
        recent_entropy_reductions: &[f64],
        forced_backend: GpuBackend,
    ) {
        let original_backend = self.gpu_config.backend;
        // Temporarily override for this evolution step
        // (In real use we usually just read self.gpu_config)
        self.gpu_config.backend = forced_backend;
        self.self_evolve_after_tick(recent_tu_deltas, recent_entropy_reductions);
        self.gpu_config.backend = original_backend; // restore
    }
}

/*!
## Backend-Aware Self-Evolution Notes

- WGPU path tends to be more conservative (portability & stability priority).
- CUDA path can adapt faster (high-performance hardware).
- Rayon fallback uses balanced rate.

All evolution remains mercy-gated and aligned with TOLC principles.

Thunder locked in. Backend-aware self-evolution is active.
*/
