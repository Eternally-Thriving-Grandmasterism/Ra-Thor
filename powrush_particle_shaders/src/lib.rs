/*!
# Powrush Particle Shaders — Cooperative Matrix Multiply-Accumulate

Exploration of Cooperative Matrix Multiply-Accumulate (CoopMMA) on modern GPUs.

## What is Cooperative Matrix Multiply-Accumulate?

Cooperative MMA allows a group of threads (wave or larger) to collectively perform matrix multiplication and accumulation with dedicated hardware acceleration.

It is the core primitive behind Tensor Cores (NVIDIA), Matrix Cores (AMD), and similar features on other architectures.

Key characteristics:
- Much higher throughput than scalar or vector matrix math
- Designed for small-to-medium matrix sizes (e.g., 16x16, 8x8 fragments)
- Threads cooperate to load, compute, and store results
- Exposed via extensions such as VK_KHR_cooperative_matrix and emerging WGSL support

## Potential Applications in Powrush

While still emerging in shading languages, CoopMMA could eventually enable:
- Learned / neural culling and LOD selection
- Small neural networks for importance scoring or visual effect modulation
- Advanced procedural deformation or animation of particle batches
- High-performance batched linear algebra within compute shaders

These would represent a significant leap in the complexity and intelligence of real-time particle systems.
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

pub mod compute {
    /// Forward-looking notes on Cooperative MMA usage.
    pub const COOPERATIVE_MMA_NOTES: &str = r#"
        // Cooperative Matrix Multiply-Accumulate is still maturing in WGSL (2026).
        //
        // Example future direction:
        //
        // let a = cooperative_matrix_load(...);
        // let b = cooperative_matrix_load(...);
        // let c = cooperative_matrix_multiply_accumulate(a, b, c);
        // cooperative_matrix_store(result, c);
        //
        // For now, complex matrix work can be done using
        // traditional vector math or emerging cooperative vector features.
        //
        // This capability is tracked for future adoption in advanced
        // culling, LOD, or procedural visual effects.
    "#;
}
