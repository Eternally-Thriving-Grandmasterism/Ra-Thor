/*!
# Powrush Particle Shaders — Cooperative Vector Operations

Investigation of cooperative vector operations on modern GPUs.

## What are Cooperative Vector Operations?

Cooperative Vector operations allow a group of threads (typically a wave or subgroup) to collectively perform vector or matrix computations more efficiently than independent threads.

This includes:
- Cooperative matrix multiply-accumulate (MMA)
- Cooperative vector reductions and scans
- Hardware-accelerated small matrix operations shared across lanes

These features are becoming available in newer GPU architectures (e.g., NVIDIA Blackwell Cooperative Vectors, AMD, Intel extensions) and are exposed through extensions in Vulkan, DX12, and emerging WGSL support.

## Relevance to Our Pipeline

While our current focus is on particle culling, visibility, and command generation, cooperative vector operations can potentially accelerate:
- Wave-local reductions and prefix sums (complementing ballot and shuffle)
- Small matrix transformations on particle data (e.g., batch transformations)
- More advanced compaction or sorting within waves
- Future work involving neural or learned culling / LOD decisions

They represent the next layer of hardware acceleration beyond traditional subgroup ballot and shuffle operations.
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

pub mod compute {
    /// Placeholder / forward-looking example of how cooperative vectors
    /// might be used in a future culling or transformation pass.
    pub const COOPERATIVE_VECTOR_NOTES: &str = r#"
        // Cooperative vector operations are still emerging in WGSL.
        // Example direction (not yet standard):
        //
        // use cooperative_matrix or cooperative_vector extensions
        //
        // let result = cooperative_vector_add(vec_a, vec_b);
        // or cooperative matrix multiply for batched transforms
        //
        // For now, we rely on ballot + shuffle for wave-local work.
        // Future versions of this shader may incorporate cooperative vectors
        // for higher throughput on vector-heavy operations.
    "#;
}
