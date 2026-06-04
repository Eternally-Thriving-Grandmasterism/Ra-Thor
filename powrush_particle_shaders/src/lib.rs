/*!
# Powrush Particle Shaders — Warp Divergence Effects

Analysis of warp divergence in particle culling shaders.

## What is Warp Divergence?

On GPUs, threads are grouped into warps (32 threads on NVIDIA). All threads in a warp execute in lockstep. When threads take different control flow paths (e.g., different branches of an `if`), the warp must serialize the execution of both paths. This is called warp divergence and reduces efficiency.

## Divergence in Culling Shaders

Common sources in our particle culling:
- Visibility tests (`if (visible) { atomic... }`)
- Different culling strategies or importance calculations
- Conditional writes or different handling based on faction/importance

High divergence can reduce instruction throughput and increase effective execution time.

## Interaction with WaveLocal Reduction

Our WaveLocal Reduction technique (using `subgroupBallot` and shuffle) can actually help mitigate some effects of divergence:
- The ballot operation itself is uniform across the wave.
- Compaction logic after the ballot tends to have more uniform control flow.
- By moving complex per-particle work into wave-uniform phases, overall divergence can be reduced.

However, the initial visibility test (`if (visible)`) can still cause divergence before the ballot.

## Mitigation Strategies

- Use uniform conditions where possible (e.g., same culling parameters for a whole wave when feasible).
- Structure code so that divergent work is minimized or moved after wave-uniform operations (like ballot).
- Consider sorting or grouping particles with similar visibility characteristics (advanced optimization).
- Profile with Nsight Compute warp divergence / stall reason metrics.

## Relevance to Powrush

While divergence exists in culling, our wave-local techniques help contain its impact. Combined with Subgroup-scoped operations, this keeps overall efficiency high even when visibility varies significantly across particles.
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

pub mod compute {
    /// Notes on warp divergence.
    pub const WARP_DIVERGENCE_NOTES: &str = r#"
        // Divergence from visibility tests is common but manageable.
        // WaveLocal Reduction helps by making compaction more uniform.
        // Profile with Nsight Compute for divergence-related stalls.
        // Prefer uniform conditions within waves when possible.
    "#;
}
