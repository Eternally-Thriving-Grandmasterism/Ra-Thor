/*!
# Powrush Particle Shaders — GPU Branch Prediction

Investigation of how GPUs handle branch prediction compared to CPUs.

## GPU vs CPU Branching

**CPUs** have sophisticated branch predictors with history tables, allowing speculative execution of likely branches to hide latency.

**GPUs** take a fundamentally different approach:
- Designed for massive parallelism and latency hiding through many threads rather than deep speculation on individual threads.
- When a warp encounters a branch, if all threads agree on the outcome, execution is efficient.
- If threads diverge, the warp serializes both paths (warp divergence).
- GPUs have limited or no traditional dynamic branch prediction like CPUs. They rely more on warp uniformity.

## Implications for Shader Code

- The best way to avoid branch penalties on GPUs is to **minimize divergence** (keep warps as uniform as possible).
- Relying on "prediction" is less effective than structuring code for uniformity.
- Operations like `subgroupBallot` are powerful because they are wave-uniform and help move work into uniform phases.

## Relevance to Powrush Culling

Visibility tests (`if (visible)`) introduce potential divergence. Our WaveLocal Reduction technique helps by performing the divergent visibility test first, then moving into wave-uniform compaction using ballot and shuffle. This structure reduces the performance impact of any divergence that does occur.

Understanding that GPUs do not have strong branch prediction reinforces why techniques that promote warp uniformity (like our wave-local methods) are so effective.
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

pub mod compute {
    /// Notes on GPU branch prediction.
    pub const GPU_BRANCH_PREDICTION_NOTES: &str = r#"
        // GPUs have limited branch prediction compared to CPUs.
        // Focus on minimizing warp divergence instead.
        // Wave-uniform operations (ballot, shuffle) help structure code for efficiency.
        // Our WaveLocal Reduction already follows this principle.
    "#;
}
