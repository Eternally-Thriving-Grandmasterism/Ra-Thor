/*!
# Powrush Particle Shaders — Warp Divergence Performance Impact

Detailed analysis of the performance costs of warp divergence.

## Performance Costs of Divergence

When a warp diverges:

1. **Reduced Throughput**: The warp executes both sides of the branch serially. In the worst case (50/50 split), throughput for that section can drop by ~50%.
2. **Increased Latency**: Divergent sections take longer because both paths must run.
3. **Wasted Resources**: Threads on the inactive path still occupy scheduler slots but perform no useful work.
4. **Compounded Effects**: Divergence often pairs with uncoalesced memory accesses, further hurting performance.
5. **Latency Hiding Degradation**: Warps spend more time on divergent branches, reducing the ability to hide memory latency with other work.

## Quantitative Intuition

- Uniform warp: Full throughput (32 threads doing useful work per cycle in the active path).
- Divergent warp (50/50): Effectively ~16 threads of useful work per cycle during the divergent section.
- In hot loops with high divergence, this can lead to 1.5x–2x or more slowdown in affected code.

## Impact in Culling Shaders

Visibility tests (`if (visible)`) are a primary source of divergence. Particles near decision boundaries (e.g., near frustum planes or distance thresholds) are most likely to cause splits within a warp.

## How WaveLocal Reduction Helps

Our WaveLocal Reduction technique mitigates some impact:
- The divergent visibility test happens first.
- Immediately after, we move into wave-uniform operations (ballot + shuffle for compaction).
- This limits the duration of divergent execution and moves more work into uniform phases.

While divergence is not eliminated, its performance penalty is contained.

## Profiling Recommendations

Use Nsight Compute to measure:
- Warp divergence metrics / stall reasons
- Instruction throughput in divergent vs uniform sections
- Overall kernel efficiency before/after divergence-mitigating changes

## Best Practices

- Structure code to minimize time spent in divergent branches.
- Move work into wave-uniform phases as early as possible (as done in WaveLocal Reduction).
- When possible, group particles with similar properties to improve warp uniformity.
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

pub mod compute {
    /// Notes on divergence impact.
    pub const DIVERGENCE_IMPACT_NOTES: &str = r#"
        // Divergence can cut effective throughput significantly.
        // WaveLocal Reduction helps contain the damage by moving to uniform phases quickly.
        // Profile with Nsight Compute for divergence metrics.
        // Minimize time spent in divergent code paths.
    "#;
}
