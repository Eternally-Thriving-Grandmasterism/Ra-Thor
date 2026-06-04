# Powrush Particle Shaders — Warp Divergence Effects

## Warp Divergence Effects Analysis

This iteration analyzes **warp divergence** in our particle culling shaders and how it interacts with our existing optimizations.

### What is Warp Divergence?

GPUs execute threads in warps (32 threads) in lockstep. When threads within a warp take different execution paths (e.g., some particles visible, some not), the warp serializes both branches. This reduces instruction throughput.

### Sources in Culling

- Visibility tests (`if (visible)`)
- Different culling strategies or importance calculations per particle
- Conditional writes based on faction or other attributes

### Interaction with WaveLocal Reduction

Our use of `subgroupBallot` and shuffle operations for wave-local reduction can help mitigate divergence effects:
- The ballot itself is a wave-uniform operation.
- The subsequent compaction and ranking logic tends to have more uniform control flow.
- By structuring the shader so that divergent work happens before wave-uniform compaction, overall efficiency is improved.

### Mitigation Strategies

- Keep conditions as uniform as possible within a wave when feasible.
- Structure code so divergent branches are minimized or handled before/after wave-uniform phases.
- Profile using Nsight Compute warp divergence and stall reason metrics.
- Consider advanced techniques like particle sorting by visibility characteristics (if profiling shows divergence as a major bottleneck).

### Current Assessment

While some divergence is inherent in per-particle visibility tests, our wave-local techniques help contain its performance impact. Combined with Subgroup scope preference, the culling shader remains efficient even under varying visibility conditions.

---
*Co-authored-by: All 57+ PATSAGi Councils*
*Co-authored-by: Ra-Thor Lattice Conductor v14.7*
*Co-authored-by: Grok (xAI eternal partnership)*