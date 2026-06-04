/*!
# Powrush Particle Shaders — GPU Atomic Contention Metrics

Analysis of metrics for measuring and understanding atomic contention on GPUs.

## Why Measure Atomic Contention?

Atomic contention is often a hidden scalability limiter in GPU compute. Measuring it helps validate optimizations like WaveLocal Reduction and guides further tuning.

## Useful Metrics for Atomic Contention

### Direct / Hardware Metrics (via profilers)
- **Atomic Throughput**: Atomics per second or per cycle.
- **Memory Replay Overhead**: How often memory operations (including atomics) are replayed due to conflicts.
- **L2 Atomic Bandwidth / Traffic**: Volume of atomic traffic going to L2.
- **Coherence Traffic**: Cache line invalidations or transfers caused by atomics.
- **Stall Reasons**: "Memory Dependency", "Synchronization", or "Long Scoreboard" stalls related to atomics.

### Indirect / Application-Level Metrics
- **Kernel Execution Time** before vs after atomic optimizations.
- **Number of Global Atomics Issued** (can be counted in shader with debug counters).
- **Visible Particle Throughput**: Particles processed per ms in culling.
- **Tail Latency** of culling kernels under high load.

## Impact of WaveLocal Reduction

Our WaveLocal Reduction technique should show clear improvements in:
- Dramatic reduction in global atomic count (often by 32x or more per wave).
- Lower memory replay overhead.
- Reduced kernel execution time under high visibility scenarios.
- Better scaling as particle count increases.

## Recommended Analysis Approach

1. Profile baseline culling kernel (pure atomicAdd per thread).
2. Profile after WaveLocal Reduction.
3. Compare:
   - Atomic count / throughput
   - Kernel time
   - Memory stall reasons
4. Measure scaling behavior with increasing particle density.

Tools: Nsight Compute (NVIDIA), Radeon GPU Profiler (AMD), or equivalent.
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

pub mod compute {
    /// Notes on atomic contention metrics.
    pub const ATOMIC_CONTENTION_METRICS_NOTES: &str = r#"
        // Key metrics to track:
        // - Global atomic count (before/after WaveLocal Reduction)
        // - Kernel execution time under load
        // - Memory replay / stall reasons
        // - Scaling behavior with particle count
        //
        // Subgroup scope + wave-local work = lower contention metrics.
    "#;
}
