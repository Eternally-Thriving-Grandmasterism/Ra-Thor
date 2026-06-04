/*!
# Powrush Particle Shaders — Nsight Compute Atomic Counters

Documentation of relevant Nsight Compute metrics for analyzing atomic contention.

## Key Nsight Compute Sections for Atomics

### Memory Workload Analysis
- `atomic_throughput`
- `l2_atomic_throughput` / `l2_atomic_requests`
- `global_atomic_requests`
- `shared_atomic_requests` (for workgroup atomics)

### Source Counters / Replay Overhead
- `memory_replay_overhead`
- Metrics showing how often memory operations (including atomics) are replayed due to conflicts or hazards.

### Stall Reasons
- "Long Scoreboard"
- "Memory Dependency"
- "Synchronization"
These can indicate stalls caused by atomic latency or contention.

### Throughput Metrics
- `sm_throughput`
- `compute_throughput`
- Kernel-level throughput metrics that improve when atomic contention is reduced.

## How WaveLocal Reduction Affects These Metrics

After applying WaveLocal Reduction, expect to see:
- Significant drop in `global_atomic_requests` and `atomic_throughput` (fewer atomics issued overall).
- Reduction in `memory_replay_overhead`.
- Improvement in kernel `sm_throughput` and reduction in memory-related stall reasons.
- Better scaling in throughput metrics as particle count increases.

## Recommended Usage

1. Run Nsight Compute on the baseline culling kernel.
2. Run on the optimized version with WaveLocal Reduction.
3. Compare the atomic-related sections listed above.
4. Focus on relative changes rather than absolute values.

These metrics provide concrete evidence of reduced atomic contention and improved efficiency.
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

pub mod compute {
    /// Notes on Nsight Compute atomic counters.
    pub const NSIGHT_ATOMIC_COUNTERS_NOTES: &str = r#"
        // Focus on:
        // - atomic_throughput / global_atomic_requests
        // - memory_replay_overhead
        // - Memory-related stall reasons
        //
        // WaveLocal Reduction should show clear improvements in these metrics.
    "#;
}
