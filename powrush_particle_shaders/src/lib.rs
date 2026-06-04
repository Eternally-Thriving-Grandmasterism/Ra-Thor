/*!
# Powrush Particle Shaders — Profiling cudaMemPrefetchAsync Overhead

Guidance on measuring and analyzing the overhead of `cudaMemPrefetchAsync`.

## What Overhead Does cudaMemPrefetchAsync Introduce?

`cudaMemPrefetchAsync` itself has relatively low CPU-side overhead (it returns quickly). The real cost comes from the **page migration work** it triggers:

- Driver processing of the prefetch request
- Actual data movement over PCIe/NVLink
- Potential impact on GPU caches and TLBs
- Interaction with concurrent memory operations

## How to Profile It

### Nsight Systems Timeline
- Capture timelines with and without `cudaMemPrefetchAsync` calls.
- Look at memory migration traffic (Host ↔ Device transfers) triggered by prefetch operations.
- Check whether migration overlaps well with compute or causes contention/gaps.
- Measure the time from prefetch call to when the data appears resident (can use CUDA events around the prefetch).

### Nsight Compute
- Compare kernel execution time and memory stall reasons with and without prefetching.
- Look for reductions in page-fault-related stalls or `memory_replay_overhead`.
- Check overall memory efficiency metrics.

### Application-Level Metrics
- Time from `cudaMemPrefetchAsync` submission to data readiness (using CUDA events).
- Kernel execution time improvement.
- End-to-end workload or frame time improvement.
- Net benefit = (time saved in kernels) - (migration cost + any contention).

## Recommended Workflow

1. Profile baseline (no prefetch or naive Unified Memory).
2. Add `cudaMemPrefetchAsync` before hot kernels.
3. Profile again and compare:
   - Migration traffic volume and timing
   - Kernel execution time and stall reasons
   - Overall throughput
4. Determine if the reduction in kernel time outweighs the migration cost.

## When the Overhead Is Worth It

`cudaMemPrefetchAsync` is usually worth the cost when:
- It significantly reduces page faults inside performance-critical kernels.
- Migration can be overlapped with other useful work.
- The data will be accessed soon after prefetching.

It is less useful when data is prefetched too early (wasted bandwidth) or when explicit memory copies would be simpler and faster.
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

pub mod compute {
    /// Notes on profiling prefetch overhead.
    pub const PREFETCH_OVERHEAD_NOTES: &str = r#"
        // Profile migration traffic in Nsight Systems.
        // Compare kernel time and stalls with/without prefetch.
        // Measure net benefit (kernel time saved vs migration cost).
        // Overlap prefetch with other work when possible.
    "#;
}
