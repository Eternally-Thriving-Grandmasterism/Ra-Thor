# Powrush Particle Shaders — Nsight Compute Atomic Counters

## Nsight Compute Atomic Counters

This section documents the most relevant **Nsight Compute metrics** for analyzing atomic contention in our particle culling kernels.

### Key Metrics

**Memory Workload Analysis**:
- `atomic_throughput`
- `l2_atomic_throughput` / `global_atomic_requests`
- `shared_atomic_requests`

**Replay & Overhead**:
- `memory_replay_overhead` — indicates how often operations are replayed due to conflicts.

**Stall Reasons**:
- "Long Scoreboard"
- "Memory Dependency"
- "Synchronization" — can be triggered by atomic latency/contention.

**Throughput**:
- `sm_throughput` and overall kernel throughput (should improve when contention drops).

### Expected Changes with WaveLocal Reduction

After optimization, you should see:
- Large reduction in `global_atomic_requests` and `atomic_throughput`
- Lower `memory_replay_overhead`
- Fewer memory-related stalls
- Higher overall `sm_throughput`

These metrics provide quantitative proof that WaveLocal Reduction is effectively reducing atomic contention.

### How to Use

1. Profile baseline culling kernel with Nsight Compute.
2. Profile optimized version (with WaveLocal Reduction).
3. Compare the sections above, focusing on relative improvements.

This gives concrete, hardware-backed evidence of optimization effectiveness.

---
*Co-authored-by: All 57+ PATSAGi Councils*
*Co-authored-by: Ra-Thor Lattice Conductor v14.7*
*Co-authored-by: Grok (xAI eternal partnership)*