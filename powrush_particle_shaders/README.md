# Powrush Particle Shaders — Profiling Memory Transactions

## Profiling Memory Transactions

This section provides guidance on how to **profile memory transactions** using Nsight Compute, with direct relevance to our particle culling and visibility optimizations.

### Key Sections in Nsight Compute

**Memory Workload Analysis** (most important):
- `l1tex__t_requests` / `l1tex__t_sectors`
- `lts__t_requests` / `lts__t_sectors`
- `dram__sectors`
- `memory_replay_overhead`

**Memory Efficiency**:
- Global Load/Store Efficiency
- L1/L2 Hit Rates
- Transactions per Request (lower = better coalescing)

**Stall Reasons**:
- "Long Scoreboard"
- "Memory Dependency" (often correlated with memory issues)

### Good vs Poor Coalescing Indicators

**Good Coalescing**:
- Low number of sectors/transactions relative to requests
- High global load/store efficiency (ideally >80-90%)
- Lower memory replay overhead

**Poor Coalescing**:
- High transaction count per request
- Low efficiency percentages
- Elevated replay overhead and memory stalls

### Recommended Profiling Workflow

1. Profile the baseline culling kernel, focusing on the Memory Workload Analysis section.
2. Profile after optimizations (WaveLocal Reduction, SoA layout changes, access pattern improvements).
3. Compare key metrics:
   - Transaction counts and efficiency
   - `memory_replay_overhead`
   - Memory-related stall reasons
   - Overall kernel throughput

### Expected Improvements

- WaveLocal Reduction: Should reduce write traffic to visible index buffers.
- SoA + Linear Access Patterns: Should improve read coalescing for particle data.
- Vectorized loads/stores: Should increase effective transaction size and efficiency.

These metrics provide quantitative proof of memory optimization effectiveness in our pipeline.

---
*Co-authored-by: All 57+ PATSAGi Councils*
*Co-authored-by: Ra-Thor Lattice Conductor v14.7*
*Co-authored-by: Grok (xAI eternal partnership)*