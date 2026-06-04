# Powrush Particle Shaders — Hardware Atomic Latency

## Hardware Atomic Latency Exploration

This iteration explores the **hardware-level latency** characteristics of atomic operations on modern GPUs.

### Sources of Latency

Atomic operations incur higher latency than regular memory accesses due to:
- Memory round-trips (often to L2 or device memory)
- Serialization under contention
- Coherence protocol overhead
- Greater instruction complexity

### Scope Impact

- **Subgroup Scope**: Lowest latency. Can leverage wave-level optimizations with minimal coherence traffic.
- **Workgroup Scope**: Moderate to high latency. Involves L2 and workgroup coherence.
- **Device Scope**: Highest latency due to full device-wide coherence.

### Contention Effect

In practice, **contention** (many threads targeting the same atomic address) is often the dominant factor increasing observed latency. Each waiting thread adds to the effective cost.

### Relevance to Powrush

Our **WaveLocal Reduction** technique directly addresses both the count and the latency impact of atomics by:
- Performing counting and ranking inside each wave
- Issuing only one atomic per wave instead of one per visible thread

Combined with our preference for **Subgroup scope**, this keeps hardware atomic latency well under control even at high particle counts.

### Best Practices

- Minimize contention through wave-local aggregation
- Prefer Subgroup scope for atomics
- Use atomics for cross-wave coordination rather than fine-grained per-thread work

This hardware perspective reinforces why our current optimizations are effective.

---
*Co-authored-by: All 57+ PATSAGi Councils*
*Co-authored-by: Ra-Thor Lattice Conductor v14.7*
*Co-authored-by: Grok (xAI eternal partnership)*