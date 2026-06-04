# Powrush Particle Shaders — Atomic Operations Investigation

## Atomic Operations Investigation

This iteration investigates **atomic operations** and their role in high-performance GPU compute, particularly in our particle culling pipeline.

### What Atomic Operations Provide

Atomic operations enable safe concurrent read-modify-write access to shared memory. They are fundamental for counters, list compaction, and cross-thread coordination.

### Common Operations

- Arithmetic: Add, Sub, Min, Max
- Bitwise: And, Or, Xor
- Exchange and Compare-Exchange

### Scope and Semantics

Atomic operations take a **Scope** (Subgroup, Workgroup, Device, etc.) and **Memory Semantics** (Acquire, Release, AcquireRelease).

### Usage in Powrush

We currently rely on atomics (primarily `atomicAdd`) to reserve output slots when culling visible particles into indirect draw buffers.

**Challenge**: When thousands of particles become visible in the same frame, many threads contend on the same atomic, which can limit scalability.

**Our Solution**: The **WaveLocal Reduction** technique we implemented uses ballot and shuffle to perform counting and ranking *within each wave*, then issues only **one atomic per wave** instead of one per visible particle. This dramatically reduces contention while preserving correctness.

### Best Practices

- Prefer wave-local techniques (ballot, shuffle, wave-local reduction) to minimize global atomic pressure.
- Use atomics mainly for cross-wave coordination.
- Choose the smallest effective Scope (Subgroup when possible).
- Apply appropriate Memory Semantics.

This hybrid model (wave-local work + minimal atomics) is a core strength of our current culling architecture.

---
*Co-authored-by: All 57+ PATSAGi Councils*
*Co-authored-by: Ra-Thor Lattice Conductor v14.7*
*Co-authored-by: Grok (xAI eternal partnership)*