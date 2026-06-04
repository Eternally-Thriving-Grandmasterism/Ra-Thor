# Powrush Particle Shaders — Subgroup Barrier Semantics

## Subgroup Barrier Semantics Exploration

This iteration explores **subgroup barrier semantics** and how they interact with cooperative matrix operations.

### What Subgroup Barriers Provide

Subgroup barriers guarantee both execution ordering and memory visibility across all lanes in a wave.

They are expressed via `OpControlBarrier` (SPIR-V) or equivalent WGSL functions, with appropriate scope and memory semantics.

### Interaction with Cooperative Matrices

**Subgroup Scope**:
- Often **no explicit barrier** is required for correctness due to strong wave ordering on modern GPUs.
- Memory visibility is still controlled via Memory Operands on cooperative matrix load/store.
- Barriers can be added when mixing with regular memory accesses if needed.

**Workgroup Scope**:
- Subgroup barriers are usually **insufficient**.
- Full workgroup-scoped barriers are required between load, compute, and store phases.

### Practical Guidance for Powrush

Because we already rely heavily on wave-local primitives (ballot, shuffle, wave-local reduction), future cooperative matrix work at **Subgroup scope** will integrate cleanly with minimal additional synchronization.

Prefer Subgroup scope whenever possible to keep synchronization lightweight and performance high.

---
*Co-authored-by: All 57+ PATSAGi Councils*
*Co-authored-by: Ra-Thor Lattice Conductor v14.7*
*Co-authored-by: Grok (xAI eternal partnership)*