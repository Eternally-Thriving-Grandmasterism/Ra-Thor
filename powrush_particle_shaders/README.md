# Powrush Particle Shaders — Memory Fence Semantics

## Memory Fence Semantics Investigation

This iteration investigates **memory fence semantics** (`OpMemoryBarrier`) and how they relate to cooperative matrix operations.

### What Memory Fences Provide

A memory fence establishes ordering and visibility guarantees across memory operations. Unlike memory operands attached to individual loads or stores, a fence applies more broadly to sequences of memory operations.

### Key Semantics

- **Acquire Fence**: Ensures later memory operations see relevant prior writes from other threads.
- **Release Fence**: Ensures prior writes become visible to threads performing later acquire operations.
- **AcquireRelease Fence**: Combines both directions.

### Fences vs Instruction Memory Operands

- Memory Operands on cooperative matrix load/store provide localized ordering.
- Standalone fences provide ordering across multiple operations or between different types of memory accesses.

### Interaction with Cooperative Matrices

When sequencing cooperative matrix load → multiply-accumulate → store (especially when mixed with regular loads/stores), fences help establish clear ordering points.

**Subgroup Scope**:
- Requirements are generally lighter.
- Memory Operands on the instructions themselves are often sufficient.
- Explicit fences can frequently be minimized.

**Workgroup Scope**:
- Stronger AcquireRelease fences are typically required between phases.

### Practical Guidance for Powrush

Our architecture favors **Subgroup-scoped** operations. This means memory fence requirements for future cooperative matrix work will generally remain lightweight, consistent with our existing wave-local synchronization style (ballot, shuffle, wave-local reduction).

---
*Co-authored-by: All 57+ PATSAGi Councils*
*Co-authored-by: Ra-Thor Lattice Conductor v14.7*
*Co-authored-by: Grok (xAI eternal partnership)*