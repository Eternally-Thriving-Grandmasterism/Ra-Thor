/*!
# Powrush Particle Shaders — Memory Barriers Investigation

Investigation of memory barriers and their role in cooperative matrix operations.

## What are Memory Barriers?

Memory barriers control the ordering and visibility of memory operations across threads. They ensure that writes performed by one thread become visible to reads performed by other threads in a well-defined order.

In SPIR-V, they are expressed through:
- `OpControlBarrier` (with memory semantics)
- `OpMemoryBarrier`
- Memory Operands on load/store instructions (Acquire, Release, AcquireRelease, etc.)

## Key Memory Semantics

- **Acquire**: A load operation with Acquire semantics ensures that subsequent loads see all writes that happened-before the releasing store in other threads.
- **Release**: A store operation with Release semantics makes all prior writes visible to threads that later perform an acquire load.
- **AcquireRelease**: Combines both Acquire and Release semantics.
- **SequentiallyConsistent**: Strongest ordering (rarely needed in graphics/compute).

## Memory Barriers and Cooperative Matrices

When performing cooperative matrix operations:

- Memory barriers (via Memory Operands or explicit `OpControlBarrier`) are used to establish ordering between:
  - Loading data into cooperative matrices
  - Performing the multiply-accumulate
  - Storing results

**Subgroup Scope**:
- Often lighter barrier requirements due to wave-level ordering guarantees.
- Memory Operands on cooperative matrix load/store are usually sufficient.

**Workgroup Scope**:
- Explicit `OpControlBarrier` with AcquireRelease semantics is typically required between phases.

## Relevance to Powrush

Because we primarily use **Subgroup-scoped** techniques, memory barrier requirements for future cooperative matrix work will generally be lighter than in workgroup-scoped designs. Our existing wave-local patterns (ballot, shuffle, wave-local reduction) already operate with relatively relaxed synchronization.
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

pub mod compute {
    /// Notes on memory barriers.
    pub const MEMORY_BARRIER_NOTES: &str = r#"
        // Subgroup scope: lighter barriers, rely on Memory Operands
        // Workgroup scope: explicit AcquireRelease barriers usually required
        //
        // Prefer Subgroup scope to keep synchronization simple.
    "#;
}
