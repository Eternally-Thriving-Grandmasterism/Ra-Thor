/*!
# Powrush Particle Shaders — Atomic Operations Investigation

Investigation of atomic operations in GPU compute shaders.

## What are Atomic Operations?

Atomic operations allow multiple threads to safely perform read-modify-write operations on shared memory locations without data races. They are essential for counters, compaction, reductions, and synchronization.

## Common Atomic Operations (SPIR-V)

- `OpAtomicIAdd` / `OpAtomicISub`
- `OpAtomicAnd`, `OpAtomicOr`, `OpAtomicXor`
- `OpAtomicSMin` / `OpAtomicUMin`, `OpAtomicSMax` / `OpAtomicUMax`
- `OpAtomicExchange`, `OpAtomicCompareExchange`
- `OpAtomicLoad`, `OpAtomicStore`

## Key Parameters

- **Scope**: Subgroup, Workgroup, Device, QueueFamily, etc.
- **Memory Semantics**: Acquire, Release, AcquireRelease, etc.
- **Storage Class**: Workgroup, StorageBuffer, Uniform, etc.

## Relevance to Powrush Particle System

We currently use atomics extensively in compute culling to reserve output slots for visible particles (e.g., `atomicAdd` on `instance_count` in `DrawIndirect`).

**Limitation of Pure Atomic Approach**:
- High contention when many threads become visible simultaneously.
- Can become a performance bottleneck at very large particle counts.

**Our Optimization**:
- WaveLocal Reduction (using ballot + shuffle) significantly reduces the number of global atomics by counting and ranking within the wave first, then performing only one atomic per wave.

This hybrid approach (wave-local work + minimal atomics) gives us the best of both worlds: correctness with high performance.
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

pub mod compute {
    /// Notes on atomic operations.
    pub const ATOMIC_OPERATIONS_NOTES: &str = r#"
        // Prefer wave-local techniques (ballot, shuffle, wave-local reduction)
        // to minimize global atomic contention.
        //
        // Use atomics primarily for cross-wave coordination
        // (e.g., reserving space in output buffers).
        //
        // Always choose appropriate Scope and Memory Semantics.
    "#;
}
