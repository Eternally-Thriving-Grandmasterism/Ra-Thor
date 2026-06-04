/*!
# Powrush Particle Shaders — Subgroup Barrier Semantics

Exploration of subgroup (wave) barrier semantics and their use with cooperative matrices.

## What are Subgroup Barriers?

Subgroup barriers ensure that all threads (lanes) within the same subgroup have reached a certain execution point before any thread is allowed to proceed.

They provide two main guarantees:

1. **Execution Barrier**: All threads reach the barrier before any continue.
2. **Memory Barrier**: Memory operations before the barrier are made visible to operations after the barrier (within the subgroup).

## Barrier Semantics in SPIR-V / Vulkan

Barriers are expressed using `OpControlBarrier` with:
- `Scope` = `Subgroup`
- Appropriate `Memory Semantics` (Acquire, Release, AcquireRelease, etc.)

In WGSL (when available), this will likely appear as `subgroupBarrier()` or similar with memory semantics.

## Interaction with Cooperative Matrices

### Subgroup-Scoped Cooperative Matrices

When using `VK_SCOPE_SUBGROUP_KHR`:
- Explicit subgroup barriers are **often not required** for correctness.
- Wave execution on most GPUs has strong implicit ordering guarantees.
- However, barriers can still be useful when mixing cooperative matrix operations with regular global/shared memory accesses to ensure visibility.

### Workgroup-Scoped Cooperative Matrices

When using `VK_SCOPE_WORKGROUP_KHR`:
- Subgroup barriers alone are usually **insufficient**.
- Full workgroup barriers (`OpControlBarrier` with Workgroup scope) are typically required between phases.

## Relevance to Powrush

Given our extensive use of wave-local techniques (ballot, shuffle, wave-local reduction), most of our future cooperative matrix usage will likely be at **Subgroup scope**.
In these cases, explicit barriers can often be avoided or used sparingly, reducing complexity and improving performance.
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

pub mod compute {
    /// Notes on subgroup barrier semantics.
    pub const SUBGROUP_BARRIER_NOTES: &str = r#"
        // Subgroup scope cooperative matrices:
        // - Often no explicit barrier needed
        // - Use memory operands for visibility when necessary
        //
        // Workgroup scope:
        // - Usually requires full workgroup barriers
        //
        // Prefer Subgroup scope to minimize synchronization complexity.
    "#;
}
