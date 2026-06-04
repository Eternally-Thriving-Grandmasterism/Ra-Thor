/*!
# Powrush Particle Shaders — Memory Fence Semantics

Investigation of memory fence semantics in SPIR-V / Vulkan.

## What is a Memory Fence?

A memory fence (`OpMemoryBarrier`) is a standalone synchronization instruction that establishes ordering and visibility between memory operations without being attached to a specific load or store.

It takes:
- A **Scope** (Subgroup, Workgroup, Device, etc.)
- **Memory Semantics** (Acquire, Release, AcquireRelease, etc.)

## Key Fence Semantics

- **Acquire Fence**: Ensures that all subsequent memory operations see writes from other threads that happened-before a releasing operation.
- **Release Fence**: Ensures that all prior memory operations are made visible to other threads that will later perform acquire operations.
- **AcquireRelease Fence**: Combines both Acquire and Release semantics.

## Fences vs Memory Operands on Instructions

- Memory Operands on individual loads/stores (e.g., on `OpCooperativeMatrixLoadKHR`) provide localized ordering.
- Standalone fences (`OpMemoryBarrier`) provide broader ordering guarantees across multiple memory operations.

Fences are useful when you need to establish ordering between groups of operations rather than single instructions.

## Relevance to Cooperative Matrices and Powrush

When performing sequences of cooperative matrix load → compute → store mixed with other memory accesses, fences can be used to establish clear ordering points.

**Subgroup Scope**:
- Fences are often lighter or can be avoided due to wave ordering.
- Memory Operands on the cooperative matrix instructions are frequently sufficient.

**Workgroup Scope**:
- Explicit AcquireRelease fences (via `OpMemoryBarrier` or `OpControlBarrier`) are more commonly needed between phases.

Our preference for Subgroup-scoped designs keeps fence usage minimal.
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

pub mod compute {
    /// Notes on memory fence semantics.
    pub const MEMORY_FENCE_NOTES: &str = r#"
        // Acquire fence: orders subsequent loads
        // Release fence: orders prior stores
        // AcquireRelease fence: both directions
        //
        // Prefer Subgroup scope to minimize fence requirements.
    "#;
}
