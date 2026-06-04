/*!
# Powrush Particle Shaders — Cooperative Matrix Synchronization Primitives

Exploration of synchronization requirements when using cooperative matrices.

## Why Synchronization Matters

Cooperative matrix operations involve multiple threads working together on matrix fragments.
Proper memory ordering and visibility are required to ensure correct results, especially when:
- Loading data into cooperative matrices
- Performing multiply-accumulate
- Storing results back to memory

## Synchronization in Cooperative Matrices

### Subgroup Scope (Recommended)

When using `VK_SCOPE_SUBGROUP_KHR`:
- Many operations have relatively relaxed synchronization requirements because execution within a wave is often implicitly ordered.
- Memory Operands on `OpCooperativeMatrixLoadKHR` / `StoreKHR` still control visibility (`MakePointerAvailable`, `MakePointerVisible`, `NonPrivatePointer`, etc.).
- Explicit subgroup barriers are usually not required for correctness within the wave.

### Workgroup Scope

When using `VK_SCOPE_WORKGROUP_KHR`:
- Stronger synchronization is typically required.
- `OpControlBarrier` with appropriate memory semantics is often needed between cooperative matrix operations.
- Higher risk of data races if synchronization is insufficient.

## Best Practices

- Prefer **Subgroup scope** for most workloads (lower synchronization cost, better performance).
- Use appropriate **Memory Operands** on load/store instructions even at subgroup scope.
- When using Workgroup scope, insert proper barriers between phases (load → compute → store).
- Test thoroughly, as insufficient synchronization can lead to subtle correctness bugs.

## Relevance to Powrush

For our particle culling, visibility buffer, and future advanced compute workloads, sticking to **Subgroup-scoped** cooperative matrices will minimize synchronization complexity while still providing excellent performance.
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

pub mod compute {
    /// Notes on cooperative matrix synchronization.
    pub const COOPERATIVE_MATRIX_SYNC_NOTES: &str = r#"
        // Subgroup scope: lighter synchronization
        // Workgroup scope: requires explicit barriers (OpControlBarrier)
        //
        // Always use correct Memory Operands on cooperative matrix
        // load and store operations.
    "#;
}
