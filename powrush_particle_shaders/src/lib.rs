/*!
# Powrush Particle Shaders — Memory Operand Details

Detailed exploration of Memory Operands used with cooperative matrix load and store operations.

## Why Memory Operands Matter

`OpCooperativeMatrixLoadKHR` and `OpCooperativeMatrixStoreKHR` accept Memory Operands that control:
- Memory ordering (Acquire/Release semantics)
- Visibility between threads
- Caching and coherence behavior
- Whether the pointer is treated as private or non-private

Incorrect memory operands can lead to data races or stale data.

## Key Memory Operands for Cooperative Matrices

### Visibility and Ordering
- `MakePointerAvailable` / `MakePointerAvailableKHR`
  - Makes the pointer's value available to other threads/scopes after a store.
- `MakePointerVisible` / `MakePointerVisibleKHR`
  - Makes the pointer's value visible to the current thread/scope before a load.

### Privacy
- `NonPrivatePointer` / `NonPrivatePointerKHR`
  - Indicates that the pointer may be accessed by multiple threads (required for most cooperative use cases).

### Other Common Operands
- `Volatile`: Prevents caching; always read from memory.
- `Aligned`: Specifies alignment of the memory access.

## Scope Interaction

When using **Subgroup scope**:
- `NonPrivatePointer` is usually required.
- `MakePointerAvailable` / `MakePointerVisible` may still be needed depending on the surrounding memory operations.

When using **Workgroup scope**:
- Stronger visibility semantics (AcquireRelease) are often necessary in combination with barriers.

## Practical Guidance

For Subgroup-scoped cooperative matrices (recommended for Powrush):
- Always include `NonPrivatePointer`.
- Use `MakePointerAvailable` after stores and `MakePointerVisible` before loads when mixing with other memory accesses.
- Start with conservative (safe) operands and optimize only after verifying correctness.
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

pub mod compute {
    /// Notes on memory operands for cooperative matrices.
    pub const MEMORY_OPERAND_NOTES: &str = r#"
        // Common safe combination for Subgroup scope:
        // NonPrivatePointer + MakePointerAvailable / MakePointerVisible as needed
        //
        // Always validate memory ordering when mixing cooperative
        // matrix operations with regular loads/stores.
    "#;
}
