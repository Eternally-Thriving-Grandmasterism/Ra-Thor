/*!
# Powrush Particle Shaders — Cooperative Matrix Scope Parameters

Exploration of scope parameters in cooperative matrix operations.

## What is Scope in Cooperative Matrices?

Every cooperative matrix operation is associated with a **Scope** that defines which group of threads cooperates to perform the operation.

The scope determines:
- How many threads participate
- Synchronization requirements
- Which hardware execution units are used
- Performance characteristics

## Available Scopes (from SPV_KHR_cooperative_matrix / VK_KHR_cooperative_matrix)

- **VK_SCOPE_SUBGROUP_KHR** (most common)
  - Threads within the same subgroup/wave cooperate.
  - Best performance and broadest hardware support.
  - Maps directly to wave-level execution.

- **VK_SCOPE_WORKGROUP_KHR**
  - Threads within the same workgroup cooperate.
  - Larger scope, more synchronization overhead.
  - Useful for larger matrices that don't fit in a single wave.

- **VK_SCOPE_QUEUE_FAMILY_KHR**
  - Even larger scope across queue family.
  - Rarely used for typical workloads.

- **VK_SCOPE_DEVICE_KHR**
  - Device-wide scope.
  - Very rare; high synchronization cost.

## Why Scope Matters for Powrush

For particle culling, visibility, and future neural/procedural effects, `VK_SCOPE_SUBGROUP_KHR` will almost always be the best choice because:
- It aligns with our existing wave-local reduction and ballot/shuffle work.
- It has the lowest overhead.
- It has the widest hardware support.

Larger scopes may be useful for very large matrix operations that exceed subgroup size.
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

pub mod compute {
    /// Notes on cooperative matrix scopes.
    pub const COOPERATIVE_MATRIX_SCOPE_NOTES: &str = r#"
        // Recommended scope for most use cases: Subgroup
        //
        // In SPIR-V: Scope Subgroup
        // In Vulkan: VK_SCOPE_SUBGROUP_KHR
        //
        // Larger scopes (Workgroup, Device) are possible but
        // come with higher synchronization cost and are
        // rarely needed for particle system workloads.
    "#;
}
