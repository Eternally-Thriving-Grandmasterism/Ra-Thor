/*!
# Powrush Particle Shaders — Vulkan Cooperative Matrix Extensions

Investigation of Vulkan cooperative matrix extensions.

## Main Extensions

### VK_KHR_cooperative_matrix (Recommended / Portable)
- The primary cross-vendor extension for cooperative matrices in Vulkan.
- Defines cooperative matrix types, scopes, and operations.
- Provides `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` to query supported matrix configurations.
- Works with SPIR-V `CooperativeMatrix` types and instructions.

### VK_NV_cooperative_matrix (NVIDIA)
- NVIDIA's original cooperative matrix extension.
- Still widely supported and used.
- Largely superseded by the KHR version for new code.

### VK_NV_cooperative_matrix2 (NVIDIA)
- Newer NVIDIA-specific extension with additional features and improved integration.

## Key Concepts

- **Matrix Scope**: Defines the group of threads that cooperate (most commonly `VK_SCOPE_SUBGROUP_KHR`).
- **Matrix Properties**: Element type, rows, columns, and stride for A, B, C, and result matrices.
- **Operations**: Load, Store, and Multiply-Accumulate performed cooperatively.

## Relationship to WGSL

When WGSL gains cooperative matrix support, implementations will typically map WGSL cooperative matrix operations down to `VK_KHR_cooperative_matrix` (or the appropriate vendor extension) under the hood.

## Relevance to Powrush

Understanding these extensions helps us prepare for when WGSL cooperative matrix support becomes available. At that point, we will be able to leverage hardware-accelerated matrix operations directly from compute shaders for advanced particle system features.
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

pub mod compute {
    /// Notes on Vulkan cooperative matrix extensions.
    pub const VULKAN_COOPERATIVE_MATRIX_NOTES: &str = r#"
        // Primary extension to target: VK_KHR_cooperative_matrix
        // Query supported properties using vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR
        // Use SPIR-V CooperativeMatrix types for shader code
        //
        // When WGSL support lands, it will abstract over these extensions.
    "#;
}
