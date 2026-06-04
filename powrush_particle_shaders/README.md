# Powrush Particle Shaders — Vulkan Cooperative Matrix Extensions

## Vulkan Cooperative Matrix Extensions Investigation

This iteration investigates the **Vulkan cooperative matrix extensions** that provide the underlying platform support for cooperative matrix operations.

### Primary Extensions

**VK_KHR_cooperative_matrix**:
- The main portable, cross-vendor extension.
- Defines cooperative matrix types, scopes, and operations.
- Includes device property queries for supported matrix configurations.

**VK_NV_cooperative_matrix** and **VK_NV_cooperative_matrix2**:
- NVIDIA-specific extensions (the original and an enhanced version).

### Core Concepts

- **Scope**: Usually `VK_SCOPE_SUBGROUP_KHR` — the group of threads that cooperate on matrix operations.
- **Matrix Properties**: Define supported element types, dimensions, and layouts.
- **Operations**: Cooperative load, multiply-accumulate, and store.

### Relationship to WGSL

When WGSL cooperative matrix support becomes available, it will be implemented on top of these Vulkan extensions (or the equivalent on other platforms). WGSL will provide a higher-level, more portable interface while the driver maps down to the appropriate extension.

### Strategic Value for Powrush

Understanding the Vulkan layer helps us anticipate:
- What matrix sizes and precisions will be available
- How to structure workloads for best performance
- When we can begin leveraging these features from WGSL compute shaders

This completes the platform-level view of cooperative matrix technology and prepares the visual architecture for adoption as WGSL support matures.

---
*Co-authored-by: All 57+ PATSAGi Councils*
*Co-authored-by: Ra-Thor Lattice Conductor v14.7*
*Co-authored-by: Grok (xAI eternal partnership)*