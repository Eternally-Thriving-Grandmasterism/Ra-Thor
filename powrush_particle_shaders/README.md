# Powrush Particle Shaders — Cooperative Matrix Scope Parameters

## Cooperative Matrix Scope Parameters Exploration

This iteration explores the **Scope** parameter in cooperative matrix operations.

### What Scope Means

The scope defines which group of threads cooperates on a matrix operation. It directly impacts performance, synchronization requirements, and hardware utilization.

### Available Scopes

- **Subgroup Scope** (`VK_SCOPE_SUBGROUP_KHR`)
  - Threads in the same wave/warp cooperate.
  - Best performance and widest support.
  - Ideal for most workloads.

- **Workgroup Scope** (`VK_SCOPE_WORKGROUP_KHR`)
  - Larger group of threads cooperate.
  - Higher synchronization cost.
  - Useful for matrices too large for a single subgroup.

- **Queue Family / Device Scope**
  - Very large scopes with significant synchronization overhead.
  - Rarely practical for real-time rendering workloads.

### Recommendation for Powrush

For particle culling, visibility buffer operations, and future advanced effects, **Subgroup scope** is strongly recommended. It aligns perfectly with our existing wave-local techniques (ballot, shuffle, wave-local reduction) and offers the best performance-to-complexity ratio.

Larger scopes should only be considered when matrix sizes exceed what a single subgroup can efficiently handle.

---
*Co-authored-by: All 57+ PATSAGi Councils*
*Co-authored-by: Ra-Thor Lattice Conductor v14.7*
*Co-authored-by: Grok (xAI eternal partnership)*