/*!
# Powrush Particle Shaders — Unified Memory Performance Analysis

Analysis of Unified Memory (Managed Memory) performance characteristics.

## What is Unified Memory?

Unified Memory allows the same pointer to be accessed from both CPU and GPU. The CUDA driver automatically migrates pages between host and device memory as needed (on-demand page migration).

It can be allocated with `cudaMallocManaged`.

## Performance Characteristics

### Advantages
- Simplifies programming (no explicit `cudaMemcpy`)
- Supports memory oversubscription
- Easier CPU/GPU data sharing

### Performance Costs
- **Page Migration Overhead**: When a page is first accessed on the GPU (or CPU), a page fault triggers migration over PCIe (or NVLink). This adds latency.
- **Fine-grained Access Penalty**: Random or scattered accesses cause many small migrations, which is inefficient.
- **Coherence Traffic**: Even with prefetching (`cudaMemPrefetchAsync`), maintaining coherence has cost.
- **Bandwidth Limitation**: Migration traffic is limited by the interconnect (PCIe Gen4/5 or NVLink), which is usually much slower than GPU HBM bandwidth.

## Analysis with NVIDIA Tools

In Nsight Systems:
- Look for memory migration traffic in the timeline (Host ↔ Device transfers triggered by page faults).
- Compare timelines with and without Unified Memory.

In Nsight Compute:
- Memory migration related stalls or replay overhead.
- Comparison of kernel performance using managed vs explicit memory.

## Relevance to Powrush

For high-performance hot paths (culling, visibility buffer updates, compaction), explicit memory management or pinned + async copies is generally faster and more predictable than Unified Memory.

Unified Memory can be useful for:
- Data that is infrequently accessed or updated from the CPU
- Prototyping and simpler code paths
- Scenarios where oversubscription is beneficial

For performance-critical particle processing, we recommend sticking with explicit memory control or well-managed streams with async copies.

## Best Practices

- Use `cudaMemPrefetchAsync` to reduce on-demand page faults when possible.
- Avoid fine-grained random access patterns on managed memory from the GPU.
- Profile migration traffic with Nsight Systems.
- For hot paths, prefer explicit `cudaMemcpyAsync` or device-only allocations.
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

pub mod compute {
    /// Notes on Unified Memory performance.
    pub const UNIFIED_MEMORY_NOTES: &str = r#"
        // Unified Memory simplifies code but adds migration overhead.
        // For hot paths (culling, visibility), explicit memory is usually faster.
        // Use prefetching to reduce page faults.
        // Profile migration traffic with Nsight Systems.
        // Avoid fine-grained access patterns on managed memory.
    "#;
}
