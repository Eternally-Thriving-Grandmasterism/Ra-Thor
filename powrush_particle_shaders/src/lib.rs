/*!
# Powrush Particle Shaders — Warp Coalescing Optimization

Exploration of warp memory coalescing and optimization strategies.

## What is Warp Coalescing?

On GPUs, a warp (32 threads) achieves maximum memory efficiency when its memory accesses are **coalesced** — meaning threads access consecutive addresses that fall within the same cache line (typically 128 bytes on NVIDIA). The hardware can then combine these into fewer, larger memory transactions.

Poor coalescing leads to many small transactions, wasting bandwidth and increasing latency.

## Why It Matters

Memory-bound kernels (common in particle culling, visibility buffer updates, and data compaction) are highly sensitive to coalescing. Good coalescing can provide 2x–4x or more effective bandwidth improvement.

## Optimization Strategies

### 1. Data Layout (SoA vs AoS)
- **Structure of Arrays (SoA)**: Store particle positions as separate arrays (x[], y[], z[]). When all threads access the same field (e.g., x), accesses are naturally coalesced.
- **Array of Structures (AoS)**: Storing full particle structs together often leads to strided accesses and poor coalescing when accessing one field.

### 2. Access Pattern in Shaders
- Ensure thread `i` in a warp accesses data element `i` (or at least consecutive elements).
- Avoid strided or random access patterns within a warp when possible.

### 3. Particle Sorting / Spatial Binning
- Group particles that are processed together (e.g., by screen-space tile, frustum region, or visibility) so that nearby particles have data in nearby memory.
- This improves both coalescing and cache locality.

### 4. Shared Memory Staging
- Use shared memory to gather/scatter data and reorganize it for fully coalesced global memory writes (useful in compaction phases).

### 5. Vectorized Loads/Stores
- Use `float4` / `int4` loads and stores when data is suitably aligned. This increases effective coalescing and reduces instruction count.

## Relevance to Powrush Pipeline

- **Culling Shader**: Reading `particle_positions` benefits from linear layout and consecutive thread indexing. Writing `visible_indices` after WaveLocal Reduction should be coalesced if indices are written in order.
- **Visibility Buffer**: Writes during rasterization or compute rasterization are highly sensitive to coalescing.
- **Indirect Draw Buffers**: Writing `DrawIndirect` commands and visible index lists benefits from good coalescing.

Our current wave-local techniques (ballot + shuffle) already help with the compaction phase by enabling ordered writes within the wave.

## Best Practices

- Prefer SoA layouts for particle data when the access pattern is field-wise.
- Keep thread-to-data mapping linear within warps.
- Consider spatial or visibility-based sorting/binning for large particle sets.
- Use vector loads/stores where alignment allows.
- Profile memory transactions and bandwidth utilization with Nsight Compute.
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

pub mod compute {
    /// Notes on warp coalescing.
    pub const WARP_COALESCING_NOTES: &str = r#"
        // Prefer SoA layouts for particle data.
        // Keep thread indexing linear with data layout.
        // Use vectorized loads/stores when possible.
        // Wave-local compaction helps with ordered writes.
        // Profile memory efficiency with Nsight Compute.
    "#;
}
