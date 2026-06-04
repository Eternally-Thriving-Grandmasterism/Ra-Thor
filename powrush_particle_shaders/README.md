# Powrush Particle Shaders — Warp Coalescing Optimization

## Warp Coalescing Optimization Exploration

This iteration explores **warp memory coalescing** and practical optimization strategies relevant to our particle system.

### What is Coalescing?

A warp achieves peak memory efficiency when its 32 threads access consecutive memory addresses that fit within a single cache line transaction (typically 128 bytes). Poor coalescing results in many small, inefficient memory transactions.

### Why It Matters

Particle culling, visibility buffer updates, and data compaction are often memory-bound. Good coalescing can provide substantial bandwidth and performance gains.

### Key Optimization Techniques

**1. Data Layout (SoA Preferred)**
- Store particle attributes as separate arrays (positions as x[], y[], z[]).
- Field-wise access by consecutive threads leads to natural coalescing.

**2. Linear Thread-to-Data Mapping**
- Ensure thread `i` accesses data element `i` within the warp.
- Avoid strided or scattered access patterns.

**3. Spatial / Visibility Binning**
- Group particles that are processed together (by screen tile, frustum region, or visibility) to improve both coalescing and cache locality.

**4. Shared Memory Staging**
- Use shared memory to reorganize data before global writes, especially during compaction phases.

**5. Vectorized Loads/Stores**
- Use `float4` / `int4` when data alignment permits. This increases effective transaction size and reduces instruction count.

### Relevance to Our Pipeline

- Reading particle positions in the culling shader benefits from linear SoA layout.
- Writing visible indices after WaveLocal Reduction is more efficient when writes are coalesced and in-order.
- Visibility buffer updates (raster or compute) are highly sensitive to coalescing quality.

Our wave-local compaction already helps by enabling more ordered writes within the wave.

### Best Practices

- Use SoA layouts for particle data.
- Maintain linear thread-to-data mapping.
- Consider binning/sorting for large datasets.
- Use vector loads/stores where possible.
- Profile memory transaction efficiency with Nsight Compute.

This optimization area complements our existing wave-local and Subgroup-scoped techniques.

---
*Co-authored-by: All 57+ PATSAGi Councils*
*Co-authored-by: Ra-Thor Lattice Conductor v14.7*
*Co-authored-by: Grok (xAI eternal partnership)*