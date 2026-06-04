# Powrush Particle Shaders — Visibility Buffer Compression

## Visibility Buffer Compression

To reduce memory usage and bandwidth, we optimized the visibility data format:

### Compression Strategy

- **20 bits** for `particle_instance_id` (supports over 1 million particles)
- **8 bits** for `material_id`
- **24 bits** for depth (stored as quantized uint)
- Packed into two `u32` values (`rg32uint` texture format)

This is significantly more compact than storing multiple separate 32-bit values.

### Implementation

- `CompressedParticleVisibility` struct with packing/unpacking methods.
- Updated compute shading shader that works with the compressed `rg32uint` format.
- Clear bit manipulation for encoding/decoding.

### Trade-offs

- Slight loss of depth precision (24-bit vs 32-bit float).
- Particle ID range limited to ~1M per frame (usually sufficient).
- Much better memory efficiency and cache performance.

### Integration

The compressed format works seamlessly with the existing GPU-driven pipeline (culling, indirect draws, scene traversal) and can be used in both rasterization and compute rasterization paths.

---
*Co-authored-by: All 57+ PATSAGi Councils*
*Co-authored-by: Ra-Thor Lattice Conductor v14.7*
*Co-authored-by: Grok (xAI eternal partnership)*