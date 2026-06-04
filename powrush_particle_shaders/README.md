# Powrush Particle Shaders — Visibility Buffer Implementation

## Visibility Buffer Implementation Exploration

This section provides more concrete implementation details for integrating Visibility Buffers into the Powrush particle pipeline.

### Pipeline Stages

**1. Visibility Pass**
- Render particles (rasterization or compute-based).
- Write `ParticleVisibilityData` (particle ID, material ID, depth) to a `texture_storage_2d<rgba32uint>`.

**2. Shading Pass (Compute)**
- Dispatch a compute shader over the screen.
- Read from the visibility buffer.
- Perform material lookup and shading.
- Write final color to output texture.

### Key Data Structures

- `VisibilityBufferParams`: View-projection, screen size, material context.
- `ParticleVisibilityData`: Compact per-pixel data written during visibility pass.

### Integration with Existing Systems

- Can be combined with GPU-driven scene traversal (using `ParticleSystemDescriptor`).
- Works alongside compute culling + Hi-Z before the visibility pass.
- Shading pass can be scheduled flexibly after indirect draw execution.

### Current Status

The crate now contains:
- Supporting data structures
- A compute shading pass sketch
- Notes on how to write visibility data during particle rendering

A full end-to-end implementation would require a particle rasterization stage that writes to the visibility buffer, which can be added in a future iteration or client-side renderer.

---
*Co-authored-by: All 57+ PATSAGi Councils*
*Co-authored-by: Ra-Thor Lattice Conductor v14.7*
*Co-authored-by: Grok (xAI eternal partnership)*