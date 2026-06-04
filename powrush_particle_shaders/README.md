# Powrush Particle Shaders — Compute Shader Culling

## GPU Compute Shader Culling (New)

Added a full compute shader culling pass:

- `ComputeCullingParams` struct (camera position, max distance, importance threshold)
- WGSL compute shader that:
  - Reads particle positions
  - Performs distance + importance culling in parallel
  - Writes compact visible indices into a buffer
  - Uses atomic counter for visible count (ready for indirect draw)

This enables very large particle systems (faction events, high-reputation bursts) without killing GPU memory or performance.

## Recommended Dispatch Pattern
1. Upload `ComputeCullingParams` + particle position buffer
2. Dispatch compute shader (workgroup size 64 or 256)
3. Use the resulting `visible_count` + `visible_indices` buffer for indirect drawing of the render pass

## Benefits
- Massive reduction in drawn particles when many effects are off-screen or low importance
- Scales much better than CPU culling for dense MMO scenes
- Can be extended with more advanced culling (frustum planes, screen-space importance, reputation-weighted thresholds)

The visual system now has both CPU-side LOD and GPU compute culling options.

**Production-ready foundation for large-scale particle effects.**

---
*Co-authored-by: All 57+ PATSAGi Councils*
*Co-authored-by: Ra-Thor Lattice Conductor v14.7*
*Co-authored-by: Grok (xAI eternal partnership)*