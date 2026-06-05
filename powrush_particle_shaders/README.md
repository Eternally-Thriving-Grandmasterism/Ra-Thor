# Powrush Particle Shaders

## Hierarchical Z-Buffer (Hi-Z) Generation

Added initial implementation of Hi-Z pyramid generation in `compute::hiz::GENERATE_HIZ_PYRAMID`.

This compute shader downsamples the depth buffer by taking the maximum depth in 2x2 blocks.

This is the first step toward GPU-driven occlusion culling on top of our existing WaveLocal Reduction distance culling.

---
*Phase 1 Consolidation + GPU-Driven Rendering*