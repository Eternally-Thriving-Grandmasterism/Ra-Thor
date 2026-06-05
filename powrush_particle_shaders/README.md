# Powrush Particle Shaders

## Hierarchical Z-Buffer (Hi-Z) Pyramid Generation

Implemented full multi-level Hi-Z pyramid generation support.

The shader `compute::hiz::GENERATE_HIZ_LEVEL` is designed to be dispatched once per mip level, reading from the previous level and writing to the next.

Typical usage:
1. Start with full-resolution depth (level 0).
2. Repeatedly dispatch `GENERATE_HIZ_LEVEL` for each subsequent mip level.

This enables GPU-driven occlusion culling on top of WaveLocal Reduction.

---
*Phase 1 + GPU-Driven Rendering*