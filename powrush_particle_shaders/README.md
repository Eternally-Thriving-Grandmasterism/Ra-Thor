# Powrush Particle Shaders

## Descriptor Set Layout Creation

Added production-grade descriptor set layout creation for the major pipeline stages.

`GpuDrivenPipeline` now includes methods to create layouts for:
- Culling / Hi-Z
- Compaction
- Visibility Pass
- Shading Pass

This enables proper, type-safe resource binding across the entire GPU-driven pipeline.

---
*GPU-Driven Rendering (Production Quality)*