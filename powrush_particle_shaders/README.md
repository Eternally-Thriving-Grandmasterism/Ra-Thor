# Powrush Particle Shaders

## Culling Architecture (Phase 1 Consolidation)

We are actively unifying the culling system around **WaveLocal Reduction**.

### Current Components

- `CullingPass`: Main abstraction for configuring and running a culling pass.
- `CullingBuffers`: Helper for preparing related GPU buffers.
- `compute::WAVE_LOCAL_REDUCTION_CULLING`: The core shader.

The goal is a clean, maintainable, and efficient culling pipeline.

---
*Part of Ra-Thor Phase 1 Consolidation*