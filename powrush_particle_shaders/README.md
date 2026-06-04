# Powrush Particle Shaders

## Culling Architecture (Phase 1 Consolidation)

### Current State

We are unifying the culling loop around **WaveLocal Reduction**.

`CullingPass` now includes:
- Shader source access
- Dispatch size calculation
- Basic indirect buffer preparation

This is the foundation for a clean, unified culling pipeline.

---
*Part of Ra-Thor Phase 1 Consolidation*