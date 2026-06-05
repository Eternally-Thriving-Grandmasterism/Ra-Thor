# Powrush Particle Shaders

## Culling Architecture (Phase 1 Consolidation)

The culling system has been unified around **WaveLocal Reduction** as the primary recommended technique.

### Structure

- `CullingPass` — High-level abstraction for a culling pass.
- `CullingConfig` — Configuration (workgroup size, etc.).
- `compute::WAVE_LOCAL_REDUCTION_CULLING` — The shader used by the unified path.

### Recommended Flow

1. Create `ComputeCullingParams`.
2. Create `CullingPass`.
3. Use `CullingPass::shader_source()` and `dispatch_size()` for dispatch.
4. After dispatch, read `DrawIndirect` and visible indices.

This structure will be further unified in subsequent steps of Phase 1.

---
*Part of Ra-Thor Phase 1 Consolidation*
