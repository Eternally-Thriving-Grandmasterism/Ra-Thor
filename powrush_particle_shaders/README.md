# Powrush Particle Shaders

## Culling Loop Unification Progress (Phase 1 Priority 2)

`CullingPass` has been extended with `prepare_dispatch()`, which returns a `CullingDispatchPreparation` struct.

This continues building a clean, professional interface that separates:
- Configuration (`CullingPass` + `CullingConfig`)
- Resource preparation (`CullingResources`)
- Dispatch information (`CullingDispatchPreparation`)

The design is intentionally extensible for future Vulkan integration.

---
*Phase 1 Consolidation*