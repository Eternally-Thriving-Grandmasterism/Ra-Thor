# Powrush Particle Shaders

## Culling Loop Unification (Phase 1 Priority 2)

`CullingDispatchPreparation` and `CullingResources` have been refined to better associate resources with dispatch information.

The architecture now clearly separates:
- Configuration
- Dispatch preparation
- Resource management

This design is being built for clean future integration with actual Vulkan buffers and command recording.

---
*Phase 1 Consolidation*