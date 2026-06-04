# Powrush Particle Shaders

## Culling Unification Progress

`CullingPass` now includes a `prepare_dispatch()` method that returns a `CullingDispatchInfo` struct.

This is part of building a clean, high-level interface for the culling system.

---
*Phase 1 Consolidation*