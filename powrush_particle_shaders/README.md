# Powrush Particle Shaders

## Structure of Arrays (SoA) Adoption

We are moving to Structure of Arrays layout for particle positions to improve memory coalescing on the GPU.

### Current State

- Positions are now accessed via separate `pos_x`, `pos_y`, `pos_z` arrays in the WaveLocal Reduction culling shader.
- This improves coalesced memory access during culling.

This change is part of the ongoing performance and architectural improvements in Phase 1.

---
*Phase 1 Consolidation*