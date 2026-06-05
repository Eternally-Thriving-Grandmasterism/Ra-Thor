# Powrush Particle Shaders

## Combined Hi-Z Occlusion + Compaction

Refactored `HIZ_OCCLUSION_AND_COMPACTION` for improved clarity:

- Clear separation between Hi-Z occlusion test (`is_occluded` function)
- WaveLocal Reduction compaction logic
- Better comments and structure

This provides a clean, efficient combined pass for distance + occlusion culling with compaction.

---
*GPU-Driven Rendering*