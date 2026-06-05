# Powrush Particle Shaders

## Hi-Z Occlusion Test

Added `compute::hiz::HIZ_OCCLUSION_TEST`.

This compute shader tests particles against the Hi-Z pyramid to determine occlusion.
It outputs visibility flags that can be used for further compaction (e.g., combined with WaveLocal Reduction).

This completes the core of GPU-driven occlusion culling on top of distance culling.

---
*GPU-Driven Rendering*