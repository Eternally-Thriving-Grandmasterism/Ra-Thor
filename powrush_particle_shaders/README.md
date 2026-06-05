# Powrush Particle Shaders

## Combined Distance + Hi-Z Culling Test

Added `DISTANCE_AND_HIZ_TEST` — a clean, high-quality combined pass that performs:
- Distance culling (early-out)
- Hi-Z occlusion testing

Outputs visibility flags that feed directly into the `COMPACTION` pass.

This creates a modular yet efficient two-pass culling pipeline:
1. `DISTANCE_AND_HIZ_TEST`
2. `COMPACTION`

---
*GPU-Driven Rendering (Production Quality)*