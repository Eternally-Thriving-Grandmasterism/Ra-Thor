# Powrush Particle Shaders

## Modular Culling Pipeline (Hi-Z + Compaction)

Following Option 2, we now have two clean, separate but tightly integrated passes:

1. `HIZ_OCCLUSION_TEST` — Tests particles against the Hi-Z pyramid and writes visibility flags.
2. `COMPACTION` — Reads visibility flags and performs WaveLocal Reduction style compaction.

This design prioritizes clarity, debuggability, and flexibility while maintaining high performance.

---
*GPU-Driven Rendering*