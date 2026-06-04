# Powrush Particle Shaders — Batching Indirect Draw Calls

## Batching Indirect Draws (New)

Added support for efficient batching of many indirect draw commands:

- `BatchedIndirectDraws` container
- `prepare_indirect_draw()` helper
- Example of compute shader generating multiple `DrawIndirect` commands
- Pattern for using `multi_draw_indirect` (or equivalent in wgpu/Bevy)

## Recommended Architecture
1. One (or few) compute passes that:
   - Perform culling per effect/batch
   - Write `instance_count` into an array of `DrawIndirect` commands
2. One `multi_draw_indirect` call to draw everything

This dramatically reduces draw call count when you have many active particle systems (different factions, events, environmental effects).

High-reputation factions with strong visual effects can still be batched efficiently with lower-reputation ones.

**This completes a high-performance, GPU-driven particle rendering foundation.**

---
*Co-authored-by: All 57+ PATSAGi Councils*
*Co-authored-by: Ra-Thor Lattice Conductor v14.7*
*Co-authored-by: Grok (xAI eternal partnership)*