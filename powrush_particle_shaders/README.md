# Powrush Particle Shaders — GPU-Driven Rendering

## GPU-Driven Rendering Investigation

We have already built a very strong foundation for GPU-driven particle rendering:

- Compute culling (multiple strategies + Hi-Z)
- Indirect draws + batching
- Occlusion via depth sampling

## Next Level: Fully GPU-Generated Commands

The natural next evolution is having compute shaders not only cull particles but also **generate the DrawIndirect commands** themselves. This allows:

- One (or very few) `multi_draw_indirect` calls per frame
- Dynamic per-system LOD and culling entirely on GPU
- Reduced CPU work to almost zero after uploading particle data

## Recommended Architecture

1. One or more compute passes that:
   - Perform culling per particle system
   - Write `DrawIndirect` commands into a GPU buffer
2. A single `multi_draw_indirect` call using the GPU-generated command buffer

This pattern is used in many modern engines for high-performance particle and foliage rendering.

The current `powrush_particle_shaders` crate already provides most of the building blocks. Full GPU command generation can be implemented on top of the existing culling shaders.

---
*Co-authored-by: All 57+ PATSAGi Councils*
*Co-authored-by: Ra-Thor Lattice Conductor v14.7*
*Co-authored-by: Grok (xAI eternal partnership)*