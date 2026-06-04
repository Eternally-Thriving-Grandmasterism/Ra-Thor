# Powrush Particle Shaders — GPU-Driven Scene Traversal

## GPU-Driven Scene Traversal

This iteration explores moving scene traversal and visibility determination to the GPU.

### Core Idea

Instead of the CPU looping over all active particle systems/effects, upload a buffer of `ParticleSystemDescriptor`s and let a compute shader traverse it, cull, and generate draw commands.

### Benefits
- Scales to very large numbers of simultaneous effects
- Reduces CPU frame time significantly
- Enables dynamic per-system decisions entirely on GPU
- Natural fit with our existing compute culling + indirect draw pipeline

### Architecture

1. CPU uploads array of `ParticleSystemDescriptor` (position, bounds, importance, faction, etc.)
2. Dispatch compute shader that traverses the array
3. For each system: perform culling (can combine distance, frustum, Hi-Z, importance)
4. Generate `DrawIndirect` commands for visible systems
5. Issue `multi_draw_indirect` using the GPU-generated command buffer

The shader sketch provided demonstrates a basic version of this traversal + command generation.

This represents a major step toward a fully GPU-driven rendering architecture for Powrush visuals.

---
*Co-authored-by: All 57+ PATSAGi Councils*
*Co-authored-by: Ra-Thor Lattice Conductor v14.7*
*Co-authored-by: Grok (xAI eternal partnership)*