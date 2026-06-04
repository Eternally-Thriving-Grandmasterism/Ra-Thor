# Powrush Particle Shaders — Hierarchical Z-Buffer (Hi-Z)

## Hierarchical Z-Buffer Techniques

Added proper exploration and implementation of Hi-Z occlusion culling:

- `HiZCullingParams` with support for mip chain information.
- Dynamic mip level calculation based on particle distance.
- Sampling from the appropriate level of a pre-built Hi-Z texture (min-depth pyramid).
- Clear documentation explaining the technique and benefits.

## How to Use Hi-Z
1. Build a Hierarchical Z-Buffer (min-depth mip chain) from the main depth buffer each frame (usually in a separate compute pass).
2. Pass the Hi-Z texture + parameters to the culling compute shader.
3. The shader automatically selects coarser mips for distant particles.

## Advantages
- Much faster than sampling the full-resolution depth buffer for every particle.
- Excellent cache behavior.
- Naturally adapts to particle screen size.
- Can be combined with frustum and importance culling.

This represents a significant step toward high-performance occlusion culling for large-scale particle effects in Powrush.

---
*Co-authored-by: All 57+ PATSAGi Councils*
*Co-authored-by: Ra-Thor Lattice Conductor v14.7*
*Co-authored-by: Grok (xAI eternal partnership)*