# Powrush Particle Shaders — Visibility Buffer Techniques

## Visibility Buffer Investigation

This iteration explores **Visibility Buffer** rendering techniques and their applicability to our GPU-driven particle pipeline.

### Core Concept

Instead of storing full material properties in a G-Buffer, a Visibility Buffer stores compact identifiers (particle ID, material ID, depth). Shading is then performed in a subsequent pass (often compute) only for pixels that are actually visible.

### Benefits for Powrush
- Lower memory bandwidth than traditional deferred rendering.
- Excellent fit with our compute culling + indirect draw + GPU scene traversal architecture.
- Enables decoupled shading (compute shading pass can be scheduled flexibly).
- Easier to handle many different faction visual styles and resonance effects.

### Integration Opportunities

- During particle rasterization (or compute rasterization), write `ParticleVisibilityData` into the visibility buffer.
- In a later compute pass, read the visibility buffer and perform shading/lighting.
- Can be combined with our existing Hi-Z occlusion culling and GPU-driven command generation.

### Recommended Direction

For high-performance particle rendering in Powrush, a Visibility Buffer + compute shading approach is very promising. It aligns well with the GPU-driven philosophy we've been building toward.

---
*Co-authored-by: All 57+ PATSAGi Councils*
*Co-authored-by: Ra-Thor Lattice Conductor v14.7*
*Co-authored-by: Grok (xAI eternal partnership)*