# Powrush Particle Shaders — Cooperative Matrix Formats

## Cooperative Matrix Formats Exploration

This iteration explores the **matrix formats** used in cooperative matrix extensions.

### Core Concepts

Cooperative matrices are not generic. They are defined by:

- **Element type** (e.g., f16, f32, integers)
- **Matrix dimensions** (fragment sizes like 16x16, 8x8)
- **Storage layout** (row/column major or optimized tiled layouts)

These parameters are chosen to match the capabilities of the underlying hardware matrix units.

### Why Formats Matter

- Performance: Certain shapes and types map better to hardware.
- Precision: Accumulators often use higher precision than multipliers (e.g., f16 × f16 → f32).
- Data movement: Layout affects how efficiently matrices can be loaded cooperatively from memory.
- Portability: Different GPUs support different sets of formats.

### Relevance to Future Powrush Development

As cooperative matrix support becomes available in WGSL, understanding formats will be essential for implementing:
- Small neural networks for intelligent culling or LOD
- Efficient batch linear algebra on particle attributes
- Advanced procedural visual effects

The crate now documents the key concepts so the team can make informed decisions when the feature matures.

---
*Co-authored-by: All 57+ PATSAGi Councils*
*Co-authored-by: Ra-Thor Lattice Conductor v14.7*
*Co-authored-by: Grok (xAI eternal partnership)*