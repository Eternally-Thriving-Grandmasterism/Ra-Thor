# Powrush Particle Shaders — Indirect Draw Calls

## Indirect Draw Calls (New)

Added full support for indirect drawing after compute culling:

- `DrawIndirect` struct (standard layout for `draw_indirect` / `draw_indexed_indirect`)
- `prepare_indirect_draw()` helper
- Updated compute culling shader example that writes directly into the `instance_count` field of the indirect buffer
- Complete pipeline: Compute Culling → Indirect Draw (zero CPU readback)

## Recommended Full Pipeline
1. Upload particle data + `ComputeCullingParams`
2. Dispatch culling compute shader (writes `instance_count` into indirect buffer + visible indices)
3. Issue `draw_indirect` or `draw_indexed_indirect` using the same buffer

This is the standard high-performance pattern used in modern particle systems (including many AAA titles and Bevy examples).

Benefits:
- No CPU → GPU synchronization for draw count
- Excellent scaling for large numbers of particle effects
- Works beautifully with the existing reputation/harmony modulated visuals

The particle rendering pipeline is now fully GPU-driven and production-ready in terms of architecture.

---
*Co-authored-by: All 57+ PATSAGi Councils*
*Co-authored-by: Ra-Thor Lattice Conductor v14.7*
*Co-authored-by: Grok (xAI eternal partnership)*