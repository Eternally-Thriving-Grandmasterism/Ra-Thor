# Powrush Particle Shaders — Advanced Culling Strategies

## Compute Shader Culling Strategies

Added multiple culling strategies that can be combined:

- **Distance Culling**: Fastest, good first pass.
- **Frustum Culling**: More accurate than distance alone (uses camera forward + FOV).
- **Importance Culling**: Filters low-reputation or low-harmony effects.
- **Combined Strategy** (recommended): Distance + Frustum + Importance.

The WGSL example shows how to implement all three in one compute pass with early-outs for performance.

## Recommendations
- Use **Distance + Frustum** for general cases.
- Add **Importance** when you have many low-impact effects (low-reputation factions or minor events).
- For very large scenes, consider hierarchical clustering or multi-pass culling.

These strategies work together with the existing indirect draw batching system.

**The culling system is now flexible and production-grade.**

---
*Co-authored-by: All 57+ PATSAGi Councils*
*Co-authored-by: Ra-Thor Lattice Conductor v14.7*
*Co-authored-by: Grok (xAI eternal partnership)*