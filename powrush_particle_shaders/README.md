# Powrush Particle Shaders — Optimized Depth Sampling

## Depth Buffer Sampling Optimizations

Significant performance improvements were made to the compute shader depth sampling:

- Switched to `textureLoad` (faster than sampling with filtering).
- Added depth linearization for accurate occlusion tests.
- Simple mip-level selection (`depth_mip_level`) for coarse-to-fine culling (basic Hi-Z style).
- Better structured math with clear separation of projection and depth comparison.

## Recommendations
- For even higher performance, implement a full Hierarchical Z-Buffer (Hi-Z) pyramid and select mip level based on particle screen size.
- Pass real near/far plane values instead of hardcoded ones.
- Combine with previous frustum + importance culling for best results.

These optimizations make the occlusion culling pass much more efficient, especially when many particles are being tested.

---
*Co-authored-by: All 57+ PATSAGi Councils*
*Co-authored-by: Ra-Thor Lattice Conductor v14.7*
*Co-authored-by: Grok (xAI eternal partnership)*