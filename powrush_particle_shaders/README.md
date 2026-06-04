# Powrush Particle Shaders — Compute Shader Depth Sampling

## Implemented: Proper Depth Sampling in Compute

Replaced the placeholder depth sampling with a correct implementation:

- Uses `view_proj` matrix to transform world position → clip space → NDC → UV
- Samples the depth texture at the projected screen position
- Compares particle depth against scene depth for occlusion test
- Integrated with existing indirect draw pipeline

## Host Responsibilities
To use this shader, the application must:
1. Provide a depth texture (usually the main depth buffer from the previous frame or current prepass).
2. Upload the current view-projection matrix.
3. Bind the depth texture and sampler.

**Note on Depth Linearization**: The current comparison uses NDC z directly. For more accurate results with perspective projection, depth linearization should be applied on both sides (particle and sampled depth). This can be added in a follow-up iteration.

This brings the occlusion culling system much closer to production readiness.

---
*Co-authored-by: All 57+ PATSAGi Councils*
*Co-authored-by: Ra-Thor Lattice Conductor v14.7*
*Co-authored-by: Grok (xAI eternal partnership)*