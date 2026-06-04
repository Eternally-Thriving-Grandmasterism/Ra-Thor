# Powrush Particle Shaders — GPU Occlusion Queries

## Investigation: GPU Occlusion Queries vs Compute Occlusion Culling

**Traditional GPU Occlusion Queries**:
- Hardware feature (query objects).
- Good for large objects (buildings, characters).
- Poor fit for particles: too many queries needed, asynchronous results, overhead.

**Compute Shader Occlusion Culling (Recommended)**:
- Sample depth texture directly in compute.
- Highly flexible and scalable.
- Can be fused with frustum + importance culling in one pass.
- No query object overhead.
- Easier to integrate with our existing indirect draw + batching pipeline.

## Implementation
Added `OcclusionCullingParams` and a compute shader sketch that samples a depth texture to perform occlusion culling on particles.

This can be combined with previous culling strategies (distance, frustum, importance) for very robust results.

**Note**: The depth sampling in the shader is simplified. A real implementation requires passing the view-projection matrix and proper screen-space projection.

The culling system continues to evolve toward production quality.

---
*Co-authored-by: All 57+ PATSAGi Councils*
*Co-authored-by: Ra-Thor Lattice Conductor v14.7*
*Co-authored-by: Grok (xAI eternal partnership)*