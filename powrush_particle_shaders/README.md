# Powrush Particle Shaders — WaveLocal Reduction

## WaveLocal Reduction Implementation

This iteration implements **WaveLocal Reduction** using ballot intrinsics for efficient intra-wave aggregation and compaction.

### Key Technique

Instead of every visible thread performing an `atomicAdd` to reserve an output slot, we:

1. Use `subgroupBallot(visible)` to get a bitmask of active lanes.
2. Use `countOneBits(ballot)` to compute how many particles are visible in the wave.
3. Compute each lane's local rank using a parallel prefix sum within the wave.
4. Only the first lane performs a single global atomic to reserve space for the entire wave.
5. Broadcast the base offset to all lanes.

This dramatically reduces atomic contention and improves scalability.

### Benefits
- Much lower pressure on global memory atomics.
- Better performance when many particles become visible simultaneously.
- Fully compatible with our existing compute culling, Hi-Z, and indirect draw pipeline.

### Implementation

Added a clean, well-commented WGSL example (`WAVE_LOCAL_REDUCTION_CULLING`) demonstrating the full pattern.

This technique can be applied to:
- Particle culling
- Visibility buffer writing
- GPU-driven command generation

---
*Co-authored-by: All 57+ PATSAGi Councils*
*Co-authored-by: Ra-Thor Lattice Conductor v14.7*
*Co-authored-by: Grok (xAI eternal partnership)*