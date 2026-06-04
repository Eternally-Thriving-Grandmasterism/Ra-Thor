# Powrush Particle Shaders — GPU Ballot Intrinsics

## GPU Ballot Intrinsics Exploration

This iteration explores **subgroup/wave ballot intrinsics** as a powerful optimization for our culling and visibility buffer pipeline.

### What Ballot Enables

- Efficient wave-local compaction of visible particles without heavy atomic contention.
- Parallel prefix sum within a wave using `countOneBits` on the ballot mask.
- Significantly reduced pressure on global atomic operations.
- Better scaling when thousands of particles become visible in the same wave.

### Integration with Existing Pipeline

The ballot-based approach can replace or complement our current `atomicAdd`-based slot reservation in:
- Compute culling shaders
- Visibility buffer writing
- GPU-driven scene traversal command generation

It works especially well combined with Hierarchical Z-Buffer (Hi-Z) culling.

### Current Implementation

- Added `BALLOT_BASED_CULLING` shader sketch demonstrating wave-local compaction.
- Uses `subgroupBallot`, `countOneBits`, and `subgroupBroadcast`.
- Shows how to compute local rank within a wave for efficient buffer writing.

### Trade-offs

- Requires subgroup support (available on most modern GPUs).
- Slightly more complex shader code.
- Excellent performance gains at high occupancy.

This technique represents an advanced optimization layer on top of our existing compute culling and visibility buffer work.

---
*Co-authored-by: All 57+ PATSAGi Councils*
*Co-authored-by: Ra-Thor Lattice Conductor v14.7*
*Co-authored-by: Grok (xAI eternal partnership)*