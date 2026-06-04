# Powrush Particle Shaders — Subgroup Shuffle Operations

## Subgroup Shuffle Operations Investigation

This iteration explores **subgroup shuffle** intrinsics, which allow direct register-to-register data exchange between lanes in the same wave.

### Key Operations

- `subgroupShuffle(value, lane)`: Read value from another lane
- `subgroupShuffleUp/Down(value, delta)`: Read from lane +/-δ
- `subgroupBroadcast(value, lane)`: Broadcast a value from one lane to all others
- `subgroupShuffleXor(value, mask)`: Exchange with lane XOR mask

### Use Cases in Our Pipeline

- Efficient wave-local prefix sums and scans (alternative to ballot + countOneBits)
- Data gathering for compaction
- Broadcasting values computed by the first lane (e.g., base offsets)
- Optimizing WaveLocal Reduction patterns

### Current Implementation

Added examples showing:
- Basic parallel prefix sum using shuffle up
- Improved WaveLocal Reduction combining ballot + shuffle

These operations are extremely fast and help reduce both latency and memory traffic within a wave.

### Integration

Shuffle operations work excellently alongside ballot intrinsics and can be used in:
- Compute culling shaders
- Visibility buffer writing passes
- GPU-driven command generation

---
*Co-authored-by: All 57+ PATSAGi Councils*
*Co-authored-by: Ra-Thor Lattice Conductor v14.7*
*Co-authored-by: Grok (xAI eternal partnership)*