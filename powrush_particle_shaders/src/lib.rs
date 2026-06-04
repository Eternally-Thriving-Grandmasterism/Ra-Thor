/*!
# Powrush Particle Shaders — Subgroup Shuffle Operations

Investigation of subgroup/wave shuffle intrinsics for efficient intra-wave data exchange.

## What are Subgroup Shuffle Operations?

Subgroup shuffle operations allow lanes within the same wave to directly read values from other lanes' registers.

Common operations:
- `subgroupShuffle(value, sourceLane)`
- `subgroupShuffleUp(value, delta)`
- `subgroupShuffleDown(value, delta)`
- `subgroupBroadcast(value, sourceLane)`
- `subgroupShuffleXor(value, mask)`

These are extremely fast because they operate entirely within the wave's register file.

## Relevance to Our Pipeline

We already used `subgroupBroadcast` in WaveLocal Reduction.
Shuffle operations can further optimize:
- Parallel prefix sums / scans
- Data gathering for compaction
- Efficient reductions without ballot + countOneBits in some cases
- Better implementation of wave-local algorithms
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

pub mod compute {
    /// Example demonstrating subgroup shuffle for wave-local prefix sum.
    /// This can be used as an alternative or complement to ballot-based ranking.
    pub const SHUFFLE_BASED_PREFIX_SUM: &str = r#"
        enable subgroups;

        @compute @workgroup_size(64)
        fn main(
            @builtin(subgroup_invocation_id) lane: u32
        ) {
            // Example: Each lane has a value
            var value: u32 = lane + 1u;  // placeholder

            // Inclusive prefix sum using shuffle up
            // This is a classic parallel scan pattern using shuffles
            for (var offset = 1u; offset < 64u; offset *= 2u) {
                let other = subgroupShuffleUp(value, offset);
                if (lane >= offset) {
                    value += other;
                }
            }

            // 'value' now contains the inclusive prefix sum within the wave
        }
    "#;

    /// Improved WaveLocal Reduction using shuffle for data exchange
    pub const SHUFFLE_WAVE_LOCAL_REDUCTION: &str = r#"
        enable subgroups;

        struct DrawIndirect { /* ... */ };

        @group(0) @binding(0) var<uniform> params: ComputeCullingParams;
        @group(0) @binding(1) var<storage, read> particle_positions: array<vec3<f32>>;
        @group(0) @binding(2) var<storage, read_write> visible_indices: array<u32>;
        @group(0) @binding(3) var<storage, read_write> draw_indirect: DrawIndirect;

        @compute @workgroup_size(64)
        fn main(
            @builtin(global_invocation_id) global_id: vec3<u32>,
            @builtin(subgroup_invocation_id) lane: u32
        ) {
            let index = global_id.x;
            if (index >= params.total_particles) { return; }

            let visible = /* culling condition */;

            // Ballot to find active lanes
            let ballot = subgroupBallot(visible);
            let wave_count = countOneBits(ballot);

            // Use shuffle to broadcast base offset
            var base: u32 = 0u;
            if (lane == 0u) {
                base = atomicAdd(&draw_indirect.instance_count, wave_count);
            }
            base = subgroupBroadcast(base, 0u);

            if (visible) {
                // Compute local rank using ballot + countOneBits (or shuffle-based scan)
                let rank = countOneBits(ballot & ((1u << lane) - 1u));
                visible_indices[base + rank] = index;
            }
        }
    "#;
}
