/*!
# Powrush Particle Shaders — GPU Ballot Intrinsics

Exploration of GPU subgroup/wave ballot intrinsics for high-performance culling and data compaction.

## What are Ballot Intrinsics?

Ballot operations allow threads within a wave (or subgroup) to collectively vote on a condition.
The result is a bitmask where each bit represents one lane's vote.

Common operations:
- `subgroupBallot(condition)` / `waveBallot(condition)`
- Returns a `u32` or `vec4<u32>` bitmask
- Extremely fast (single instruction on most modern GPUs)

## Why Ballot for Particles?

Our current culling and visibility logic relies heavily on `atomicAdd` to reserve slots in output buffers.
This can cause contention when many threads become visible at the same time.

**Ballot intrinsics** enable:
- Wave-local compaction without atomics
- Efficient parallel prefix sums within a wave
- Reduced atomic pressure on global memory
- Better performance at very high particle counts
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

pub mod compute {
    /// Example of using ballot intrinsics for efficient visible particle compaction.
    /// This is more advanced than pure atomicAdd-based approaches.
    pub const BALLOT_BASED_CULLING: &str = r#"
        enable subgroups;  // WGSL subgroup extension

        struct DrawIndirect {
            vertex_count: u32,
            instance_count: u32,
            first_vertex: u32,
            first_instance: u32,
        };

        @group(0) @binding(0) var<uniform> params: ComputeCullingParams;
        @group(0) @binding(1) var<storage, read> particle_positions: array<vec3<f32>>;
        @group(0) @binding(2) var<storage, read_write> visible_indices: array<u32>;
        @group(0) @binding(3) var<storage, read_write> draw_indirect: DrawIndirect;

        var<workgroup> shared_offset: atomic<u32>;

        @compute @workgroup_size(64)
        fn main(
            @builtin(global_invocation_id) global_id: vec3<u32>,
            @builtin(subgroup_invocation_id) lane_id: u32,
            @builtin(subgroup_size) wave_size: u32
        ) {
            let index = global_id.x;
            if (index >= params.total_particles) { return; }

            let pos = particle_positions[index];
            let visible = distance(pos, params.camera_position) < params.max_cull_distance;

            // Ballot: which lanes in this wave are visible?
            let ballot = subgroupBallot(visible);
            let local_count = countOneBits(ballot);

            // Compute local offset within the wave using ballot
            let local_rank = countOneBits(ballot & ((1u << lane_id) - 1u));

            // First lane in wave reserves space using atomic
            var wave_offset: u32;
            if (lane_id == 0u) {
                wave_offset = atomicAdd(&draw_indirect.instance_count, local_count);
            }
            wave_offset = subgroupBroadcast(wave_offset, 0u);

            if (visible) {
                let global_slot = wave_offset + local_rank;
                visible_indices[global_slot] = index;
            }
        }
    "#;
}

pub mod wgsl { /* existing shaders */ }
