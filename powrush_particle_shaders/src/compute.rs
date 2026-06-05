/*!
# Compute Shaders

Culling and compaction shaders with vkCmdDrawIndirectCount support.
*/

pub mod hiz {
    pub const DISTANCE_AND_HIZ_TEST: &str = r#" ... "#;

    /// WaveLocal Reduction Compaction with Draw Count support
    ///
    /// This version writes both the compacted indices and the actual
    /// draw count into a separate buffer, enabling vkCmdDrawIndirectCount.
    pub const COMPACTION_WITH_DRAW_COUNT: &str = r#"
        enable subgroups;

        struct DrawIndirect {
            vertex_count: u32,
            instance_count: u32,
            first_vertex: u32,
            first_instance: u32,
        };

        @group(0) @binding(0) var<storage, read> visible_flags: array<u32>;
        @group(0) @binding(1) var<storage, read_write> visible_indices: array<u32>;
        @group(0) @binding(2) var<storage, read_write> draw_indirect: DrawIndirect;
        @group(0) @binding(3) var<storage, read_write> draw_count: array<u32>; // For vkCmdDrawIndirectCount
        @group(0) @binding(4) var<uniform> params: CompactionParams;

        struct CompactionParams {
            total_particles: u32,
        };

        @compute @workgroup_size(64)
        fn main(
            @builtin(global_invocation_id) global_id: vec3<u32>,
            @builtin(subgroup_invocation_id) lane: u32
        ) {
            let index = global_id.x;
            if (index >= params.total_particles) { return; }

            let visible = visible_flags[index] == 1u;

            let ballot = subgroupBallot(visible);
            let local_rank = countOneBits(ballot & ((1u << lane) - 1u));

            var base_offset: u32 = 0u;
            if (lane == 0u) {
                let wave_count = countOneBits(ballot);
                base_offset = atomicAdd(&draw_indirect.instance_count, wave_count);

                // Atomically update the draw count buffer (for vkCmdDrawIndirectCount)
                atomicAdd(&draw_count[0], wave_count);
            }

            base_offset = subgroupBroadcast(base_offset, 0u);

            if (visible) {
                visible_indices[base_offset + local_rank] = index;
            }
        }
    "#;
}
