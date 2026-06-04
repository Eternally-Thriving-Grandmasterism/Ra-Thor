/*!
# Compute Shaders

WGSL/GLSL compute shader sources for the particle system.
*/

pub mod culling {
    /// WaveLocal Reduction culling shader using Structure of Arrays for positions.
    ///
    /// This is the recommended primary culling technique.
    pub const WAVE_LOCAL_REDUCTION_CULLING: &str = r#"
        enable subgroups;

        struct ComputeCullingParams {
            view_proj: mat4x4<f32>,
            camera_position: vec3<f32>,
            max_cull_distance: f32,
            total_particles: u32,
        };

        struct DrawIndirect {
            vertex_count: u32,
            instance_count: u32,
            first_vertex: u32,
            first_instance: u32,
        };

        @group(0) @binding(0) var<uniform> params: ComputeCullingParams;

        // Structure of Arrays for positions (better coalescing)
        @group(0) @binding(1) var<storage, read> pos_x: array<f32>;
        @group(0) @binding(2) var<storage, read> pos_y: array<f32>;
        @group(0) @binding(3) var<storage, read> pos_z: array<f32>;

        @group(0) @binding(4) var<storage, read_write> visible_indices: array<u32>;
        @group(0) @binding(5) var<storage, read_write> draw_indirect: DrawIndirect;

        @compute @workgroup_size(64)
        fn main(
            @builtin(global_invocation_id) global_id: vec3<u32>,
            @builtin(subgroup_invocation_id) lane: u32
        ) {
            let index = global_id.x;
            if (index >= params.total_particles) { return; }

            // Structure of Arrays access
            let position = vec3<f32>(
                pos_x[index],
                pos_y[index],
                pos_z[index]
            );

            let visible = distance(position, params.camera_position) < params.max_cull_distance;

            let ballot = subgroupBallot(visible);
            let wave_visible_count = countOneBits(ballot);
            let local_rank = countOneBits(ballot & ((1u << lane) - 1u));

            var base_offset: u32 = 0u;
            if (lane == 0u) {
                base_offset = atomicAdd(&draw_indirect.instance_count, wave_visible_count);
            }

            base_offset = subgroupBroadcast(base_offset, 0u);

            if (visible) {
                let output_index = base_offset + local_rank;
                visible_indices[output_index] = index;
            }
        }
    "#;
}
