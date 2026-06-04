/*!
# Compute Shaders

WGSL/GLSL compute shader sources.
*/

pub mod culling {
    /// WaveLocal Reduction culling shader (register-pressure optimized).
    ///
    /// Uses Structure of Arrays + squared distance + tightened variable lifetimes.
    pub const WAVE_LOCAL_REDUCTION_CULLING: &str = r#"
        enable subgroups;

        struct ComputeCullingParams {
            view_proj: mat4x4<f32>,
            camera_position: vec3<f32>,
            max_cull_distance_squared: f32,
            total_particles: u32,
        };

        struct DrawIndirect {
            vertex_count: u32,
            instance_count: u32,
            first_vertex: u32,
            first_instance: u32,
        };

        @group(0) @binding(0) var<uniform> params: ComputeCullingParams;

        // Structure of Arrays for better coalescing
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

            // Load position components separately to reduce vec3 pressure
            let px = pos_x[index];
            let py = pos_y[index];
            let pz = pos_z[index];

            // Squared distance (avoids sqrt and vec3 construction)
            let dx = px - params.camera_position.x;
            let dy = py - params.camera_position.y;
            let dz = pz - params.camera_position.z;
            let dist_squared = dx * dx + dy * dy + dz * dz;

            let visible = dist_squared < params.max_cull_distance_squared;

            let ballot = subgroupBallot(visible);
            let local_rank = countOneBits(ballot & ((1u << lane) - 1u));

            var base_offset: u32 = 0u;
            if (lane == 0u) {
                let wave_visible_count = countOneBits(ballot);
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
