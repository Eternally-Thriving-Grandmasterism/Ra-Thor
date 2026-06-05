/*!
# Compute Module

WGSL shader sources for high-performance particle culling.

Part of the unified culling architecture around WaveLocal Reduction.
*/

/// WaveLocal Reduction culling shader.
///
/// This is the recommended primary culling technique.
/// It uses subgroup ballot + shuffle to minimize global atomic contention.
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

        let pos = particle_positions[index];
        let visible = distance(pos, params.camera_position) < params.max_cull_distance;

        // Wave-local ballot: which lanes in this wave are visible?
        let ballot = subgroupBallot(visible);
        let wave_visible_count = countOneBits(ballot);

        // Compute local rank within the wave using parallel prefix
        let local_rank = countOneBits(ballot & ((1u << lane) - 1u));

        // Only lane 0 performs the global atomic
        var base_offset: u32 = 0u;
        if (lane == 0u) {
            base_offset = atomicAdd(&draw_indirect.instance_count, wave_visible_count);
        }

        // Broadcast base offset to all lanes in the wave
        base_offset = subgroupBroadcast(base_offset, 0u);

        if (visible) {
            let output_index = base_offset + local_rank;
            visible_indices[output_index] = index;
        }
    }
"#;
