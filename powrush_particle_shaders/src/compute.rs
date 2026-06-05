/*!
# Compute Shaders

Modular compute shaders for GPU-driven culling.

Design: Hi-Z Occlusion Test and Compaction are kept as separate passes
for clarity, debuggability, and flexibility.
*/

pub mod hiz {
    /// Hi-Z Occlusion Test Pass
    /// Outputs visibility flags (1 = visible, 0 = occluded)
    pub const HIZ_OCCLUSION_TEST: &str = r#"
        @group(0) @binding(0) var hiz_pyramid: texture_2d_array<f32>;
        @group(0) @binding(1) var<storage, read> pos_x: array<f32>;
        @group(0) @binding(2) var<storage, read> pos_y: array<f32>;
        @group(0) @binding(3) var<storage, read> pos_z: array<f32>;
        @group(0) @binding(4) var<uniform> params: HiZTestParams;
        @group(0) @binding(5) var<storage, read_write> visible_flags: array<u32>;

        struct HiZTestParams {
            view_proj: mat4x4<f32>,
            screen_size: vec2<f32>,
            max_mip_level: u32,
            particle_radius: f32,
            total_particles: u32,
        };

        fn is_occluded(world_pos: vec3<f32>, params: HiZTestParams) -> bool {
            let clip = params.view_proj * vec4<f32>(world_pos, 1.0);
            if (clip.w <= 0.0) { return true; }

            let ndc = clip.xyz / clip.w;
            let screen_uv = ndc.xy * 0.5 + 0.5;
            let screen_radius = params.particle_radius / clip.w;

            let mip = u32(clamp(
                log2(max(params.screen_size.x, params.screen_size.y) * screen_radius),
                0.0, f32(params.max_mip_level)
            ));

            let hiz_depth = textureLoad(hiz_pyramid, vec2<i32>(screen_uv * params.screen_size), i32(mip), 0).r;
            let particle_max_depth = ndc.z + screen_radius;

            return hiz_depth < particle_max_depth;
        }

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let index = id.x;
            if (index >= params.total_particles) { return; }

            let pos = vec3<f32>(pos_x[index], pos_y[index], pos_z[index]);
            visible_flags[index] = select(1u, 0u, is_occluded(pos, params));
        }
    "#;

    /// WaveLocal Reduction Compaction Pass
    /// Reads visibility flags and compacts visible particles.
    pub const COMPACTION: &str = r#"
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
        @group(0) @binding(3) var<uniform> params: CompactionParams;

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
            }

            base_offset = subgroupBroadcast(base_offset, 0u);

            if (visible) {
                visible_indices[base_offset + local_rank] = index;
            }
        }
    "#;
}
