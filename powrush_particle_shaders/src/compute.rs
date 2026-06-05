/*!
# Compute Shaders

Clean and well-structured compute shaders for culling and Hi-Z occlusion.
*/

pub mod hiz {
    /// Combined Hi-Z Occlusion Test + WaveLocal Reduction Compaction.
    ///
    /// This shader performs two operations:
    /// 1. Tests particles for occlusion using the Hi-Z pyramid.
    /// 2. Compacts visible particles using efficient wave-local techniques.
    pub const HIZ_OCCLUSION_AND_COMPACTION: &str = r#"
        enable subgroups;

        struct HiZParams {
            view_proj: mat4x4<f32>,
            screen_size: vec2<f32>,
            max_mip_level: u32,
            particle_radius: f32,
            total_particles: u32,
        };

        @group(0) @binding(0) var hiz_pyramid: texture_2d_array<f32>;
        @group(0) @binding(1) var<storage, read> pos_x: array<f32>;
        @group(0) @binding(2) var<storage, read> pos_y: array<f32>;
        @group(0) @binding(3) var<storage, read> pos_z: array<f32>;
        @group(0) @binding(4) var<uniform> params: HiZParams;
        @group(0) @binding(5) var<storage, read_write> visible_indices: array<u32>;
        @group(0) @binding(6) var<storage, read_write> draw_indirect: DrawIndirect;

        // === Hi-Z Occlusion Test ===
        fn is_occluded(world_pos: vec3<f32>) -> bool {
            let clip_pos = params.view_proj * vec4<f32>(world_pos, 1.0);
            if (clip_pos.w <= 0.0) { return true; } // Behind camera

            let ndc = clip_pos.xyz / clip_pos.w;
            let screen_pos = ndc.xy * 0.5 + 0.5;

            // Conservative screen-space radius
            let screen_radius = params.particle_radius / clip_pos.w;

            // Select appropriate mip level
            let mip = u32(clamp(
                log2(max(params.screen_size.x, params.screen_size.y) * screen_radius),
                0.0,
                f32(params.max_mip_level)
            ));

            // Sample Hi-Z pyramid
            let hiz_depth = textureLoad(
                hiz_pyramid,
                vec2<i32>(screen_pos * params.screen_size),
                i32(mip),
                0
            ).r;

            // Particle's maximum depth (conservative)
            let particle_max_depth = ndc.z + screen_radius;

            return hiz_depth < particle_max_depth;
        }

        // === Main Entry ===
        @compute @workgroup_size(64)
        fn main(
            @builtin(global_invocation_id) global_id: vec3<u32>,
            @builtin(subgroup_invocation_id) lane: u32
        ) {
            let index = global_id.x;
            if (index >= params.total_particles) { return; }

            let world_pos = vec3<f32>(
                pos_x[index],
                pos_y[index],
                pos_z[index]
            );

            // Step 1: Hi-Z Occlusion Test
            let occluded = is_occluded(world_pos);
            let visible = !occluded;

            // Step 2: Wave-local compaction (WaveLocal Reduction pattern)
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
