/*!
# Compute Shaders

High-quality, modular GPU-driven culling shaders.
*/

pub mod hiz {
    /// Combined Distance + Hi-Z Occlusion Test
    ///
    /// Performs both distance culling and Hi-Z occlusion testing in one pass.
    /// Outputs visibility flags for the Compaction pass.
    pub const DISTANCE_AND_HIZ_TEST: &str = r#"
        @group(0) @binding(0) var hiz_pyramid: texture_2d_array<f32>;
        @group(0) @binding(1) var<storage, read> pos_x: array<f32>;
        @group(0) @binding(2) var<storage, read> pos_y: array<f32>;
        @group(0) @binding(3) var<storage, read> pos_z: array<f32>;
        @group(0) @binding(4) var<uniform> params: CullingParams;
        @group(0) @binding(5) var<storage, read_write> visible_flags: array<u32>;

        struct CullingParams {
            view_proj: mat4x4<f32>,
            camera_position: vec3<f32>,
            max_cull_distance_squared: f32,
            screen_size: vec2<f32>,
            max_mip_level: u32,
            particle_radius: f32,
            total_particles: u32,
        };

        // Returns true if the particle should be culled (distance or occlusion)
        fn should_cull(world_pos: vec3<f32>, params: CullingParams) -> bool {
            // === Distance Culling (cheap early-out) ===
            let dist_sq = dot(world_pos - params.camera_position,
                              world_pos - params.camera_position);
            if (dist_sq > params.max_cull_distance_squared) {
                return true;
            }

            // === Hi-Z Occlusion Test ===
            let clip = params.view_proj * vec4<f32>(world_pos, 1.0);
            if (clip.w <= 0.0) { return true; }

            let ndc = clip.xyz / clip.w;
            let screen_uv = ndc.xy * 0.5 + 0.5;
            let screen_radius = params.particle_radius / clip.w;

            let mip = u32(clamp(
                log2(max(params.screen_size.x, params.screen_size.y) * screen_radius),
                0.0,
                f32(params.max_mip_level)
            ));

            let hiz_depth = textureLoad(
                hiz_pyramid,
                vec2<i32>(screen_uv * params.screen_size),
                i32(mip),
                0
            ).r;

            let particle_max_depth = ndc.z + screen_radius;

            return hiz_depth < particle_max_depth;
        }

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let index = id.x;
            if (index >= params.total_particles) { return; }

            let world_pos = vec3<f32>(
                pos_x[index],
                pos_y[index],
                pos_z[index]
            );

            let culled = should_cull(world_pos, params);
            visible_flags[index] = select(1u, 0u, culled);
        }
    "#;

    /// WaveLocal Reduction Compaction (unchanged, consumes visibility flags)
    pub const COMPACTION: &str = r#" ... "#;
}
