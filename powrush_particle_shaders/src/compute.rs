/*!
# Compute Shaders

WaveLocal Reduction culling + Single-pass Hi-Z generation + Hi-Z Occlusion Test.
*/

pub mod hiz {
    /// Single-pass Hi-Z pyramid generation
    pub const GENERATE_HIZ_SINGLE_PASS: &str = r#" ... "#;

    /// Hi-Z Occlusion Test
    ///
    /// Tests particles against the Hi-Z pyramid to determine occlusion.
    /// Outputs visibility flags or can be combined with compaction.
    pub const HIZ_OCCLUSION_TEST: &str = r#"
        @group(0) @binding(0) var hiz_pyramid: texture_2d_array<f32>;
        @group(0) @binding(1) var<storage, read> pos_x: array<f32>;
        @group(0) @binding(2) var<storage, read> pos_y: array<f32>;
        @group(0) @binding(3) var<storage, read> pos_z: array<f32>;
        @group(0) @binding(4) var<uniform> params: HiZParams;
        @group(0) @binding(5) var<storage, read_write> visible_flags: array<u32>;

        struct HiZParams {
            view_proj: mat4x4<f32>,
            screen_size: vec2<f32>,
            max_mip_level: u32,
            particle_radius: f32,
            total_particles: u32,
        }

        fn get_hiz_mip_level(screen_size: vec2<f32>, particle_size: f32) -> u32 {
            // Simple mip level selection based on screen-space size
            let size = max(screen_size.x, screen_size.y) * particle_size;
            return u32(clamp(log2(size), 0.0, f32(params.max_mip_level)));
        }

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let index = global_id.x;
            if (index >= params.total_particles) { return; }

            let world_pos = vec3<f32>(pos_x[index], pos_y[index], pos_z[index]);

            // Project to clip space
            let clip_pos = params.view_proj * vec4<f32>(world_pos, 1.0);
            let ndc = clip_pos.xyz / clip_pos.w;

            // Convert to screen space [0, 1]
            let screen_pos = ndc.xy * 0.5 + 0.5;

            // Simple conservative bounding sphere size in screen space
            let screen_radius = params.particle_radius / clip_pos.w;

            // Choose mip level
            let mip = get_hiz_mip_level(params.screen_size, screen_radius);

            // Sample Hi-Z at appropriate mip
            let hiz_depth = textureLoad(hiz_pyramid, vec2<i32>(screen_pos * params.screen_size), i32(mip), 0).r;

            // Particle's maximum depth (conservative)
            let particle_max_depth = ndc.z + screen_radius;

            // If Hi-Z depth is closer than particle, it may be occluded
            let occluded = hiz_depth < particle_max_depth;

            visible_flags[index] = select(1u, 0u, occluded);
        }
    "#;
}
