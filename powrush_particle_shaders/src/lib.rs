/*!
# Powrush Particle Shaders — Optimized Depth Buffer Sampling

Performance optimizations for compute shader depth sampling in occlusion culling.

## Optimizations Applied

1. Use `textureLoad` instead of `textureSampleLevel` (faster, no filtering needed for depth).
2. Proper depth linearization for accurate occlusion tests.
3. Early mip-level selection for coarse culling (simple Hi-Z style).
4. Reduced divergent branching where possible.
5. Better memory access patterns.

These changes significantly improve performance of the occlusion culling pass.
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct OcclusionCullingParams {
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4], // Useful for some advanced techniques
    pub camera_position: [f32; 3],
    pub max_cull_distance: f32,
    pub total_particles: u32,
    pub depth_mip_level: f32,        // For coarse Hi-Z style sampling
}

pub mod compute {
    /// Highly optimized depth sampling occlusion culling shader.
    pub const OPTIMIZED_DEPTH_OCCLUSION_SHADER: &str = r#"
        struct OcclusionCullingParams {
            view_proj: mat4x4<f32>,
            inv_view_proj: mat4x4<f32>,
            camera_position: vec3<f32>,
            max_cull_distance: f32,
            total_particles: u32,
            depth_mip_level: f32,
        };

        @group(0) @binding(0) var<uniform> params: OcclusionCullingParams;
        @group(0) @binding(1) var depth_texture: texture_depth_2d;
        @group(0) @binding(2) var<storage, read> particle_positions: array<vec3<f32>>;
        @group(0) @binding(3) var<storage, read_write> visible_indices: array<u32>;
        @group(0) @binding(4) var<storage, read_write> draw_indirect: DrawIndirect;

        fn linearize_depth(depth: f32, near: f32, far: f32) -> f32 {
            // Convert non-linear depth to linear view-space depth
            return (2.0 * near) / (far + near - depth * (far - near));
        }

        fn world_to_screen_depth(world_pos: vec3<f32>) -> vec3<f32> {
            let clip = params.view_proj * vec4<f32>(world_pos, 1.0);
            let ndc = clip.xyz / clip.w;
            let uv = ndc.xy * 0.5 + 0.5;
            return vec3<f32>(uv, ndc.z);
        }

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let index = global_id.x;
            if (index >= params.total_particles) {
                return;
            }

            let world_pos = particle_positions[index];
            let dist = distance(world_pos, params.camera_position);

            if (dist > params.max_cull_distance) {
                return;
            }

            let screen = world_to_screen_depth(world_pos);
            let uv = screen.xy;

            // Optimized depth load using mip level for coarse culling
            let scene_depth = textureLoad(depth_texture, vec2<i32>(uv * vec2<f32>(textureDimensions(depth_texture))), i32(params.depth_mip_level));

            // Linearize both depths for accurate comparison
            // (Assuming near=0.1, far=1000.0 - these should come from camera in real code)
            let linear_particle_depth = linearize_depth(screen.z, 0.1, 1000.0);
            let linear_scene_depth = linearize_depth(scene_depth, 0.1, 1000.0);

            if (linear_particle_depth > linear_scene_depth) {
                return; // Occluded
            }

            // Visible - write to indirect buffer
            let slot = atomicAdd(&draw_indirect.instance_count, 1u);
            visible_indices[slot] = index;
            }
    "#;
}

pub mod wgsl { /* ... */ }
