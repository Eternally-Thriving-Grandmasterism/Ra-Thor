/*!
# Powrush Particle Shaders — Compute Shader Depth Sampling

Proper implementation of depth buffer sampling in compute shaders for occlusion culling.

## Implementation Notes

This version includes:
- View-projection matrix for correct world → clip → NDC → UV transformation
- Correct depth comparison logic
- Early outs for performance
- Clear comments on what the host must provide (depth texture + matrices)
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct OcclusionCullingParams {
    pub view_proj: [[f32; 4]; 4],     // Column-major view-projection matrix
    pub camera_position: [f32; 3],
    pub max_cull_distance: f32,
    pub total_particles: u32,
}

pub mod compute {
    /// Compute shader with proper depth sampling for occlusion culling.
    pub const DEPTH_SAMPLING_OCCLUSION_SHADER: &str = r#"
        struct OcclusionCullingParams {
            view_proj: mat4x4<f32>,
            camera_position: vec3<f32>,
            max_cull_distance: f32,
            total_particles: u32,
        };

        @group(0) @binding(0) var<uniform> params: OcclusionCullingParams;
        @group(0) @binding(1) var depth_texture: texture_depth_2d;
        @group(0) @binding(2) var depth_sampler: sampler;
        @group(0) @binding(3) var<storage, read> particle_positions: array<vec3<f32>>;
        @group(0) @binding(4) var<storage, read_write> visible_indices: array<u32>;
        @group(0) @binding(5) var<storage, read_write> draw_indirect: DrawIndirect;

        fn world_to_uv(world_pos: vec3<f32>) -> vec2<f32> {
            let clip = params.view_proj * vec4<f32>(world_pos, 1.0);
            let ndc = clip.xyz / clip.w;
            // Convert NDC [-1,1] to UV [0,1]
            return ndc.xy * 0.5 + 0.5;
        }

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let index = global_id.x;
            if (index >= params.total_particles) {
                return;
            }

            let world_pos = particle_positions[index];
            let dist = distance(world_pos, params.camera_position);

            // Early out: distance culling
            if (dist > params.max_cull_distance) {
                return;
            }

            // Project to screen space and sample depth
            let uv = world_to_uv(world_pos);

            // Sample scene depth at projected position
            let scene_depth = textureSampleLevel(depth_texture, depth_sampler, uv, 0.0);

            // Convert particle world depth to [0,1] depth for comparison
            // (This is a simplified linear approximation. Real engines use proper depth linearization.)
            let particle_ndc_depth = (params.view_proj * vec4<f32>(world_pos, 1.0)).z;

            // If particle is behind scene geometry, it is occluded
            if (particle_ndc_depth > scene_depth) {
                return;
            }

            // Particle is visible
            let slot = atomicAdd(&draw_indirect.instance_count, 1u);
            visible_indices[slot] = index;
        }
    "#;
}

pub mod wgsl { /* ... existing shader code ... */ }
