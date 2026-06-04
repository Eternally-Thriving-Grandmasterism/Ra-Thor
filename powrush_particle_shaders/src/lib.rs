/*!
# Powrush Particle Shaders — Hierarchical Z-Buffer (Hi-Z) Techniques

Exploration and implementation of Hierarchical Z-Buffer occlusion culling.

## What is Hierarchical Z-Buffer (Hi-Z)?

Hi-Z is an advanced occlusion culling technique that uses a mipmapped depth pyramid.
Each mip level stores the **minimum** depth value over a 2x2 (or larger) tile.

When testing a particle or bounding volume:
1. Project it to screen space.
2. Estimate its screen size.
3. Select the appropriate mip level (coarser for larger/farther objects).
4. Sample the Hi-Z texture at that level.

This allows very fast occlusion tests with excellent cache behavior and fewer samples.

## Benefits for Particle Systems
- Extremely fast culling of many small particles.
- Naturally handles varying particle screen sizes.
- Can be combined with frustum + importance culling.
- Scales well to very large particle counts.
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct HiZCullingParams {
    pub view_proj: [[f32; 4]; 4],
    pub camera_position: [f32; 3],
    pub max_cull_distance: f32,
    pub total_particles: u32,
    pub hi_z_texture_size: [u32; 2],  // Size of mip 0
    pub max_mip_level: u32,
}

pub mod compute {
    /// Hierarchical Z-Buffer occlusion culling shader.
    /// This version calculates the appropriate mip level based on particle distance.
    pub const HIZ_OCCLUSION_SHADER: &str = r#"
        struct HiZCullingParams {
            view_proj: mat4x4<f32>,
            camera_position: vec3<f32>,
            max_cull_distance: f32,
            total_particles: u32,
            hi_z_texture_size: vec2<u32>,
            max_mip_level: u32,
        };

        @group(0) @binding(0) var<uniform> params: HiZCullingParams;
        @group(0) @binding(1) var hi_z_texture: texture_depth_2d;
        @group(0) @binding(2) var hi_z_sampler: sampler;
        @group(0) @binding(3) var<storage, read> particle_positions: array<vec3<f32>>;
        @group(0) @binding(4) var<storage, read_write> visible_indices: array<u32>;
        @group(0) @binding(5) var<storage, read_write> draw_indirect: DrawIndirect;

        fn calculate_mip_level(world_pos: vec3<f32>) -> f32 {
            // Simple mip level selection based on distance
            // More advanced versions use screen-space size of the particle
            let dist = distance(world_pos, params.camera_position);
            let normalized_dist = dist / params.max_cull_distance;
            // Coarser mip for farther particles
            let mip = normalized_dist * f32(params.max_mip_level);
            return clamp(mip, 0.0, f32(params.max_mip_level));
        }

        fn world_to_uv(world_pos: vec3<f32>) -> vec2<f32> {
            let clip = params.view_proj * vec4<f32>(world_pos, 1.0);
            let ndc = clip.xyz / clip.w;
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

            if (dist > params.max_cull_distance) {
                return;
            }

            let uv = world_to_uv(world_pos);
            let mip = calculate_mip_level(world_pos);

            // Sample from the appropriate mip level of the Hi-Z texture
            let scene_min_depth = textureSampleLevel(hi_z_texture, hi_z_sampler, uv, mip);

            // Simplified depth test (should use proper linearization in production)
            let particle_depth = dist / params.max_cull_distance;

            if (particle_depth > scene_min_depth) {
                return; // Occluded by Hi-Z
            }

            // Visible
            let slot = atomicAdd(&draw_indirect.instance_count, 1u);
            visible_indices[slot] = index;
        }
    "#;
}

pub mod wgsl { /* ... */ }
