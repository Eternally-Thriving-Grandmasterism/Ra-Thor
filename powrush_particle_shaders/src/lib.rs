/*!
# Powrush Particle Shaders — Advanced Compute Shader Culling Strategies

Multiple culling strategies for compute shader particle culling.

## Available Strategies

1. **Distance Culling** (simple & fast)
2. **Frustum Culling** (more accurate, still fast)
3. **Importance + Distance** (reputation/harmony aware)
4. **Combined Frustum + Importance** (recommended for production)

This module provides the building blocks and WGSL examples for each.
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct ParticleShaderParams { /* ... */ }

#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct ComputeCullingParams {
    pub camera_position: [f32; 3],
    pub camera_forward: [f32; 3],      // for frustum culling
    pub max_cull_distance: f32,
    pub importance_threshold: f32,
    pub total_particles: u32,
    pub fov_y: f32,                    // for simple frustum approximation
    pub _padding: [f32; 2],
}

pub mod compute {
    /// Advanced culling shader supporting multiple strategies.
    /// Strategy can be selected via specialization constants or uniforms in a real implementation.
    pub const ADVANCED_CULLING_SHADER: &str = r#"
        struct ComputeCullingParams {
            camera_position: vec3<f32>,
            camera_forward: vec3<f32>,
            max_cull_distance: f32,
            importance_threshold: f32,
            total_particles: u32,
            fov_y: f32,
        };

        @group(0) @binding(0) var<uniform> params: ComputeCullingParams;
        @group(0) @binding(1) var<storage, read> particle_positions: array<vec3<f32>>;
        @group(0) @binding(2) var<storage, read_write> visible_indices: array<u32>;
        @group(0) @binding(3) var<storage, read_write> draw_indirect: DrawIndirect;

        fn is_in_frustum(pos: vec3<f32>) -> bool {
            let to_particle = normalize(pos - params.camera_position);
            let dot_product = dot(to_particle, params.camera_forward);
            let angle = acos(dot_product);
            return angle < (params.fov_y * 0.6); // approximate frustum
        }

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let index = global_id.x;
            if (index >= params.total_particles) { return; }

            let pos = particle_positions[index];
            let dist = distance(pos, params.camera_position);

            // Strategy 1: Distance culling
            if (dist > params.max_cull_distance) {
                return;
            }

            // Strategy 2: Frustum culling (approximate)
            if (!is_in_frustum(pos)) {
                return;
            }

            // Strategy 3: Importance culling (can be extended with per-particle reputation)
            let importance = 1.0; // placeholder
            if (importance < params.importance_threshold) {
                return;
            }

            // Passed all active strategies
            let slot = atomicAdd(&draw_indirect.instance_count, 1u);
            visible_indices[slot] = index;
        }
    "#;
}

pub mod wgsl { /* existing shaders */ }
