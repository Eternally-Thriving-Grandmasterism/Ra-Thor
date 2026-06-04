/*!
# Powrush Particle Shaders — Indirect Draw Calls

Complete GPU-driven particle rendering pipeline using indirect draw calls.

## Why Indirect Draw Calls?
After compute shader culling writes the visible particle count and indices,
we want to draw **without** reading the count back to the CPU.
Indirect draw calls let the GPU read the instance count directly from a buffer.

This is the final piece for a fully GPU-driven particle system:
Compute Culling → Indirect Draw
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct ParticleShaderParams { /* ... same as before ... */ }

#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct ComputeCullingParams { /* ... same as before ... */ }

/// Standard DrawIndirect structure for non-indexed draws.
/// This is what the GPU reads for indirect drawing.
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct DrawIndirect {
    pub vertex_count: u32,
    pub instance_count: u32,   // <-- Comes from culling atomic counter
    pub first_vertex: u32,
    pub first_instance: u32,
}

/// Prepares a DrawIndirect command for particle instancing.
/// Typically `vertex_count` is the number of vertices in your particle quad/mesh.
/// `instance_count` will be overwritten by the compute shader's visible count.
pub fn prepare_indirect_draw(vertex_count_per_particle: u32) -> DrawIndirect {
    DrawIndirect {
        vertex_count: vertex_count_per_particle,
        instance_count: 0, // Will be set by compute shader
        first_vertex: 0,
        first_instance: 0,
    }
}

pub mod compute {
    pub const CULLING_SHADER: &str = r#"
        // ... (same culling shader as before, but now writes to indirect buffer)

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
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let index = global_id.x;
            if (index >= params.total_particles) { return; }

            let pos = particle_positions[index];
            let dist = distance(pos, params.camera_position);

            if (dist < params.max_cull_distance) {
                let importance = 1.0; // Extend with reputation/harmony here
                if (importance >= params.importance_threshold) {
                    let slot = atomicAdd(&draw_indirect.instance_count, 1u);
                    visible_indices[slot] = index;
                }
            }
        }
    "#;
}

pub mod wgsl { /* ... same shader snippets ... */ }

pub fn get_resonance_effect(params: &ParticleShaderParams) -> &'static str {
    if params.resonance_field_strength > 0.4 {
        wgsl::RESONANCE_TRAIL
    } else {
        wgsl::BURST_RESONANCE
    }
}
