/*!
# Powrush Particle Shaders — GPU-Driven Scene Traversal

Investigation into GPU-driven scene traversal for particle and effect rendering.

## What is GPU-Driven Scene Traversal?

Traditional rendering has the CPU traverse the scene graph, perform culling, and issue draw calls.

**GPU-driven scene traversal** moves this logic to the GPU:
- The GPU receives a description of the scene (or list of effects).
- Compute shaders traverse/cull the list.
- The GPU generates the final draw commands (indirect or direct).
- Minimal CPU involvement per frame.

This is a major step toward low-CPU-overhead, highly parallel rendering.

## Application to Powrush Particles

Instead of the CPU iterating over all active particle systems every frame, we can:
1. Upload an array of `ParticleSystemDescriptor`s to the GPU.
2. Dispatch a compute shader that traverses this array.
3. For each system: perform culling (frustum, occlusion, importance).
4. Generate `DrawIndirect` commands for visible systems.
5. Execute with `multi_draw_indirect`.

This scales extremely well as the number of effects grows.
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

/// Descriptor for a particle system / visual effect.
/// This can be uploaded to the GPU as part of a "scene" buffer.
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct ParticleSystemDescriptor {
    pub position: [f32; 3],
    pub bounding_radius: f32,
    pub particle_count: u32,
    pub importance: f32,           // e.g. based on reputation/harmony
    pub faction: u32,              // index into Faction enum
    pub shader_params_index: u32,  // index into ParticleShaderParams buffer
    pub _padding: [u32; 2],
}

/// Parameters for a GPU-driven traversal compute pass.
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct GPUTraversalParams {
    pub view_proj: [[f32; 4]; 4],
    pub camera_position: [f32; 3],
    pub max_cull_distance: f32,
    pub total_systems: u32,
}

pub mod compute {
    /// Compute shader that traverses a list of particle systems on the GPU,
    /// performs culling, and generates DrawIndirect commands.
    pub const GPU_SCENE_TRAVERSAL_SHADER: &str = r#"
        struct ParticleSystemDescriptor {
            position: vec3<f32>,
            bounding_radius: f32,
            particle_count: u32,
            importance: f32,
            faction: u32,
            shader_params_index: u32,
        };

        struct GPUTraversalParams {
            view_proj: mat4x4<f32>,
            camera_position: vec3<f32>,
            max_cull_distance: f32,
            total_systems: u32,
        };

        struct DrawIndirect {
            vertex_count: u32,
            instance_count: u32,
            first_vertex: u32,
            first_instance: u32,
        };

        @group(0) @binding(0) var<uniform> params: GPUTraversalParams;
        @group(0) @binding(1) var<storage, read> systems: array<ParticleSystemDescriptor>;
        @group(0) @binding(2) var<storage, read_write> indirect_commands: array<DrawIndirect>;
        @group(0) @binding(3) var<storage, read_write> visible_system_count: atomic<u32>;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let index = global_id.x;
            if (index >= params.total_systems) { return; }

            let system = systems[index];
            let dist = distance(system.position, params.camera_position);

            // Basic culling (can be extended with frustum, Hi-Z, importance, etc.)
            if (dist > params.max_cull_distance) {
                return;
            }

            // Passed culling - generate indirect command
            let cmd_index = atomicAdd(&visible_system_count, 1u);
            indirect_commands[cmd_index].vertex_count = 4; // assume quad particles
            indirect_commands[cmd_index].instance_count = system.particle_count;
            indirect_commands[cmd_index].first_vertex = 0;
            indirect_commands[cmd_index].first_instance = 0;
        }
    "#;
}

pub mod wgsl { /* existing shaders */ }
