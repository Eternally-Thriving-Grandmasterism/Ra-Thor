/*!
# Powrush Particle Shaders

Core shaders and utilities for the particle system.

## Data Layout

We are moving toward Structure of Arrays (SoA) for better GPU memory coalescing.
*/

pub mod compute;
pub mod culling;
pub mod pipeline_manager;

pub use culling::{CullingConfig, CullingPass, CullingResources};
pub use pipeline_manager::{
    ComputePipelineManager, ComputePipelineType, SpecializationConstant, SpecializationValue,
};

/// Parameters for compute culling passes.
///
/// Note: max_cull_distance_squared should be the squared value of the desired distance.
pub struct ComputeCullingParams {
    pub view_proj: [[f32; 4]; 4],
    pub camera_position: [f32; 3],
    pub max_cull_distance_squared: f32,
    pub total_particles: u32,
}

/// DrawIndirect structure used by culling output.
pub struct DrawIndirect {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}
