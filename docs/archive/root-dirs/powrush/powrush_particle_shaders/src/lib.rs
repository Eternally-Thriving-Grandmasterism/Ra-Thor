/*!
# Powrush Particle Shaders

Core shaders and utilities for the particle system.
*/

pub mod compute;
pub mod culling;
pub mod pipeline_manager;

pub use culling::{CullingConfig, CullingPass, CullingResources};
pub use pipeline_manager::{
    ComputePipelineManager,
    ComputePipelineType,
    PipelineError,
    SpecializationConstant,
    SpecializationValue,
};

pub struct ComputeCullingParams {
    pub view_proj: [[f32; 4]; 4],
    pub camera_position: [f32; 3],
    pub max_cull_distance_squared: f32,
    pub total_particles: u32,
}

pub struct DrawIndirect {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}
