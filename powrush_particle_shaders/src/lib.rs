/*!
# Powrush Particle Shaders

Core shaders and utilities for the particle system.

## Architecture

The culling system is unified around **WaveLocal Reduction** as the primary technique.
See `culling` module for the unified dispatch structure.
*/

mod culling;

pub use culling::{CullingConfig, CullingPass};

pub mod compute;

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
