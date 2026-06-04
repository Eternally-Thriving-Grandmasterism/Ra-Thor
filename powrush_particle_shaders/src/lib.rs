/*!
# Powrush Particle Shaders

Core GPU compute shaders for high-performance particle culling and visibility.

## Architecture

The culling system is unified around **WaveLocal Reduction** as the primary technique.
See `culling` module for the unified dispatch structure.
*/

mod culling;

pub use culling::{CullingConfig, CullingPass};

pub mod compute;

/// Parameters for compute culling passes.
pub struct ComputeCullingParams {
    pub view_proj: [[f32; 4]; 4],
    pub camera_position: [f32; 3],
    pub max_cull_distance: f32,
    pub total_particles: u32,
}

/// DrawIndirect structure used by culling output.
pub struct DrawIndirect {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}
