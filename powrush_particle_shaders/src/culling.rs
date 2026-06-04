/*!
# Culling Module

Unified compute culling system centered around WaveLocal Reduction.

Goal: Clean, maintainable, and efficient culling pipeline.
*/

use crate::{ComputeCullingParams, DrawIndirect};

/// Configuration options for culling.
pub struct CullingConfig {
    pub workgroup_size: u32,
}

impl Default for CullingConfig {
    fn default() -> Self {
        Self { workgroup_size: 64 }
    }
}

/// High-level representation of a culling pass.
pub struct CullingPass {
    pub params: ComputeCullingParams,
    pub config: CullingConfig,
}

impl CullingPass {
    pub fn new(params: ComputeCullingParams) -> Self {
        Self {
            params,
            config: CullingConfig::default(),
        }
    }

    pub fn shader_source(&self) -> &'static str {
        crate::compute::WAVE_LOCAL_REDUCTION_CULLING
    }

    pub fn dispatch_size(&self) -> u32 {
        (self.params.total_particles + self.config.workgroup_size - 1)
            / self.config.workgroup_size
    }

    pub fn create_indirect_buffer(&self) -> DrawIndirect {
        DrawIndirect {
            vertex_count: 6,
            instance_count: 0,
            first_vertex: 0,
            first_instance: 0,
        }
    }
}

/// Container for buffers used during culling.
///
/// This helps separate resource management from culling logic.
pub struct CullingResources {
    pub indirect: DrawIndirect,
}

impl CullingResources {
    pub fn new(pass: &CullingPass) -> Self {
        Self {
            indirect: pass.create_indirect_buffer(),
        }
    }
}
