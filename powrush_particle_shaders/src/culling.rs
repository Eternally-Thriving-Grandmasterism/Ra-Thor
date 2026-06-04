/*!
# Culling Module

Unified culling system with clean architecture.
*/

use crate::{ComputeCullingParams, DrawIndirect};

pub struct CullingConfig {
    pub workgroup_size: u32,
}

impl Default for CullingConfig {
    fn default() -> Self {
        Self { workgroup_size: 64 }
    }
}

/// Unified culling pass abstraction.
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

    /// Prepares everything needed for a culling dispatch.
    pub fn prepare_dispatch(&self) -> CullingDispatchInfo {
        CullingDispatchInfo {
            shader: self.shader_source(),
            dispatch_size: self.dispatch_size(),
            indirect: self.create_indirect_buffer(),
        }
    }
}

/// Contains all information needed to perform a culling dispatch.
pub struct CullingDispatchInfo {
    pub shader: &'static str,
    pub dispatch_size: u32,
    pub indirect: DrawIndirect,
}
