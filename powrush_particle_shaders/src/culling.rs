/*!
# Culling Module

Unified and professional culling pipeline architecture.

Focus: Clean separation of configuration, resources, and dispatch preparation.
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

/// High-level abstraction for a WaveLocal Reduction culling pass.
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
        crate::compute::culling::WAVE_LOCAL_REDUCTION_CULLING
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

    /// Prepares a complete dispatch package with all necessary information.
    pub fn prepare_dispatch(&self) -> CullingDispatchPreparation {
        CullingDispatchPreparation {
            shader_source: self.shader_source(),
            dispatch_size: self.dispatch_size(),
            indirect_buffer: self.create_indirect_buffer(),
        }
    }
}

/// Contains everything needed to record a culling dispatch.
///
/// This struct is designed to be extended with actual buffer handles
/// when integrating with a real Vulkan backend.
pub struct CullingDispatchPreparation {
    pub shader_source: &'static str,
    pub dispatch_size: u32,
    pub indirect_buffer: DrawIndirect,
}

/// Container for culling-related GPU resources.
///
/// In a full implementation, this would hold actual buffer handles.
pub struct CullingResources {
    pub indirect: DrawIndirect,
}

impl CullingResources {
    pub fn from_pass(pass: &CullingPass) -> Self {
        Self {
            indirect: pass.create_indirect_buffer(),
        }
    }
}
