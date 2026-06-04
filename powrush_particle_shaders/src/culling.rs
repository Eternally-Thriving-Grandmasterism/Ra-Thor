/*!
# Culling Module

Professional unification of the compute culling pipeline.

Emphasis on clean separation and clear resource-to-dispatch association.
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

    /// Prepares a complete dispatch package, including resource associations.
    pub fn prepare_dispatch(&self) -> CullingDispatchPreparation {
        CullingDispatchPreparation {
            shader_source: self.shader_source(),
            dispatch_size: self.dispatch_size(),
            indirect_buffer: self.create_indirect_buffer(),
            // Resource associations will be expanded here in future iterations
        }
    }
}

/// Contains all information needed to record a culling dispatch.
///
/// Designed to be extended with actual buffer handles and resource bindings.
pub struct CullingDispatchPreparation {
    pub shader_source: &'static str,
    pub dispatch_size: u32,
    pub indirect_buffer: DrawIndirect,
}

/// Represents the set of resources required for a culling pass.
///
/// In a real implementation, this would contain actual GPU buffer handles
/// for positions (SoA), visible indices, and the indirect buffer.
pub struct CullingResources {
    pub indirect: DrawIndirect,
    // Future: pos_x, pos_y, pos_z, visible_indices buffers
}

impl CullingResources {
    pub fn from_pass(pass: &CullingPass) -> Self {
        Self {
            indirect: pass.create_indirect_buffer(),
        }
    }
}
