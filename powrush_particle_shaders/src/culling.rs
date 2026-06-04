/*!
# Culling Module

Unified compute culling pipeline built around WaveLocal Reduction.

This module provides a clean structure for particle culling.
*/

use crate::{ComputeCullingParams, DrawIndirect};

/// Configuration for a culling pass.
pub struct CullingConfig {
    pub workgroup_size: u32,
}

impl Default for CullingConfig {
    fn default() -> Self {
        Self { workgroup_size: 64 }
    }
}

/// Represents a culling pass using WaveLocal Reduction.
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

    /// Returns the shader source for WaveLocal Reduction culling.
    pub fn shader_source(&self) -> &'static str {
        crate::compute::WAVE_LOCAL_REDUCTION_CULLING
    }

    /// Calculates the number of workgroups needed.
    pub fn dispatch_size(&self) -> u32 {
        (self.params.total_particles + self.config.workgroup_size - 1)
            / self.config.workgroup_size
    }

    /// Prepares a DrawIndirect buffer with default values.
    pub fn prepare_indirect_buffer(&self) -> DrawIndirect {
        DrawIndirect {
            vertex_count: 6, // Assuming triangle list for now
            instance_count: 0,
            first_vertex: 0,
            first_instance: 0,
        }
    }
}
