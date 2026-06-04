/*!
# Culling Module

Unified compute culling pipeline built around WaveLocal Reduction.

This module is being unified for clarity, maintainability, and performance.
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

    /// Returns the recommended WaveLocal Reduction shader source.
    pub fn shader_source(&self) -> &'static str {
        crate::compute::WAVE_LOCAL_REDUCTION_CULLING
    }

    /// Calculates dispatch size (number of workgroups).
    pub fn dispatch_size(&self) -> u32 {
        (self.params.total_particles + self.config.workgroup_size - 1)
            / self.config.workgroup_size
    }

    /// Creates a default DrawIndirect buffer for this pass.
    pub fn create_indirect_buffer(&self) -> DrawIndirect {
        DrawIndirect {
            vertex_count: 6, // Placeholder for triangle rendering
            instance_count: 0,
            first_vertex: 0,
            first_instance: 0,
        }
    }
}

/// Helper for preparing culling-related GPU buffers.
///
/// This struct helps centralize buffer-related logic for better separation of concerns.
pub struct CullingBuffers {
    pub indirect: DrawIndirect,
}

impl CullingBuffers {
    pub fn new(pass: &CullingPass) -> Self {
        Self {
            indirect: pass.create_indirect_buffer(),
        }
    }
}
