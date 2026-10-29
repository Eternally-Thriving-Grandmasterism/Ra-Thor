/*!
# Culling Module

Unified compute culling pipeline built around WaveLocal Reduction.

This module provides a clean, maintainable structure for particle culling.
*/

use crate::{ComputeCullingParams, DrawIndirect};

/// High-level culling pipeline configuration.
pub struct CullingConfig {
    pub workgroup_size: u32,
}

impl Default for CullingConfig {
    fn default() -> Self {
        Self { workgroup_size: 64 }
    }
}

/// Represents a culling pass.
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

    /// Returns the shader source for the recommended WaveLocal Reduction culling.
    pub fn shader_source(&self) -> &'static str {
        crate::compute::WAVE_LOCAL_REDUCTION_CULLING
    }

    /// Prepares the dispatch size based on total particles.
    pub fn dispatch_size(&self) -> u32 {
        (self.params.total_particles + self.config.workgroup_size - 1) / self.config.workgroup_size
    }
}
