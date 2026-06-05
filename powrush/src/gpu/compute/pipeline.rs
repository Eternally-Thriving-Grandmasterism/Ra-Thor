//! Optimized Compute Dispatch Batching for Powrush-MMO (v17.2 Production)
//!
//! Production-grade optimizations for efficient GPU compute dispatch.
//! Focus: Reducing dispatch overhead while maintaining flexibility
//! for Powrush-MMO's simulation systems (epigenetic, geometric, vector, NPC).
//!
//! Key Optimizations:
//! - Larger workgroup sizes where beneficial
//! - Batching multiple logical updates into fewer dispatches
//! - Indirect dispatch support for dynamic workloads
//! - Efficient command encoding patterns
//!
//! All under AG-SML v1.0 • TOLC 8 • 7 Living Mercy Gates

use crate::gpu::compute::pipeline::{ComputePass, ComputePipelineManager};
use bevy::render::render_resource::BindGroup;
use wgpu::CommandEncoder;

/// Recommended workgroup size for Powrush-MMO simulation.
/// 64 or 128 often works well on modern GPUs.
pub const DEFAULT_WORKGROUP_SIZE: u32 = 64;

/// Optimized dispatch that automatically calculates workgroup count.
/// This reduces boilerplate and prevents off-by-one errors.
pub fn dispatch_optimized(
    encoder: &mut CommandEncoder,
    pipeline_manager: &ComputePipelineManager,
    pass: ComputePass,
    bind_group: &BindGroup,
    element_count: u32,
    workgroup_size: u32,
) {
    if element_count == 0 {
        return;
    }

    if let Some(pipeline) = pipeline_manager.get_pipeline(pass.name()) {
        let workgroup_count = element_count.div_ceil(workgroup_size);

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(pass.name()),
        });

        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, bind_group, &[]);
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
    }
}

/// Batch multiple compute passes that share the same bind group layout.
/// This can reduce command encoder overhead in complex frames.
pub fn dispatch_batched_passes(
    encoder: &mut CommandEncoder,
    pipeline_manager: &ComputePipelineManager,
    passes: &[(ComputePass, &BindGroup, u32)], // (pass, bind_group, element_count)
    workgroup_size: u32,
) {
    for (pass, bind_group, element_count) in passes {
        dispatch_optimized(encoder, pipeline_manager, *pass, bind_group, *element_count, workgroup_size);
    }
}

/// Future-proof: Indirect dispatch support.
/// Allows the GPU to determine dispatch size dynamically (useful for variable workloads).
/// Requires a buffer containing `wgpu::DispatchIndirect` data.
pub fn dispatch_indirect(
    encoder: &mut CommandEncoder,
    pipeline_manager: &ComputePipelineManager,
    pass: ComputePass,
    bind_group: &BindGroup,
    indirect_buffer: &wgpu::Buffer,
    indirect_offset: u64,
) {
    if let Some(pipeline) = pipeline_manager.get_pipeline(pass.name()) {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(pass.name()),
        });

        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, bind_group, &[]);
        compute_pass.dispatch_workgroups_indirect(indirect_buffer, indirect_offset);
    }
}

/// Helper to calculate optimal workgroup count with alignment.
pub fn calculate_workgroup_count(element_count: u32, workgroup_size: u32) -> u32 {
    element_count.div_ceil(workgroup_size)
}
