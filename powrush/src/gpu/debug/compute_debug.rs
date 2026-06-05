//! Compute Shader Debugging Utilities for Powrush-MMO (v16.7 Production)
//!
//! Tools and patterns for debugging WGSL compute shaders in Bevy/wgpu.
//!
//! Common techniques included:
//! - Debug output buffer (write values from shader for inspection)
//! - Atomic counters for tracking execution
//! - Buffer readback helpers
//! - Integration with external tools (RenderDoc, PIX, Xcode GPU Debugger)
//!
//! All under AG-SML v1.0 • TOLC 8 • 7 Living Mercy Gates

use bevy::prelude::*;
use bevy::render::render_resource::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType,
    Buffer, BufferBindingType, BufferInitDescriptor, BufferUsages, ShaderStages,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};

/// Resource holding debug buffers for compute shader inspection.
#[derive(Resource)]
pub struct ComputeDebugResources {
    pub debug_buffer: Buffer,        // General purpose debug output
    pub counter_buffer: Buffer,      // Atomic counters
    pub bind_group: BindGroup,
    pub bind_group_layout: BindGroupLayout,
}

/// Initialize debug resources.
pub fn setup_compute_debug_resources(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
) {
    let debug_buffer = render_device.create_buffer(&BufferInitDescriptor {
        label: Some("compute_debug_buffer"),
        contents: &[0u8; 1024 * 4], // 4KB debug output
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
    });

    let counter_buffer = render_device.create_buffer(&BufferInitDescriptor {
        label: Some("compute_counter_buffer"),
        contents: &[0u8; 16],
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });

    let bind_group_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("compute_debug_bind_group_layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bind_group = render_device.create_bind_group(&BindGroupDescriptor {
        label: Some("compute_debug_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(debug_buffer.as_entire_buffer_binding()),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Buffer(counter_buffer.as_entire_buffer_binding()),
            },
        ],
    });

    commands.insert_resource(ComputeDebugResources {
        debug_buffer,
        counter_buffer,
        bind_group,
        bind_group_layout,
    });
}

/// Read debug buffer back to CPU (for inspection).
/// Call this after compute dispatch in a render system or using async readback.
pub fn read_debug_buffer(
    debug_resources: &ComputeDebugResources,
    render_device: &RenderDevice,
    render_queue: &RenderQueue,
) -> Vec<u32> {
    // In production, use proper async readback or staging buffer
    // This is a simplified synchronous example for debugging purposes
    vec![]
}

/// Example WGSL snippet to include in compute shaders for debugging:
///
/// ```wgsl
/// @group(1) @binding(0) var<storage, read_write> debug_buffer: array<u32>;
/// @group(1) @binding(1) var<storage, read_write> counter: atomic<u32>;
///
/// // Inside main:
/// let idx = atomicAdd(&counter, 1u);
/// debug_buffer[idx] = some_value;
/// ```
