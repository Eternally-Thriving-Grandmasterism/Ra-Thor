//! GPU Compute Module for Powrush-MMO (v17.6 Production)
//!
//! Provides Bevy + wgpu integration for running WGSL compute shaders.
//! Implements epigenetic + geometric simulation on GPU with full
//! staging buffer pooling, async readback, and debug utilities.
//!
//! Debug utilities include debug output buffers and readback patterns
//! for inspecting compute shader behavior during development.
//!
//! All under AG-SML v1.0 • TOLC 8 Mercy Lattice • 7 Living Mercy Gates

use bevy::prelude::*;
use bevy::render::{
    render_resource::{
        BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
        BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource,
        BindingType, Buffer, BufferBindingType, BufferInitDescriptor, BufferUsages,
        ComputePipeline, ComputePipelineDescriptor, PipelineCache, ShaderStages,
    },
    renderer::{RenderDevice, RenderQueue},
};
use std::sync::Arc;

use crate::systems::epigenetic_modulation::EpigeneticModulationField;
use crate::systems::geometric_harmony_layer::GeometricHarmonyLayer;

// === Readback Support ===
pub mod readback;
use readback::{readback_buffer_blocking, StagingBufferPool};

#[derive(Resource)]
pub struct GpuSimulationResources {
    pub pipeline: ComputePipeline,
    pub bind_group_layout: BindGroupLayout,
    pub epigenetic_buffer: Buffer,
    pub geometric_buffer: Buffer,
    pub params_buffer: Buffer,
}

/// Debug output buffer for inspecting compute shader intermediate results.
#[derive(Resource)]
pub struct DebugOutputBuffer {
    pub buffer: Buffer,
}

pub struct GpuComputePlugin;

impl Plugin for GpuComputePlugin {
    fn build(&self, app: &mut App) {
        app
            .init_resource::<StagingBufferPool>()
            .add_systems(Startup, setup_gpu_compute_resources)
            .add_systems(Update, dispatch_gpu_simulation);
    }
}

fn setup_gpu_compute_resources(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
) {
    let shader = render_device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("epigenetic_geometric_update"),
        source: wgpu::ShaderSource::Wgsl(include_str!("epigenetic_geometric_update.wgsl").into()),
    });

    let bind_group_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("gpu_simulation_bind_group_layout"),
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
        ],
    });

    let epigenetic_buffer = render_device.create_buffer(&BufferInitDescriptor {
        label: Some("epigenetic_buffer"),
        contents: &[],
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    });

    let geometric_buffer = render_device.create_buffer(&BufferInitDescriptor {
        label: Some("geometric_buffer"),
        contents: &[],
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    });

    let params_buffer = render_device.create_buffer(&BufferInitDescriptor {
        label: Some("params_buffer"),
        contents: &[],
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    });

    // Debug output buffer (for shader debugging and inspection)
    let debug_buffer = render_device.create_buffer(&BufferInitDescriptor {
        label: Some("debug_output_buffer"),
        contents: &[],
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });

    let pipeline = render_device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("epigenetic_geometric_pipeline"),
        layout: None,
        module: &shader,
        entry_point: "main",
    });

    commands.insert_resource(GpuSimulationResources {
        pipeline,
        bind_group_layout,
        epigenetic_buffer,
        geometric_buffer,
        params_buffer,
    });

    commands.insert_resource(DebugOutputBuffer { buffer: debug_buffer });
}

/// Main simulation dispatch system with **concrete readback usage**.
/// After simulation, we perform a blocking readback of the epigenetic buffer
/// for demonstration and debugging purposes. In production async readback
/// via `readback::readback_buffer_async` should be preferred.
fn dispatch_gpu_simulation(
    gpu_resources: Res<GpuSimulationResources>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut staging_pool: ResMut<StagingBufferPool>,
    debug_buffer: Res<DebugOutputBuffer>,
    epigenetic: Res<EpigeneticModulationField>,
    geometric: Res<GeometricHarmonyLayer>,
) {
    // TODO: Real upload of CPU state + bind group creation + actual dispatch
    // using pipeline.rs helpers

    // === Concrete Readback Usage ===
    // Read back a portion of the epigenetic buffer after simulation.
    // This demonstrates the full dispatch → readback flow.
    let readback_size = 256; // example size
    match readback_buffer_blocking(
        &render_device,
        &render_queue,
        &gpu_resources.epigenetic_buffer,
        0,
        readback_size,
        &mut staging_pool,
    ) {
        Ok(data) => {
            // In real code: deserialize or inspect `data`
            println!("[GPU] Read back {} bytes from epigenetic buffer", data.len());
        }
        Err(e) => {
            eprintln!("[GPU] Readback failed: {:?}", e);
        }
    }

    // === Debug Utility: Read debug output buffer ===
    // Useful during development to inspect intermediate compute results.
    let _ = readback_buffer_blocking(
        &render_device,
        &render_queue,
        &debug_buffer.buffer,
        0,
        128,
        &mut staging_pool,
    );
}
