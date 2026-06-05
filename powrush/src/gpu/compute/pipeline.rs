//! GPU Compute Shader Pipeline Management for Powrush-MMO (v17.1 Production)
//!
//! Professional investigation and implementation of structured compute shader pipelines.
//! This enables clean, scalable, and efficient multi-pass GPU computation
//! for Powrush-MMO simulation systems (epigenetic, geometric, vector, NPC memory, etc.).
//!
//! Key Concepts Covered:
//! - ComputePipeline creation and caching
//! - BindGroupLayout organization
//! - Multi-pass compute dispatch patterns
//! - Resource management across compute passes
//! - Integration with Bevy's render system
//!
//! Designed for use with Ra-Thor AGI and PATSAGi Councils for intelligent NPC behavior.
//!
//! All under AG-SML v1.0 • TOLC 8 • 7 Living Mercy Gates

use bevy::prelude::*;
use bevy::render::render_resource::{
    BindGroup, BindGroupLayout, ComputePipeline, ComputePipelineDescriptor,
    PipelineCache, ShaderModule,
};
use std::collections::HashMap;

/// Manages multiple compute pipelines in a clean, cache-friendly way.
#[derive(Resource)]
pub struct ComputePipelineManager {
    pipelines: HashMap<String, ComputePipeline>,
    bind_group_layouts: HashMap<String, BindGroupLayout>,
}

impl ComputePipelineManager {
    pub fn new() -> Self {
        Self {
            pipelines: HashMap::new(),
            bind_group_layouts: HashMap::new(),
        }
    }

    /// Register a new compute pipeline with a unique name.
    pub fn register_pipeline(
        &mut self,
        name: &str,
        pipeline: ComputePipeline,
        bind_group_layout: BindGroupLayout,
    ) {
        self.pipelines.insert(name.to_string(), pipeline);
        self.bind_group_layouts.insert(name.to_string(), bind_group_layout);
    }

    pub fn get_pipeline(&self, name: &str) -> Option<&ComputePipeline> {
        self.pipelines.get(name)
    }

    pub fn get_bind_group_layout(&self, name: &str) -> Option<&BindGroupLayout> {
        self.bind_group_layouts.get(name)
    }
}

/// Example of creating a compute pipeline from a WGSL shader.
/// This pattern should be used for all Powrush simulation compute shaders.
pub fn create_compute_pipeline(
    device: &wgpu::Device,
    shader: &ShaderModule,
    entry_point: &str,
    bind_group_layout: &BindGroupLayout,
    label: &str,
) -> ComputePipeline {
    device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(
            &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{}_layout", label)),
                bind_group_layouts: &[bind_group_layout],
                push_constant_ranges: &[],
            }),
        ),
        module: shader,
        entry_point,
    })
}

/// Recommended multi-pass compute dispatch order for Powrush-MMO:
///
/// 1. Epigenetic Profile Update (player state)
/// 2. Geometric Harmony / Layer Update (world state)
/// 3. Vector Embedding Update (for similarity search)
/// 4. NPC Memory Formation & Recall (using vector search results)
///
/// This ordering ensures dependencies are respected.
pub enum ComputePass {
    EpigeneticUpdate,
    GeometricUpdate,
    VectorEmbeddingUpdate,
    NpcMemoryUpdate,
}

impl ComputePass {
    pub fn name(&self) -> &'static str {
        match self {
            ComputePass::EpigeneticUpdate => "epigenetic_update",
            ComputePass::GeometricUpdate => "geometric_update",
            ComputePass::VectorEmbeddingUpdate => "vector_embedding_update",
            ComputePass::NpcMemoryUpdate => "npc_memory_update",
        }
    }
}

/// Helper to dispatch a named compute pass.
/// In production, this would be called from a dedicated GPU simulation system.
pub fn dispatch_compute_pass(
    encoder: &mut wgpu::CommandEncoder,
    pipeline_manager: &ComputePipelineManager,
    pass: ComputePass,
    bind_group: &BindGroup,
    workgroup_count: u32,
) {
    if let Some(pipeline) = pipeline_manager.get_pipeline(pass.name()) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(pass.name()),
        });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(workgroup_count, 1, 1);
    }
}
