/*!
# Compute Pipeline Manager

A centralized manager for creating, caching, and retrieving Vulkan compute pipelines.

This is designed to work well with the culling and visibility systems in Powrush.
*/

use std::collections::HashMap;

use ash::vk;

use crate::{ComputeCullingParams, culling::CullingPass};

/// Identifies different types of compute pipelines used in the particle system.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ComputePipelineType {
    /// Primary particle culling using WaveLocal Reduction
    CullingPrimary,
    /// Visibility buffer writing and compaction
    VisibilityWrite,
    // Future:
    // Sorting,
    // ImportanceScoring,
}

/// Value for a specialization constant.
#[derive(Clone, Copy, Debug)]
pub enum SpecializationValue {
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
}

/// A single specialization constant entry.
#[derive(Clone, Debug)]
pub struct SpecializationConstant {
    pub constant_id: u32,
    pub value: SpecializationValue,
}

/// Information required to create or retrieve a compute pipeline.
#[derive(Clone, Debug)]
pub struct ComputePipelineCreateInfo {
    pub pipeline_type: ComputePipelineType,
    pub specialization_constants: Vec<SpecializationConstant>,
}

/// Manages compute pipelines with caching and specialization constant support.
pub struct ComputePipelineManager {
    device: ash::Device,
    pipeline_cache: vk::PipelineCache,
    pipelines: HashMap<ComputePipelineType, vk::Pipeline>,
    pipeline_layouts: HashMap<ComputePipelineType, vk::PipelineLayout>,
}

impl ComputePipelineManager {
    pub fn new(device: ash::Device) -> Self {
        // In a real implementation, you would create/load a VkPipelineCache here.
        let pipeline_cache = vk::PipelineCache::null(); // Placeholder

        Self {
            device,
            pipeline_cache,
            pipelines: HashMap::new(),
            pipeline_layouts: HashMap::new(),
        }
    }

    /// Returns an existing pipeline or creates a new one.
    pub fn get_pipeline(
        &mut self,
        create_info: &ComputePipelineCreateInfo,
    ) -> vk::Pipeline {
        if let Some(&pipeline) = self.pipelines.get(&create_info.pipeline_type) {
            return pipeline;
        }

        let pipeline = self.create_pipeline(create_info);
        self.pipelines.insert(create_info.pipeline_type, pipeline);
        pipeline
    }

    fn create_pipeline(&self, create_info: &ComputePipelineCreateInfo) -> vk::Pipeline {
        // TODO: Build VkSpecializationInfo from create_info.specialization_constants
        // TODO: Select shader module and pipeline layout based on pipeline_type
        // TODO: Call vkCreateComputePipelines

        // Placeholder - in real code this would create an actual pipeline
        vk::Pipeline::null()
    }

    /// Returns the pipeline layout for a given pipeline type.
    pub fn get_pipeline_layout(&self, ty: ComputePipelineType) -> vk::PipelineLayout {
        // In a real implementation, layouts would be created and cached here.
        vk::PipelineLayout::null()
    }

    pub fn destroy(&mut self) {
        // Destroy pipelines and pipeline cache
        for (_, pipeline) in self.pipelines.drain() {
            unsafe {
                self.device.destroy_pipeline(pipeline, None);
            }
        }

        if self.pipeline_cache != vk::PipelineCache::null() {
            unsafe {
                self.device.destroy_pipeline_cache(self.pipeline_cache, None);
            }
        }
    }
}
