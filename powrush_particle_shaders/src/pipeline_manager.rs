/*!
# Compute Pipeline Manager

Centralized management of Vulkan compute pipelines with support for
specialization constants and caching.

Designed for the Powrush particle system (culling, visibility, etc.).
*/

use std::collections::HashMap;

use ash::vk;

/// Types of compute pipelines used in the system.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ComputePipelineType {
    CullingPrimary,
    VisibilityWrite,
}

/// Supported specialization constant value types.
#[derive(Clone, Copy, Debug)]
pub enum SpecializationValue {
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
}

/// Represents one specialization constant.
#[derive(Clone, Debug)]
pub struct SpecializationConstant {
    pub constant_id: u32,
    pub value: SpecializationValue,
}

/// Input for requesting a compute pipeline.
#[derive(Clone, Debug)]
pub struct ComputePipelineCreateInfo {
    pub pipeline_type: ComputePipelineType,
    pub specialization_constants: Vec<SpecializationConstant>,
}

/// Manages creation, caching and lifetime of compute pipelines.
pub struct ComputePipelineManager {
    device: ash::Device,
    pipeline_cache: vk::PipelineCache,
    pipelines: HashMap<ComputePipelineType, vk::Pipeline>,
    layouts: HashMap<ComputePipelineType, vk::PipelineLayout>,
}

impl ComputePipelineManager {
    pub fn new(device: ash::Device) -> Self {
        // TODO: Create or load a real VkPipelineCache
        let pipeline_cache = vk::PipelineCache::null();

        Self {
            device,
            pipeline_cache,
            pipelines: HashMap::new(),
            layouts: HashMap::new(),
        }
    }

    /// Get or create a pipeline for the requested type + specialization constants.
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
        // Build specialization info
        let spec_entries: Vec<vk::SpecializationMapEntry> = create_info
            .specialization_constants
            .iter()
            .enumerate()
            .map(|(i, c)| vk::SpecializationMapEntry {
                constant_id: c.constant_id,
                offset: (i * std::mem::size_of::<u32>()) as u32,
                size: std::mem::size_of::<u32>() as u64,
            })
            .collect();

        let spec_data: Vec<u8> = create_info
            .specialization_constants
            .iter()
            .flat_map(|c| match c.value {
                SpecializationValue::U32(v) => v.to_ne_bytes().to_vec(),
                SpecializationValue::I32(v) => v.to_ne_bytes().to_vec(),
                SpecializationValue::F32(v) => v.to_ne_bytes().to_vec(),
                SpecializationValue::Bool(v) => {
                    if v { 1u32.to_ne_bytes().to_vec() } else { 0u32.to_ne_bytes().to_vec() }
                }
            })
            .collect();

        let spec_info = if !spec_entries.is_empty() {
            Some(vk::SpecializationInfo {
                map_entry_count: spec_entries.len() as u32,
                p_map_entries: spec_entries.as_ptr(),
                data_size: spec_data.len(),
                p_data: spec_data.as_ptr() as *const _,
            })
        } else {
            None
        };

        // TODO: Select shader module and pipeline layout based on pipeline_type
        // TODO: Actually call vkCreateComputePipelines

        vk::Pipeline::null() // Placeholder
    }

    pub fn get_pipeline_layout(&self, ty: ComputePipelineType) -> vk::PipelineLayout {
        // TODO: Return or create appropriate layout
        vk::PipelineLayout::null()
    }

    pub fn destroy(&mut self) {
        for (_, pipeline) in self.pipelines.drain() {
            unsafe { self.device.destroy_pipeline(pipeline, None); }
        }

        if self.pipeline_cache != vk::PipelineCache::null() {
            unsafe { self.device.destroy_pipeline_cache(self.pipeline_cache, None); }
        }
    }
}
