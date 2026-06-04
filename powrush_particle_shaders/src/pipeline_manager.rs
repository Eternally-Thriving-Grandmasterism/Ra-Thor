/*!
# Compute Pipeline Manager

A centralized, cache-friendly manager for Vulkan compute pipelines.

Features:
- Specialization constant support
- Pipeline layout caching per type
- Designed for easy integration with the culling and visibility systems

This is still a work in progress. Real `vkCreateComputePipelines` calls will be added
when the full Vulkan device + shader module context is available.
*/

use std::collections::HashMap;

use ash::vk;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ComputePipelineType {
    CullingPrimary,
    VisibilityWrite,
}

#[derive(Clone, Copy, Debug)]
pub enum SpecializationValue {
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
}

#[derive(Clone, Debug)]
pub struct SpecializationConstant {
    pub constant_id: u32,
    pub value: SpecializationValue,
}

#[derive(Clone, Debug)]
pub struct ComputePipelineCreateInfo {
    pub pipeline_type: ComputePipelineType,
    pub specialization_constants: Vec<SpecializationConstant>,
}

pub struct ComputePipelineManager {
    device: ash::Device,
    pipeline_cache: vk::PipelineCache,
    pipelines: HashMap<ComputePipelineType, vk::Pipeline>,
    layouts: HashMap<ComputePipelineType, vk::PipelineLayout>,
}

impl ComputePipelineManager {
    pub fn new(device: ash::Device) -> Self {
        // TODO: Create or load a persistent VkPipelineCache from disk
        let pipeline_cache = vk::PipelineCache::null();

        Self {
            device,
            pipeline_cache,
            pipelines: HashMap::new(),
            layouts: HashMap::new(),
        }
    }

    /// Returns a pipeline for the given type, creating it if necessary.
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
        // === Specialization Constants ===
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

        let _spec_info = if !spec_entries.is_empty() {
            Some(vk::SpecializationInfo {
                map_entry_count: spec_entries.len() as u32,
                p_map_entries: spec_entries.as_ptr(),
                data_size: spec_data.len(),
                p_data: spec_data.as_ptr() as *const _,
            })
        } else {
            None
        };

        // === TODOs for full implementation ===
        // 1. Select shader module based on create_info.pipeline_type
        // 2. Get or create VkPipelineLayout using get_pipeline_layout()
        // 3. Build VkComputePipelineCreateInfo
        // 4. Call vkCreateComputePipelines(self.device, self.pipeline_cache, ...)
        // 5. Handle errors properly

        vk::Pipeline::null()
    }

    /// Returns (or creates) a pipeline layout for the given pipeline type.
    pub fn get_pipeline_layout(&mut self, ty: ComputePipelineType) -> vk::PipelineLayout {
        if let Some(&layout) = self.layouts.get(&ty) {
            return layout;
        }

        // TODO: Create a real VkPipelineLayout with appropriate descriptor set layouts
        // and push constant ranges based on the needs of this pipeline type.
        let layout = vk::PipelineLayout::null();
        self.layouts.insert(ty, layout);
        layout
    }

    pub fn destroy(&mut self) {
        for (_, pipeline) in self.pipelines.drain() {
            unsafe { self.device.destroy_pipeline(pipeline, None); }
        }

        for (_, layout) in self.layouts.drain() {
            unsafe { self.device.destroy_pipeline_layout(layout, None); }
        }

        if self.pipeline_cache != vk::PipelineCache::null() {
            unsafe { self.device.destroy_pipeline_cache(self.pipeline_cache, None); }
        }
    }
}
