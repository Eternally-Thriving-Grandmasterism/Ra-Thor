/*!
# Compute Pipeline Manager

Centralized manager for Vulkan compute pipelines with support for:
- Specialization constants
- Pipeline layout caching
- Pipeline cache persistence (save/load)

Designed for the Powrush particle system.
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
    /// Creates a new pipeline manager.
    ///
    /// If `initial_cache_data` is provided, it will be used to initialize the
    /// `VkPipelineCache` (enables persistence across runs).
    pub fn new(device: ash::Device, initial_cache_data: Option<&[u8]>) -> Self {
        let cache_create_info = vk::PipelineCacheCreateInfo {
            s_type: vk::StructureType::PIPELINE_CACHE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::PipelineCacheCreateFlags::empty(),
            initial_data_size: initial_cache_data.map_or(0, |d| d.len()),
            p_initial_data: initial_cache_data.map_or(std::ptr::null(), |d| d.as_ptr() as *const _),
        };

        let pipeline_cache = unsafe {
            device
                .create_pipeline_cache(&cache_create_info, None)
                .expect("Failed to create pipeline cache")
        };

        Self {
            device,
            pipeline_cache,
            pipelines: HashMap::new(),
            layouts: HashMap::new(),
        }
    }

    /// Returns pipeline cache data that can be saved to disk for persistence.
    pub fn get_cache_data(&self) -> Vec<u8> {
        unsafe {
            self.device
                .get_pipeline_cache_data(self.pipeline_cache)
                .expect("Failed to get pipeline cache data")
        }
    }

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
        // Build specialization info (same as before)
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

        // TODO: Select shader module and pipeline layout
        // TODO: Call vkCreateComputePipelines with self.pipeline_cache

        vk::Pipeline::null()
    }

    pub fn get_pipeline_layout(&mut self, ty: ComputePipelineType) -> vk::PipelineLayout {
        if let Some(&layout) = self.layouts.get(&ty) {
            return layout;
        }

        let layout = vk::PipelineLayout::null(); // TODO: Create real layout
        self.layouts.insert(ty, layout);
        layout
    }

    /// Destroys all pipelines and the pipeline cache.
    ///
    /// Call `get_cache_data()` before destroy if you want to persist it.
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
