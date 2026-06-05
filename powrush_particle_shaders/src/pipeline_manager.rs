/*!
# Compute Pipeline Manager

Centralized manager for Vulkan compute pipelines with:
- Specialization constant support
- Pipeline layout caching
- Automatic pipeline cache persistence (optional)

When a `cache_path` is provided, the manager will automatically save
and load the pipeline cache on creation/destruction.
*/

use std::collections::HashMap;
use std::path::PathBuf;

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
    /// Optional path for automatic cache persistence
    cache_path: Option<PathBuf>,
}

impl ComputePipelineManager {
    /// Creates a new `ComputePipelineManager`.
    ///
    /// - `device`: The Vulkan device.
    /// - `cache_path`: Optional path to a file for automatic cache persistence.
    ///   If the file exists, it will be loaded on creation.
    ///   The cache will be saved automatically on `destroy()`.
    pub fn new(device: ash::Device, cache_path: Option<PathBuf>) -> Self {
        let initial_data = cache_path
            .as_ref()
            .and_then(|path| std::fs::read(path).ok());

        let initial_data_ref = initial_data.as_deref();

        let cache_create_info = vk::PipelineCacheCreateInfo {
            s_type: vk::StructureType::PIPELINE_CACHE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::PipelineCacheCreateFlags::empty(),
            initial_data_size: initial_data_ref.map_or(0, |d| d.len()),
            p_initial_data: initial_data_ref.map_or(std::ptr::null(), |d| d.as_ptr() as *const _),
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
            cache_path,
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
        // ... (specialization constant handling remains the same)
        vk::Pipeline::null()
    }

    pub fn get_pipeline_layout(&mut self, ty: ComputePipelineType) -> vk::PipelineLayout {
        if let Some(&layout) = self.layouts.get(&ty) {
            return layout;
        }

        let layout = vk::PipelineLayout::null();
        self.layouts.insert(ty, layout);
        layout
    }

    /// Saves the current pipeline cache to disk (if a path was provided).
    fn save_cache(&self) {
        if let Some(path) = &self.cache_path {
            if let Ok(data) = unsafe { self.device.get_pipeline_cache_data(self.pipeline_cache) } {
                let _ = std::fs::write(path, data);
            }
        }
    }

    pub fn destroy(&mut self) {
        self.save_cache();

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

impl Drop for ComputePipelineManager {
    fn drop(&mut self) {
        // Ensure cache is saved even if destroy() was not called explicitly
        self.save_cache();
    }
}
