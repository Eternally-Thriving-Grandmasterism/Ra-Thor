/*!
# Compute Pipeline Manager

A robust Vulkan compute pipeline manager with:
- Real pipeline creation
- Specialization constants
- Automatic cache persistence
- Shader module loading from SPIR-V bytecode
- Proper error handling
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

#[derive(Debug)]
pub enum PipelineError {
    ShaderModuleNotFound(ComputePipelineType),
    PipelineCreationFailed(vk::Result),
    PipelineLayoutNotFound(ComputePipelineType),
    ShaderModuleCreationFailed(vk::Result),
}

pub struct ComputePipelineManager {
    device: ash::Device,
    pipeline_cache: vk::PipelineCache,
    pipelines: HashMap<ComputePipelineType, vk::Pipeline>,
    layouts: HashMap<ComputePipelineType, vk::PipelineLayout>,
    shader_modules: HashMap<ComputePipelineType, vk::ShaderModule>,
    cache_path: Option<PathBuf>,
}

impl ComputePipelineManager {
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
            shader_modules: HashMap::new(),
            cache_path,
        }
    }

    /// Load a shader module from SPIR-V bytecode and register it for a pipeline type.
    pub fn load_shader_module(
        &mut self,
        pipeline_type: ComputePipelineType,
        spirv_code: &[u8],
    ) -> Result<(), PipelineError> {
        let shader_module_create_info = vk::ShaderModuleCreateInfo {
            s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::ShaderModuleCreateFlags::empty(),
            code_size: spirv_code.len(),
            p_code: spirv_code.as_ptr() as *const u32,
        };

        let shader_module = unsafe {
            self.device
                .create_shader_module(&shader_module_create_info, None)
                .map_err(PipelineError::ShaderModuleCreationFailed)?
        };

        self.shader_modules.insert(pipeline_type, shader_module);
        Ok(())
    }

    /// Load a shader module from a file path.
    pub fn load_shader_module_from_file(
        &mut self,
        pipeline_type: ComputePipelineType,
        path: &std::path::Path,
    ) -> Result<(), PipelineError> {
        let spirv_code = std::fs::read(path)
            .map_err(|_| PipelineError::ShaderModuleCreationFailed(vk::Result::ERROR_UNKNOWN))?; // Simplified error

        self.load_shader_module(pipeline_type, &spirv_code)
    }

    pub fn register_pipeline_layout(
        &mut self,
        pipeline_type: ComputePipelineType,
        layout: vk::PipelineLayout,
    ) {
        self.layouts.insert(pipeline_type, layout);
    }

    pub fn get_pipeline(
        &mut self,
        create_info: &ComputePipelineCreateInfo,
    ) -> Result<vk::Pipeline, PipelineError> {
        if let Some(&pipeline) = self.pipelines.get(&create_info.pipeline_type) {
            return Ok(pipeline);
        }

        let pipeline = self.create_pipeline(create_info)?;
        self.pipelines.insert(create_info.pipeline_type, pipeline);
        Ok(pipeline)
    }

    fn create_pipeline(
        &self,
        create_info: &ComputePipelineCreateInfo,
    ) -> Result<vk::Pipeline, PipelineError> {
        let shader_module = self
            .shader_modules
            .get(&create_info.pipeline_type)
            .copied()
            .ok_or(PipelineError::ShaderModuleNotFound(create_info.pipeline_type))?;

        let pipeline_layout = self
            .layouts
            .get(&create_info.pipeline_type)
            .copied()
            .ok_or(PipelineError::PipelineLayoutNotFound(create_info.pipeline_type))?;

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
            vk::SpecializationInfo {
                map_entry_count: spec_entries.len() as u32,
                p_map_entries: spec_entries.as_ptr(),
                data_size: spec_data.len(),
                p_data: spec_data.as_ptr() as *const _,
            }
        } else {
            vk::SpecializationInfo::default()
        };

        let stage_create_info = vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::PipelineShaderStageCreateFlags::empty(),
            stage: vk::ShaderStageFlags::COMPUTE,
            module: shader_module,
            p_name: b"main\0".as_ptr() as *const i8,
            p_specialization_info: if spec_entries.is_empty() {
                std::ptr::null()
            } else {
                &spec_info
            },
        };

        let create_info_vk = vk::ComputePipelineCreateInfo {
            s_type: vk::StructureType::COMPUTE_PIPELINE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::PipelineCreateFlags::empty(),
            stage: stage_create_info,
            layout: pipeline_layout,
            base_pipeline_handle: vk::Pipeline::null(),
            base_pipeline_index: 0,
        };

        let pipelines = unsafe {
            self.device
                .create_compute_pipelines(self.pipeline_cache, &[create_info_vk], None)
                .map_err(PipelineError::PipelineCreationFailed)?
        };

        Ok(pipelines[0])
    }

    pub fn get_pipeline_layout(&mut self, ty: ComputePipelineType) -> vk::PipelineLayout {
        if let Some(&layout) = self.layouts.get(&ty) {
            return layout;
        }

        let layout = vk::PipelineLayout::null();
        self.layouts.insert(ty, layout);
        layout
    }

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

        // Optionally destroy shader modules if we own them
        for (_, module) in self.shader_modules.drain() {
            unsafe { self.device.destroy_shader_module(module, None); }
        }
    }
}

impl Drop for ComputePipelineManager {
    fn drop(&mut self) {
        self.save_cache();
    }
}
