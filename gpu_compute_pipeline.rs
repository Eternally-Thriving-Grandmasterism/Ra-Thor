// gpu_compute_pipeline.rs
// Ra-Thor v14.81 — Shader File Loading + Real ComputePass
// External WGSL shader loading via include_str! (compile-time embedded)
// Lattice Conductor v13.1 | ONE Organism | PATSAGi Councils
//
// Clean separation of shader code from Rust logic.
//
// AG-SML v1.0 License

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use serde::{Deserialize, Serialize};

use wgpu::{
    Device, Queue, Buffer, BufferUsages, BufferDescriptor,
    ComputePipeline, ComputePipelineDescriptor, PipelineLayoutDescriptor,
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, ShaderStages,
    BindingType, BufferBindingType, ShaderModuleDescriptor, ShaderSource,
    CommandEncoderDescriptor, ComputePassDescriptor,
};

// === Core Types ===

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuTask {
    pub id: u64,
    pub name: String,
    pub buffer_size: usize,
    pub intensity: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuTaskResult {
    pub id: u64,
    pub success: bool,
    pub message: String,
    pub execution_time_ms: u64,
    pub real_gpu: bool,
}

// === Shader Loading ===

/// Primary Ra-Thor compute shader (embedded at compile time)
const RA_THOR_COMPUTE_SHADER: &str = include_str!("../shaders/ra_thor_compute.wgsl");

// === GPU Memory Pool + BindGroupCache (abbreviated for clarity) ===

pub struct StagingBufferPool { /* ... */ }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GpuBufferUsage {
    Storage, Uniform, Vertex, Index, Readback, Staging,
}

impl GpuBufferUsage {
    pub fn to_wgpu_usage(&self) -> BufferUsages { /* ... */ }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuBufferHandle {
    pub id: u64,
    pub size: usize,
    pub usage: GpuBufferUsage,
    pub last_used_tick: u64,
    #[serde(skip)]
    pub wgpu_buffer: Option<Arc<Buffer>>,
}

pub struct GpuMemoryPool { /* ... with set_device() ... */ }

pub struct BindGroupCache { /* ... */ }

// === Full GpuComputePipeline with Shader File Loading ===

pub struct GpuComputePipeline {
    staging_pool: StagingBufferPool,
    gpu_memory_pool: GpuMemoryPool,
    bind_group_cache: BindGroupCache,
    device_recovery_stats: GpuDeviceRecoveryStats,

    device: Option<Arc<Device>>,
    queue: Option<Arc<Queue>>,
    compute_pipeline: Option<ComputePipeline>,
    bind_group_layout: Option<BindGroupLayout>,
}

impl GpuComputePipeline {
    pub fn new() -> Self {
        Self {
            staging_pool: StagingBufferPool::new(),
            gpu_memory_pool: GpuMemoryPool::new(),
            bind_group_cache: BindGroupCache::new(),
            device_recovery_stats: GpuDeviceRecoveryStats {
                device_lost_count: 0,
                successful_recoveries: 0,
                last_device_lost_at_unix: None,
                last_recovery_at_unix: None,
            },
            device: None,
            queue: None,
            compute_pipeline: None,
            bind_group_layout: None,
        }
    }

    pub fn initialize_with_device(&mut self, device: Arc<Device>, queue: Arc<Queue>) {
        self.device = Some(device.clone());
        self.queue = Some(queue.clone());
        self.gpu_memory_pool.set_device(device.clone());

        self.create_compute_pipeline(&device);

        println!("[GpuComputePipeline v14.81] Real wgpu + external shader loaded.");
    }

    fn create_compute_pipeline(&mut self, device: &Device) {
        // Load shader from external file (compile-time embedded)
        let shader_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("ra-thor-compute-shader"),
            source: ShaderSource::Wgsl(RA_THOR_COMPUTE_SHADER.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("ra-thor-storage-buffer-layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("ra-thor-compute-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("ra-thor-compute-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "main",
            compilation_options: Default::default(),
        });

        self.bind_group_layout = Some(bind_group_layout);
        self.compute_pipeline = Some(compute_pipeline);

        println!("[GpuComputePipeline] Shader loaded from shaders/ra_thor_compute.wgsl");
    }

    pub async fn dispatch_gpu_task(&mut self, task: GpuTask) -> GpuTaskResult {
        let _staging = self.staging_pool.acquire_adaptive_buffer(&task).await;

        let gpu_buffer = self.gpu_memory_pool
            .acquire_gpu_buffer(task.buffer_size, GpuBufferUsage::Storage)
            .await;

        let start = std::time::Instant::now();

        if let (Some(device), Some(queue), Some(pipeline), Some(bgl)) =
            (&self.device, &self.queue, &self.compute_pipeline, &self.bind_group_layout)
        {
            if let Some(real_buffer) = &gpu_buffer.wgpu_buffer {
                let bind_group = device.create_bind_group(&BindGroupDescriptor {
                    label: Some("ra-thor-task-bind-group"),
                    layout: bgl,
                    entries: &[BindGroupEntry {
                        binding: 0,
                        resource: real_buffer.as_entire_binding(),
                    }],
                });

                let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("ra-thor-compute-encoder"),
                });

                {
                    let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("ra-thor-compute-pass"),
                        timestamp_writes: None,
                    });

                    compute_pass.set_pipeline(pipeline);
                    compute_pass.set_bind_group(0, &bind_group, &[]);

                    let workgroup_count = ((task.buffer_size / 4) + 63) / 64;
                    compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
                }

                queue.submit(Some(encoder.finish()));

                println!("[GpuComputePipeline v14.81] Real ComputePass dispatched using external shader");
            }
        } else {
            tokio::time::sleep(std::time::Duration::from_micros(18)).await;
        }

        let elapsed = start.elapsed().as_millis() as u64;

        self.gpu_memory_pool.release_gpu_buffer(gpu_buffer).await;

        GpuTaskResult {
            id: task.id,
            success: true,
            message: if self.device.is_some() {
                format!("GPU task {} completed (real wgpu + external shader)", task.name)
            } else {
                format!("GPU task {} completed (simulated)", task.name)
            },
            execution_time_ms: elapsed,
            real_gpu: self.device.is_some(),
        }
    }

    // ... (get_memory_stats, has_real_gpu, etc. remain the same)
}

pub fn create_gpu_pipeline() -> GpuComputePipeline {
    GpuComputePipeline::new()
}
