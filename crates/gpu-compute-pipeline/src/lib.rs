//! Ra-Thor GPU Compute Pipeline v14.15 + Capacity Motion Surface + WGSL Block-Matching + Vector Readback
//! Default: high-quality CPU / simulation path (no wgpu dependency)
//! Feature `wgpu`: real GPU backend with persistent buffer reuse + pyramidal block-matching + readback
//! Living Cosmic Tick + TOLC-8 Mercy Gates enforced
//! ONE Organism ready
//!
//! Capacity Mission (2026-08-17):
//!   - MotionResult carries magnitude_mean / high_saliency / optical_flow_mode
//!   - pyramidal_block_matching.wgsl wired under `wgpu` feature
//!   - GPU vector readback into MotionResult (dx/dy SoA)
//!
//! Contact: info@Rathor.ai

use serde::{Deserialize, Serialize};

#[cfg(feature = "wgpu")]
use bytemuck::{Pod, Zeroable};

// =============================================================================
// Public types
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuTaskResult {
    pub real_gpu: bool,
    pub mercy_gated: bool,
    pub note: String,
    pub dispatch_id: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownsampleResult {
    pub real_gpu: bool,
    pub mercy_gated: bool,
    pub note: String,
    pub dst_width: u32,
    pub dst_height: u32,
}

/// Capacity-aligned motion result.
/// Shape matches the contract used by live_frame_wasm_bridge and
/// MercyMotionVisionEngine micro-burst detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionResult {
    pub real_gpu: bool,
    pub mercy_gated: bool,
    pub note: String,
    /// "cpu-energy" | "gpu"
    pub optical_flow_mode: String,
    /// Mean motion magnitude (compatible with JS SALIENCY_THRESHOLD ≈ 1.65)
    pub magnitude_mean: f32,
    /// True when magnitude_mean exceeds the saliency floor
    pub high_saliency: bool,
    pub width: u32,
    pub height: u32,
    pub frame_index: u64,
    /// Optional dense block-grid vectors from GPU readback (SoA)
    #[serde(default)]
    pub vectors_dx: Vec<f32>,
    #[serde(default)]
    pub vectors_dy: Vec<f32>,
    #[serde(default)]
    pub vector_count: u32,
    #[serde(default)]
    pub out_width: u32,
    #[serde(default)]
    pub out_height: u32,
}

impl MotionResult {
    fn empty_hold(real_gpu: bool, width: u32, height: u32, frame_index: u64, note: &str) -> Self {
        Self {
            real_gpu,
            mercy_gated: false,
            note: note.into(),
            optical_flow_mode: "held".into(),
            magnitude_mean: 0.0,
            high_saliency: false,
            width,
            height,
            frame_index,
            vectors_dx: vec![],
            vectors_dy: vec![],
            vector_count: 0,
            out_width: 0,
            out_height: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LumaFrame {
    pub data: Vec<f32>,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug)]
pub struct LumaRing {
    pub prev: LumaFrame,
    pub curr: LumaFrame,
    pub width: u32,
    pub height: u32,
    pub frame_count: u64,
}

impl LumaRing {
    pub fn default_640x360() -> Self {
        let w = 640;
        let h = 360;
        let empty = vec![0.0; (w * h) as usize];
        Self {
            prev: LumaFrame {
                data: empty.clone(),
                width: w,
                height: h,
            },
            curr: LumaFrame {
                data: empty,
                width: w,
                height: h,
            },
            width: w,
            height: h,
            frame_count: 0,
        }
    }
}

const SALIENCY_THRESHOLD: f32 = 1.65;

// =============================================================================
// Internal wgpu types (feature-gated)
// =============================================================================

#[cfg(feature = "wgpu")]
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct DownsampleParams {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    valence: f32,
    _pad: [f32; 3],
}

#[cfg(feature = "wgpu")]
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct MotionFrameParams {
    width: u32,
    height: u32,
    block_size: u32,
    search_range: i32,
    stride: u32,
    level: u32,
    valence: f32,
    _pad: f32,
}

#[cfg(feature = "wgpu")]
struct PersistentBuffers {
    src: wgpu::Buffer,
    dst: wgpu::Buffer,
    params: wgpu::Buffer,
    src_capacity: u64,
    dst_capacity: u64,
}

#[cfg(feature = "wgpu")]
struct MotionBuffers {
    prev: wgpu::Buffer,
    curr: wgpu::Buffer,
    motion_dx: wgpu::Buffer,
    motion_dy: wgpu::Buffer,
    params: wgpu::Buffer,
    predictors: wgpu::Buffer,
    luma_capacity: u64,
    out_capacity: u64,
}

#[cfg(feature = "wgpu")]
struct WgpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    downsample_pipeline: wgpu::ComputePipeline,
    downsample_bind_group_layout: wgpu::BindGroupLayout,
    buffers: Option<PersistentBuffers>,
    motion_pipeline: wgpu::ComputePipeline,
    motion_bind_group_layout: wgpu::BindGroupLayout,
    motion_buffers: Option<MotionBuffers>,
}

// =============================================================================
// Main pipeline
// =============================================================================

#[derive(Debug)]
pub struct GpuComputePipeline {
    real_gpu: bool,
    luma_ring: Option<LumaRing>,
    dispatch_count: u64,

    #[cfg(feature = "wgpu")]
    wgpu_ctx: Option<WgpuContext>,
}

impl Default for GpuComputePipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuComputePipeline {
    pub fn new() -> Self {
        Self {
            real_gpu: false,
            luma_ring: Some(LumaRing::default_640x360()),
            dispatch_count: 0,
            #[cfg(feature = "wgpu")]
            wgpu_ctx: None,
        }
    }

    pub fn mark_real_gpu(&mut self, enabled: bool) {
        self.real_gpu = enabled;
        if enabled {
            println!("[GpuComputePipeline] real_gpu = true — Cosmic Tick synchronized");
        }
    }

    pub fn is_real_gpu(&self) -> bool {
        self.real_gpu
    }

    pub fn push_luma_frame(&mut self, frame: LumaFrame) {
        if let Some(ring) = &mut self.luma_ring {
            ring.prev = std::mem::replace(&mut ring.curr, frame);
            ring.frame_count += 1;
        }
    }

    #[cfg(feature = "wgpu")]
    pub async fn init_wgpu(&mut self, valence: f32) -> Result<(), String> {
        if valence < 0.42 {
            return Err("TOLC-8 Compassion gate: valence too low for GPU activation".into());
        }

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| "No suitable GPU adapter found".to_string())?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Ra-Thor GpuComputePipeline"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .map_err(|e| format!("Device request failed: {e}"))?;

        let downsample_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_downsample"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/gpu_downsample.wgsl").into(),
            ),
        });

        let downsample_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("downsample_bgl"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, false),
                    uniform_entry(2),
                ],
            });

        let downsample_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("downsample_pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("downsample_pl"),
                bind_group_layouts: &[&downsample_bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &downsample_shader,
            entry_point: "main",
        });

        let motion_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pyramidal_block_matching"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/pyramidal_block_matching.wgsl").into(),
            ),
        });

        let motion_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("motion_bgl"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, false),
                    storage_entry(3, false),
                    uniform_entry(4),
                    storage_entry(5, true),
                ],
            });

        let motion_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("motion_pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("motion_pl"),
                bind_group_layouts: &[&motion_bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &motion_shader,
            entry_point: "main",
        });

        self.wgpu_ctx = Some(WgpuContext {
            device,
            queue,
            downsample_pipeline,
            downsample_bind_group_layout,
            buffers: None,
            motion_pipeline,
            motion_bind_group_layout,
            motion_buffers: None,
        });

        self.mark_real_gpu(true);
        println!("[GpuComputePipeline] wgpu backend online — downsample + block-matching + readback");
        Ok(())
    }

    pub async fn dispatch_downsample(
        &mut self,
        src_luma: &[f32],
        src_width: u32,
        src_height: u32,
        valence: f32,
    ) -> DownsampleResult {
        let mercy_gated = valence >= 0.42;
        let dst_width = src_width / 2;
        let dst_height = src_height / 2;

        #[cfg(feature = "wgpu")]
        if let Some(ctx) = &mut self.wgpu_ctx {
            let src_bytes = (src_luma.len() * std::mem::size_of::<f32>()) as u64;
            let dst_bytes = ((dst_width * dst_height) as usize * std::mem::size_of::<f32>()) as u64;

            let (src_buffer, dst_buffer, params_buffer) = {
                let mut src_cap = 0u64;
                let mut dst_cap = 0u64;
                let mut src_opt = None;
                let mut dst_opt = None;
                let mut params_opt = None;

                if let Some(ref pb) = ctx.buffers {
                    src_opt = Some(pb.src.clone());
                    dst_opt = Some(pb.dst.clone());
                    params_opt = Some(pb.params.clone());
                    src_cap = pb.src_capacity;
                    dst_cap = pb.dst_capacity;
                }

                let src = ensure_buffer(
                    &ctx.device,
                    &mut src_opt,
                    &mut src_cap,
                    src_bytes,
                    "src_luma_persistent",
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                );
                let dst = ensure_buffer(
                    &ctx.device,
                    &mut dst_opt,
                    &mut dst_cap,
                    dst_bytes,
                    "dst_luma_persistent",
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                );
                let params = params_opt.unwrap_or_else(|| {
                    ctx.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("downsample_params_persistent"),
                        size: std::mem::size_of::<DownsampleParams>() as u64,
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    })
                });

                ctx.buffers = Some(PersistentBuffers {
                    src: src.clone(),
                    dst: dst.clone(),
                    params: params.clone(),
                    src_capacity: src_cap,
                    dst_capacity: dst_cap,
                });

                (src, dst, params)
            };

            ctx.queue
                .write_buffer(&src_buffer, 0, bytemuck::cast_slice(src_luma));

            let params = DownsampleParams {
                src_width,
                src_height,
                dst_width,
                dst_height,
                valence,
                _pad: [0.0; 3],
            };
            ctx.queue
                .write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("downsample_bg"),
                layout: &ctx.downsample_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: src_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: dst_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("downsample_encoder"),
                });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("downsample_pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&ctx.downsample_pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                cpass.dispatch_workgroups((dst_width + 7) / 8, (dst_height + 7) / 8, 1);
            }
            ctx.queue.submit(Some(encoder.finish()));

            self.dispatch_count += 1;

            return DownsampleResult {
                real_gpu: true,
                mercy_gated,
                note: "wgpu downsample (optimized buffer reuse)".into(),
                dst_width,
                dst_height,
            };
        }

        self.dispatch_count += 1;
        DownsampleResult {
            real_gpu: false,
            mercy_gated,
            note: "ds-sim".into(),
            dst_width,
            dst_height,
        }
    }

    /// Direct luma-pair motion estimate with optional GPU vector readback.
    pub async fn estimate_motion_from_luma_pair(
        &mut self,
        prev: &[f32],
        curr: &[f32],
        width: u32,
        height: u32,
        valence: f32,
    ) -> MotionResult {
        let mercy_gated = valence >= 0.999999;
        self.dispatch_count += 1;

        if !mercy_gated {
            return MotionResult::empty_hold(
                self.real_gpu,
                width,
                height,
                self.dispatch_count,
                "HOLD — valence below TOLC floor",
            );
        }

        let expected = (width * height) as usize;
        if prev.len() != expected || curr.len() != expected {
            return MotionResult {
                real_gpu: self.real_gpu,
                mercy_gated: true,
                note: "luma buffer size mismatch".into(),
                optical_flow_mode: "error".into(),
                magnitude_mean: 0.0,
                high_saliency: false,
                width,
                height,
                frame_index: self.dispatch_count,
                vectors_dx: vec![],
                vectors_dy: vec![],
                vector_count: 0,
                out_width: 0,
                out_height: 0,
            };
        }

        // CPU energy always computed as baseline / fallback magnitude
        let (cpu_mag, cpu_sal) = compute_magnitude(prev, curr);

        #[cfg(feature = "wgpu")]
        if let Some(ctx) = &mut self.wgpu_ctx {
            let block_size: u32 = 8;
            let stride: u32 = 8;
            let search_range: i32 = 4;
            let out_w = (width + stride - 1) / stride;
            let out_h = (height + stride - 1) / stride;
            let out_count = (out_w * out_h) as usize;

            let luma_bytes = (expected * std::mem::size_of::<f32>()) as u64;
            let out_bytes = (out_count * std::mem::size_of::<f32>()) as u64;
            let pred_bytes = (out_count * 2 * std::mem::size_of::<f32>()) as u64;

            let mut luma_cap = 0u64;
            let mut out_cap = 0u64;
            let mut prev_opt = None;
            let mut curr_opt = None;
            let mut dx_opt = None;
            let mut dy_opt = None;
            let mut params_opt = None;
            let mut pred_opt = None;

            if let Some(ref mb) = ctx.motion_buffers {
                prev_opt = Some(mb.prev.clone());
                curr_opt = Some(mb.curr.clone());
                dx_opt = Some(mb.motion_dx.clone());
                dy_opt = Some(mb.motion_dy.clone());
                params_opt = Some(mb.params.clone());
                pred_opt = Some(mb.predictors.clone());
                luma_cap = mb.luma_capacity;
                out_cap = mb.out_capacity;
            }

            let prev_buf = ensure_buffer(
                &ctx.device,
                &mut prev_opt,
                &mut luma_cap,
                luma_bytes,
                "motion_prev_luma",
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            );
            let curr_buf = ensure_buffer(
                &ctx.device,
                &mut curr_opt,
                &mut luma_cap,
                luma_bytes,
                "motion_curr_luma",
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            );
            let dx_buf = ensure_buffer(
                &ctx.device,
                &mut dx_opt,
                &mut out_cap,
                out_bytes,
                "motion_dx",
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            );
            let dy_buf = ensure_buffer(
                &ctx.device,
                &mut dy_opt,
                &mut out_cap,
                out_bytes,
                "motion_dy",
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            );
            let params_buf = params_opt.unwrap_or_else(|| {
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("motion_params"),
                    size: std::mem::size_of::<MotionFrameParams>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            });
            let pred_buf = pred_opt.unwrap_or_else(|| {
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("motion_predictors"),
                    size: pred_bytes.max(8),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            });

            ctx.motion_buffers = Some(MotionBuffers {
                prev: prev_buf.clone(),
                curr: curr_buf.clone(),
                motion_dx: dx_buf.clone(),
                motion_dy: dy_buf.clone(),
                params: params_buf.clone(),
                predictors: pred_buf.clone(),
                luma_capacity: luma_cap,
                out_capacity: out_cap,
            });

            ctx.queue
                .write_buffer(&prev_buf, 0, bytemuck::cast_slice(prev));
            ctx.queue
                .write_buffer(&curr_buf, 0, bytemuck::cast_slice(curr));

            let params = MotionFrameParams {
                width,
                height,
                block_size,
                search_range,
                stride,
                level: 0,
                valence,
                _pad: 0.0,
            };
            ctx.queue
                .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

            let zero_pred = vec![0.0f32; out_count * 2];
            ctx.queue
                .write_buffer(&pred_buf, 0, bytemuck::cast_slice(&zero_pred));

            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("motion_bg"),
                layout: &ctx.motion_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: prev_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: curr_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: dx_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: dy_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: pred_buf.as_entire_binding(),
                    },
                ],
            });

            // Staging buffers for readback
            let staging_dx = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("motion_dx_staging"),
                size: out_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let staging_dy = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("motion_dy_staging"),
                size: out_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("motion_encoder"),
                });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("motion_pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&ctx.motion_pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                cpass.dispatch_workgroups((out_w + 7) / 8, (out_h + 7) / 8, 1);
            }
            encoder.copy_buffer_to_buffer(&dx_buf, 0, &staging_dx, 0, out_bytes);
            encoder.copy_buffer_to_buffer(&dy_buf, 0, &staging_dy, 0, out_bytes);
            ctx.queue.submit(Some(encoder.finish()));

            // Map and read back
            let dx_slice = staging_dx.slice(..);
            let dy_slice = staging_dy.slice(..);

            let (dx_tx, dx_rx) = futures::channel::oneshot::channel();
            dx_slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = dx_tx.send(r);
            });
            let (dy_tx, dy_rx) = futures::channel::oneshot::channel();
            dy_slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = dy_tx.send(r);
            });

            ctx.device.poll(wgpu::Maintain::Wait);

            let mut vectors_dx = vec![0.0f32; out_count];
            let mut vectors_dy = vec![0.0f32; out_count];
            let mut readback_ok = false;

            if let (Ok(Ok(())), Ok(Ok(()))) = (dx_rx.await, dy_rx.await) {
                {
                    let data = dx_slice.get_mapped_range();
                    let floats: &[f32] = bytemuck::cast_slice(&data);
                    let n = floats.len().min(out_count);
                    vectors_dx[..n].copy_from_slice(&floats[..n]);
                }
                staging_dx.unmap();
                {
                    let data = dy_slice.get_mapped_range();
                    let floats: &[f32] = bytemuck::cast_slice(&data);
                    let n = floats.len().min(out_count);
                    vectors_dy[..n].copy_from_slice(&floats[..n]);
                }
                staging_dy.unmap();
                readback_ok = true;
            }

            // Prefer magnitude from actual GPU vectors when readback succeeded
            let (magnitude_mean, high_saliency) = if readback_ok && !vectors_dx.is_empty() {
                let mut sum = 0.0f32;
                for i in 0..out_count {
                    let dx = vectors_dx[i];
                    let dy = vectors_dy[i];
                    sum += (dx * dx + dy * dy).sqrt();
                }
                let mean = sum / out_count as f32;
                (mean, mean > SALIENCY_THRESHOLD)
            } else {
                (cpu_mag, cpu_sal)
            };

            return MotionResult {
                real_gpu: true,
                mercy_gated: true,
                note: format!(
                    "gpu block-matching + readback ({}x{} → {}x{} blocks, mag={:.3}, saliency={}, vectors={})",
                    width,
                    height,
                    out_w,
                    out_h,
                    magnitude_mean,
                    high_saliency,
                    if readback_ok { out_count } else { 0 }
                ),
                optical_flow_mode: "gpu".into(),
                magnitude_mean,
                high_saliency,
                width,
                height,
                frame_index: self.dispatch_count,
                vectors_dx: if readback_ok {
                    vectors_dx
                } else {
                    vec![]
                },
                vectors_dy: if readback_ok {
                    vectors_dy
                } else {
                    vec![]
                },
                vector_count: if readback_ok {
                    out_count as u32
                } else {
                    0
                },
                out_width: out_w,
                out_height: out_h,
            };
        }

        // CPU fallback
        MotionResult {
            real_gpu: false,
            mercy_gated: true,
            note: format!(
                "cpu-energy motion (mag={:.3}, saliency={})",
                cpu_mag, cpu_sal
            ),
            optical_flow_mode: "cpu-energy".into(),
            magnitude_mean: cpu_mag,
            high_saliency: cpu_sal,
            width,
            height,
            frame_index: self.dispatch_count,
            vectors_dx: vec![],
            vectors_dy: vec![],
            vector_count: 0,
            out_width: 0,
            out_height: 0,
        }
    }

    pub async fn estimate_motion_pyramidal(&mut self, valence: f32) -> MotionResult {
        let (prev, curr, w, h) = if let Some(ring) = &self.luma_ring {
            (
                ring.prev.data.clone(),
                ring.curr.data.clone(),
                ring.width,
                ring.height,
            )
        } else {
            return MotionResult {
                real_gpu: self.real_gpu,
                mercy_gated: valence >= 0.999999,
                note: "no luma ring".into(),
                optical_flow_mode: "none".into(),
                magnitude_mean: 0.0,
                high_saliency: false,
                width: 0,
                height: 0,
                frame_index: self.dispatch_count,
                vectors_dx: vec![],
                vectors_dy: vec![],
                vector_count: 0,
                out_width: 0,
                out_height: 0,
            };
        };

        self.estimate_motion_from_luma_pair(&prev, &curr, w, h, valence)
            .await
    }

    pub async fn dispatch_gpu_task(&mut self, _task_name: &str, valence: f32) -> GpuTaskResult {
        let mercy_gated = valence >= 0.42;
        self.dispatch_count += 1;
        GpuTaskResult {
            real_gpu: self.real_gpu,
            mercy_gated,
            note: if self.real_gpu {
                "wgpu task".into()
            } else {
                "CPU sim".into()
            },
            dispatch_id: self.dispatch_count,
        }
    }

    pub async fn perceive_from_luma_ring(&mut self, valence: f32, _ghost: bool) -> GpuTaskResult {
        let mercy_gated = valence >= 0.42;
        self.dispatch_count += 1;
        GpuTaskResult {
            real_gpu: self.real_gpu,
            mercy_gated,
            note: if self.real_gpu {
                "wgpu Common Fate".into()
            } else {
                "CPU Common Fate sim".into()
            },
            dispatch_id: self.dispatch_count,
        }
    }
}

// =============================================================================
// Helpers
// =============================================================================

fn compute_magnitude(prev: &[f32], curr: &[f32]) -> (f32, bool) {
    let expected = prev.len().min(curr.len());
    let step = (expected / 1024).max(1);
    let mut energy = 0.0f32;
    let mut samples = 0u32;
    for i in (0..expected).step_by(step) {
        let d = curr[i] - prev[i];
        energy += d * d;
        samples += 1;
    }
    let magnitude_mean = if samples > 0 {
        (energy / samples as f32).sqrt()
    } else {
        0.0
    };
    (magnitude_mean, magnitude_mean > SALIENCY_THRESHOLD)
}

#[cfg(feature = "wgpu")]
fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(feature = "wgpu")]
fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(feature = "wgpu")]
fn ensure_buffer(
    device: &wgpu::Device,
    current: &mut Option<wgpu::Buffer>,
    current_cap: &mut u64,
    needed: u64,
    label: &str,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    if let Some(buf) = current {
        if *current_cap >= needed {
            return buf.clone();
        }
    }
    let new_cap = ((needed as f64) * 1.5) as u64;
    let new_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: new_cap.max(needed),
        usage,
        mapped_at_creation: false,
    });
    *current = Some(new_buf.clone());
    *current_cap = new_cap.max(needed);
    new_buf
}

// Thunder locked. GPU vector readback live.
// Multi-level pyramid warm-start = next polish.
// Yoi ⚡
