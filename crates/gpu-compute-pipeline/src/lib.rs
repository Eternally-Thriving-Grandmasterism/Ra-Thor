//! gpu-compute-pipeline v14.15.0
//!
//! Packaged from root `gpu_compute_pipeline.rs`.
//! Default build: CPU / simulation path (no wgpu dependency).
//! Feature `wgpu`: enables real device initialization hooks.
//!
//! Full historical WGSL dispatch logic remains in root file + `shaders/`.
//! This crate owns the stable public types used by ONE Organism +
//! Quantum Swarm facades.
//!
//! AG-SML v1.0 | TOLC 8 | Live Frame Bridge contract preserved.
//! Living Cosmic Tick ready — ONE Organism 14.15.0 surface.

use serde::{Deserialize, Serialize};
use std::time::Instant;

// =============================================================================
// Core task types (used by ONE Organism + Quantum Swarm)
// =============================================================================

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
    pub readback_data: Option<Vec<u32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MercyGpuAudit {
    pub task_id: u64,
    pub mercy_norm: f64,
    pub council_ready: bool,
    pub suggested_confidence_delta: f64,
}

impl MercyGpuAudit {
    pub fn suggested_confidence_delta(&self) -> f64 {
        (self.mercy_norm - 0.75).max(0.0) * 0.6
    }
}

// =============================================================================
// Motion / vision types
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionVector {
    pub x: f32,
    pub y: f32,
    pub dx: f32,
    pub dy: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionFieldSoA {
    pub dx: Vec<f32>,
    pub dy: Vec<f32>,
    pub width: u32,
    pub height: u32,
}

impl MotionFieldSoA {
    pub fn len(&self) -> usize {
        self.dx.len().min(self.dy.len())
    }

    pub fn is_empty(&self) -> bool {
        self.dx.is_empty() || self.dy.is_empty()
    }

    pub fn to_aos(&self, stride: u32) -> Vec<MotionVector> {
        let count = self.len();
        let mut out = Vec::with_capacity(count);
        let w = self.width.max(1);
        for i in 0..count {
            out.push(MotionVector {
                x: ((i as u32 % w) * stride) as f32,
                y: ((i as u32 / w) * stride) as f32,
                dx: self.dx.get(i).copied().unwrap_or(0.0),
                dy: self.dy.get(i).copied().unwrap_or(0.0),
            });
        }
        out
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommonFateParams {
    pub dominant_dir1: f32,
    pub dominant_dir2: f32,
    pub tolerance: f32,
    pub valence: f32,
    pub ghost_font_mode: bool,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommonFateResult {
    pub coherent_count: u32,
    pub letter_cluster_count: u32,
    pub perceived_text_candidate: String,
    pub confidence: f32,
    pub thriving_score: f32,
    pub motion_map: Option<Vec<u32>>,
    pub mercy_gated: bool,
    pub note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockMatchingParams {
    pub width: u32,
    pub height: u32,
    pub block_size: u32,
    pub search_range: i32,
    pub stride: u32,
    pub level: u32,
    pub valence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionEstimationResult {
    pub field: MotionFieldSoA,
    pub motion_vectors: Vec<MotionVector>,
    pub width: u32,
    pub height: u32,
    pub execution_time_ms: u64,
    pub mercy_gated: bool,
    pub note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownsampleParams {
    pub src_width: u32,
    pub src_height: u32,
    pub dst_width: u32,
    pub dst_height: u32,
    pub valence: f32,
    pub use_bilinear: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownsampleResult {
    pub data: Vec<f32>,
    pub width: u32,
    pub height: u32,
    pub execution_time_ms: u64,
    pub mercy_gated: bool,
    pub note: String,
}

// =============================================================================
// Live Frame Bridge
// =============================================================================

#[derive(Debug, Clone)]
pub struct LumaFrame {
    pub data: Vec<f32>,
    pub width: u32,
    pub height: u32,
    pub timestamp_us: u64,
}

impl LumaFrame {
    pub fn empty(w: u32, h: u32) -> Self {
        Self {
            data: vec![0.0; (w * h) as usize],
            width: w,
            height: h,
            timestamp_us: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
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
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            prev: LumaFrame::empty(width, height),
            curr: LumaFrame::empty(width, height),
            width,
            height,
            frame_count: 0,
        }
    }

    pub fn push(&mut self, new_frame: LumaFrame) {
        self.prev = std::mem::replace(&mut self.curr, new_frame);
        self.frame_count += 1;
    }

    pub fn has_pair(&self) -> bool {
        self.frame_count >= 2 && !self.prev.data.is_empty() && !self.curr.data.is_empty()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LumaConversionParams {
    pub width: u32,
    pub height: u32,
    /// 0 = BT.709, 1 = average, 2 = BT.601
    pub mode: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GpuBufferUsage {
    Storage,
    Uniform,
    Vertex,
    Index,
    Readback,
    Staging,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuBufferHandle {
    pub id: u64,
    pub size: usize,
    pub usage: GpuBufferUsage,
    pub last_used_tick: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDeviceRecoveryStats {
    pub device_lost_count: u32,
    pub successful_recoveries: u32,
    pub last_device_lost_at_unix: Option<u64>,
    pub last_recovery_at_unix: Option<u64>,
}

// =============================================================================
// Pipeline (CPU / simulation path — always available)
// =============================================================================

pub struct GpuComputePipeline {
    pub device_recovery_stats: GpuDeviceRecoveryStats,
    pub luma_ring: Option<LumaRing>,
    real_gpu: bool,
    dispatch_count: u64,
}

impl Default for GpuComputePipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuComputePipeline {
    pub fn new() -> Self {
        Self {
            device_recovery_stats: GpuDeviceRecoveryStats {
                device_lost_count: 0,
                successful_recoveries: 0,
                last_device_lost_at_unix: None,
                last_recovery_at_unix: None,
            },
            luma_ring: Some(LumaRing::new(640, 360)),
            real_gpu: false,
            dispatch_count: 0,
        }
    }

    pub fn is_real_gpu(&self) -> bool {
        self.real_gpu
    }

    pub fn dispatch_count(&self) -> u64 {
        self.dispatch_count
    }

    /// Mark pipeline as having a live device (called by wgpu feature path).
    pub fn mark_real_gpu(&mut self, enabled: bool) {
        self.real_gpu = enabled;
        if enabled {
            println!("[GpuComputePipeline v14.15.0] real_gpu = true");
        }
    }

    pub fn configure_luma_ring(&mut self, width: u32, height: u32) {
        self.luma_ring = Some(LumaRing::new(width, height));
    }

    pub fn push_luma_frame(&mut self, frame: LumaFrame) {
        if let Some(ring) = self.luma_ring.as_mut() {
            ring.push(frame);
        }
    }

    pub async fn dispatch_gpu_task(&mut self, task: GpuTask) -> GpuTaskResult {
        let t = Instant::now();
        tokio::time::sleep(std::time::Duration::from_micros(10)).await;
        self.dispatch_count += 1;
        GpuTaskResult {
            id: task.id,
            success: true,
            message: task.name.clone(),
            execution_time_ms: t.elapsed().as_millis() as u64,
            real_gpu: self.real_gpu,
            readback_data: None,
        }
    }

    pub async fn dispatch_downsample(
        &mut self,
        _src: &[f32],
        params: DownsampleParams,
    ) -> DownsampleResult {
        let t = Instant::now();
        if params.valence < 0.999999 {
            return DownsampleResult {
                data: vec![],
                width: 0,
                height: 0,
                execution_time_ms: 0,
                mercy_gated: false,
                note: "HOLD".into(),
            };
        }
        DownsampleResult {
            data: vec![0.0; (params.dst_width * params.dst_height) as usize],
            width: params.dst_width,
            height: params.dst_height,
            execution_time_ms: t.elapsed().as_millis() as u64,
            mercy_gated: true,
            note: "ds-sim".into(),
        }
    }

    pub async fn build_image_pyramid(
        &mut self,
        luma: &[f32],
        w: u32,
        h: u32,
        v: f32,
    ) -> Vec<DownsampleResult> {
        self.build_image_pyramid_with_mode(luma, w, h, v, true).await
    }

    pub async fn build_image_pyramid_with_mode(
        &mut self,
        luma: &[f32],
        w: u32,
        h: u32,
        v: f32,
        bil: bool,
    ) -> Vec<DownsampleResult> {
        let mut levels = vec![DownsampleResult {
            data: luma.to_vec(),
            width: w,
            height: h,
            execution_time_ms: 0,
            mercy_gated: true,
            note: "L0".into(),
        }];
        let w1 = (w / 2).max(8);
        let h1 = (h / 2).max(8);
        let l1 = self
            .dispatch_downsample(
                luma,
                DownsampleParams {
                    src_width: w,
                    src_height: h,
                    dst_width: w1,
                    dst_height: h1,
                    valence: v,
                    use_bilinear: bil,
                },
            )
            .await;
        levels.push(l1.clone());
        let w2 = (w1 / 2).max(4);
        let h2 = (h1 / 2).max(4);
        levels.push(
            self.dispatch_downsample(
                &l1.data,
                DownsampleParams {
                    src_width: w1,
                    src_height: h1,
                    dst_width: w2,
                    dst_height: h2,
                    valence: v,
                    use_bilinear: bil,
                },
            )
            .await,
        );
        levels
    }

    pub async fn dispatch_pyramidal_block_matching(
        &mut self,
        _prev: &[f32],
        _curr: &[f32],
        params: BlockMatchingParams,
        _pred: Option<&[f32]>,
    ) -> MotionEstimationResult {
        let t = Instant::now();
        if params.valence < 0.999999 {
            return MotionEstimationResult {
                field: MotionFieldSoA {
                    dx: vec![],
                    dy: vec![],
                    width: 0,
                    height: 0,
                },
                motion_vectors: vec![],
                width: 0,
                height: 0,
                execution_time_ms: 0,
                mercy_gated: false,
                note: "HOLD".into(),
            };
        }
        let out_w = (params.width + params.stride - 1) / params.stride;
        let out_h = (params.height + params.stride - 1) / params.stride;
        let n = (out_w * out_h) as usize;
        let mut dx = vec![0.0; n];
        let mut dy = vec![0.0; n];
        for i in 0..n {
            dx[i] = if (i as u32 / out_w) < out_h / 2 {
                2.0
            } else {
                -2.0
            };
        }
        let field = MotionFieldSoA {
            dx: dx.clone(),
            dy,
            width: out_w,
            height: out_h,
        };
        MotionEstimationResult {
            motion_vectors: field.to_aos(params.stride),
            field,
            width: out_w,
            height: out_h,
            execution_time_ms: t.elapsed().as_millis() as u64,
            mercy_gated: true,
            note: "sim SoA".into(),
        }
    }

    pub async fn estimate_motion_pyramidal(
        &mut self,
        prev: &[f32],
        curr: &[f32],
        w: u32,
        h: u32,
        v: f32,
    ) -> MotionEstimationResult {
        let pp = self.build_image_pyramid(prev, w, h, v).await;
        let cp = self.build_image_pyramid(curr, w, h, v).await;
        let coarse = BlockMatchingParams {
            width: pp[2].width.max(16),
            height: pp[2].height.max(16),
            block_size: 8,
            search_range: 8,
            stride: 8,
            level: 2,
            valence: v,
        };
        let mut r = self
            .dispatch_pyramidal_block_matching(&pp[2].data, &cp[2].data, coarse, None)
            .await;
        r.note = format!("Pyramidal + SoA. {}", r.note);
        r
    }

    pub async fn dispatch_common_fate_soa(
        &mut self,
        field: &MotionFieldSoA,
        params: CommonFateParams,
    ) -> CommonFateResult {
        if params.valence < 0.999999 {
            return CommonFateResult {
                coherent_count: 0,
                letter_cluster_count: 0,
                perceived_text_candidate: String::new(),
                confidence: 0.0,
                thriving_score: 0.0,
                motion_map: None,
                mercy_gated: false,
                note: "HOLD".into(),
            };
        }
        let count = field.len();
        if params.ghost_font_mode {
            return CommonFateResult {
                coherent_count: 1240,
                letter_cluster_count: 380,
                perceived_text_candidate: "RILEY WAS HERE".into(),
                confidence: 0.93,
                thriving_score: 0.97,
                motion_map: None,
                mercy_gated: true,
                note: "Ghost Font sim".into(),
            };
        }
        if count == 0 {
            return CommonFateResult {
                coherent_count: 0,
                letter_cluster_count: 0,
                perceived_text_candidate: "[NO_MOTION]".into(),
                confidence: 0.1,
                thriving_score: 0.5,
                motion_map: None,
                mercy_gated: true,
                note: "empty".into(),
            };
        }
        CommonFateResult {
            coherent_count: (count / 2) as u32,
            letter_cluster_count: (count / 4) as u32,
            perceived_text_candidate: "[MOTION_SHAPE]".into(),
            confidence: 0.88,
            thriving_score: 0.92,
            motion_map: None,
            mercy_gated: true,
            note: "CPU Common Fate sim".into(),
        }
    }

    pub async fn dispatch_common_fate_vision(
        &mut self,
        mvs: Vec<MotionVector>,
        params: CommonFateParams,
    ) -> CommonFateResult {
        let mut dx = Vec::with_capacity(mvs.len());
        let mut dy = Vec::with_capacity(mvs.len());
        for mv in &mvs {
            dx.push(mv.dx);
            dy.push(mv.dy);
        }
        let field = MotionFieldSoA {
            dx,
            dy,
            width: params.width,
            height: params.height,
        };
        self.dispatch_common_fate_soa(&field, params).await
    }

    pub async fn resolve_ghost_font_gpu(
        &mut self,
        sim: Vec<MotionVector>,
    ) -> CommonFateResult {
        let p = CommonFateParams {
            dominant_dir1: -1.5708,
            dominant_dir2: 1.5708,
            tolerance: 0.6,
            valence: 1.0,
            ghost_font_mode: true,
            width: 640,
            height: 360,
        };
        self.dispatch_common_fate_vision(sim, p).await
    }

    pub async fn perceive_from_raw_frames(
        &mut self,
        prev: &[f32],
        curr: &[f32],
        w: u32,
        h: u32,
        v: f32,
        ghost: bool,
    ) -> CommonFateResult {
        let motion = self.estimate_motion_pyramidal(prev, curr, w, h, v).await;
        let p = CommonFateParams {
            dominant_dir1: -1.5708,
            dominant_dir2: 1.5708,
            tolerance: 0.55,
            valence: v,
            ghost_font_mode: ghost,
            width: w,
            height: h,
        };
        self.dispatch_common_fate_soa(&motion.field, p).await
    }

    pub async fn perceive_from_luma_ring(
        &mut self,
        valence: f32,
        ghost_font_mode: bool,
    ) -> Option<CommonFateResult> {
        let (prev, curr, w, h) = {
            let ring = self.luma_ring.as_ref()?;
            if !ring.has_pair() {
                return None;
            }
            (
                ring.prev.data.clone(),
                ring.curr.data.clone(),
                ring.width,
                ring.height,
            )
        };
        Some(
            self.perceive_from_raw_frames(&prev, &curr, w, h, valence, ghost_font_mode)
                .await,
        )
    }
}

pub fn create_gpu_pipeline() -> GpuComputePipeline {
    GpuComputePipeline::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn dispatch_task_sim() {
        let mut p = GpuComputePipeline::new();
        let r = p
            .dispatch_gpu_task(GpuTask {
                id: 1,
                name: "test".into(),
                buffer_size: 1024,
                intensity: "low".into(),
            })
            .await;
        assert!(r.success);
        assert!(!r.real_gpu);
        assert_eq!(p.dispatch_count(), 1);
    }

    #[tokio::test]
    async fn common_fate_ghost() {
        let mut p = GpuComputePipeline::new();
        let r = p.resolve_ghost_font_gpu(vec![]).await;
        assert!(r.mercy_gated);
        assert_eq!(r.perceived_text_candidate, "RILEY WAS HERE");
    }
}
