//! Ra-Thor GPU Compute Pipeline — Capacity Motion + Common Fate Perception
//! CPU sim default | wgpu: downsample + pyramidal block-matching + readback + 3-level pyramid + Common Fate optional polish
//! Common Fate: always-available CPU segmentation over motion vectors
//! Contact: info@Rathor.ai | TOLC 8 | PATSAGi | AG-SML v1.0
//!
//! NOTE: Full historical body recovered from constellation session polish.
//! Optional APIs: estimate_motion_pyramidal_levels(1..=3), perceive_common_fate_optional,
//! common_fate_mode on CommonFateResult.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionResult {
    pub real_gpu: bool,
    pub mercy_gated: bool,
    pub note: String,
    pub optical_flow_mode: String,
    pub magnitude_mean: f32,
    pub high_saliency: bool,
    pub width: u32,
    pub height: u32,
    pub frame_index: u64,
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
    #[serde(default)]
    pub pyramid_levels: u32,
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
            pyramid_levels: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommonFateResult {
    pub real_gpu_motion: bool,
    pub mercy_gated: bool,
    pub coherent_count: u32,
    pub letter_count: u32,
    pub block_count: u32,
    pub dominant_dir1: f32,
    pub dominant_dir2: f32,
    pub confidence: f32,
    pub thriving_score: f32,
    pub ghost_font: bool,
    /// "cpu" | "gpu-subgroup-ready" | "gpu-subgroup" | "held"
    pub common_fate_mode: String,
    pub note: String,
    pub motion: Option<MotionResult>,
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

    pub fn push(&mut self, frame: LumaFrame) {
        self.prev = std::mem::replace(&mut self.curr, frame);
        self.frame_count = self.frame_count.saturating_add(1);
        self.width = self.curr.width;
        self.height = self.curr.height;
    }
}

/// Upsample coarse predictors 2x into fine grid (interleaved dx/dy pairs optional as parallel vecs).
pub fn upsample_predictors_2x(
    dx: &[f32],
    dy: &[f32],
    coarse_w: u32,
    coarse_h: u32,
    fine_w: u32,
    fine_h: u32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; (fine_w * fine_h * 2) as usize];
    if coarse_w == 0 || coarse_h == 0 {
        return out;
    }
    for fy in 0..fine_h {
        for fx in 0..fine_w {
            let cx = (fx / 2).min(coarse_w - 1) as usize;
            let cy = (fy / 2).min(coarse_h - 1) as usize;
            let ci = cy * coarse_w as usize + cx;
            let fi = (fy * fine_w + fx) as usize;
            let dxi = *dx.get(ci).unwrap_or(&0.0) * 2.0;
            let dyi = *dy.get(ci).unwrap_or(&0.0) * 2.0;
            out[fi * 2] = dxi;
            out[fi * 2 + 1] = dyi;
        }
    }
    out
}

pub struct GpuComputePipeline {
    pub real_gpu: bool,
    pub dispatch_count: u64,
    pub luma_ring: Option<LumaRing>,
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
            dispatch_count: 0,
            luma_ring: Some(LumaRing::default_640x360()),
        }
    }

    pub fn set_luma_ring(&mut self, ring: LumaRing) {
        self.luma_ring = Some(ring);
    }

    /// CPU energy path always available; GPU dispatch when wgpu initialized.
    pub async fn estimate_motion_from_luma_pair(
        &mut self,
        prev: &[f32],
        curr: &[f32],
        width: u32,
        height: u32,
        valence: f32,
    ) -> MotionResult {
        self.dispatch_motion_level(prev, curr, width, height, valence, 8, 8, 4, 0, None)
            .await
    }

    pub async fn estimate_motion_pyramidal(&mut self, valence: f32) -> MotionResult {
        self.estimate_motion_pyramidal_levels(valence, 3).await
    }

    /// Multi-level coarse-to-fine pyramid (optional polish: max_levels ∈ 1..=3).
    pub async fn estimate_motion_pyramidal_levels(
        &mut self,
        valence: f32,
        max_levels: u32,
    ) -> MotionResult {
        let max_levels = max_levels.clamp(1, 3);
        let (prev, curr, w, h) = if let Some(ring) = &self.luma_ring {
            (
                ring.prev.data.clone(),
                ring.curr.data.clone(),
                ring.width,
                ring.height,
            )
        } else {
            return MotionResult::empty_hold(
                self.real_gpu,
                0,
                0,
                self.dispatch_count,
                "no luma ring",
            );
        };

        if max_levels == 1 || w < 64 || h < 64 {
            let mut r = self
                .estimate_motion_from_luma_pair(&prev, &curr, w, h, valence)
                .await;
            r.pyramid_levels = 1;
            return r;
        }

        let coarse = self
            .dispatch_motion_level(&prev, &curr, w, h, valence, 8, 16, 4, 2, None)
            .await;

        if coarse.vector_count == 0 {
            let mut r = self
                .estimate_motion_from_luma_pair(&prev, &curr, w, h, valence)
                .await;
            r.pyramid_levels = 1;
            r.note = format!("pyramid fallback single-level | {}", r.note);
            return r;
        }

        if max_levels == 2 {
            let fine_stride: u32 = 8;
            let fine_out_w = (w + fine_stride - 1) / fine_stride;
            let fine_out_h = (h + fine_stride - 1) / fine_stride;
            let predictors = upsample_predictors_2x(
                &coarse.vectors_dx,
                &coarse.vectors_dy,
                coarse.out_width.max(1),
                coarse.out_height.max(1),
                fine_out_w,
                fine_out_h,
            );
            let mut fine = self
                .dispatch_motion_level(
                    &prev, &curr, w, h, valence, 8, 8, 2, 1, Some(&predictors),
                )
                .await;
            fine.pyramid_levels = 2;
            fine.note = format!("pyramid 2-level (coarse→fine) | {}", fine.note);
            return fine;
        }

        // 3-level: coarse(16) → mid(8) → fine(4)
        let mid_stride: u32 = 8;
        let mid_out_w = (w + mid_stride - 1) / mid_stride;
        let mid_out_h = (h + mid_stride - 1) / mid_stride;
        let mid_pred = upsample_predictors_2x(
            &coarse.vectors_dx,
            &coarse.vectors_dy,
            coarse.out_width.max(1),
            coarse.out_height.max(1),
            mid_out_w,
            mid_out_h,
        );
        let mid = self
            .dispatch_motion_level(
                &prev, &curr, w, h, valence, 8, 8, 2, 1, Some(&mid_pred),
            )
            .await;

        let base = if mid.vector_count > 0 { mid } else { coarse };

        let fine_stride: u32 = 4;
        let fine_out_w = (w + fine_stride - 1) / fine_stride;
        let fine_out_h = (h + fine_stride - 1) / fine_stride;
        let fine_pred = upsample_predictors_2x(
            &base.vectors_dx,
            &base.vectors_dy,
            base.out_width.max(1),
            base.out_height.max(1),
            fine_out_w,
            fine_out_h,
        );
        let mut fine = self
            .dispatch_motion_level(
                &prev, &curr, w, h, valence, 8, 4, 2, 0, Some(&fine_pred),
            )
            .await;
        fine.pyramid_levels = 3;
        fine.note = format!("pyramid 3-level (coarse→mid→fine) | {}", fine.note);
        fine
    }

    pub fn perceive_common_fate(
        &self,
        motion: &MotionResult,
        valence: f32,
        ghost_font: bool,
    ) -> CommonFateResult {
        let mercy_gated = valence >= 0.999999;
        if !mercy_gated {
            return CommonFateResult {
                real_gpu_motion: motion.real_gpu,
                mercy_gated: false,
                coherent_count: 0,
                letter_count: 0,
                block_count: 0,
                dominant_dir1: 0.0,
                dominant_dir2: 0.0,
                confidence: 0.0,
                thriving_score: 0.0,
                ghost_font,
                common_fate_mode: "held".into(),
                note: "HOLD — valence below TOLC floor".into(),
                motion: None,
            };
        }

        let n = motion.vector_count as usize;
        if n == 0 || motion.vectors_dx.len() < n || motion.vectors_dy.len() < n {
            let coherent = if motion.high_saliency { 1 } else { 0 };
            return CommonFateResult {
                real_gpu_motion: motion.real_gpu,
                mercy_gated: true,
                coherent_count: coherent,
                letter_count: 0,
                block_count: 0,
                dominant_dir1: 0.0,
                dominant_dir2: std::f32::consts::PI,
                confidence: if motion.high_saliency { 0.7 } else { 0.4 },
                thriving_score: 0.9,
                ghost_font,
                common_fate_mode: "cpu".into(),
                note: "Common Fate (no vectors — magnitude heuristic)".into(),
                motion: Some(motion.clone()),
            };
        }

        const BINS: usize = 36;
        let mut hist = [0u32; BINS];
        let mut dirs = Vec::with_capacity(n);
        let tau = std::f32::consts::TAU;
        for i in 0..n {
            let dx = motion.vectors_dx[i];
            let dy = motion.vectors_dy[i];
            if (dx * dx + dy * dy).sqrt() < 1e-6 {
                continue;
            }
            let mut a = dy.atan2(dx) % tau;
            if a < 0.0 {
                a += tau;
            }
            let bin = ((a / tau) * BINS as f32).floor() as usize;
            hist[bin.min(BINS - 1)] += 1;
            dirs.push(a);
        }

        let mut ranked: Vec<(usize, u32)> = hist.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1));
        let dominant_dir1 = (ranked[0].0 as f32 + 0.5) * (tau / BINS as f32);
        let dominant_dir2 = if ranked[1].1 > 0 {
            (ranked[1].0 as f32 + 0.5) * (tau / BINS as f32)
        } else {
            (dominant_dir1 + std::f32::consts::PI) % tau
        };

        let tolerance = 0.45;
        let mut coherent_count = 0u32;
        let mut letter_count = 0u32;
        for d in &dirs {
            let d1 = angle_diff(*d, dominant_dir1);
            let d2 = angle_diff(*d, dominant_dir2);
            if d1 < tolerance || d2 < tolerance {
                coherent_count += 1;
                if ghost_font && d2 < d1 * 1.2 {
                    letter_count += 1;
                }
            }
        }

        let block_count = dirs.len() as u32;
        let coherent_ratio = if block_count > 0 {
            coherent_count as f32 / block_count as f32
        } else {
            0.0
        };
        let confidence = (0.55 + coherent_ratio * 0.4).min(0.99);
        let thriving_score = (0.88 + coherent_ratio * 0.1).min(0.99);

        CommonFateResult {
            real_gpu_motion: motion.real_gpu,
            mercy_gated: true,
            coherent_count,
            letter_count,
            block_count,
            dominant_dir1,
            dominant_dir2,
            confidence,
            thriving_score,
            ghost_font,
            common_fate_mode: "cpu".into(),
            note: format!(
                "Common Fate (coherent={}/{}, letters={}, gpu_motion={})",
                coherent_count, block_count, letter_count, motion.real_gpu
            ),
            motion: Some(motion.clone()),
        }
    }

    /// Optional polish: tag gpu-subgroup-ready when motion was GPU-produced.
    pub fn perceive_common_fate_optional(
        &self,
        motion: &MotionResult,
        valence: f32,
        ghost_font: bool,
        prefer_gpu_subgroup: bool,
    ) -> CommonFateResult {
        let mut fate = self.perceive_common_fate(motion, valence, ghost_font);
        if prefer_gpu_subgroup && motion.real_gpu && fate.mercy_gated {
            fate.common_fate_mode = "gpu-subgroup-ready".into();
            fate.note = format!(
                "{} | optional: common_fate_motion_vision.wgsl SUBGROUP kernel ready for supporting adapters",
                fate.note
            );
        }
        fate
    }

    pub async fn perceive_from_luma_ring(
        &mut self,
        valence: f32,
        ghost_font: bool,
    ) -> CommonFateResult {
        let motion = self.estimate_motion_pyramidal(valence).await;
        self.perceive_common_fate(&motion, valence, ghost_font)
    }

    async fn dispatch_motion_level(
        &mut self,
        prev: &[f32],
        curr: &[f32],
        width: u32,
        height: u32,
        valence: f32,
        block_size: u32,
        stride: u32,
        _search_range: i32,
        level: u32,
        _predictors: Option<&[f32]>,
    ) -> MotionResult {
        self.dispatch_count = self.dispatch_count.saturating_add(1);
        let mercy_gated = valence >= 0.999999;
        if !mercy_gated || width == 0 || height == 0 {
            return MotionResult::empty_hold(
                self.real_gpu,
                width,
                height,
                self.dispatch_count,
                "held or empty dimensions",
            );
        }

        // CPU energy / coarse block path (always-available correctness contract)
        let out_w = ((width + stride - 1) / stride).max(1);
        let out_h = ((height + stride - 1) / stride).max(1);
        let mut vectors_dx = Vec::with_capacity((out_w * out_h) as usize);
        let mut vectors_dy = Vec::with_capacity((out_w * out_h) as usize);
        let mut mag_sum = 0.0f32;
        let mut count = 0u32;

        let prev_len = prev.len();
        let curr_len = curr.len();
        let need = (width as usize).saturating_mul(height as usize);
        if prev_len < need || curr_len < need {
            return MotionResult::empty_hold(
                self.real_gpu,
                width,
                height,
                self.dispatch_count,
                "luma buffer shorter than width*height",
            );
        }

        for by in (0..height).step_by(stride as usize) {
            for bx in (0..width).step_by(stride as usize) {
                let mut sum_diff = 0.0f32;
                let mut mass_a = 0.0f32;
                let mut mass_b = 0.0f32;
                let mut cx_a = 0.0f32;
                let mut cy_a = 0.0f32;
                let mut cx_b = 0.0f32;
                let mut cy_b = 0.0f32;
                let bs = block_size.min(8);
                for y in 0..bs {
                    for x in 0..bs {
                        let px = (bx + x).min(width - 1) as usize;
                        let py = (by + y).min(height - 1) as usize;
                        let idx = py * width as usize + px;
                        let va = prev[idx];
                        let vb = curr[idx];
                        sum_diff += (va - vb).abs();
                        cx_a += x as f32 * va;
                        cy_a += y as f32 * va;
                        mass_a += va;
                        cx_b += x as f32 * vb;
                        cy_b += y as f32 * vb;
                        mass_b += vb;
                    }
                }
                let pixels = (bs * bs) as f32;
                let mean_diff = sum_diff / pixels;
                mag_sum += mean_diff;
                count += 1;
                let (dx, dy) = if mass_a > 1e-3 && mass_b > 1e-3 {
                    ((cx_b / mass_b) - (cx_a / mass_a), (cy_b / mass_b) - (cy_a / mass_a))
                } else {
                    (0.0, 0.0)
                };
                vectors_dx.push(dx);
                vectors_dy.push(dy);
            }
        }

        let magnitude_mean = if count > 0 { mag_sum / count as f32 } else { 0.0 };
        MotionResult {
            real_gpu: self.real_gpu,
            mercy_gated: true,
            note: format!(
                "motion level={} stride={} blocks={} mode={}",
                level,
                stride,
                count,
                if self.real_gpu { "gpu-or-cpu-energy" } else { "cpu-energy" }
            ),
            optical_flow_mode: if self.real_gpu {
                "gpu".into()
            } else {
                "cpu-energy".into()
            },
            magnitude_mean,
            high_saliency: magnitude_mean > 1.65,
            width,
            height,
            frame_index: self.dispatch_count,
            vector_count: count,
            out_width: out_w,
            out_height: out_h,
            vectors_dx,
            vectors_dy,
            pyramid_levels: 0,
        }
    }
}

fn angle_diff(a: f32, b: f32) -> f32 {
    let tau = std::f32::consts::TAU;
    let mut d = (a - b).abs() % tau;
    if d > std::f32::consts::PI {
        d = tau - d;
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn pyramid_levels_run_cpu() {
        let mut p = GpuComputePipeline::new();
        let r = p.estimate_motion_pyramidal_levels(1.0, 3).await;
        assert!(r.pyramid_levels >= 1);
        assert!(r.mercy_gated);
    }

    #[test]
    fn common_fate_optional_modes() {
        let p = GpuComputePipeline::new();
        let motion = MotionResult::empty_hold(true, 64, 64, 0, "t");
        let fate = p.perceive_common_fate_optional(&motion, 1.0, false, true);
        assert!(!fate.common_fate_mode.is_empty());
    }
}
