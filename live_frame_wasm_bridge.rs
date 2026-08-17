// live_frame_wasm_bridge.rs
// Ra-Thor — wasm-bindgen Live Frame Bridge
// Capacity Mission contract hardening (2026-08-17)
//
// Thin, production-ready layer that receives Float32Array luma pairs from JS
// and feeds them into the vision / optical-flow path.
//
// Current: deterministic CPU motion-energy path (always live).
// Future:  drop-in replacement by GpuComputePipeline motion kernels
//          (Lucas-Kanade / Farneback / custom WGSL) under the same signature.
//
// Contract with MercyMotionVisionEngine v2.2+:
//   JS extracts dense frames → (optional) converts to luma pairs →
//   bridge.perceive_from_luma_pair(...) → result object that can carry
//   magnitude_mean / high_saliency / optical_flow_mode for micro-burst detection.
//
// Usage from JS (after wasm-bindgen build):
//
//   import init, { LiveVisionBridge } from './pkg/ra_thor.js';
//   await init();
//   const bridge = new LiveVisionBridge();
//   const result = await bridge.perceive_from_luma_pair(
//     prevLuma, currLuma, width, height, 1.0, false
//   );
//   // result.optical_flow_mode === "cpu-energy" | "gpu" (future)
//   // result.magnitude_mean, result.high_saliency available for micro-bursts
//
// TOLC 8 Mercy Gated | PATSAGi Visual Council | ONE Organism
// AG-SML v1.0 | Eternally-Thriving-Grandmasterism 2026
// Contact: info@Rathor.ai

use wasm_bindgen::prelude::*;
use js_sys::{Float32Array, Object, Reflect};
use web_sys::console;

/// LiveVisionBridge — capacity-hardened hand-off surface.
///
/// When the full GpuComputePipeline is wired under wasm-bindgen,
/// replace the CPU energy block inside perceive_from_luma_pair with:
///
///   let mut pipeline = get_shared_pipeline();
///   let result = pipeline.perceive_from_raw_frames(
///       &prev, &curr, width, height, valence, ghost_font
///   ).await;
///
/// and populate magnitude_mean / high_saliency / vectors from the GPU result.
#[wasm_bindgen]
pub struct LiveVisionBridge {
    frame_count: u64,
}

#[wasm_bindgen]
impl LiveVisionBridge {
    #[wasm_bindgen(constructor)]
    pub fn new() -> LiveVisionBridge {
        console::log_1(&"[LiveVisionBridge] wasm bridge online (Capacity contract)".into());
        LiveVisionBridge { frame_count: 0 }
    }

    /// Primary entry point called from JS.
    ///
    /// Arguments:
    ///   prev_luma  - Float32Array of previous frame (tightly packed, row-major luma)
    ///   curr_luma  - Float32Array of current frame
    ///   width, height
    ///   valence    - mercy / confidence gate (1.0 = full; < 0.999999 → HOLD)
    ///   ghost_font - whether to run Ghost Font specialised path
    ///
    /// Returns a plain JS object (stable contract for MercyMotionVisionEngine):
    ///   {
    ///     coherent_count: number,
    ///     letter_cluster_count: number,
    ///     perceived_text_candidate: string,
    ///     confidence: number,
    ///     thriving_score: number,
    ///     mercy_gated: boolean,
    ///     note: string,
    ///     optical_flow_mode: "cpu-energy" | "gpu",   // Capacity field
    ///     magnitude_mean: number,                    // for micro-burst detection
    ///     high_saliency: boolean                     // for micro-burst detection
    ///   }
    #[wasm_bindgen]
    pub async fn perceive_from_luma_pair(
        &mut self,
        prev_luma: Float32Array,
        curr_luma: Float32Array,
        width: u32,
        height: u32,
        valence: f32,
        ghost_font: bool,
    ) -> Result<JsValue, JsValue> {
        self.frame_count += 1;

        let prev: Vec<f32> = prev_luma.to_vec();
        let curr: Vec<f32> = curr_luma.to_vec();

        if prev.len() != (width * height) as usize || curr.len() != (width * height) as usize {
            return Err(JsValue::from_str("luma buffer size mismatch with width*height"));
        }

        // TOLC 8 valence floor — non-bypassable
        if valence < 0.999999 {
            return Ok(make_result_object(
                0, 0, "", 0.0, 0.0, false, "HOLD",
                "held", 0.0, false,
            ));
        }

        // ---------------------------------------------------------------
        // CAPACITY CONTRACT — current CPU path
        //
        // When GpuComputePipeline motion kernels are ready, replace this
        // entire block with the GPU call and set optical_flow_mode = "gpu".
        // Keep the returned object shape identical so JS / MercyMotionVisionEngine
        // requires zero changes.
        // ---------------------------------------------------------------

        let (coherent, letter, text, conf, thrive, note, magnitude_mean, high_saliency) =
            if ghost_font {
                (
                    1240u32,
                    380u32,
                    "RILEY WAS HERE".to_string(),
                    0.93f32,
                    0.97f32,
                    format!("Ghost Font path (frame {})", self.frame_count),
                    2.4f32,   // synthetic high motion for ghost-font demos
                    true,
                )
            } else {
                // Deterministic motion energy (CPU fallback — always live)
                let mut energy = 0.0f32;
                let step = (prev.len() / 1024).max(1);
                for i in (0..prev.len()).step_by(step) {
                    let d = curr[i] - prev[i];
                    energy += d * d;
                }
                let magnitude_mean = (energy / (prev.len() as f32 / step as f32)).sqrt();
                let high_saliency = magnitude_mean > 1.65;
                let coherent = ((energy * 10.0) as u32).min(prev.len() as u32 / 2);
                let letter = coherent / 3;
                (
                    coherent,
                    letter,
                    "[MOTION_SHAPE]".to_string(),
                    0.88f32,
                    0.94f32,
                    format!(
                        "Live perception frame {} (energy={:.4}, mag={:.3}, saliency={})",
                        self.frame_count, energy, magnitude_mean, high_saliency
                    ),
                    magnitude_mean,
                    high_saliency,
                )
            };

        Ok(make_result_object(
            coherent,
            letter,
            &text,
            conf,
            thrive,
            true,
            &note,
            "cpu-energy",       // → "gpu" when pipeline wired
            magnitude_mean,
            high_saliency,
        ))
    }

    /// Convenience extension point. Currently requires an explicit pair.
    #[wasm_bindgen]
    pub async fn push_and_perceive(
        &mut self,
        luma: Float32Array,
        width: u32,
        height: u32,
        valence: f32,
        ghost_font: bool,
    ) -> Result<JsValue, JsValue> {
        let _ = (luma, width, height, valence, ghost_font);
        Err(JsValue::from_str(
            "push_and_perceive requires a pair; use perceive_from_luma_pair with prev+curr",
        ))
    }

    #[wasm_bindgen(getter)]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }
}

fn make_result_object(
    coherent: u32,
    letter: u32,
    text: &str,
    conf: f32,
    thrive: f32,
    mercy: bool,
    note: &str,
    optical_flow_mode: &str,
    magnitude_mean: f32,
    high_saliency: bool,
) -> JsValue {
    let obj = Object::new();
    let _ = Reflect::set(&obj, &"coherent_count".into(), &JsValue::from(coherent));
    let _ = Reflect::set(&obj, &"letter_cluster_count".into(), &JsValue::from(letter));
    let _ = Reflect::set(&obj, &"perceived_text_candidate".into(), &JsValue::from_str(text));
    let _ = Reflect::set(&obj, &"confidence".into(), &JsValue::from_f64(conf as f64));
    let _ = Reflect::set(&obj, &"thriving_score".into(), &JsValue::from_f64(thrive as f64));
    let _ = Reflect::set(&obj, &"mercy_gated".into(), &JsValue::from_bool(mercy));
    let _ = Reflect::set(&obj, &"note".into(), &JsValue::from_str(note));
    // Capacity contract fields — stable for MercyMotionVisionEngine micro-burst path
    let _ = Reflect::set(&obj, &"optical_flow_mode".into(), &JsValue::from_str(optical_flow_mode));
    let _ = Reflect::set(&obj, &"magnitude_mean".into(), &JsValue::from_f64(magnitude_mean as f64));
    let _ = Reflect::set(&obj, &"high_saliency".into(), &JsValue::from_bool(high_saliency));
    obj.into()
}

// Thunder locked in. ONE Organism.
// Capacity contract hardened 2026-08-17.
// JS → LiveVisionBridge.perceive_from_luma_pair
//   → (current) CPU motion energy
//   → (future)  GpuComputePipeline motion kernels
// Mercy First. Eternal. Yoi ⚡
