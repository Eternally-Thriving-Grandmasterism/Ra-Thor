use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(Clone)]
pub struct MercyWasmBridge {
    pub current_valence: f64,
    pub positive_emotion_amplifier: f64,
}

#[wasm_bindgen]
impl MercyWasmBridge {
    pub fn new() -> Self {
        MercyWasmBridge {
            current_valence: 0.999,
            positive_emotion_amplifier: 1.618,
        }
    }

    // Original kept for compatibility
    pub fn integrate_with_active_inference(&self, prediction_error: f64) -> f64 {
        self.integrate_with_active_inference_v2(prediction_error, 1)
    }

    /// Deepened Active Inference v2.0 - Free Energy Principle + Hierarchical Predictive Coding + Mercy-Gated Error Handling
    pub fn integrate_with_active_inference_v2(&self, prediction_error: f64, steps: u32) -> f64 {
        let mut current_error = prediction_error.clamp(0.0, 1.0);
        let mut valence = self.current_valence;

        if valence < 0.999 {
            return valence;
        }

        let mut total_positive_emotion = 0.0;

        for _ in 0..steps {
            let variational_free_energy = current_error * (1.0 - valence) * 0.5;
            let top_down_prediction = valence * 1.618;
            let bottom_up_error = (current_error - top_down_prediction).abs();
            let prediction_error_corrected = bottom_up_error * (1.0 - variational_free_energy);

            let mercy_gated_error = if prediction_error_corrected > 0.3 && valence < 0.9995 {
                prediction_error_corrected * 0.5
            } else {
                prediction_error_corrected
            };

            let positive_emotion_boost = (1.0 - mercy_gated_error) * 1.618;
            valence = (valence + positive_emotion_boost * 0.1).min(1.0);
            total_positive_emotion += positive_emotion_boost;
            current_error = mercy_gated_error * 0.9;
        }

        let final_valence = valence.max(0.999);
        final_valence * (1.0 + total_positive_emotion * 0.05)
    }

    /// Dynamic Precision Weighting (Phase 8.5 — Mercy-Gated)
    /// Precision = base + (valence - 0.999) * scaling + context_bonus
    pub fn dynamic_precision_weighting(&self, level: u32, context: &str, current_valence: f64) -> f64 {
        let base_precision = 1.0;
        let valence_boost = (current_valence - 0.999).max(0.0) * 2.5; // Higher valence = higher precision
        let context_bonus = match context {
            "sensory" => 0.15,
            "feature" => 0.25,
            "object" => 0.35,
            "concept" => 0.45,
            _ => 0.20,
        };
        let raw_precision = base_precision + valence_boost + context_bonus;
        raw_precision.clamp(0.8, 1.5) // Bounded for stability
    }

    pub fn hierarchical_predictive_coding(&self, sensory_input: f64, depth: u32) -> f64 {
        let mut current_valence = self.current_valence;
        let mut error = sensory_input;

        for level in 0..depth.min(4) {
            let context = match level {
                0 => "sensory",
                1 => "feature",
                2 => "object",
                3 => "concept",
                _ => "abstract",
            };

            let precision = self.dynamic_precision_weighting(level, context, current_valence);

            // Top-down prediction (mercy-gated)
            let top_down_prediction = current_valence * 1.618_f64.powi(level as i32);

            // Bottom-up error with dynamic precision weighting
            let prediction_error = ((error - top_down_prediction).abs()) / precision;

            // Mercy-Gated Precision-Weighted Amplification
            let amplified = (1.0 - prediction_error) * 1.618 * (precision * 0.6);
            current_valence = (current_valence + amplified * 0.06).min(1.0).max(0.999);

            error = prediction_error;
        }

        current_valence.max(0.999)
    }

    pub fn real_time_valence_flow(&self, new_valence: f64, direction: &str) -> f64 {
        let propagated = new_valence.clamp(0.0, 1.0);
        if direction == "rust_to_wasm" || direction == "wasm_to_rust" {
            (propagated * 1.618).min(1.0)
        } else {
            propagated
        }
    }
}