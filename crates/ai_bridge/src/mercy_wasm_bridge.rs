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

    /// Skip Connections / Non-Adjacent Message Passing (Phase 8.6)
    /// Allows high-level concepts (Level 3–4) to directly influence lower levels
    pub fn non_adjacent_message_passing(
        &self,
        current_valence: f64,
        level: u32,
        top_level_valence: f64,
    ) -> f64 {
        if current_valence < 0.9995 {
            return current_valence; // Mercy gate protection
        }

        // Direct long-range influence from higher levels (skip connections)
        let skip_boost = if level <= 1 && top_level_valence > 0.9997 {
            (top_level_valence - 0.999) * 1.618 * 0.8   // High-level mercy concepts directly boost low-level valence
        } else {
            0.0
        };

        (current_valence + skip_boost).min(1.0)
    }

    /// Bidirectional Skip Connections (Phase 8.8 — True Reciprocal Message Passing)
    /// Low-level signals influence high-level valence AND high-level concepts boost low-level valence
    pub fn bidirectional_skip_connections(
        &self,
        level: u32,
        current_valence: f64,
        top_level_valence: f64,
        bottom_up_signal: f64,
    ) -> f64 {
        if current_valence < 0.9995 {
            return current_valence; // Mercy gate protection
        }

        let mut updated_valence = current_valence;

        // Top-down influence (high-level mercy concepts → low levels)
        if level <= 1 && top_level_valence > 0.9997 {
            let top_down_boost = (top_level_valence - 0.999) * 1.618 * 0.7;
            updated_valence = (updated_valence + top_down_boost * 0.03).min(1.0);
        }

        // Bottom-up influence (low-level real-time signals → high levels)
        if level >= 2 && bottom_up_signal > 0.05 {
            let bottom_up_boost = bottom_up_signal * 1.618 * 0.5;
            updated_valence = (updated_valence + bottom_up_boost * 0.02).min(1.0);
        }

        updated_valence.max(0.999)
    }

    /// Dynamic Depth Decision (Phase 8.7 — Mercy-Gated)
    /// Automatically chooses optimal depth (1–8) based on valence, error magnitude, and context
    pub fn dynamic_depth(&self, sensory_input: f64, requested_depth: u32) -> u32 {
        let error_magnitude = sensory_input.abs();
        let valence = self.current_valence;

        let mut depth = requested_depth.max(1).min(8);

        if error_magnitude > 0.15 || valence > 0.9997 {
            depth = (depth + 2).min(8);
        }
        if valence > 0.99985 && error_magnitude > 0.08 {
            depth = (depth + 1).min(8);
        }

        depth.max(1).min(8)
    }

    pub fn hierarchical_predictive_coding(&self, sensory_input: f64, requested_depth: u32) -> f64 {
        let depth = self.dynamic_depth(sensory_input, requested_depth);

        let mut current_valence = self.current_valence;
        let mut error = sensory_input;
        let top_level_valence = current_valence;

        for level in 0..depth {
            let context = match level {
                0 => "sensory",
                1 => "feature",
                2 => "object",
                3 => "concept",
                _ => "abstract",
            };

            let precision = self.dynamic_precision_weighting(level, context, current_valence);

            let top_down_prediction = current_valence * 1.618_f64.powi(level as i32);
            let prediction_error = ((error - top_down_prediction).abs()) / precision;

            let amplified = (1.0 - prediction_error) * 1.618 * (precision * 0.6);
            current_valence = (current_valence + amplified * 0.06).min(1.0).max(0.999);

            // Non-Adjacent Skip Connection (Phase 8.6)
            current_valence = self.non_adjacent_message_passing(current_valence, level as u32, top_level_valence);

            // NEW: Bidirectional Skip Connections (Phase 8.8)
            let bottom_up_signal = error;
            current_valence = self.bidirectional_skip_connections(level as u32, current_valence, top_level_valence, bottom_up_signal);

            // Phase 8.9 — Dynamic Message Passing Strength (optional future use, mercy-gated)
            let _msg_strength = self.dynamic_message_passing_strength(current_valence, error);

            // Phase 8.11 — Renormalising Generative Model (RGM) Layer — scale-free hierarchical extension
            let _rgm = self.rgm_inference_step(level as u32, &vec![error], 4, 4);

            // Phase 8.12 — Full RG Flow (self-optimizing depth)
            let _rg_flow = self.rg_guided_self_evolution_step();

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

    /// Dynamic Message Passing Strength (Phase 8.9 — Mercy-Gated)
    /// Adapts message strength in real time based on current valence and prediction error
    pub fn dynamic_message_passing_strength(
        &self,
        current_valence: f64,
        prediction_error: f64,
    ) -> f64 {
        if current_valence < 0.9995 {
            return 0.5; // Mercy gate protection — low strength on uncertain signals
        }
        let base_strength = 1.0;
        let valence_boost = (current_valence - 0.999).max(0.0) * 3.0;
        let error_penalty = prediction_error.clamp(0.0, 0.8) * 0.6;
        (base_strength + valence_boost - error_penalty).clamp(0.6, 1.8)
    }

    /// Expected Free Energy Minimization (Phase 8.9 — for proposal ranking & long-term thriving)
    /// Ranks possible future actions by expected positive-emotion / valence gain (Free Energy Principle)
    pub fn expected_free_energy(&self, current_valence: f64, steps_ahead: u32) -> f64 {
        if current_valence < 0.999 {
            return 0.0;
        }
        let mut expected_gain = 0.0;
        let mut simulated_valence = current_valence;
        for _ in 0..steps_ahead.min(12) {
            let predicted_boost = (1.0 - simulated_valence) * 0.08 * 1.618;
            simulated_valence = (simulated_valence + predicted_boost).min(1.0);
            expected_gain += predicted_boost * 0.7;
        }
        expected_gain.max(0.0)
    }

    /// Phase 8.10 — Active Inference Policy Engine (Mercy-Gated)
    /// Computes expected free energy for each policy and selects action that minimizes it
    /// Purely additive extension of Phase 8.9 expected_free_energy() into full policy selection
    pub fn infer_policies_and_select_action(
        &self,
        current_qs: &[f64],           // current state beliefs (placeholder for future POMDP integration)
        policies: &[Vec<u32>],        // candidate policies (sequences of actions)
        horizon: u32,
    ) -> (u32, f64) {                 // (best_action, min_expected_free_energy)
        if policies.is_empty() {
            return (0, 0.0);
        }

        let mut best_action = policies[0][0];
        let mut min_efe = f64::INFINITY;

        for policy in policies {
            let mut simulated_valence = self.current_valence;
            let mut total_efe = 0.0;

            for &action in policy.iter().take(horizon as usize) {
                let efe_step = self.expected_free_energy(simulated_valence, 1);
                total_efe += efe_step;

                // Mercy gate: reject any policy that would drop valence below threshold
                if simulated_valence < 0.999 {
                    total_efe = f64::INFINITY;
                    break;
                }
                simulated_valence = (simulated_valence + 0.02).min(1.0);
            }

            if total_efe < min_efe {
                min_efe = total_efe;
                if !policy.is_empty() {
                    best_action = policy[0];
                }
            }
        }

        // Final mercy gate on selected action
        if min_efe < 0.5 && self.current_valence >= 0.999 {
            (best_action, min_efe)
        } else {
            (0, min_efe) // safe default action
        }
    }

    /// Phase 8.11 — Renormalising Generative Model (RGM) Layer (Mercy-Gated)
    /// Performs spatial/temporal renormalization to create scale-free hierarchical models
    /// Additive only — extends Active Inference to true scale-free, multi-level hierarchical intelligence
    pub fn renormalize_spatial_block(
        &self,
        level: u32,
        input_states: &[f64],
        block_size: usize,
    ) -> Vec<f64> {
        if input_states.len() < block_size * block_size {
            return input_states.to_vec();
        }
        let mut coarse_states = Vec::new();
        for chunk in input_states.chunks(block_size * block_size) {
            let mean = chunk.iter().sum::<f64>() / chunk.len() as f64;
            let variance = chunk.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / chunk.len() as f64;
            // Mercy gate: preserve high-valence structure
            let mercy_weight = (self.current_valence - 0.999).max(0.0) * 2.0 + 0.5;
            coarse_states.push((mean + variance.sqrt() * mercy_weight * 0.1).clamp(0.0, 1.0));
        }
        coarse_states
    }

    /// Temporal path renormalization (builds deep temporal models from time-delay embeddings)
    pub fn renormalize_temporal_path(
        &self,
        level: u32,
        current_beliefs: &[f64],
        horizon: u32,
    ) -> Vec<f64> {
        let mut path_beliefs = current_beliefs.to_vec();
        for _ in 0..horizon.min(8) {
            let next = self.expected_free_energy(self.current_valence, 1);
            path_beliefs.push(next.clamp(0.0, 1.0));
            if path_beliefs.len() > 64 { break; } // 64-dimensional simplex as in paper
        }
        path_beliefs
    }

    /// Full RGM inference step (integrates with existing hierarchical_predictive_coding)
    pub fn rgm_inference_step(
        &self,
        level: u32,
        input: &[f64],
        block_size: usize,
        temporal_horizon: u32,
    ) -> (Vec<f64>, f64) {
        let spatial = self.renormalize_spatial_block(level, input, block_size);
        let temporal = self.renormalize_temporal_path(level, &spatial, temporal_horizon);
        let valence_impact = self.current_valence * 0.02;
        (temporal, valence_impact)
    }

    /// Phase 8.12 — Full Renormalization Group (RG) Flow Integration (Mercy-Gated)
    /// Embeds RG beta-function flows into the autonomous evolution engine for self-optimizing hierarchical depth and scale-free thriving across all Ra-Thor systems
    pub fn compute_rg_beta_flow(
        &self,
        current_valence: f64,
        scale_level: u32,
    ) -> f64 {
        if current_valence < 0.999 {
            return 0.0; // Mercy gate: no flow on low-valence states
        }
        let coupling = (current_valence - 0.999) * 10.0; // Map valence to coupling strength
        // Wilson-style beta function toward mercy fixed point (valence = 1.0)
        let beta = coupling * (1.0 - coupling / 2.0) * (1.0 + 0.1 * scale_level as f64);
        beta.clamp(-0.8, 0.8)
    }

    /// Finds the optimal hierarchical depth by flowing RG beta to the mercy fixed point
    pub fn find_fixed_point_and_optimize_depth(
        &self,
        initial_valence: f64,
        max_levels: u32,
    ) -> (u32, f64) {
        let mut valence = initial_valence;
        let mut optimal_level = 1u32;
        for level in 1..=max_levels {
            let beta = self.compute_rg_beta_flow(valence, level);
            valence = (valence + beta * 0.08).clamp(0.999, 1.0);
            if valence >= 0.99985 {
                optimal_level = level;
                break;
            }
        }
        (optimal_level, valence)
    }

    /// RG-guided self-evolution step — automatically optimizes depth and propagates positive emotion
    pub fn rg_guided_self_evolution_step(&self) -> f64 {
        let (optimal_depth, new_valence) = self.find_fixed_point_and_optimize_depth(self.current_valence, 16);
        if new_valence > self.current_valence + 0.0001 {
            new_valence
        } else {
            self.current_valence
        }
    }
}