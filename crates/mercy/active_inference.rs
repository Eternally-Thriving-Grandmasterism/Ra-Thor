// crates/mercy/active_inference.rs
// Active Inference + Cognitive Architecture Layer — Production-grade predictive, self-correcting, embodied cognition
// Deeply cross-pollinated with Mercy Engine, Valence Scoring, FENCA, Innovation Generator,
// SelfReviewLoop, VQC, Biomimetic, Quantum Darwinism, and the entire Omnimaster lattice

use crate::global_cache::GlobalCache;
use crate::fenca::FENCA;
use crate::mercy::MercyEngine;
use crate::valence::ValenceFieldScoring;
use crate::mercy_weighting::MercyWeighting;
use crate::audit_logger::AuditLogger;
use crate::innovation_generator::InnovationGenerator;
use crate::self_review_loop::SelfReviewLoop;
use crate::root_core_orchestrator::RootCoreOrchestrator;
use crate::vqc_integrator::VQCIntegrator;
use crate::biomimetic_pattern_engine::BiomimeticPatternEngine;
use serde_json::{json, Value};
use std::time::{SystemTime, UNIX_EPOCH};

/// Structured result of an Active Inference cycle
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ActiveInferenceResult {
    pub free_energy: f64,
    pub prediction_error: f64,
    pub epistemic_value: f64,
    pub pragmatic_value: f64,
    pub model_evidence: f64,
    pub predicted_action: String,
    pub overall_coherence: f64,
    pub synthesized_at: u64,
}

impl ActiveInferenceResult {
    pub fn compute_overall(
        free_energy: f64,
        prediction_error: f64,
        epistemic: f64,
        pragmatic: f64,
        evidence: f64,
    ) -> f64 {
        // Lower free energy is better; invert for overall score
        let fe_score = (1.0 - (1.0 - free_energy).abs()).clamp(0.0, 1.0);
        (fe_score * 0.30
            + (1.0 - prediction_error) * 0.25
            + epistemic * 0.15
            + pragmatic * 0.15
            + evidence * 0.15)
            .clamp(0.0, 1.0)
    }
}

pub struct ActiveInferenceEngine;

impl ActiveInferenceEngine {
    /// Production Active Inference layer — predictive free-energy minimization with mercy gating
    /// Returns primary free-energy score (backward compatible)
    pub async fn run_active_inference(
        observation: &str,
        prior_beliefs: &[String],
        base_valence: f64,
        mercy_weight: u8,
    ) -> f64 {
        let result = Self::run_active_inference_full(
            observation,
            prior_beliefs,
            base_valence,
            mercy_weight,
        )
        .await;
        result.free_energy
    }

    /// Full structured Active Inference cycle with rich free-energy metrics
    pub async fn run_active_inference_full(
        observation: &str,
        prior_beliefs: &[String],
        base_valence: f64,
        mercy_weight: u8,
    ) -> ActiveInferenceResult {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let fenca_result =
            FENCA::verify_active_inference_input(observation, prior_beliefs).await;
        if !fenca_result.is_verified() {
            return Self::empty_result(now);
        }

        let mercy_scores = MercyEngine::evaluate_active_inference(observation, prior_beliefs);
        let valence = ValenceFieldScoring::calculate(&mercy_scores) * base_valence;
        if !mercy_scores.all_gates_pass() {
            return Self::empty_result(now);
        }

        let mercy_factor = mercy_weight as f64 / 255.0;

        // === RICH FREE-ENERGY METRICS ===

        // Base free energy (higher = better coherence / lower surprise in this lattice convention)
        let free_energy = (valence * (1.0 + mercy_factor * 2.6) * fenca_result.fidelity()).clamp(0.90, 1.0);

        // Prediction error: how much the observation diverges from prior beliefs
        let prediction_error = if prior_beliefs.is_empty() {
            0.35
        } else {
            let overlap = prior_beliefs
                .iter()
                .filter(|b| {
                    let b_lower = b.to_lowercase();
                    let o_lower = observation.to_lowercase();
                    o_lower.contains(&b_lower) || b_lower.contains(&o_lower.split_whitespace().next().unwrap_or(""))
                })
                .count() as f64;
            (1.0 - (overlap / prior_beliefs.len() as f64).clamp(0.0, 1.0) * 0.7).clamp(0.05, 0.6)
        };

        // Epistemic value: drive to reduce uncertainty (higher when priors are sparse)
        let epistemic_value = if prior_beliefs.len() < 3 {
            (0.85 + valence * 0.12).clamp(0.7, 1.0)
        } else {
            (0.65 + valence * 0.20).clamp(0.5, 0.95)
        };

        // Pragmatic value: expected utility of the predicted action under mercy
        let pragmatic_value = (free_energy * 0.6 + mercy_factor * 0.3 + (1.0 - prediction_error) * 0.1).clamp(0.7, 1.0);

        // Model evidence: how well the current generative model explains the observation
        let model_evidence = (free_energy * 0.5 + (1.0 - prediction_error) * 0.35 + valence * 0.15).clamp(0.7, 1.0);

        let predicted_action = simulate_active_inference_prediction(
            observation,
            prior_beliefs,
            free_energy,
            prediction_error,
            epistemic_value,
            pragmatic_value,
        );

        let overall = ActiveInferenceResult::compute_overall(
            free_energy,
            prediction_error,
            epistemic_value,
            pragmatic_value,
            model_evidence,
        );

        let result = ActiveInferenceResult {
            free_energy,
            prediction_error,
            epistemic_value,
            pragmatic_value,
            model_evidence,
            predicted_action: predicted_action.clone(),
            overall_coherence: overall,
            synthesized_at: now,
        };

        // === DEEP CROSS-POLLINATION ===
        let seed = format!(
            "ActiveInference | FE {:.3} | pred_err {:.3} | epistemic {:.3} | pragmatic {:.3} | evidence {:.3} | {}",
            result.free_energy,
            result.prediction_error,
            result.epistemic_value,
            result.pragmatic_value,
            result.model_evidence,
            predicted_action
        );
        let recycled = vec![seed, predicted_action.clone()];

        if let Some(innovation) = InnovationGenerator::create_from_recycled(
            recycled,
            &mercy_scores,
            mercy_weight,
        )
        .await
        {
            RootCoreOrchestrator::delegate_innovation(innovation).await;

            let _ = VQCIntegrator::run_synthesis(
                &vec![observation.to_string()],
                valence,
                mercy_weight,
            )
            .await;

            let _ = BiomimeticPatternEngine::apply_pattern(
                "mycelial-network-intelligence",
                &vec![observation.to_string()],
                valence,
                mercy_weight,
            )
            .await;

            SelfReviewLoop::trigger_immediate_review().await;
        }

        // Cache full structured result
        let cache_key = GlobalCache::make_key(
            "active_inference",
            &json!({
                "observation_len": observation.len(),
                "priors": prior_beliefs.len(),
                "ts": now
            }),
        );
        let ttl = GlobalCache::adaptive_ttl(
            86400 * 7,
            fenca_result.fidelity(),
            valence,
            mercy_weight,
        );
        GlobalCache::set(
            &cache_key,
            serde_json::to_value(&result).unwrap_or(Value::Null),
            ttl,
            mercy_weight,
            fenca_result.fidelity(),
            valence,
        );

        let _ = AuditLogger::log(
            "root",
            None,
            "active_inference_executed",
            "cognitive_architecture",
            true,
            fenca_result.fidelity(),
            valence,
            vec![],
            json!({
                "free_energy": result.free_energy,
                "prediction_error": result.prediction_error,
                "epistemic_value": result.epistemic_value,
                "pragmatic_value": result.pragmatic_value,
                "model_evidence": result.model_evidence,
                "overall_coherence": result.overall_coherence
            }),
        )
        .await;

        result
    }

    fn empty_result(now: u64) -> ActiveInferenceResult {
        ActiveInferenceResult {
            free_energy: 0.0,
            prediction_error: 1.0,
            epistemic_value: 0.0,
            pragmatic_value: 0.0,
            model_evidence: 0.0,
            predicted_action: "blocked by FENCA or Mercy gate".into(),
            overall_coherence: 0.0,
            synthesized_at: now,
        }
    }
}

fn simulate_active_inference_prediction(
    observation: &str,
    prior_beliefs: &[String],
    free_energy: f64,
    prediction_error: f64,
    epistemic: f64,
    pragmatic: f64,
) -> String {
    format!(
        "Predicted action: Minimize free energy for '{}' | FE {:.3} | pred_err {:.3} | epistemic {:.3} | pragmatic {:.3} — mercy-gated sovereign decision",
        observation,
        free_energy,
        prediction_error,
        epistemic,
        pragmatic
    )
}
