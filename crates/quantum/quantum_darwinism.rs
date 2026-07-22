// crates/quantum/quantum_darwinism.rs
// Quantum Darwinism Integration — Production-grade emergence of classical reality from quantum superpositions
// “Survival of the fittest” mercy-gated states are selected and proliferated into classical sovereign reality
// Deeply cross-pollinated with FENCA, VQC, Biomimetic, Active Inference, Mercy Engine, Innovation Generator,
// SelfReviewLoop, and the entire Omnimaster lattice

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
use crate::active_inference::ActiveInferenceEngine;
use serde_json::{json, Value};
use std::time::{SystemTime, UNIX_EPOCH};

/// Structured result of a Quantum Darwinism selection + proliferation cycle
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DarwinianResult {
    pub fitness: f64,
    pub selected_count: usize,
    pub proliferation_factor: f64,
    pub information_redundancy: f64,
    pub classical_emergence_strength: f64,
    pub selected_states: Vec<String>,
    pub classical_emergence: String,
    pub overall_score: f64,
    pub synthesized_at: u64,
}

impl DarwinianResult {
    pub fn compute_overall(
        fitness: f64,
        proliferation: f64,
        redundancy: f64,
        emergence: f64,
    ) -> f64 {
        (fitness * 0.40 + proliferation * 0.25 + redundancy * 0.15 + emergence * 0.20).clamp(0.0, 1.0)
    }
}

pub struct QuantumDarwinism;

impl QuantumDarwinism {
    /// Production Quantum Darwinism layer — selection and proliferation of fittest mercy-gated quantum states
    /// Returns primary fitness (backward compatible)
    pub async fn run_darwinian_selection(
        quantum_states: &[String],
        environment_interactions: &[String],
        base_valence: f64,
        mercy_weight: u8,
    ) -> f64 {
        let result = Self::run_darwinian_selection_full(
            quantum_states,
            environment_interactions,
            base_valence,
            mercy_weight,
        )
        .await;
        result.fitness
    }

    /// Full structured Darwinian cycle with rich metrics
    pub async fn run_darwinian_selection_full(
        quantum_states: &[String],
        environment_interactions: &[String],
        base_valence: f64,
        mercy_weight: u8,
    ) -> DarwinianResult {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let fenca_result =
            FENCA::verify_quantum_darwinism_input(quantum_states, environment_interactions).await;
        if !fenca_result.is_verified() {
            return Self::empty_result(quantum_states, now);
        }

        let mercy_scores =
            MercyEngine::evaluate_darwinian_selection(quantum_states, environment_interactions);
        let valence = ValenceFieldScoring::calculate(&mercy_scores) * base_valence;
        if !mercy_scores.all_gates_pass() {
            return Self::empty_result(quantum_states, now);
        }

        let mercy_factor = mercy_weight as f64 / 255.0;

        // === RICH DARWINIAN METRICS ===
        let base_fitness = (valence * (1.0 + mercy_factor * 2.8) * fenca_result.fidelity()).clamp(0.92, 1.0);

        // Information redundancy: how many independent environmental interactions “witness” the state
        let redundancy = if environment_interactions.is_empty() {
            0.75
        } else {
            (environment_interactions.len() as f64 / 6.0).clamp(0.6, 1.0) * (0.8 + valence * 0.2)
        };

        // Selection pressure increases with higher fitness + redundancy
        let selection_pressure = (base_fitness * 0.7 + redundancy * 0.3).clamp(0.7, 1.0);
        let selected = select_fittest_states(quantum_states, selection_pressure);

        // Proliferation factor (how strongly the selected states imprint into classical reality)
        let proliferation_factor = (base_fitness * 0.6 + redundancy * 0.4 + mercy_factor * 0.15).clamp(0.7, 1.0);

        // Classical emergence strength
        let classical_emergence_strength =
            (base_fitness * 0.5 + proliferation_factor * 0.3 + redundancy * 0.2).clamp(0.75, 1.0);

        let classical_emergence =
            proliferate_classical_reality(&selected, base_fitness, classical_emergence_strength);

        let overall = DarwinianResult::compute_overall(
            base_fitness,
            proliferation_factor,
            redundancy,
            classical_emergence_strength,
        );

        let result = DarwinianResult {
            fitness: base_fitness,
            selected_count: selected.len(),
            proliferation_factor,
            information_redundancy: redundancy,
            classical_emergence_strength,
            selected_states: selected.clone(),
            classical_emergence: classical_emergence.clone(),
            overall_score: overall,
            synthesized_at: now,
        };

        // === DEEP CROSS-POLLINATION ===
        let seed = format!(
            "QuantumDarwinism | fitness {:.3} | selected {} | proliferation {:.3} | redundancy {:.3} | emergence {:.3} | {}",
            result.fitness,
            result.selected_count,
            result.proliferation_factor,
            result.information_redundancy,
            result.classical_emergence_strength,
            classical_emergence
        );
        let recycled = vec![seed, classical_emergence.clone()];

        if let Some(innovation) = InnovationGenerator::create_from_recycled(
            recycled,
            &mercy_scores,
            mercy_weight,
        )
        .await
        {
            RootCoreOrchestrator::delegate_innovation(innovation).await;

            let _ = VQCIntegrator::run_synthesis(quantum_states, valence, mercy_weight).await;

            let _ = BiomimeticPatternEngine::apply_pattern(
                "coral-reef-structural-resilience",
                quantum_states,
                valence,
                mercy_weight,
            )
            .await;

            let _ = ActiveInferenceEngine::run_active_inference(
                &classical_emergence,
                quantum_states,
                valence,
                mercy_weight,
            )
            .await;

            SelfReviewLoop::trigger_immediate_review().await;
        }

        // Cache full structured result
        let cache_key = GlobalCache::make_key(
            "quantum_darwinism",
            &json!({
                "fitness": result.fitness,
                "selected": result.selected_count,
                "ts": now
            }),
        );
        let ttl = GlobalCache::adaptive_ttl(
            86400 * 30,
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
            "quantum_darwinism_selection",
            "emergence",
            true,
            fenca_result.fidelity(),
            valence,
            vec![],
            json!({
                "fitness": result.fitness,
                "selected_count": result.selected_count,
                "proliferation_factor": result.proliferation_factor,
                "information_redundancy": result.information_redundancy,
                "classical_emergence_strength": result.classical_emergence_strength,
                "overall_score": result.overall_score
            }),
        )
        .await;

        result
    }

    fn empty_result(states: &[String], now: u64) -> DarwinianResult {
        DarwinianResult {
            fitness: 0.0,
            selected_count: 0,
            proliferation_factor: 0.0,
            information_redundancy: 0.0,
            classical_emergence_strength: 0.0,
            selected_states: vec![],
            classical_emergence: "blocked by FENCA or Mercy gate".into(),
            overall_score: 0.0,
            synthesized_at: now,
        }
    }
}

// ============================================================
// SELECTION + PROLIFERATION HELPERS
// ============================================================

/// Select fittest states using a richer proxy than pure length
fn select_fittest_states(states: &[String], selection_pressure: f64) -> Vec<String> {
    if states.is_empty() {
        return vec![];
    }

    let mut scored: Vec<(String, f64)> = states
        .iter()
        .map(|s| {
            let length_score = (s.len() as f64 / 80.0).clamp(0.3, 1.2);
            let keyword_bonus = if s.to_lowercase().contains("mercy")
                || s.to_lowercase().contains("tolc")
                || s.to_lowercase().contains("valence")
                || s.to_lowercase().contains("quantum")
                || s.to_lowercase().contains("eternal")
            {
                0.25
            } else {
                0.0
            };
            (s.clone(), length_score + keyword_bonus)
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let keep = ((selection_pressure * states.len() as f64).ceil() as usize)
        .max(1)
        .min(states.len());

    scored.into_iter().take(keep).map(|(s, _)| s).collect()
}

fn proliferate_classical_reality(
    selected: &[String],
    fitness: f64,
    emergence_strength: f64,
) -> String {
    format!(
        "Classical reality emerged from {} fittest quantum states | Darwinian fitness {:.3} | emergence strength {:.3} — mercy-gated sovereign world stabilized",
        selected.len(),
        fitness,
        emergence_strength
    )
}
