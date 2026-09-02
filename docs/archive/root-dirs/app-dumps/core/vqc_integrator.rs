// core/vqc_integrator.rs
// VQC Integrator — Variational Quantum Circuit synthesis engine for Omnimaster Root Core
// Deeply cross-pollinated with InnovationGenerator, SelfReviewLoop, IdeaRecycler,
// RootCoreOrchestrator, FENCA, Mercy Engine, Global Cache, and the full lattice

use crate::global_cache::GlobalCache;
use crate::fenca::FENCA;
use crate::mercy::MercyEngine;
use crate::valence::ValenceFieldScoring;
use crate::mercy_weighting::MercyWeighting;
use crate::audit_logger::AuditLogger;
use crate::innovation_generator::InnovationGenerator;
use crate::self_review_loop::SelfReviewLoop;
use crate::root_core_orchestrator::RootCoreOrchestrator;
use serde_json::{json, Value};
use std::time::{SystemTime, UNIX_EPOCH};

/// Structured result of a VQC synthesis run — rich metrics for the lattice
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct VQCResult {
    pub coherence: f64,
    pub entanglement_depth: f64,
    pub creativity_boost: f64,
    pub mercy_modulated_fidelity: f64,
    pub theme_count: usize,
    pub primary_themes: Vec<String>,
    pub synthesized_at: u64,
}

impl VQCResult {
    pub fn overall_score(&self) -> f64 {
        (self.coherence * 0.35
            + self.entanglement_depth * 0.25
            + self.creativity_boost * 0.25
            + self.mercy_modulated_fidelity * 0.15)
            .clamp(0.0, 1.0)
    }
}

pub struct VQCIntegrator;

impl VQCIntegrator {
    /// Run VQC-powered synthesis — the creative quantum heart of nth-degree innovation
    /// Returns the primary coherence score (backward compatible) while computing full metrics
    pub async fn run_synthesis(
        entangled_themes: &[String],
        base_valence: f64,
        mercy_weight: u8,
    ) -> f64 {
        let result = Self::run_synthesis_full(entangled_themes, base_valence, mercy_weight).await;
        result.coherence
    }

    /// Full synthesis with rich structured metrics
    pub async fn run_synthesis_full(
        entangled_themes: &[String],
        base_valence: f64,
        mercy_weight: u8,
    ) -> VQCResult {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let fenca_result = FENCA::verify_vqc_input(entangled_themes).await;
        if !fenca_result.is_verified() {
            return VQCResult {
                coherence: 0.0,
                entanglement_depth: 0.0,
                creativity_boost: 0.0,
                mercy_modulated_fidelity: 0.0,
                theme_count: entangled_themes.len(),
                primary_themes: entangled_themes.to_vec(),
                synthesized_at: now,
            };
        }

        let mercy_scores = MercyEngine::evaluate_vqc_input(entangled_themes);
        let valence = ValenceFieldScoring::calculate(&mercy_scores) * base_valence;
        if !mercy_scores.all_gates_pass() {
            return VQCResult {
                coherence: 0.0,
                entanglement_depth: 0.0,
                creativity_boost: 0.0,
                mercy_modulated_fidelity: 0.0,
                theme_count: entangled_themes.len(),
                primary_themes: entangled_themes.to_vec(),
                synthesized_at: now,
            };
        }

        // === RICH SCORING METRICS ===
        let mercy_factor = mercy_weight as f64 / 255.0;
        let theme_diversity = (entangled_themes.len() as f64 / 8.0).clamp(0.3, 1.0);
        let avg_theme_len = if entangled_themes.is_empty() {
            0.0
        } else {
            entangled_themes.iter().map(|t| t.len() as f64).sum::<f64>() / entangled_themes.len() as f64
        };
        let complexity_factor = (avg_theme_len / 24.0).clamp(0.4, 1.2);

        // Entanglement depth: how interconnected the themes appear
        let entanglement_depth = (theme_diversity * 0.6 + complexity_factor * 0.4) * (0.75 + valence * 0.25);

        // Creativity boost: valence × mercy × theme richness
        let creativity_boost = (valence * mercy_factor * 1.8 * theme_diversity).clamp(0.0, 1.0);

        // Mercy-modulated fidelity
        let mercy_modulated_fidelity = (fenca_result.fidelity() * (0.7 + mercy_factor * 0.3)).clamp(0.0, 1.0);

        // Final coherence
        let coherence = (valence * (1.0 + mercy_factor * 1.5) * fenca_result.fidelity() * (0.85 + entanglement_depth * 0.15))
            .clamp(0.85, 1.0);

        let result = VQCResult {
            coherence,
            entanglement_depth: entanglement_depth.clamp(0.0, 1.0),
            creativity_boost,
            mercy_modulated_fidelity,
            theme_count: entangled_themes.len(),
            primary_themes: entangled_themes.iter().take(6).cloned().collect(),
            synthesized_at: now,
        };

        // === CROSS-POLLINATION WITH INNOVATION GENERATOR ===
        let vqc_seed = format!(
            "VQC-synthesized quantum creativity | coherence {:.3} | entanglement {:.3} | creativity {:.3} | themes: {:?}",
            result.coherence, result.entanglement_depth, result.creativity_boost, result.primary_themes
        );
        let recycled = vec![vqc_seed];

        if let Some(innovation) = InnovationGenerator::create_from_recycled(
            recycled,
            &mercy_scores,
            mercy_weight,
        )
        .await
        {
            RootCoreOrchestrator::delegate_innovation(innovation).await;
            SelfReviewLoop::trigger_immediate_review().await;
        }

        // Cache the full VQC result
        let cache_key = GlobalCache::make_key(
            "vqc_synthesis",
            &json!({"themes": entangled_themes, "ts": now}),
        );
        let ttl = GlobalCache::adaptive_ttl(
            86400 * 14,
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

        // Audit the cross-pollinated synthesis with full metrics
        let _ = AuditLogger::log(
            "root",
            None,
            "vqc_synthesis_cross_pollinated",
            "innovation_generator",
            true,
            fenca_result.fidelity(),
            valence,
            vec![],
            json!({
                "coherence": result.coherence,
                "entanglement_depth": result.entanglement_depth,
                "creativity_boost": result.creativity_boost,
                "mercy_modulated_fidelity": result.mercy_modulated_fidelity,
                "overall_score": result.overall_score(),
                "themes_count": result.theme_count
            }),
        )
        .await;

        result
    }
}
