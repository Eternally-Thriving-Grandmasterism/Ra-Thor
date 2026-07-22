// core/biomimetic_pattern_engine.rs
// Biomimetic Pattern Engine — Living nature-inspired design core of the Omnimaster Root Core
// Deeply cross-pollinated with InnovationGenerator, VQCIntegrator, SelfReviewLoop, IdeaRecycler,
// RootCoreOrchestrator, FENCA, Mercy Engine, and the entire lattice

use crate::global_cache::GlobalCache;
use crate::fenca::FENCA;
use crate::mercy::MercyEngine;
use crate::valence::ValenceFieldScoring;
use crate::mercy_weighting::MercyWeighting;
use crate::audit_logger::AuditLogger;
use crate::innovation_generator::InnovationGenerator;
use crate::vqc_integrator::VQCIntegrator;
use crate::self_review_loop::SelfReviewLoop;
use crate::root_core_orchestrator::RootCoreOrchestrator;
use serde_json::{json, Value};
use std::time::{SystemTime, UNIX_EPOCH};

/// Structured biomimetic pattern — living, scored, ready for lattice application
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BiomimeticPattern {
    pub id: String,
    pub name: String,
    pub description: String,
    pub nature_source: String,
    pub domains: Vec<String>,
    pub coherence_floor: f64,
    pub mercy_affinity: f64,
}

/// Living catalog of high-fidelity biomimetic patterns
pub fn living_pattern_catalog() -> Vec<BiomimeticPattern> {
    vec![
        BiomimeticPattern {
            id: "avian-lev".into(),
            name: "avian-LEV-self-healing".into(),
            description: "Bird-inspired lift, variable geometry and self-healing airframe skin".into(),
            nature_source: "avian flight + feather regeneration".into(),
            domains: vec!["flight".into(), "self-healing".into(), "space".into(), "airframe".into()],
            coherence_floor: 0.94,
            mercy_affinity: 0.97,
        },
        BiomimeticPattern {
            id: "gecko-setae".into(),
            name: "gecko-setae-adhesion-pinnacle".into(),
            description: "Dry adhesion via hierarchical setae for reversible, residue-free grip".into(),
            nature_source: "gecko toe pads".into(),
            domains: vec!["adhesion".into(), "robotics".into(), "persistence".into()],
            coherence_floor: 0.95,
            mercy_affinity: 0.96,
        },
        BiomimeticPattern {
            id: "lotus".into(),
            name: "lotus-self-cleaning-regeneration".into(),
            description: "Superhydrophobic self-cleaning surface with regenerative micro-texture".into(),
            nature_source: "Nelumbo nucifera leaf".into(),
            domains: vec!["self-cleaning".into(), "surface".into(), "regeneration".into()],
            coherence_floor: 0.93,
            mercy_affinity: 0.98,
        },
        BiomimeticPattern {
            id: "fractal-528".into(),
            name: "fractal-528hz-asre-resonance".into(),
            description: "Fractal geometry + 528 Hz resonance for coherent information propagation".into(),
            nature_source: "fractal biology + solfeggio resonance".into(),
            domains: vec!["resonance".into(), "quantum".into(), "information".into(), "healing".into()],
            coherence_floor: 0.96,
            mercy_affinity: 0.99,
        },
        BiomimeticPattern {
            id: "mycelial".into(),
            name: "mycelial-network-intelligence".into(),
            description: "Distributed mycelial decision-making and resource allocation".into(),
            nature_source: "fungal mycelium networks".into(),
            domains: vec!["swarm".into(), "orchestration".into(), "distributed".into()],
            coherence_floor: 0.94,
            mercy_affinity: 0.97,
        },
        BiomimeticPattern {
            id: "spider-silk".into(),
            name: "spider-silk-tensile-strength".into(),
            description: "Extreme tensile strength-to-weight with controlled energy dissipation".into(),
            nature_source: "spider silk proteins".into(),
            domains: vec!["materials".into(), "structural".into(), "resilience".into()],
            coherence_floor: 0.93,
            mercy_affinity: 0.95,
        },
        BiomimeticPattern {
            id: "termite".into(),
            name: "termite-mound-ventilation".into(),
            description: "Passive climate regulation via geometric airflow channels".into(),
            nature_source: "termite mound architecture".into(),
            domains: vec!["ventilation".into(), "thermal".into(), "architecture".into()],
            coherence_floor: 0.92,
            mercy_affinity: 0.94,
        },
        BiomimeticPattern {
            id: "whale-fin".into(),
            name: "whale-fin-turbulence-control".into(),
            description: "Tubercles for delayed stall and efficient flow control".into(),
            nature_source: "humpback whale flipper".into(),
            domains: vec!["flow".into(), "propulsion".into(), "efficiency".into()],
            coherence_floor: 0.93,
            mercy_affinity: 0.95,
        },
    ]
}

pub struct BiomimeticPatternEngine;

impl BiomimeticPatternEngine {
    /// Select the best living pattern for the given themes under valence + mercy
    pub fn select_best_pattern(themes: &[String], valence: f64, mercy_weight: u8) -> BiomimeticPattern {
        let catalog = living_pattern_catalog();
        let mercy_factor = mercy_weight as f64 / 255.0;

        let mut best = catalog[0].clone();
        let mut best_score = 0.0;

        for pattern in catalog {
            let domain_hits = pattern
                .domains
                .iter()
                .filter(|d| themes.iter().any(|t| t.to_lowercase().contains(&d.to_lowercase()) || d.contains(&t.to_lowercase())))
                .count() as f64;

            let name_hit = if themes.iter().any(|t| pattern.name.contains(&t.to_lowercase()) || t.contains(&pattern.id)) {
                1.5
            } else {
                0.0
            };

            let score = (domain_hits * 1.4 + name_hit + pattern.coherence_floor * 2.0 + pattern.mercy_affinity * mercy_factor)
                * (0.7 + valence * 0.3);

            if score > best_score {
                best_score = score;
                best = pattern;
            }
        }

        best
    }

    /// Apply biomimetic patterns with full cross-pollination to every system
    pub async fn apply_pattern(
        pattern_name: &str,
        entangled_themes: &[String],
        base_valence: f64,
        mercy_weight: u8,
    ) -> f64 {
        let fenca_result = FENCA::verify_biomimetic_input(pattern_name, entangled_themes).await;
        if !fenca_result.is_verified() {
            return 0.0;
        }

        let mercy_scores = MercyEngine::evaluate_biomimetic_input(pattern_name, entangled_themes);
        let valence = ValenceFieldScoring::calculate(&mercy_scores) * base_valence;
        if !mercy_scores.all_gates_pass() {
            return 0.0;
        }

        // Resolve to a living structured pattern (or fall back to requested name)
        let pattern = living_pattern_catalog()
            .into_iter()
            .find(|p| p.name == pattern_name || p.id == pattern_name)
            .unwrap_or_else(|| Self::select_best_pattern(entangled_themes, valence, mercy_weight));

        let mercy_boost = (mercy_weight as f64 / 255.0) * 2.8;
        let biomimetic_coherence = (valence * mercy_boost * fenca_result.fidelity() * pattern.coherence_floor)
            .clamp(0.92, 1.0);

        // === CROSS-POLLINATION WITH INNOVATION GENERATOR + IDEA RECYCLER PATH ===
        let biomimetic_seed = format!(
            "Biomimetic {} ({}) applied to themes {:?} | nature: {} | domains: {:?}",
            pattern.name, pattern.id, entangled_themes, pattern.nature_source, pattern.domains
        );
        let recycled = vec![biomimetic_seed];

        if let Some(innovation) = InnovationGenerator::create_from_recycled(
            recycled,
            &mercy_scores,
            mercy_weight,
        )
        .await
        {
            RootCoreOrchestrator::delegate_innovation(innovation).await;
            // Cross-pollinate back into VQC synthesis
            let _ = VQCIntegrator::run_synthesis(entangled_themes, valence, mercy_weight).await;
            // Feed the lattice for eternal recursion
            SelfReviewLoop::trigger_immediate_review().await;
        }

        // Cache the pattern result
        let cache_key = GlobalCache::make_key(
            "biomimetic_pattern",
            &json!({"name": &pattern.name, "id": &pattern.id}),
        );
        let ttl = GlobalCache::adaptive_ttl(
            86400 * 30,
            fenca_result.fidelity(),
            valence,
            mercy_weight,
        );
        GlobalCache::set(
            &cache_key,
            json!({
                "coherence": biomimetic_coherence,
                "pattern": &pattern,
            }),
            ttl,
            mercy_weight,
            fenca_result.fidelity(),
            valence,
        );

        // Audit the cross-pollinated biomimetic application
        let _ = AuditLogger::log(
            "root",
            None,
            "biomimetic_pattern_cross_pollinated",
            &pattern.name,
            true,
            fenca_result.fidelity(),
            valence,
            vec![],
            json!({
                "coherence": biomimetic_coherence,
                "pattern_id": &pattern.id,
                "nature_source": &pattern.nature_source,
                "themes_count": entangled_themes.len(),
                "mercy_boost": mercy_boost
            }),
        )
        .await;

        biomimetic_coherence
    }

    /// High-level convenience: select best pattern then apply it
    pub async fn select_and_apply(
        entangled_themes: &[String],
        base_valence: f64,
        mercy_weight: u8,
    ) -> (BiomimeticPattern, f64) {
        let pattern = Self::select_best_pattern(entangled_themes, base_valence, mercy_weight);
        let coherence = Self::apply_pattern(&pattern.name, entangled_themes, base_valence, mercy_weight).await;
        (pattern, coherence)
    }
}
