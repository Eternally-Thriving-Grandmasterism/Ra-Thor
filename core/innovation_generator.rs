// core/innovation_generator.rs
// Innovation Generator — The creative engine of the Omnimaster Root Core
// Takes recycled ideas from /docs codices, applies FENCA + Mercy Engine, and generates new innovations
// Fully mercy-weighted, auditable, and delegatable

use crate::global_cache::GlobalCache;
use crate::fenca::FENCA;
use crate::mercy::MercyEngine;
use crate::valence::ValenceFieldScoring;
use crate::mercy_weighting::MercyWeighting;
use crate::audit_logger::AuditLogger;
use crate::root_core_orchestrator::RootCoreOrchestrator;
use serde_json::Value;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Innovation {
    pub id: String,
    pub target: Target,
    pub description: String,
    pub code_snippet: Option<String>,
    pub mercy_level: u8,
    pub valence_score: f64,
    pub created_at: u64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum Target {
    Quantum,
    Mercy,
    Access,
    Persistence,
    Orchestration,
    Cache,
    Kernel,
}

pub struct InnovationGenerator;

impl InnovationGenerator {
    /// Generate innovation from recycled ideas
    pub async fn create_from_recycled(
        recycled_ideas: Vec<String>,
        mercy_scores: &Vec<crate::mercy::GateScore>,
        mercy_weight: u8,
    ) -> Option<Innovation> {

        // 1. FENCA verification of recycled ideas
        let fenca_result = FENCA::verify_recycled_ideas(&recycled_ideas).await;
        if !fenca_result.is_verified() {
            return None;
        }

        // 2. Mercy Engine evaluation
        let valence = ValenceFieldScoring::calculate(mercy_scores);
        if !mercy_scores.all_gates_pass() {
            return None;
        }

        // 3. Generate innovation idea (creative synthesis)
        let innovation_description = synthesize_innovation(&recycled_ideas, valence, mercy_weight);

        // 4. Create innovation object
        let innovation = Innovation {
            id: format!("inn_{}", SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()),
            target: determine_target(&innovation_description),
            description: innovation_description,
            code_snippet: None, // can be filled later by generator
            mercy_level: mercy_weight,
            valence_score: valence,
            created_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        };

        // 5. Audit log the innovation generation
        let _ = AuditLogger::log(
            "root",
            None,
            "innovation_generated",
            &innovation.id,
            true,
            fenca_result.fidelity(),
            valence,
            vec![],
            serde_json::json!({"target": format!("{:?}", innovation.target)}),
        ).await;

        // 6. Cache the innovation
        let cache_key = GlobalCache::make_key("innovation", &json!({"id": &innovation.id}));
        let ttl = GlobalCache::adaptive_ttl(86400, fenca_result.fidelity(), valence, mercy_weight);
        GlobalCache::set(&cache_key, serde_json::to_value(&innovation).unwrap(), ttl, mercy_weight as u8, fenca_result.fidelity(), valence);

        Some(innovation)
    }

    /// Delegate the generated innovation back to Root Core
    pub async fn delegate(innovation: Innovation) {
        RootCoreOrchestrator::delegate_innovation(innovation).await;
    }
}

// Helper to synthesize new innovation from recycled ideas
fn synthesize_innovation(recycled: &[String], valence: f64, mercy_weight: u8) -> String {
    // Creative synthesis logic (can be expanded with VQC later)
    format!("New innovation synthesized from {} recycled ideas with valence {:.2} and mercy weight {}", 
            recycled.len(), valence, mercy_weight)
}

// Helper to determine target crate for delegation
fn determine_target(description: &str) -> Target {
    if description.contains("quantum") || description.contains("entanglement") {
        Target::Quantum
    } else if description.contains("mercy") || description.contains("valence") {
        Target::Mercy
    } else if description.contains("access") || description.contains("rebac") {
        Target::Access
    } else {
        Target::Kernel
    }
}
